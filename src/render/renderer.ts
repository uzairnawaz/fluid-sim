import common from "../gpu/common.wgsl?raw";
import particlesShader from "./particles.wgsl?raw";
import thicknessSplatShader from "./thickness_splat.wgsl?raw";
import fullscreenVs from "./fullscreen_vs.wgsl?raw";
import copyDepthFs from "./copy_depth.wgsl?raw";
import bilateralFs from "./bilateral.wgsl?raw";
import blurFs from "./blur.wgsl?raw";
import reconstructNormalFs from "./reconstruct_normal.wgsl?raw";
import showDepthFs from "./show_depth.wgsl?raw";
import type { Camera } from "./camera";

const CAMERA_UNIFORM_BYTES = 256; // 4 × mat4x4<f32> (view, proj, invProj, invView)
const DEPTH_FORMAT: GPUTextureFormat = "depth32float";
const SMOOTH_FORMAT: GPUTextureFormat = "r32float";
// r16float is blendable in WebGPU without any feature. Half-precision is
// plenty for per-particle Gaussian splats summed along the view ray.
const THICKNESS_FORMAT: GPUTextureFormat = "r16float";
// Normals in [-1, 1] → full-precision half-float. No feature required.
const NORMAL_FORMAT: GPUTextureFormat = "rgba16float";

// Bilateral filter tuning.
const SPATIAL_SIGMA_PX = 8.0; // pixel-space Gaussian σ
// Depth values cluster tightly in roughly [0.993, 0.997] after perspective
// (camera dist ≈ 2, particles in a ~1 m cube). σ much below the full-surface
// range so the filter smooths within-surface jitter but keeps silhouettes.
const DEPTH_SIGMA = 0.0008;

export class Renderer {
  readonly device: GPUDevice;
  readonly format: GPUTextureFormat;

  // Pipelines.
  private particlesPipeline: GPURenderPipeline;
  private thicknessPipeline: GPURenderPipeline;
  private copyDepthPipeline: GPURenderPipeline;
  private bilateralPipeline: GPURenderPipeline;
  private blurPipeline: GPURenderPipeline;
  private reconstructNormalPipeline: GPURenderPipeline;
  private showDepthPipeline: GPURenderPipeline;

  // Stable uniform buffers.
  private cameraBuffer: GPUBuffer;
  private bilateralHBuffer: GPUBuffer;
  private bilateralVBuffer: GPUBuffer;

  // Size-dependent textures recreated on resize. TEXTURE_BINDING is here so
  // later Phase 7 passes (normal reconstruction, composite) can sample them.
  private depthTexture: GPUTexture | null = null;
  private depthView: GPUTextureView | null = null;
  private smoothedA: GPUTexture | null = null;
  private smoothedAView: GPUTextureView | null = null;
  private smoothedB: GPUTexture | null = null;
  private smoothedBView: GPUTextureView | null = null;
  private thicknessTexture: GPUTexture | null = null;
  private thicknessView: GPUTextureView | null = null;
  private thicknessBlurTexture: GPUTexture | null = null;
  private thicknessBlurView: GPUTextureView | null = null;
  private normalTexture: GPUTexture | null = null;
  private normalView: GPUTextureView | null = null;
  private width = 0;
  private height = 0;

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.format = format;

    // ---- Particle sphere-imposter pass (from step 7.1). ----
    const particlesModule = device.createShaderModule({
      label: "particles",
      code: `${common}\n${particlesShader}`,
    });
    this.particlesPipeline = device.createRenderPipeline({
      label: "particles",
      layout: "auto",
      vertex: { module: particlesModule, entryPoint: "vs_main" },
      fragment: {
        module: particlesModule,
        entryPoint: "fs_main",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list" },
      depthStencil: {
        format: DEPTH_FORMAT,
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });

    // ---- Thickness splat pass (step 7.3). ----
    // No depth test/write — total water along the ray, not just the nearest
    // surface — with additive blending summing per-particle Gaussians.
    const thicknessModule = device.createShaderModule({
      label: "thickness_splat",
      code: `${common}\n${thicknessSplatShader}`,
    });
    this.thicknessPipeline = device.createRenderPipeline({
      label: "thickness_splat",
      layout: "auto",
      vertex: { module: thicknessModule, entryPoint: "vs_main" },
      fragment: {
        module: thicknessModule,
        entryPoint: "fs_main",
        targets: [
          {
            format: THICKNESS_FORMAT,
            blend: {
              color: { srcFactor: "one", dstFactor: "one", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list" },
    });

    // ---- Fullscreen post-process passes (step 7.2). ----
    const mkFullscreenPipeline = (
      label: string,
      fsCode: string,
      targetFormat: GPUTextureFormat,
    ): GPURenderPipeline => {
      const module = device.createShaderModule({
        label,
        code: `${fullscreenVs}\n${fsCode}`,
      });
      return device.createRenderPipeline({
        label,
        layout: "auto",
        vertex: { module, entryPoint: "vs_main" },
        fragment: {
          module,
          entryPoint: "fs_main",
          targets: [{ format: targetFormat }],
        },
        primitive: { topology: "triangle-list" },
      });
    };
    this.copyDepthPipeline = mkFullscreenPipeline(
      "copy_depth",
      copyDepthFs,
      SMOOTH_FORMAT,
    );
    this.bilateralPipeline = mkFullscreenPipeline(
      "bilateral",
      bilateralFs,
      SMOOTH_FORMAT,
    );
    this.blurPipeline = mkFullscreenPipeline(
      "blur",
      blurFs,
      THICKNESS_FORMAT,
    );
    this.reconstructNormalPipeline = mkFullscreenPipeline(
      "reconstruct_normal",
      reconstructNormalFs,
      NORMAL_FORMAT,
    );
    this.showDepthPipeline = mkFullscreenPipeline(
      "show_depth",
      showDepthFs,
      format,
    );

    this.cameraBuffer = device.createBuffer({
      label: "camera",
      size: CAMERA_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Bilateral uniforms: vec2f direction + 2×f32 sigmas = 16 B (aligned).
    const mkBilateralBuffer = (
      label: string,
      dx: number,
      dy: number,
    ): GPUBuffer => {
      const buf = device.createBuffer({
        label,
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(
        buf,
        0,
        new Float32Array([dx, dy, SPATIAL_SIGMA_PX, DEPTH_SIGMA]),
      );
      return buf;
    };
    this.bilateralHBuffer = mkBilateralBuffer("bilateral_h", 1, 0);
    this.bilateralVBuffer = mkBilateralBuffer("bilateral_v", 0, 1);
  }

  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;

    this.depthTexture?.destroy();
    this.smoothedA?.destroy();
    this.smoothedB?.destroy();
    this.thicknessTexture?.destroy();
    this.thicknessBlurTexture?.destroy();
    this.normalTexture?.destroy();

    this.depthTexture = this.device.createTexture({
      label: "depth",
      size: { width, height },
      format: DEPTH_FORMAT,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.depthView = this.depthTexture.createView();

    const mkSmoothedTex = (label: string): GPUTexture =>
      this.device.createTexture({
        label,
        size: { width, height },
        format: SMOOTH_FORMAT,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
    this.smoothedA = mkSmoothedTex("smoothed_a");
    this.smoothedAView = this.smoothedA.createView();
    this.smoothedB = mkSmoothedTex("smoothed_b");
    this.smoothedBView = this.smoothedB.createView();

    this.thicknessTexture = this.device.createTexture({
      label: "thickness",
      size: { width, height },
      format: THICKNESS_FORMAT,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.thicknessView = this.thicknessTexture.createView();
    this.thicknessBlurTexture = this.device.createTexture({
      label: "thickness_blur",
      size: { width, height },
      format: THICKNESS_FORMAT,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.thicknessBlurView = this.thicknessBlurTexture.createView();

    this.normalTexture = this.device.createTexture({
      label: "normal",
      size: { width, height },
      format: NORMAL_FORMAT,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.normalView = this.normalTexture.createView();
  }

  render(
    view: GPUTextureView,
    particleBuffer: GPUBuffer,
    particleCount: number,
    camera: Camera,
  ): void {
    if (
      !this.depthView ||
      !this.smoothedAView ||
      !this.smoothedBView ||
      !this.thicknessView ||
      !this.thicknessBlurView ||
      !this.normalView
    ) {
      throw new Error("Renderer.render called before resize");
    }

    // Camera uniform: view, proj, invProj, invView (256 bytes).
    const cam = new Float32Array(64);
    cam.set(camera.view, 0);
    cam.set(camera.proj, 16);
    cam.set(camera.invProj, 32);
    cam.set(camera.invView, 48);
    this.device.queue.writeBuffer(
      this.cameraBuffer,
      0,
      cam as unknown as GPUAllowSharedBufferSource,
    );

    // ---- Bind groups. ----
    const particlesBG = this.device.createBindGroup({
      layout: this.particlesPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: particleBuffer } },
        { binding: 1, resource: { buffer: this.cameraBuffer } },
      ],
    });
    const thicknessBG = this.device.createBindGroup({
      layout: this.thicknessPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: particleBuffer } },
        { binding: 1, resource: { buffer: this.cameraBuffer } },
      ],
    });
    const copyDepthBG = this.device.createBindGroup({
      layout: this.copyDepthPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: this.depthView }],
    });
    // Horizontal: read smoothedA → write smoothedB.
    const bilateralHBG = this.device.createBindGroup({
      layout: this.bilateralPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.smoothedAView },
        { binding: 1, resource: { buffer: this.bilateralHBuffer } },
      ],
    });
    // Vertical: read smoothedB → write smoothedA.
    const bilateralVBG = this.device.createBindGroup({
      layout: this.bilateralPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.smoothedBView },
        { binding: 1, resource: { buffer: this.bilateralVBuffer } },
      ],
    });
    // Thickness blur: H reads the raw-accumulated thickness texture and
    // writes to the blur scratch; V reads back from scratch and writes
    // into the thickness texture in-place, so downstream (show_depth)
    // samples the smoothed result without any extra binding change.
    const blurHBG = this.device.createBindGroup({
      layout: this.blurPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.thicknessView },
        { binding: 1, resource: { buffer: this.bilateralHBuffer } },
      ],
    });
    const blurVBG = this.device.createBindGroup({
      layout: this.blurPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.thicknessBlurView },
        { binding: 1, resource: { buffer: this.bilateralVBuffer } },
      ],
    });

    const reconstructNormalBG = this.device.createBindGroup({
      layout: this.reconstructNormalPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.smoothedAView },
        { binding: 1, resource: { buffer: this.cameraBuffer } },
      ],
    });
    const showDepthBG = this.device.createBindGroup({
      layout: this.showDepthPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.smoothedAView },
        { binding: 1, resource: this.thicknessView },
        { binding: 2, resource: this.normalView },
        { binding: 3, resource: { buffer: this.cameraBuffer } },
      ],
    });

    const encoder = this.device.createCommandEncoder({ label: "render" });

    // 1. Particle sphere imposters: color → canvas, depth → depthTexture.
    {
      const pass = encoder.beginRenderPass({
        label: "particles",
        colorAttachments: [
          {
            view,
            clearValue: { r: 0.05, g: 0.06, b: 0.08, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
        depthStencilAttachment: {
          view: this.depthView,
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });
      pass.setPipeline(this.particlesPipeline);
      pass.setBindGroup(0, particlesBG);
      pass.draw(6, particleCount);
      pass.end();
    }

    // 1.5. Thickness splat: additive Gaussian per particle, no depth test.
    {
      const pass = encoder.beginRenderPass({
        label: "thickness",
        colorAttachments: [
          {
            view: this.thicknessView,
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
          },
        ],
      });
      pass.setPipeline(this.thicknessPipeline);
      pass.setBindGroup(0, thicknessBG);
      pass.draw(6, particleCount);
      pass.end();
    }

    // 1.75. Blur the thickness texture separably (H then V). Non-bilateral —
    //       thickness is continuous, we want it smooth across the whole
    //       fluid silhouette so Beer's-law color varies smoothly. V writes
    //       back into `thickness` so show_depth sees the smoothed result.
    {
      const pass = encoder.beginRenderPass({
        label: "thickness_blur_h",
        colorAttachments: [
          {
            view: this.thicknessBlurView,
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
          },
        ],
      });
      pass.setPipeline(this.blurPipeline);
      pass.setBindGroup(0, blurHBG);
      pass.draw(3);
      pass.end();
    }
    {
      const pass = encoder.beginRenderPass({
        label: "thickness_blur_v",
        colorAttachments: [
          {
            view: this.thicknessView,
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
          },
        ],
      });
      pass.setPipeline(this.blurPipeline);
      pass.setBindGroup(0, blurVBG);
      pass.draw(3);
      pass.end();
    }

    // 2. Copy depth32float → r32float (so the filter can sample it as a
    //    regular texture_2d<f32>).
    {
      const pass = encoder.beginRenderPass({
        label: "copy_depth",
        colorAttachments: [
          {
            view: this.smoothedAView,
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 1, g: 0, b: 0, a: 0 },
          },
        ],
      });
      pass.setPipeline(this.copyDepthPipeline);
      pass.setBindGroup(0, copyDepthBG);
      pass.draw(3);
      pass.end();
    }

    // 3. Bilateral horizontal: smoothedA → smoothedB.
    {
      const pass = encoder.beginRenderPass({
        label: "bilateral_h",
        colorAttachments: [
          {
            view: this.smoothedBView,
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 1, g: 0, b: 0, a: 0 },
          },
        ],
      });
      pass.setPipeline(this.bilateralPipeline);
      pass.setBindGroup(0, bilateralHBG);
      pass.draw(3);
      pass.end();
    }

    // 4. Bilateral vertical: smoothedB → smoothedA.
    {
      const pass = encoder.beginRenderPass({
        label: "bilateral_v",
        colorAttachments: [
          {
            view: this.smoothedAView,
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 1, g: 0, b: 0, a: 0 },
          },
        ],
      });
      pass.setPipeline(this.bilateralPipeline);
      pass.setBindGroup(0, bilateralVBG);
      pass.draw(3);
      pass.end();
    }

    // 4.5. Reconstruct view-space normals from smoothed depth.
    {
      const pass = encoder.beginRenderPass({
        label: "reconstruct_normal",
        colorAttachments: [
          {
            view: this.normalView,
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
          },
        ],
      });
      pass.setPipeline(this.reconstructNormalPipeline);
      pass.setBindGroup(0, reconstructNormalBG);
      pass.draw(3);
      pass.end();
    }

    // 5. Composite (step 7.5): Fresnel-blended reflection/refraction against
    //    a procedural skybox, Beer-Lambert through thickness, direct specular.
    //    Covers the whole frame, so loadOp "clear" — overwrites the particle
    //    pass's canvas color (only its depth output is still used by 7.2).
    {
      const pass = encoder.beginRenderPass({
        label: "composite",
        colorAttachments: [
          {
            view,
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          },
        ],
      });
      pass.setPipeline(this.showDepthPipeline);
      pass.setBindGroup(0, showDepthBG);
      pass.draw(3);
      pass.end();
    }

    this.device.queue.submit([encoder.finish()]);
  }
}
