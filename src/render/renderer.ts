import common from "../gpu/common.wgsl?raw";
import particlesShader from "./particles.wgsl?raw";
import type { Camera } from "./camera";

const CAMERA_UNIFORM_BYTES = 128; // 2 × mat4x4<f32>

export class Renderer {
  readonly device: GPUDevice;
  readonly format: GPUTextureFormat;
  private pipeline: GPURenderPipeline;
  private cameraBuffer: GPUBuffer;

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.format = format;

    const module = device.createShaderModule({
      label: "particles",
      code: `${common}\n${particlesShader}`,
    });
    this.pipeline = device.createRenderPipeline({
      label: "particles",
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });

    this.cameraBuffer = device.createBuffer({
      label: "camera",
      size: CAMERA_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  render(
    view: GPUTextureView,
    particleBuffer: GPUBuffer,
    particleCount: number,
    camera: Camera,
  ): void {
    const cam = new Float32Array(32);
    cam.set(camera.view, 0);
    cam.set(camera.proj, 16);
    this.device.queue.writeBuffer(
      this.cameraBuffer,
      0,
      cam as unknown as GPUAllowSharedBufferSource,
    );

    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: particleBuffer } },
        { binding: 1, resource: { buffer: this.cameraBuffer } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          clearValue: { r: 0.05, g: 0.06, b: 0.08, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6, particleCount);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}
