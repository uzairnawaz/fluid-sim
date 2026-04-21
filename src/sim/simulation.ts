import type { SimParams } from "./params";
import common from "../gpu/common.wgsl?raw";
import applyForceShader from "./apply_force.wgsl?raw";
import computeDensityShader from "./compute_density.wgsl?raw";

// SimParams struct in common.wgsl has 5 f32s (20 bytes); the uniform address
// space rounds struct size up to a multiple of 16, so allocate 32 bytes.
const SIM_PARAMS_UBO_BYTES = 32;

export class Simulation {
    readonly device: GPUDevice;
    readonly params: SimParams;
    readonly particleBuffer: GPUBuffer;
    readonly paramsBuffer: GPUBuffer;
    private applyForcePipeline: GPUComputePipeline;
    private densityPipeline: GPUComputePipeline;

    constructor(device: GPUDevice, params: SimParams) {
        this.device = device;
        this.params = params;
        const PARTICLE_STRIDE_BYTES = 48; // size of Particle in common.wgsl
        this.particleBuffer = device.createBuffer({
            label: "particles",
            size: params.particleCount * PARTICLE_STRIDE_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const init = new Float32Array(params.particleCount * 12);
        this.initializeParticles(init);
        this.device.queue.writeBuffer(this.particleBuffer, 0, init);

        this.paramsBuffer = device.createBuffer({
            label: "sim_params",
            size: SIM_PARAMS_UBO_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        // Layout must match SimParams in common.wgsl.
        const paramsData = new Float32Array(SIM_PARAMS_UBO_BYTES / 4);
        paramsData[0] = params.particleCount;
        paramsData[1] = params.h;
        paramsData[2] = params.restDensity;
        paramsData[3] = params.dt;
        paramsData[4] = params.solverIters;
        this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

        const applyForceModule = device.createShaderModule({
            label: "apply_force",
            code: `${common}\n${applyForceShader}`,
        });
        this.applyForcePipeline = device.createComputePipeline({
            label: "apply_force",
            layout: "auto",
            compute: { module: applyForceModule, entryPoint: "main" },
        });

        const densityModule = device.createShaderModule({
            label: "compute_density",
            code: `${common}\n${computeDensityShader}`,
        });
        this.densityPipeline = device.createComputePipeline({
            label: "compute_density",
            layout: "auto",
            compute: { module: densityModule, entryPoint: "main" },
        });
    }

    initializeParticles(particle_buf: Float32Array): void {
        for (let i = 0; i < this.params.particleCount; i++) {
            const o = i * 12;
            // pos in [-0.5, 0.5]^3
            particle_buf[o + 0] = Math.random() - 0.5;
            particle_buf[o + 1] = Math.random() - 0.5;
            particle_buf[o + 2] = Math.random() - 0.5;
            // vel in [-0.1, 0.1]^3
            particle_buf[o + 4] = (Math.random() - 0.5) * 0.2;
            particle_buf[o + 5] = (Math.random() - 0.5) * 0.2;
            particle_buf[o + 6] = (Math.random() - 0.5) * 0.2;
            // padding (3, 7) and predPos/lambda (8..11) stay 0
        }
    }

    step(): void {
        // TODO: predict → grid → solver iters → finalize
        const workgroups = Math.ceil(this.params.particleCount / 64);

        const densityBindGroup = this.device.createBindGroup({
            layout: this.densityPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.paramsBuffer } },
            ],
        });
        const forceBindGroup = this.device.createBindGroup({
            layout: this.applyForcePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
            ],
        });

        const encoder = this.device.createCommandEncoder({ label: "sim.step" });

        const densityPass = encoder.beginComputePass({ label: "compute_density" });
        densityPass.setPipeline(this.densityPipeline);
        densityPass.setBindGroup(0, densityBindGroup);
        densityPass.dispatchWorkgroups(workgroups);
        densityPass.end();

        const forcePass = encoder.beginComputePass({ label: "apply_force" });
        forcePass.setPipeline(this.applyForcePipeline);
        forcePass.setBindGroup(0, forceBindGroup);
        forcePass.dispatchWorkgroups(workgroups);
        forcePass.end();

        this.device.queue.submit([encoder.finish()]);
    }
}
