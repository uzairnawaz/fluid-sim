import type { SimParams } from "./params";
import common from "../gpu/common.wgsl?raw";
import applyForceShader from "./apply_force.wgsl?raw";

export class Simulation {
    readonly device: GPUDevice;
    readonly params: SimParams;
    readonly particleBuffer: GPUBuffer;
    private applyForcePipeline: GPUComputePipeline;

    constructor(device: GPUDevice, params: SimParams) {
        this.device = device;
        this.params = params;
        const PARTICLE_STRIDE_BYTES = 48; // size of Particle in common.wgsl
        this.particleBuffer = device.createBuffer({
            size: params.particleCount * PARTICLE_STRIDE_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const init = new Float32Array(params.particleCount * 12);
        this.initializeParticles(init);
        this.device.queue.writeBuffer(this.particleBuffer, 0, init);

        const module = device.createShaderModule({
            label: "apply_force",
            code: `${common}\n${applyForceShader}`,
        });
        this.applyForcePipeline = device.createComputePipeline({
            label: "apply_force",
            layout: "auto",
            compute: { module, entryPoint: "main" },
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
        const bindGroup = this.device.createBindGroup({
            layout: this.applyForcePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
            ],
        });
        const encoder = this.device.createCommandEncoder({ label: "sim.step" });
        const pass = encoder.beginComputePass({ label: "apply_force" });
        pass.setPipeline(this.applyForcePipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.params.particleCount / 64));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }
}
