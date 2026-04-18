import type { SimParams } from "./params";

export class Simulation {
    readonly device: GPUDevice;
    readonly params: SimParams;
    readonly particleBuffer: GPUBuffer;

    constructor(device: GPUDevice, params: SimParams) {
        this.device = device;
        this.params = params;
        const PARTICLE_STRIDE_BYTES = 48; // size of Particle in common.wgsl
        this.particleBuffer = device.createBuffer({
            size: params.particleCount * PARTICLE_STRIDE_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        const init = new Float32Array(params.particleCount * 12);
        this.initializeParticles(init);
        device.queue.writeBuffer(this.particleBuffer, 0, init);
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
    }
}
