import type { SimParams } from "./params";
import common from "../gpu/common.wgsl?raw";
import predictShader from "./predict.wgsl?raw";
import computeDensityShader from "./compute_density.wgsl?raw";
import computeLambdaShader from "./compute_lambda.wgsl?raw";
import applyCorrectionShader from "./apply_correction.wgsl?raw";
import updateVelocityShader from "./update_velocity.wgsl?raw";
import xsphShader from "./xsph.wgsl?raw";
import commitShader from "./commit.wgsl?raw";

// SimParams struct in common.wgsl has 5 f32s (20 bytes); uniform address-space
// rounding bumps the struct size to a multiple of 16, so allocate 32 bytes.
const SIM_PARAMS_UBO_BYTES = 32;
const PARTICLE_STRIDE_BYTES = 48;
const WORKGROUP_SIZE = 64;

export class Simulation {
    readonly device: GPUDevice;
    readonly params: SimParams;
    readonly particleBuffer: GPUBuffer;
    readonly paramsBuffer: GPUBuffer;

    private predictPipeline: GPUComputePipeline;
    private densityPipeline: GPUComputePipeline;
    private lambdaPipeline: GPUComputePipeline;
    private correctionPipeline: GPUComputePipeline;
    private updateVelocityPipeline: GPUComputePipeline;
    private xsphPipeline: GPUComputePipeline;
    private commitPipeline: GPUComputePipeline;

    constructor(device: GPUDevice, params: SimParams) {
        this.device = device;
        this.params = params;

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

        const mkCompute = (label: string, code: string): GPUComputePipeline => {
            const module = device.createShaderModule({
                label,
                code: `${common}\n${code}`,
            });
            return device.createComputePipeline({
                label,
                layout: "auto",
                compute: { module, entryPoint: "main" },
            });
        };
        this.predictPipeline = mkCompute("predict", predictShader);
        this.densityPipeline = mkCompute(
            "compute_density",
            computeDensityShader,
        );
        this.lambdaPipeline = mkCompute("compute_lambda", computeLambdaShader);
        this.correctionPipeline = mkCompute(
            "apply_correction",
            applyCorrectionShader,
        );
        this.updateVelocityPipeline = mkCompute(
            "update_velocity",
            updateVelocityShader,
        );
        this.xsphPipeline = mkCompute("xsph", xsphShader);
        this.commitPipeline = mkCompute("commit", commitShader);
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
            // density (3), _pad1 (7), predPos + lambda (8..11) stay 0
        }
    }

    step(): void {
        const workgroups = Math.ceil(
            this.params.particleCount / WORKGROUP_SIZE,
        );

        // Bind groups. Resources are stable, so these would be safe to cache —
        // recreated per frame for symmetry with the renderer's convention.
        const particles = {
            binding: 0,
            resource: { buffer: this.particleBuffer },
        };
        const paramsUBO = {
            binding: 1,
            resource: { buffer: this.paramsBuffer },
        };

        const predictBG = this.device.createBindGroup({
            layout: this.predictPipeline.getBindGroupLayout(0),
            entries: [particles, paramsUBO],
        });
        const densityBG = this.device.createBindGroup({
            layout: this.densityPipeline.getBindGroupLayout(0),
            entries: [particles, paramsUBO],
        });
        const lambdaBG = this.device.createBindGroup({
            layout: this.lambdaPipeline.getBindGroupLayout(0),
            entries: [particles, paramsUBO],
        });
        const correctionBG = this.device.createBindGroup({
            layout: this.correctionPipeline.getBindGroupLayout(0),
            entries: [particles, paramsUBO],
        });
        const updateVelBG = this.device.createBindGroup({
            layout: this.updateVelocityPipeline.getBindGroupLayout(0),
            entries: [particles, paramsUBO],
        });
        const xsphBG = this.device.createBindGroup({
            layout: this.xsphPipeline.getBindGroupLayout(0),
            entries: [particles, paramsUBO],
        });
        const commitBG = this.device.createBindGroup({
            layout: this.commitPipeline.getBindGroupLayout(0),
            entries: [particles],
        });

        const encoder = this.device.createCommandEncoder({ label: "sim.step" });

        const dispatch = (
            pipeline: GPUComputePipeline,
            bg: GPUBindGroup,
            label: string,
        ) => {
            const pass = encoder.beginComputePass({ label });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(workgroups);
            pass.end();
        };

        // 1. Apply gravity, predict position.
        dispatch(this.predictPipeline, predictBG, "predict");

        // 2. (Neighbor build goes here once we have one.)
        // 3. Constraint iterations: density → lambda → correction.
        const iters = this.params.solverIters | 0;
        for (let iter = 0; iter < iters; iter++) {
            dispatch(this.densityPipeline, densityBG, `density[${iter}]`);
            dispatch(this.lambdaPipeline, lambdaBG, `lambda[${iter}]`);
            dispatch(
                this.correctionPipeline,
                correctionBG,
                `correction[${iter}]`,
            );
        }

        // 4. v ← (p* − p) / dt   (must precede commit — reads the old p).
        dispatch(this.updateVelocityPipeline, updateVelBG, "update_velocity");

        // 4.5. XSPH viscosity: damp velocity toward neighbor average.
        dispatch(this.xsphPipeline, xsphBG, "xsph");

        // 5. p ← p*.
        dispatch(this.commitPipeline, commitBG, "commit");

        this.device.queue.submit([encoder.finish()]);
    }
}
