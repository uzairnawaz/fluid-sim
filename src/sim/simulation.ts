import type { SimParams } from "./params";
import common from "../gpu/common.wgsl?raw";
import grid from "./grid.wgsl?raw";
import predictShader from "./predict.wgsl?raw";
import computeDensityShader from "./compute_density.wgsl?raw";
import computeLambdaShader from "./compute_lambda.wgsl?raw";
import applyCorrectionShader from "./apply_correction.wgsl?raw";
import updateVelocityShader from "./update_velocity.wgsl?raw";
import xsphShader from "./xsph.wgsl?raw";
import commitShader from "./commit.wgsl?raw";
import gridClearShader from "./grid_clear.wgsl?raw";
import gridCountShader from "./grid_count.wgsl?raw";
import gridPrefixSumShader from "./grid_prefix_sum.wgsl?raw";
import gridScatterShader from "./grid_scatter.wgsl?raw";

// SimParams struct in common.wgsl has 5 f32s (20 bytes); uniform address-space
// rounding bumps the struct size to a multiple of 16, so allocate 32 bytes.
const SIM_PARAMS_UBO_BYTES = 32;
const PARTICLE_STRIDE_BYTES = 48;
const WORKGROUP_SIZE = 64;

// Must stay in lockstep with the constants at the top of grid.wgsl. 10 cells
// per axis × h = 0.1 → covers the [-0.5, 0.5]³ box exactly.
const NUM_CELLS = 1000;

export class Simulation {
    readonly device: GPUDevice;
    readonly params: SimParams;
    readonly particleBuffer: GPUBuffer;
    readonly paramsBuffer: GPUBuffer;

    // Grid buffers. cellCount is repurposed across the grid build (scratch
    // counter during scatter, final per-cell populations afterwards).
    private cellCountBuffer: GPUBuffer;
    private cellStartBuffer: GPUBuffer;
    private sortedIdxBuffer: GPUBuffer;

    private predictPipeline: GPUComputePipeline;
    private densityPipeline: GPUComputePipeline;
    private lambdaPipeline: GPUComputePipeline;
    private correctionPipeline: GPUComputePipeline;
    private updateVelocityPipeline: GPUComputePipeline;
    private xsphPipeline: GPUComputePipeline;
    private commitPipeline: GPUComputePipeline;
    private gridClearPipeline: GPUComputePipeline;
    private gridCountPipeline: GPUComputePipeline;
    private gridPrefixSumPipeline: GPUComputePipeline;
    private gridScatterPipeline: GPUComputePipeline;

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

        this.cellCountBuffer = device.createBuffer({
            label: "cell_count",
            size: NUM_CELLS * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.cellStartBuffer = device.createBuffer({
            label: "cell_start",
            size: NUM_CELLS * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.sortedIdxBuffer = device.createBuffer({
            label: "sorted_idx",
            size: params.particleCount * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        // Shaders that touch the grid get grid.wgsl prepended after common.wgsl
        // so the constants + cellCoord/cellLinearIdx helpers are in scope.
        const mkCompute = (
            label: string,
            code: string,
            includeGrid: boolean,
        ): GPUComputePipeline => {
            const prelude = includeGrid ? `${common}\n${grid}` : common;
            const module = device.createShaderModule({
                label,
                code: `${prelude}\n${code}`,
            });
            return device.createComputePipeline({
                label,
                layout: "auto",
                compute: { module, entryPoint: "main" },
            });
        };
        this.predictPipeline = mkCompute("predict", predictShader, false);
        this.updateVelocityPipeline = mkCompute(
            "update_velocity",
            updateVelocityShader,
            false,
        );
        this.commitPipeline = mkCompute("commit", commitShader, false);

        this.densityPipeline = mkCompute(
            "compute_density",
            computeDensityShader,
            true,
        );
        this.lambdaPipeline = mkCompute(
            "compute_lambda",
            computeLambdaShader,
            true,
        );
        this.correctionPipeline = mkCompute(
            "apply_correction",
            applyCorrectionShader,
            true,
        );
        this.xsphPipeline = mkCompute("xsph", xsphShader, true);

        this.gridClearPipeline = mkCompute("grid_clear", gridClearShader, true);
        this.gridCountPipeline = mkCompute("grid_count", gridCountShader, true);
        this.gridPrefixSumPipeline = mkCompute(
            "grid_prefix_sum",
            gridPrefixSumShader,
            true,
        );
        this.gridScatterPipeline = mkCompute(
            "grid_scatter",
            gridScatterShader,
            true,
        );
    }

    initializeParticles(particle_buf: Float32Array): void {
        // Dam-break initial condition: pack all particles into a slab on the
        // −x end of the box (x ∈ [−0.5, −0.1], y/z full range). Under gravity
        // + constraint solve they collapse and spread across the floor.
        // Slab volume (0.4 m³) is matched to the fluid's natural rest volume
        // (particleCount / restDensity ≈ 0.33 m³) — slightly over-packed so
        // release produces visible motion.
        for (let i = 0; i < this.params.particleCount; i++) {
            const o = i * 12;
            particle_buf[o + 0] = -0.5 + Math.random() * 0.4;
            particle_buf[o + 1] = Math.random() - 0.5;
            particle_buf[o + 2] = Math.random() - 0.5;
            // velocity (4..6) and all other fields stay 0 — dam-break starts
            // at rest.
        }
    }

    step(): void {
        const particleWG = Math.ceil(
            this.params.particleCount / WORKGROUP_SIZE,
        );
        const cellWG = Math.ceil(NUM_CELLS / WORKGROUP_SIZE);

        // Reusable entries.
        const particles = {
            binding: 0,
            resource: { buffer: this.particleBuffer },
        };
        const paramsUBO = {
            binding: 1,
            resource: { buffer: this.paramsBuffer },
        };
        const cellStartR = {
            binding: 2,
            resource: { buffer: this.cellStartBuffer },
        };
        const cellCountR = {
            binding: 3,
            resource: { buffer: this.cellCountBuffer },
        };
        const sortedIdxR = {
            binding: 4,
            resource: { buffer: this.sortedIdxBuffer },
        };

        // Sim-time bind groups.
        const predictBG = this.device.createBindGroup({
            layout: this.predictPipeline.getBindGroupLayout(0),
            entries: [particles, paramsUBO],
        });
        const gridQueryEntries = [
            particles,
            paramsUBO,
            cellStartR,
            cellCountR,
            sortedIdxR,
        ];
        const densityBG = this.device.createBindGroup({
            layout: this.densityPipeline.getBindGroupLayout(0),
            entries: gridQueryEntries,
        });
        const lambdaBG = this.device.createBindGroup({
            layout: this.lambdaPipeline.getBindGroupLayout(0),
            entries: gridQueryEntries,
        });
        const correctionBG = this.device.createBindGroup({
            layout: this.correctionPipeline.getBindGroupLayout(0),
            entries: gridQueryEntries,
        });
        const xsphBG = this.device.createBindGroup({
            layout: this.xsphPipeline.getBindGroupLayout(0),
            entries: gridQueryEntries,
        });
        const updateVelBG = this.device.createBindGroup({
            layout: this.updateVelocityPipeline.getBindGroupLayout(0),
            entries: [particles, paramsUBO],
        });
        const commitBG = this.device.createBindGroup({
            layout: this.commitPipeline.getBindGroupLayout(0),
            entries: [particles],
        });

        // Grid-build bind groups (different layouts per pipeline — bindings
        // below match the declarations in grid_*.wgsl one-for-one).
        const gridClearBG = this.device.createBindGroup({
            layout: this.gridClearPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.cellCountBuffer } },
            ],
        });
        const gridCountBG = this.device.createBindGroup({
            layout: this.gridCountPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.cellCountBuffer } },
            ],
        });
        const gridPrefixSumBG = this.device.createBindGroup({
            layout: this.gridPrefixSumPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.cellCountBuffer } },
                { binding: 1, resource: { buffer: this.cellStartBuffer } },
            ],
        });
        const gridScatterBG = this.device.createBindGroup({
            layout: this.gridScatterPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.cellCountBuffer } },
                { binding: 2, resource: { buffer: this.cellStartBuffer } },
                { binding: 3, resource: { buffer: this.sortedIdxBuffer } },
            ],
        });

        const encoder = this.device.createCommandEncoder({ label: "sim.step" });

        const dispatch = (
            pipeline: GPUComputePipeline,
            bg: GPUBindGroup,
            label: string,
            workgroups: number,
        ) => {
            const pass = encoder.beginComputePass({ label });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(workgroups);
            pass.end();
        };

        // 1. Apply gravity, predict position.
        dispatch(this.predictPipeline, predictBG, "predict", particleWG);

        // 2. Build neighbor grid from predPos.
        dispatch(this.gridClearPipeline, gridClearBG, "grid_clear", cellWG);
        dispatch(this.gridCountPipeline, gridCountBG, "grid_count", particleWG);
        dispatch(
            this.gridPrefixSumPipeline,
            gridPrefixSumBG,
            "grid_prefix_sum",
            1,
        );
        dispatch(
            this.gridScatterPipeline,
            gridScatterBG,
            "grid_scatter",
            particleWG,
        );

        // 3. Constraint iterations: density → lambda → correction.
        //    Grid is not rebuilt between iters — correction-induced cell
        //    migrations are small and the error washes out over the iterations.
        const iters = this.params.solverIters | 0;
        for (let iter = 0; iter < iters; iter++) {
            dispatch(
                this.densityPipeline,
                densityBG,
                `density[${iter}]`,
                particleWG,
            );
            dispatch(
                this.lambdaPipeline,
                lambdaBG,
                `lambda[${iter}]`,
                particleWG,
            );
            dispatch(
                this.correctionPipeline,
                correctionBG,
                `correction[${iter}]`,
                particleWG,
            );
        }

        // 4. v ← (p* − p) / dt   (must precede commit — reads the old p).
        dispatch(
            this.updateVelocityPipeline,
            updateVelBG,
            "update_velocity",
            particleWG,
        );

        // 4.5. XSPH viscosity: damp velocity toward neighbor average.
        dispatch(this.xsphPipeline, xsphBG, "xsph", particleWG);

        // 5. p ← p*.
        dispatch(this.commitPipeline, commitBG, "commit", particleWG);

        this.device.queue.submit([encoder.finish()]);
    }
}
