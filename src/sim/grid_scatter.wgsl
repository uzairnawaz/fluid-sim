// Step 4 of the grid build: each particle writes its own index into the
// sorted-by-cell array. Uses cellCount (which grid_prefix_sum zeroed) as a
// running per-cell offset counter.
//
// After this dispatch:
//   • sortedIdx[cellStart[c] .. cellStart[c] + cellCount[c]] lists the
//     particle indices in cell c.
//   • cellCount once again holds the actual per-cell populations, so sim
//     shaders can read it directly as array<u32>.
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> cellStart: array<u32>;
@group(0) @binding(3) var<storage, read_write> sortedIdx: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&particles)) { return; }
    let cell = cellLinearIdx(cellCoord(particles[i].predPos));
    let offset = atomicAdd(&cellCount[cell], 1u);
    sortedIdx[cellStart[cell] + offset] = i;
}
