// Step 2 of the grid build: for each particle, increment the atomic counter in
// its cell. After this dispatch, cellCount[c] = number of particles in cell c.
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cellCount: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&particles)) { return; }
    let cell = cellLinearIdx(cellCoord(particles[i].predPos));
    atomicAdd(&cellCount[cell], 1u);
}
