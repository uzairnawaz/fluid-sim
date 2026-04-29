// Step 1 of the grid build: zero cellCount so grid_count can atomicAdd into it
// without accumulating values from the previous frame.
@group(0) @binding(0) var<storage, read_write> cellCount: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= NUM_CELLS) { return; }
    atomicStore(&cellCount[i], 0u);
}
