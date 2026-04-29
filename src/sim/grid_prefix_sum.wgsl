// Step 3 of the grid build: exclusive prefix sum of cellCount → cellStart.
// Simultaneously resets cellCount to 0 so grid_scatter can reuse it as a
// per-cell write-offset counter.
//
// Single-threaded. For NUM_CELLS = 1000 that's ~1 µs — negligible compared to
// the O(N·k) solver passes that would benefit from a parallel prefix sum.
// If NUM_CELLS grows much past ~4096, swap this for a Hillis-Steele scan.
@group(0) @binding(0) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> cellStart: array<u32>;

@compute @workgroup_size(1)
fn main() {
    var sum: u32 = 0u;
    for (var i: u32 = 0u; i < NUM_CELLS; i = i + 1u) {
        let c = atomicLoad(&cellCount[i]);
        cellStart[i] = sum;
        sum = sum + c;
        atomicStore(&cellCount[i], 0u);
    }
}
