// Step 3 of the grid build: exclusive prefix sum of cellCount → cellStart.
// Simultaneously resets cellCount to 0 so grid_scatter can reuse it as a
// per-cell write-offset counter.
//
// Single-workgroup parallel scan: 256 threads, each handling
// ITEMS_PER_THREAD = 4 cells, so the workgroup covers 1024 ≥ NUM_CELLS = 1000.
// Threads whose cells fall past NUM_CELLS contribute 0 and write nothing.
// Three phases, separated by workgroup barriers:
//   1. Each thread serially exclusive-scans its own 4 cells, then reports
//      its total to a 256-entry workgroup-memory array `blockTotals`.
//   2. Hillis-Steele inclusive scan of `blockTotals` (log₂(256) = 8 passes).
//   3. Each thread reads its block's prefix offset (= blockTotals[tid - 1]
//      after the inclusive scan), adds it to its 4 local exclusive sums,
//      and writes the 4 final values to cellStart. Resets cellCount.
//
// If NUM_CELLS ever exceeds 1024, bump ITEMS_PER_THREAD (8 → up to 2048)
// or switch to a multi-block scan with carry propagation.

@group(0) @binding(0) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> cellStart: array<u32>;

const THREADS:          u32 = 256u;
const ITEMS_PER_THREAD: u32 = 4u;

var<workgroup> blockTotals: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let baseIdx = tid * ITEMS_PER_THREAD;

    // ---- Phase 1: serial exclusive scan of this thread's 4 cells. ----
    var local: array<u32, 4>;
    var localSum: u32 = 0u;
    for (var i: u32 = 0u; i < ITEMS_PER_THREAD; i = i + 1u) {
        let idx = baseIdx + i;
        var v: u32 = 0u;
        if (idx < NUM_CELLS) {
            v = atomicLoad(&cellCount[idx]);
        }
        local[i] = localSum;        // exclusive: stash running sum BEFORE add
        localSum = localSum + v;
    }
    blockTotals[tid] = localSum;
    workgroupBarrier();

    // ---- Phase 2: Hillis-Steele inclusive scan of blockTotals. ----
    // After the loop, blockTotals[tid] holds the inclusive prefix sum of the
    // first (tid + 1) per-thread totals.
    for (var step: u32 = 1u; step < THREADS; step = step << 1u) {
        var prev: u32 = 0u;
        if (tid >= step) {
            prev = blockTotals[tid - step];
        }
        workgroupBarrier();         // all reads finish before any writes
        if (tid >= step) {
            blockTotals[tid] = blockTotals[tid] + prev;
        }
        workgroupBarrier();         // all writes finish before next iter's reads
    }

    // Convert inclusive scan → exclusive block offset for this thread.
    var blockOffset: u32 = 0u;
    if (tid > 0u) {
        blockOffset = blockTotals[tid - 1u];
    }

    // ---- Phase 3: write cellStart values and reset cellCount. ----
    for (var i: u32 = 0u; i < ITEMS_PER_THREAD; i = i + 1u) {
        let idx = baseIdx + i;
        if (idx < NUM_CELLS) {
            cellStart[idx] = blockOffset + local[i];
            atomicStore(&cellCount[idx], 0u);
        }
    }
}
