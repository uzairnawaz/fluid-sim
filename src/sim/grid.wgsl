// Uniform spatial hash grid for neighbor queries. Prepended (after common.wgsl)
// to every compute shader that builds or queries the grid.
//
// Cell size = h = 0.1 (= SPH smoothing radius) so that the 27-cell neighborhood
// (3×3×3 around a query cell) covers every particle within h of any point in
// the query cell. Grid anchored to the box min and sized to cover the whole
// [-0.5, 0.5]³ simulation domain. If either h or the box changes, the
// constants below AND the buffer sizes in simulation.ts must move in lockstep.

const GRID_MIN: vec3f = vec3f(-0.5);
const GRID_CELL_SIZE: f32 = 0.1;
const GRID_DIM: i32 = 10;
const NUM_CELLS: u32 = 1000u;

fn cellCoord(p: vec3f) -> vec3i {
    let c = vec3i(floor((p - GRID_MIN) / GRID_CELL_SIZE));
    return clamp(c, vec3i(0), vec3i(GRID_DIM - 1));
}

fn cellLinearIdx(c: vec3i) -> u32 {
    return u32(c.x + c.y * GRID_DIM + c.z * GRID_DIM * GRID_DIM);
}
