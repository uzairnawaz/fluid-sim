
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var<storage, read> cellStart: array<u32>;
@group(0) @binding(3) var<storage, read> cellCount: array<u32>;
@group(0) @binding(4) var<storage, read> sortedIdx: array<u32>;

const PI: f32 = 3.14159265358979;
const XSPH_C: f32 = 0.01;

// PBF step 4.5 (Macklin & Müller eq. 17), density-normalised form:
//   v_i ← v_i + c Σ_j (m_j / ρ_j) (v_j − v_i) W(p_i − p_j, h)
// Grid-accelerated. Intra-dispatch race on vel is accepted — same trade-off
// as apply_correction.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&particles);
    if (i >= n) { return; }

    let h = params.h;
    let h2 = h * h;
    let poly6_coef = 315.0 / (64.0 * PI * pow(h, 9.0));

    let pos_i = particles[i].predPos;
    let vel_i = particles[i].vel;
    let cc = cellCoord(pos_i);
    var v_accum = vec3f(0.0);

    for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                let nc = cc + vec3i(dx, dy, dz);
                if (any(nc < vec3i(0)) || any(nc >= vec3i(GRID_DIM))) { continue; }
                let cell = cellLinearIdx(nc);
                let start = cellStart[cell];
                let cnt = cellCount[cell];
                for (var k: u32 = 0u; k < cnt; k = k + 1u) {
                    let j = sortedIdx[start + k];
                    if (j == i) { continue; }
                    let r = pos_i - particles[j].predPos;
                    let r2 = dot(r, r);
                    if (r2 < h2) {
                        let x = h2 - r2;
                        let W = poly6_coef * x * x * x;
                        let rho_j = particles[j].density;
                        if (rho_j > 0.0) {
                            v_accum = v_accum + (particles[j].vel - vel_i) * (W / rho_j);
                        }
                    }
                }
            }
        }
    }
    particles[i].vel = vel_i + XSPH_C * v_accum;
}
