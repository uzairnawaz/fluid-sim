
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var<storage, read> cellStart: array<u32>;
@group(0) @binding(3) var<storage, read> cellCount: array<u32>;
@group(0) @binding(4) var<storage, read> sortedIdx: array<u32>;

const PI: f32 = 3.14159265358979;

// Tensile instability correction (Macklin & Müller 2013, eq. 13).
const S_CORR_K: f32 = 0.001;
const S_CORR_DQ_FRAC: f32 = 0.2;

// PBF step 3c (eq. 14, with eq. 13 tensile term):
//   Δp_i = (1/ρ₀) Σ_j (λ_i + λ_j + s_corr) ∇W_spiky(p_i − p_j, h)
// Grid-accelerated 27-cell neighborhood.
// Note: writes particles[i].predPos while other threads read their neighbors'
// predPos from the same dispatch — deliberate race, as in Macklin's reference.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&particles);
    if (i >= n) { return; }

    let h = params.h;
    let h2 = h * h;
    let rho0 = params.restDensity;
    let spiky_coef = -45.0 / (PI * pow(h, 6.0));
    let poly6_coef = 315.0 / (64.0 * PI * pow(h, 9.0));

    let dq = S_CORR_DQ_FRAC * h;
    let dq_x = h2 - dq * dq;
    let W_dq = poly6_coef * dq_x * dq_x * dq_x;

    let pos_i = particles[i].predPos;
    let lambda_i = particles[i].lambda;
    let cc = cellCoord(pos_i);
    var dp = vec3f(0.0);

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
                    if (r2 < h2 && r2 > 0.0) {
                        let rLen = sqrt(r2);
                        let sx = h - rLen;
                        let gradW = spiky_coef * sx * sx * (r / rLen);
                        let px = h2 - r2;
                        let W_ij = poly6_coef * px * px * px;
                        let ratio = W_ij / W_dq;
                        let ratio2 = ratio * ratio;
                        let s_corr = -S_CORR_K * ratio2 * ratio2; // n = 4
                        dp = dp + (lambda_i + particles[j].lambda + s_corr) * gradW;
                    }
                }
            }
        }
    }
    dp = dp / rho0;

    var newPos = pos_i + dp;
    newPos = clamp(newPos, vec3f(-0.5), vec3f(0.5));
    particles[i].predPos = newPos;
}
