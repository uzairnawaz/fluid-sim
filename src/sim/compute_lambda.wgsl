
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var<storage, read> cellStart: array<u32>;
@group(0) @binding(3) var<storage, read> cellCount: array<u32>;
@group(0) @binding(4) var<storage, read> sortedIdx: array<u32>;

const PI: f32 = 3.14159265358979;

// CFM relaxation (Macklin & Müller 2013, §3). Prevents λ from exploding in
// regions with few neighbors where Σ|∇C|² approaches zero.
const EPSILON_CFM: f32 = 600.0;

// PBF step 3b (eq. 11):
//   C_i = ρ_i / ρ₀ − 1
//   λ_i = −C_i / (Σ_k |∇_{p_k} C_i|² + ε)
// Spiky gradient (∇W) via grid-accelerated 27-cell query.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&particles);
    if (i >= n) { return; }

    let h = params.h;
    let rho0 = params.restDensity;
    let spiky_coef = -45.0 / (PI * pow(h, 6.0));

    let pos_i = particles[i].predPos;
    let cc = cellCoord(pos_i);

    var gradCi_self = vec3f(0.0);
    var sum_grad_sq = 0.0;

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
                    let rLen = length(r);
                    if (rLen > 0.0 && rLen < h) {
                        let x = h - rLen;
                        let gradW = spiky_coef * x * x * (r / rLen);
                        let gradCi_j = -gradW / rho0;
                        sum_grad_sq = sum_grad_sq + dot(gradCi_j, gradCi_j);
                        gradCi_self = gradCi_self + gradW;
                    }
                }
            }
        }
    }
    gradCi_self = gradCi_self / rho0;
    sum_grad_sq = sum_grad_sq + dot(gradCi_self, gradCi_self);

    let C_i = particles[i].density / rho0 - 1.0;
    particles[i].lambda = -C_i / (sum_grad_sq + EPSILON_CFM);
}
