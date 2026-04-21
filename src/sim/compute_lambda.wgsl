
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;

const PI: f32 = 3.14159265358979;

// CFM relaxation (Macklin & Müller 2013, §3). Prevents λ from exploding in
// regions with few neighbors where Σ|∇C|² approaches zero.
const EPSILON_CFM: f32 = 600.0;

// PBF step 3b: for each particle i, evaluate density constraint
//   C_i = ρ_i / ρ₀ − 1                          (eq. 1)
// and Lagrange multiplier
//   λ_i = −C_i / (Σ_k |∇_{p_k} C_i|² + ε)       (eq. 11)
//
// Gradient of the constraint (eq. 8), using the Spiky kernel:
//   ∇_{p_i} C_i = (1/ρ₀) Σ_j ∇W(p_i−p_j,h)
//   ∇_{p_j} C_i = −(1/ρ₀) ∇W(p_i−p_j,h)           for j ≠ i
// The denominator sums over every k, including k = i.
//
// Spiky kernel gradient:
//   ∇W(r,h) = −(45 / (π h⁶)) (h − |r|)² (r / |r|)
// (Poly6 for density magnitude, Spiky for gradients — standard SPH practice.)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&particles);
    if (i >= n) { return; }

    let h = params.h;
    let rho0 = params.restDensity;
    let spiky_coef = -45.0 / (PI * pow(h, 6.0));

    let pos_i = particles[i].predPos;

    var gradCi_self = vec3f(0.0);  // accumulator for ∇_{p_i} C_i (before 1/ρ₀)
    var sum_grad_sq = 0.0;         // Σ_{j≠i} |∇_{p_j} C_i|²

    for (var j: u32 = 0u; j < n; j = j + 1u) {
        if (j == i) { continue; }
        let r = pos_i - particles[j].predPos;
        let rLen = length(r);
        if (rLen > 0.0 && rLen < h) {
            let x = h - rLen;
            let gradW = spiky_coef * x * x * (r / rLen);
            // ∇_{p_j} C_i = -(1/ρ₀) ∇W
            let gradCi_j = -gradW / rho0;
            sum_grad_sq = sum_grad_sq + dot(gradCi_j, gradCi_j);
            // ∇_{p_i} C_i accumulates +∇W over all j
            gradCi_self = gradCi_self + gradW;
        }
    }
    gradCi_self = gradCi_self / rho0;
    sum_grad_sq = sum_grad_sq + dot(gradCi_self, gradCi_self);

    let C_i = particles[i].density / rho0 - 1.0;
    particles[i].lambda = -C_i / (sum_grad_sq + EPSILON_CFM);
}
