
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;

const PI: f32 = 3.14159265358979;

// Tensile instability correction (Macklin & Müller 2013, eq. 13).
//   s_corr_{ij} = −k · ( W_poly6(|p_i − p_j|, h) / W_poly6(Δq, h) )^n
// An artificial inter-particle repulsion that prevents clumping in the tensile
// regime (C < 0) — surface particles would otherwise pull into droplets and
// bounce off the denser pile. With s_corr they integrate smoothly instead.
// Macklin's paper uses k = 0.1, but at our density scale (ρ₀ = 10000, λ ~ 1e-3)
// that dominates λ entirely. 0.001 is a conservative starting point — raise it
// if clumping is still visible.
const S_CORR_K: f32 = 0.001;
const S_CORR_DQ_FRAC: f32 = 0.2; // Δq = 0.2·h  (Macklin uses 0.1h–0.3h)
// n = 4 is hardcoded below as two squarings.

// PBF step 3c: compute position correction (Macklin & Müller eq. 14, now with
// the tensile-instability term from eq. 13):
//   Δp_i = (1/ρ₀) Σ_j (λ_i + λ_j + s_corr_{ij}) ∇W_spiky(p_i − p_j, h)
//   p*_i ← p*_i + Δp_i
// Poly6 for s_corr magnitude, Spiky for the gradient — standard SPH practice.
//
// Note: writes particles[i].predPos while other threads read their neighbors'
// predPos from the same dispatch — deliberate race, same pattern as Macklin's
// reference. The solver iterates, so per-iter error washes out.
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

    // Denominator of the s_corr ratio — W_poly6 at the fixed offset Δq.
    let dq = S_CORR_DQ_FRAC * h;
    let dq_x = h2 - dq * dq;
    let W_dq = poly6_coef * dq_x * dq_x * dq_x;

    let pos_i = particles[i].predPos;
    let lambda_i = particles[i].lambda;
    var dp = vec3f(0.0);

    for (var j: u32 = 0u; j < n; j = j + 1u) {
        if (j == i) { continue; }
        let r = pos_i - particles[j].predPos;
        let r2 = dot(r, r);
        if (r2 < h2 && r2 > 0.0) {
            let rLen = sqrt(r2);

            // Spiky kernel gradient ∇W(r, h).
            let sx = h - rLen;
            let gradW = spiky_coef * sx * sx * (r / rLen);

            // Poly6 W(r, h) for s_corr.
            let px = h2 - r2;
            let W_ij = poly6_coef * px * px * px;
            let ratio = W_ij / W_dq;
            let ratio2 = ratio * ratio;
            let s_corr = -S_CORR_K * ratio2 * ratio2; // ratio^n, n = 4

            dp = dp + (lambda_i + particles[j].lambda + s_corr) * gradW;
        }
    }
    dp = dp / rho0;

    var newPos = pos_i + dp;
    newPos = clamp(newPos, vec3f(-0.5), vec3f(0.5));
    particles[i].predPos = newPos;
}
