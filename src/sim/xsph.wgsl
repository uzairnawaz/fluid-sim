
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;

const PI: f32 = 3.14159265358979;

// XSPH viscosity coefficient. Higher = more damping. 0.01 is a gentle default;
// push up toward 0.05 if the surface still looks jittery.
const XSPH_C: f32 = 0.01;

// PBF step 4.5 — Macklin & Müller 2013 eq. 17, density-normalised form:
//   v_i ← v_i + c Σ_j (m_j / ρ_j) (v_j − v_i) W(p_i − p_j, h)
// Uses Poly6 W (same kernel as density). Unit mass (m_j = 1). The 1/ρ_j factor
// keeps the sum O(1) regardless of neighbor count — without it the naïve
// Macklin form ΣW blows up at high particle counts.
//
// Must run AFTER update_velocity (reads the post-constraint velocity) and
// BEFORE commit. Reads vel of neighbors while writing our own — same
// intra-dispatch race as apply_correction. For small c it washes out.
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
    var v_accum = vec3f(0.0);

    for (var j: u32 = 0u; j < n; j = j + 1u) {
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
    particles[i].vel = vel_i + XSPH_C * v_accum;
}
