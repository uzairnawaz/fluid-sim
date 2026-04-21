
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;

const PI: f32 = 3.14159265358979;

// Poly6 kernel: W(r, h) = (315 / (64 π h^9)) (h² - r²)³ for 0 ≤ |r| ≤ h, else 0.
// Brute-force: every particle considers every other particle as a candidate neighbor.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&particles);
    if (i >= n) { return; }

    let h = params.h;
    let h2 = h * h;
    let coef = 315.0 / (64.0 * PI * pow(h, 9.0));

    let pos_i = particles[i].pos;
    var rho: f32 = 0.0;
    for (var j: u32 = 0u; j < n; j = j + 1u) {
        let r = pos_i - particles[j].pos;
        let r2 = dot(r, r);
        if (r2 < h2) {
            let x = h2 - r2;
            rho = rho + coef * x * x * x;   // unit mass
        }
    }
    particles[i].density = rho;
}
