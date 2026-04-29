
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var<storage, read> cellStart: array<u32>;
@group(0) @binding(3) var<storage, read> cellCount: array<u32>;
@group(0) @binding(4) var<storage, read> sortedIdx: array<u32>;

const PI: f32 = 3.14159265358979;

// Poly6 kernel: W(r, h) = (315 / (64 π h^9)) (h² - r²)³ for 0 ≤ |r| ≤ h, else 0.
// Self-contribution (j == i) is included (correct SPH practice).
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&particles);
    if (i >= n) { return; }

    let h = params.h;
    let h2 = h * h;
    let coef = 315.0 / (64.0 * PI * pow(h, 9.0));

    let pos_i = particles[i].predPos;
    let cc = cellCoord(pos_i);
    var rho: f32 = 0.0;

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
                    let r = pos_i - particles[j].predPos;
                    let r2 = dot(r, r);
                    if (r2 < h2) {
                        let x = h2 - r2;
                        rho = rho + coef * x * x * x;
                    }
                }
            }
        }
    }
    particles[i].density = rho;
}
