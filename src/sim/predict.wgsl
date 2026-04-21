
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;

// PBF step 1: apply gravity to velocity (tentative) and predict position by
// forward Euler. We do NOT persist the tentative velocity — the final velocity
// is recovered after the solver by `update_velocity` from (predPos - pos) / dt,
// which implicitly includes gravity and any constraint impulses.
//
// We also clamp predPos to the box so the solver starts from a valid state.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&particles)) { return; }

    let g = vec3f(0.0, -9.8, 0.0);
    let v = particles[i].vel + g * params.dt;
    var p = particles[i].pos + v * params.dt;
    p = clamp(p, vec3f(-0.5), vec3f(0.5));
    particles[i].predPos = p;
}
