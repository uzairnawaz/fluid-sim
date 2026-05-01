
struct Interaction {
    center: vec3f,
    strength: f32,   // 0 means inactive; positive = outward push
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var<uniform> interact: Interaction;

const REPEL_RADIUS: f32 = 0.25;

// PBF step 1: apply gravity (and any mouse interaction impulse) to the
// velocity, predict position by forward Euler. The tentative velocity is not
// persisted; the final velocity is recovered after the solver by
// update_velocity from (predPos - pos) / dt, which implicitly includes
// gravity, the interaction force, and all constraint impulses.
//
// We clamp predPos to the box so the solver starts from a valid state.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&particles)) { return; }

    var v = particles[i].vel + vec3f(0.0, -9.8, 0.0) * params.dt;

    // Mouse repel: push particles within REPEL_RADIUS of interact.center
    // radially outward, falling off linearly to zero at the radius edge.
    if (interact.strength > 0.0) {
        let toParticle = particles[i].pos - interact.center;
        let dist = length(toParticle);
        if (dist > 1e-4 && dist < REPEL_RADIUS) {
            let dir = toParticle / dist;
            let falloff = 1.0 - dist / REPEL_RADIUS;
            v = v + dir * (interact.strength * falloff) * params.dt;
        }
    }

    var p = particles[i].pos + v * params.dt;
    p = clamp(p, vec3f(-0.5), vec3f(0.5));
    particles[i].predPos = p;
}
