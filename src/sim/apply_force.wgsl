
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&particles)) {
        return;
    }
    particles[i].vel = select(vec3f(0, 0, 0), 
                              particles[i].vel + (9.8 * 1.0 / 200.0) * vec3f(0, -1, 0),
                              particles[i].pos.y > -0.5);
    particles[i].pos = particles[i].pos + particles[i].vel * (1.0 / 200.0);
}
