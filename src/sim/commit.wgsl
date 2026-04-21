
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;

// PBF step 5: p ← p*.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&particles)) { return; }
    particles[i].pos = particles[i].predPos;
}
