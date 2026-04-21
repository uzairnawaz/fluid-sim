
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;

// PBF step 4: v ← (p* − p) / dt.
// Must run BEFORE commit — this reads the old `pos`, then commit overwrites it.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&particles)) { return; }
    particles[i].vel = (particles[i].predPos - particles[i].pos) / params.dt;
}
