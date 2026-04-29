// Full-screen triangle. One oversized triangle covers NDC [-1,1]² — no
// diagonal seam, no duplicate-vertex cost. Prepended to every fullscreen
// fragment shader in Phase 7.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    let p = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0),
    );
    return vec4f(p[vi], 0.0, 1.0);
}
