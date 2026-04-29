// Copies depth32float → r32float so later passes can sample depth as a regular
// texture_2d<f32> instead of texture_depth_2d (which the bilateral shader,
// reused for both H and V passes, can't polymorphically declare).
@group(0) @binding(0) var srcDepth: texture_depth_2d;

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let coord = vec2i(pos.xy);
    let d = textureLoad(srcDepth, coord, 0);
    return vec4f(d, 0.0, 0.0, 1.0);
}
