// Reconstruct view-space surface normals from the smoothed depth
// buffer. We unproject five texels (center + 4 neighbors) back to view-space
// positions, then take the cross product of two tangent vectors.
//
// Using textureLoad + unprojection rather than ddx/ddy because:
//   (1) ddx/ddy on non-linear depth is noisy and scale-dependent;
//   (2) at silhouettes it spans a huge depth jump, producing garbage normals.
// With explicit neighbor sampling we can pick the closer side on each axis
// (van der Laan 2009) so the tangent never straddles a silhouette.

struct CameraU {
    view: mat4x4f,
    proj: mat4x4f,
    invProj: mat4x4f,
    invView: mat4x4f,
};

@group(0) @binding(0) var srcDepth: texture_2d<f32>;
@group(0) @binding(1) var<uniform> cam: CameraU;

// Pixel (framebuffer, y-down) + non-linear depth → view-space position.
fn viewSpaceFromDepth(coord: vec2i, dims: vec2i, d: f32) -> vec3f {
    let uv = (vec2f(coord) + 0.5) / vec2f(dims);
    // Framebuffer y is down, clip/NDC y is up → flip y for unprojection.
    let ndc = vec3f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d);
    let clip = cam.invProj * vec4f(ndc, 1.0);
    return clip.xyz / clip.w;
}

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let dims = vec2i(textureDimensions(srcDepth));
    let c = vec2i(pos.xy);
    let d = textureLoad(srcDepth, c, 0).r;
    if (d >= 0.9999) { discard; }

    let P = viewSpaceFromDepth(c, dims, d);

    // Sample four neighbors' depths at stride S; larger S = smoother normals
    // at the cost of silhouette precision.
    const S: i32 = 2;
    let dL = textureLoad(srcDepth, c + vec2i(-S,  0), 0).r;
    let dR = textureLoad(srcDepth, c + vec2i( S,  0), 0).r;
    let dU = textureLoad(srcDepth, c + vec2i( 0, -S), 0).r;
    let dD = textureLoad(srcDepth, c + vec2i( 0,  S), 0).r;

    let PL = viewSpaceFromDepth(c + vec2i(-S,  0), dims, dL);
    let PR = viewSpaceFromDepth(c + vec2i( S,  0), dims, dR);
    let PU = viewSpaceFromDepth(c + vec2i( 0, -S), dims, dU);
    let PD = viewSpaceFromDepth(c + vec2i( 0,  S), dims, dD);

    // Pick the closer-depth side on each axis so the tangent doesn't cross a
    // silhouette. Tangent X points view +x; tangent Y (screen-down) → view -y.
    let tX = select(PR - P, P - PL, abs(dL - d) < abs(dR - d));
    let tY = select(PD - P, P - PU, abs(dU - d) < abs(dD - d));

    // Screen-y is down, view-y is up → cross(tY, tX) (not tX × tY) for a
    // normal that points toward the camera (+z in view space).
    let n = normalize(cross(tY, tX));
    return vec4f(n, 1.0);
}
