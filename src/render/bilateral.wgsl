// Separable 1-D bilateral filter on depth. Run twice per frame: once with
// direction = (1, 0), once with (0, 1). Spatial weight is a Gaussian in pixel
// distance; range weight is a Gaussian in depth difference — that's what
// preserves silhouettes (big depth jumps get near-zero weight, so the filter
// doesn't smear background into foreground).

struct BilateralParams {
    direction: vec2f,   // (1,0) horizontal; (0,1) vertical  (pixel units)
    spatialSigma: f32,  // σ of spatial Gaussian, in pixels
    depthSigma: f32,    // σ of depth-difference Gaussian (non-linear z units)
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: BilateralParams;

const KERNEL_RADIUS: i32 = 16;  // 33-tap kernel (−16..+16)
const FAR_EPS: f32 = 0.9999;    // anything ≥ this is "background, no fluid"

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let dims = vec2i(textureDimensions(srcTex));
    let center = vec2i(pos.xy);
    let centerDepth = textureLoad(srcTex, center, 0).r;

    // Empty pixel: pass through unchanged. Also avoids pulling filter weight
    // toward background during the composite.
    if (centerDepth >= FAR_EPS) {
        return vec4f(centerDepth, 0.0, 0.0, 1.0);
    }

    let sigS2 = params.spatialSigma * params.spatialSigma;
    let sigD2 = params.depthSigma   * params.depthSigma;

    var totalDepth:  f32 = 0.0;
    var totalWeight: f32 = 0.0;

    for (var k: i32 = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k = k + 1) {
        let offs = vec2i(params.direction * f32(k));
        let c = center + offs;
        // textureLoad returns zero for out-of-bounds, which would look like
        // "near-plane fluid" and poison the weights — bail explicitly.
        if (c.x < 0 || c.y < 0 || c.x >= dims.x || c.y >= dims.y) { continue; }

        let sd = textureLoad(srcTex, c, 0).r;
        if (sd >= FAR_EPS) { continue; }

        let ws = exp(-f32(k * k) / (2.0 * sigS2));
        let dd = sd - centerDepth;
        let wd = exp(-(dd * dd) / (2.0 * sigD2));
        let w = ws * wd;

        totalDepth  = totalDepth  + sd * w;
        totalWeight = totalWeight + w;
    }

    let result = select(centerDepth, totalDepth / totalWeight, totalWeight > 0.0);
    return vec4f(result, 0.0, 0.0, 1.0);
}
