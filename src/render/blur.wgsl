// Separable 1-D Gaussian blur on an r16float/r32float texture. Unlike
// bilateral.wgsl there's no depth-difference term — used for smoothing the
// thickness texture, which is a continuous field with no silhouettes to
// preserve. Run twice per frame (H then V) for a full 2-D blur.
//
// Uniform struct layout matches BilateralParams exactly, so we can reuse the
// existing bilateralH/V uniform buffers for this pipeline.

struct BlurParams {
    direction: vec2f,
    spatialSigma: f32,
    _pad: f32,
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: BlurParams;

const KERNEL_RADIUS: i32 = 12;

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let dims = vec2i(textureDimensions(srcTex));
    let center = vec2i(pos.xy);
    let sigS2 = params.spatialSigma * params.spatialSigma;

    var total: f32 = 0.0;
    var totalW: f32 = 0.0;
    for (var k: i32 = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k = k + 1) {
        let offs = vec2i(params.direction * f32(k));
        let c = center + offs;
        if (c.x < 0 || c.y < 0 || c.x >= dims.x || c.y >= dims.y) { continue; }
        let s = textureLoad(srcTex, c, 0).r;
        let w = exp(-f32(k * k) / (2.0 * sigS2));
        total = total + s * w;
        totalW = totalW + w;
    }
    return vec4f(total / totalW, 0.0, 0.0, 1.0);
}
