// common.wgsl (prepended by renderer.ts) provides the Particle struct.
// Thickness pass for step 7.3: each particle splats a Gaussian onto a single-
// channel float texture. With additive blending configured on the pipeline,
// the texture accumulates total water along the view ray — used later for
// Beer's-law absorption and foam.

struct CameraU {
    view: mat4x4f,
    proj: mat4x4f,
    invProj: mat4x4f,   // unused here; kept so the struct matches the UBO layout
    invView: mat4x4f,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> cam: CameraU;

const PARTICLE_RADIUS: f32 = 0.012;
// Peak contribution at the splat centre. ~one sphere diameter, so stacked
// splats approximate total water volume along the ray.
const PEAK_THICKNESS: f32 = 0.024;
// Gaussian tightness. Higher = sharper splat (closer to a disk).
const GAUSS_K: f32 = 5.0;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) quadUV: vec2f,
};

@vertex
fn vs_main(
    @builtin(vertex_index)   vi: u32,
    @builtin(instance_index) ii: u32,
) -> VsOut {
    let quad = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
    );
    let q = quad[vi];
    let p = particles[ii].pos;
    let viewCenter = (cam.view * vec4f(p, 1.0)).xyz;
    let viewBillboard = viewCenter + vec3f(q * PARTICLE_RADIUS, 0.0);

    var out: VsOut;
    out.pos    = cam.proj * vec4f(viewBillboard, 1.0);
    out.quadUV = q;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    let r2 = dot(in.quadUV, in.quadUV);
    if (r2 > 1.0) { discard; }
    let g = exp(-r2 * GAUSS_K) * PEAK_THICKNESS;
    // Only .r is consumed by the r16float target.
    return vec4f(g, 0.0, 0.0, 1.0);
}
