// common.wgsl (prepended by renderer.ts) provides the Particle struct.

struct CameraU {
  view: mat4x4f,
  proj: mat4x4f,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> cam: CameraU;

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) density: f32,
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
  let radius = 0.01;
  let p = particles[ii].pos;

  // View-space billboard: quad is offset in camera space, then projected.
  let viewCenter = cam.view * vec4f(p, 1.0);
  let viewBillboard = viewCenter.xyz + vec3f(quad[vi] * radius, 0.0);

  var out: VsOut;
  out.pos = cam.proj * vec4f(viewBillboard, 1.0);
  out.density = particles[ii].density;
  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
  // Color by density. Range tuned for default h = 0.1, unit mass:
  //   ~1566   = Poly6 self-contribution (isolated particle)  → blue
  //   ~10000+ = dense neighborhood                           → red
  let t = smoothstep(2000.0, 10000.0, in.density);
  let blue = vec3f(0.1, 0.3, 1.0);
  let red  = vec3f(1.0, 0.15, 0.1);
  return vec4f(mix(blue, red, t), 1.0);
}
