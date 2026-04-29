// common.wgsl (prepended by renderer.ts) provides the Particle struct.

struct CameraU {
  view: mat4x4f,
  proj: mat4x4f,
  invProj: mat4x4f,   // unused here; kept so the struct matches the UBO layout
  invView: mat4x4f,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> cam: CameraU;

const PARTICLE_RADIUS: f32 = 0.012;

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) density: f32,
  @location(1) quadUV: vec2f,      // local disk coord in [-1, 1]²
  @location(2) viewCenter: vec3f,  // particle centre in view space
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

  // View-space billboard. Quad spans [-r, r]² in view-space X/Y so the sphere
  // of radius PARTICLE_RADIUS fits inscribed.
  let viewCenter = (cam.view * vec4f(p, 1.0)).xyz;
  let viewBillboard = viewCenter + vec3f(q * PARTICLE_RADIUS, 0.0);

  var out: VsOut;
  out.pos        = cam.proj * vec4f(viewBillboard, 1.0);
  out.density    = particles[ii].density;
  out.quadUV     = q;
  out.viewCenter = viewCenter;
  return out;
}

struct FsOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
};

// Sphere imposter. Each quad fragment is interpreted as a point on a unit
// sphere in local (x, y, z) coords (with x, y = quadUV). Pixels outside the
// unit disk are discarded; pixels inside write the sphere-surface depth, not
// the flat quad's depth. That's what makes overlapping billboards occlude
// each other as actual spheres.
@fragment
fn fs_main(in: VsOut) -> FsOut {
  let r2 = dot(in.quadUV, in.quadUV);
  if (r2 > 1.0) { discard; }
  let z = sqrt(1.0 - r2);                // sphere height, normalized

  // View-space position on the sphere's front face. Camera looks down −Z, so
  // the front face is at viewCenter.z + z·r (closer to camera).
  let viewSpherePos = in.viewCenter + vec3f(in.quadUV, z) * PARTICLE_RADIUS;

  // Reproject to emit correct sphere-surface depth.
  let clip = cam.proj * vec4f(viewSpherePos, 1.0);
  let depth = clip.z / clip.w;

  // View-space surface normal = local sphere coord (already unit-length).
  let normal   = vec3f(in.quadUV, z);
  let lightDir = normalize(vec3f(0.4, 0.6, 1.0));
  let lambert  = max(dot(normal, lightDir), 0.0);

  // Density tint (same range as before).
  let t    = smoothstep(2000.0, 10000.0, in.density);
  let blue = vec3f(0.1, 0.3, 1.0);
  let red  = vec3f(1.0, 0.15, 0.1);
  let base = mix(blue, red, t);

  var out: FsOut;
  out.color = vec4f(base * (0.25 + 0.75 * lambert), 1.0);
  out.depth = depth;
  return out;
}
