// Step 7.5: screen-space fluid composite.
//
// Previously a debug viz. Now the final frame: for each pixel, reconstruct
// world-space position + normal from the smoothed depth + normal textures,
// sample a procedural skybox for reflection and refraction directions,
// Beer-Lambert through the thickness texture, Fresnel-blend, add specular.
//
// Non-fluid pixels get the skybox directly — this pass now covers the entire
// frame (load op is "clear" in the render pass, not "load").

struct CameraU {
    view: mat4x4f,
    proj: mat4x4f,
    invProj: mat4x4f,
    invView: mat4x4f,
};

@group(0) @binding(0) var srcDepth: texture_2d<f32>;
@group(0) @binding(1) var srcThickness: texture_2d<f32>;
@group(0) @binding(2) var srcNormal: texture_2d<f32>;
@group(0) @binding(3) var<uniform> cam: CameraU;

// ---- Scene constants (tune to taste). ----
const LIGHT_DIR_WORLD: vec3f = vec3f(0.3, 0.8, 0.4);
const LIGHT_COLOR:     vec3f = vec3f(1.0, 0.95, 0.9);
// Per-unit-thickness absorption. R absorbed fast → deep water looks blue.
const WATER_ABSORPTION: vec3f = vec3f(8.0, 2.0, 0.5);
const F0:              f32   = 0.02;    // water Fresnel at normal incidence
const SHININESS:       f32   = 128.0;
const SPEC_STRENGTH:   f32   = 0.8;
const IOR_AIR_WATER:   f32   = 1.0 / 1.33;
const FAR_EPS:         f32   = 0.9999;

// ---- Procedural skybox. ----
// Clear-afternoon palette: deep-blue zenith, paler horizon, dark nadir, a
// warm haze band along the horizon, and a sharp sun disc + soft bloom in the
// light direction. Everything derived from the view-ray direction; no cubemap.
fn sampleEnv(dir: vec3f) -> vec3f {
    let up = clamp(dir.y, -1.0, 1.0);

    // Base three-zone gradient. Power curves concentrate variation near the
    // horizon (matches atmospheric perception — zenith is a large near-flat
    // region of deep blue; most contrast happens approaching horizon).
    let zenith  = vec3f(0.15, 0.35, 0.75);
    let horizon = vec3f(0.85, 0.92, 1.00);
    let nadir   = vec3f(0.50, 0.42, 0.32);  // warm sand

    var color: vec3f;
    if (up >= 0.0) {
        color = mix(horizon, zenith, pow(up, 0.4));
    } else {
        color = mix(horizon, nadir, pow(-up, 0.6));
    }

    // Warm haze band hugging the horizon line.
    let horizonGlow = vec3f(1.0, 0.75, 0.55);
    color = color + horizonGlow * exp(-abs(up) * 7.0) * 0.2;

    // Sun disc + bloom in the light direction. `cosSun` ≈ 1 when looking at
    // the sun; tighter smoothstep range makes a smaller, sharper disc.
    let L = normalize(LIGHT_DIR_WORLD);
    let cosSun = dot(dir, L);
    let sunColor = vec3f(1.0, 0.95, 0.80);
    let disc  = smoothstep(0.9965, 0.9990, cosSun) * 6.0;
    let bloom = pow(max(cosSun, 0.0), 32.0) * 0.4;
    color = color + sunColor * (disc + bloom);

    return color;
}

fn viewSpaceFromDepth(coord: vec2i, dims: vec2i, d: f32) -> vec3f {
    let uv  = (vec2f(coord) + 0.5) / vec2f(dims);
    // Framebuffer y is down; NDC y is up — flip y on unprojection.
    let ndc = vec3f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d);
    let p   = cam.invProj * vec4f(ndc, 1.0);
    return p.xyz / p.w;
}

// Camera-ray direction in world space for the given pixel (any in-frustum z).
fn rayDirWorld(coord: vec2i, dims: vec2i) -> vec3f {
    let uv  = (vec2f(coord) + 0.5) / vec2f(dims);
    let ndc = vec3f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5);
    let vp  = cam.invProj * vec4f(ndc, 1.0);
    let vDir = normalize(vp.xyz / vp.w);
    // w=0 → rotate-only transform for a direction.
    return normalize((cam.invView * vec4f(vDir, 0.0)).xyz);
}

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let dims  = vec2i(textureDimensions(srcDepth));
    let coord = vec2i(pos.xy);
    let d     = textureLoad(srcDepth, coord, 0).r;

    let ray = rayDirWorld(coord, dims);

    // Background: skybox in the camera-ray direction.
    if (d >= FAR_EPS) {
        return vec4f(sampleEnv(ray), 1.0);
    }

    let thick = textureLoad(srcThickness, coord, 0).r;
    let nView = textureLoad(srcNormal, coord, 0).xyz;
    let nWorld = normalize((cam.invView * vec4f(nView, 0.0)).xyz);

    let V = -ray;                           // surface → camera (world)
    let R = reflect(-V, nWorld);            // mirror direction
    let reflected = sampleEnv(R);

    let Rf = refract(-V, nWorld, IOR_AIR_WATER);
    let bgRefr = sampleEnv(Rf);
    let tint = exp(-thick * WATER_ABSORPTION);
    let refracted = bgRefr * tint;

    // Schlick Fresnel — reflection dominates at grazing angles.
    let cosTheta = clamp(dot(nWorld, V), 0.0, 1.0);
    let fresnel = F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);

    // Direct specular (Blinn-Phong).
    let L = normalize(LIGHT_DIR_WORLD);
    let H = normalize(L + V);
    let specular = LIGHT_COLOR * SPEC_STRENGTH
                   * pow(max(dot(nWorld, H), 0.0), SHININESS);

    let color = mix(refracted, reflected, fresnel) + specular;
    return vec4f(color, 1.0);
}
