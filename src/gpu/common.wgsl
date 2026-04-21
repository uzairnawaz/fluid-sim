// Shared types. Prepend this to every compute and render shader.

struct Particle {
  pos:     vec3f, density: f32,
  vel:     vec3f, _pad1: f32,
  predPos: vec3f, lambda: f32,
};

struct SimParams {
  particleCount: f32,
  h: f32,
  restDensity: f32,
  dt: f32,
  solverIters: f32,
};
