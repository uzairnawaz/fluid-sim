// Shared types. Prepend this to every compute and render shader.

struct Particle {
  pos:     vec3f, _pad0: f32,
  vel:     vec3f, _pad1: f32,
  predPos: vec3f, lambda: f32,
};
