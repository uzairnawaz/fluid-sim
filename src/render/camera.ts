import { mat4, vec3, type Mat4, type Vec3 } from "wgpu-matrix";

export class Camera {
  private aspect = 1;
  private fovY = (60 * Math.PI) / 180;
  private near = 0.01;
  private far = 100;

  private target: Vec3 = vec3.create(0, 0, 0);
  private distance = 2;
  private yaw = 0;
  // Negative pitch = eye below the target (looking up); positive = eye above
  // (looking down). Original default was -0.2 (≈ −11.5°); add 25° in the
  // looking-down direction at startup.
  private pitch = -0.2 + (25 * Math.PI) / 180;

  readonly view: Mat4 = mat4.identity();
  readonly proj: Mat4 = mat4.identity();
  readonly invView: Mat4 = mat4.identity();
  readonly invProj: Mat4 = mat4.identity();

  constructor() {
    this.recompute();
  }

  resize(width: number, height: number): void {
    this.aspect = width / Math.max(1, height);
    this.recompute();
  }

  rotate(dx: number, dy: number): void {
    this.yaw   -= dx * 0.005;
    this.pitch -= dy * 0.005;
    this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch));
    this.recompute();
  }

  zoom(direction: number): void {
    this.distance *= 1 + Math.sign(direction) * 0.1;
    this.distance = Math.max(0.1, this.distance);
    this.recompute();
  }

  // Map a normalized-device-coords click (x,y in [-1,1], y up) to a world
  // point that lies at the camera's current orbit radius. Used by mouse
  // interaction so a click near the centre of the screen lands near the
  // scene origin where the fluid lives.
  worldFromNDC(ndcX: number, ndcY: number, dst: Vec3): Vec3 {
    const tanY = Math.tan(this.fovY / 2);
    const vx = ndcX * tanY * this.aspect * this.distance;
    const vy = ndcY * tanY * this.distance;
    const vz = -this.distance;
    return vec3.transformMat4(vec3.create(vx, vy, vz), this.invView, dst);
  }

  private recompute(): void {
    const cp = Math.cos(this.pitch);
    const eye = vec3.create(
      this.target[0] + this.distance * cp * Math.sin(this.yaw),
      this.target[1] + this.distance * Math.sin(this.pitch),
      this.target[2] + this.distance * cp * Math.cos(this.yaw),
    );
    mat4.lookAt(eye, this.target, vec3.create(0, 1, 0), this.view);
    mat4.perspective(this.fovY, this.aspect, this.near, this.far, this.proj);
    mat4.inverse(this.view, this.invView);
    mat4.inverse(this.proj, this.invProj);
  }
}
