import { mat4, vec3, type Mat4, type Vec3 } from "wgpu-matrix";

export class Camera {
  private aspect = 1;
  private fovY = (60 * Math.PI) / 180;
  private near = 0.01;
  private far = 100;

  private target: Vec3 = vec3.create(0, 0, 0);
  private distance = 2;
  private yaw = 0;
  private pitch = -0.2;

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
