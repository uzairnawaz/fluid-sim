import type { SimParams } from "./params";

export class Simulation {
  readonly device: GPUDevice;
  readonly params: SimParams;

  constructor(device: GPUDevice, params: SimParams) {
    this.device = device;
    this.params = params;
  }

  step(): void {
    // TODO: predict → grid → solver iters → finalize
  }
}
