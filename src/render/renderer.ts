export class Renderer {
  readonly device: GPUDevice;
  readonly format: GPUTextureFormat;

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.format = format;
  }

  render(view: GPUTextureView): void {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          clearValue: { r: 0.05, g: 0.06, b: 0.08, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    // TODO: draw particles
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}
