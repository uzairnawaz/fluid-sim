export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
}

export async function initGpu(canvas: HTMLCanvasElement): Promise<GpuContext> {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No GPU adapter found.");
  const device = await adapter.requestDevice();
  device.addEventListener("uncapturederror", (e) => {
    console.error("[WebGPU]", (e as GPUUncapturedErrorEvent).error.message);
  });
  device.lost.then((info) => {
    console.error("[WebGPU] device lost:", info.reason, info.message);
  });

  const context = canvas.getContext("webgpu") as GPUCanvasContext;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });
  
  console.log(adapter.info);

  return { adapter, device, context, format };
}
