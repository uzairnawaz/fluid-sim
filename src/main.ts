import { initGpu } from "./gpu/context";
import { defaultParams } from "./sim/params";
import { Simulation } from "./sim/simulation";
import { Renderer } from "./render/renderer";

async function main() {
  const canvas = document.getElementById("gfx") as HTMLCanvasElement;
  if (!navigator.gpu) throw new Error("WebGPU is not supported in this browser.");

  const { device, context, format } = await initGpu(canvas);
  const sim = new Simulation(device, defaultParams());
  const renderer = new Renderer(device, format);

  function frame() {
    sim.step();
    renderer.render(context.getCurrentTexture().createView());
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch((err) => {
  console.error(err);
  document.body.innerText = String(err);
});
