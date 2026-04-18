import { initGpu } from "./gpu/context";
import { defaultParams } from "./sim/params";
import { Simulation } from "./sim/simulation";
import { Renderer } from "./render/renderer";
import { Camera } from "./render/camera";

async function main() {
  const canvas = document.getElementById("gfx") as HTMLCanvasElement;
  if (!navigator.gpu) throw new Error("WebGPU is not supported in this browser.");

  const { device, context, format } = await initGpu(canvas);
  const sim = new Simulation(device, defaultParams());
  const renderer = new Renderer(device, format);
  const camera = new Camera();

  let dragging = false;
  let prevX = 0;
  let prevY = 0;
  canvas.addEventListener("mousedown", (e) => {
    dragging = true;
    prevX = e.clientX;
    prevY = e.clientY;
  });
  window.addEventListener("mouseup", () => { dragging = false; });
  canvas.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - prevX;
    const dy = e.clientY - prevY;
    prevX = e.clientX;
    prevY = e.clientY;
    camera.rotate(dx, dy);
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    camera.zoom(e.deltaY);
  }, { passive: false });

  function resize() {
    const dpr = 1; //window.devicePixelRatio || 1;
    const w = Math.floor(canvas.clientWidth * dpr);
    const h = Math.floor(canvas.clientHeight * dpr);
    canvas.width = w;
    canvas.height = h;
    camera.resize(w, h);
  }
  window.addEventListener("resize", resize);
  resize();

  function frame() {
    sim.step();
    renderer.render(
      context.getCurrentTexture().createView(),
      sim.particleBuffer,
      sim.params.particleCount,
      camera,
    );
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch((err) => {
  console.error(err);
  document.body.innerText = String(err);
});
