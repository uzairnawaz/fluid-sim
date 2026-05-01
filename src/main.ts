import { vec3 } from "wgpu-matrix";
import { initGpu } from "./gpu/context";
import { defaultParams } from "./sim/params";
import { Simulation } from "./sim/simulation";
import { Renderer } from "./render/renderer";
import { Camera } from "./render/camera";

const REPEL_STRENGTH = 300; // velocity impulse magnitude (pre-falloff, pre-dt)

async function main() {
  const canvas = document.getElementById("gfx") as HTMLCanvasElement;
  if (!navigator.gpu) throw new Error("WebGPU is not supported in this browser.");

  const { device, context, format } = await initGpu(canvas);
  const sim = new Simulation(device, defaultParams());
  const renderer = new Renderer(device, format);
  const camera = new Camera();

  // Left-button drag rotates the camera. While the cursor is over the canvas
  // and the camera is NOT being dragged, the particle sim is fed a repel
  // point at the cursor's world-space location every frame the mouse moves —
  // i.e. just hovering over the fluid pushes it.
  let dragging = false;
  let prevX = 0;
  let prevY = 0;
  let hovering = false;
  const repelWorld = vec3.create();

  function pushRepel(px: number, py: number) {
    const ndcX = (px / canvas.width) * 2 - 1;
    const ndcY = 1 - (py / canvas.height) * 2;
    camera.worldFromNDC(ndcX, ndcY, repelWorld);
    sim.setInteraction(
      repelWorld[0],
      repelWorld[1],
      repelWorld[2],
      REPEL_STRENGTH,
    );
  }

  canvas.addEventListener("mouseenter", (e) => {
    hovering = true;
    if (!dragging) pushRepel(e.clientX, e.clientY);
  });
  canvas.addEventListener("mouseleave", () => {
    hovering = false;
    sim.clearInteraction();
  });

  canvas.addEventListener("mousedown", (e) => {
    if (e.button === 0) {
      dragging = true;
      prevX = e.clientX;
      prevY = e.clientY;
      // Freeze the repel during a camera orbit — otherwise rotating the view
      // would smear particles around because the cursor's world-space target
      // moves with the camera.
      sim.clearInteraction();
    }
  });
  window.addEventListener("mouseup", (e) => {
    if (e.button === 0) {
      dragging = false;
      if (hovering) pushRepel(e.clientX, e.clientY);
    }
  });

  canvas.addEventListener("mousemove", (e) => {
    if (dragging) {
      const dx = e.clientX - prevX;
      const dy = e.clientY - prevY;
      prevX = e.clientX;
      prevY = e.clientY;
      camera.rotate(dx, dy);
    } else {
      pushRepel(e.clientX, e.clientY);
    }
  });

  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      camera.zoom(e.deltaY);
    },
    { passive: false },
  );

  function resize() {
    const dpr = 1; //window.devicePixelRatio || 1;
    const w = Math.floor(canvas.clientWidth * dpr);
    const h = Math.floor(canvas.clientHeight * dpr);
    canvas.width = w;
    canvas.height = h;
    camera.resize(w, h);
    renderer.resize(w, h);
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
