import { postJSON } from "./services/api.js";
import { buildControls } from "./components/controls.js";
import { presets } from "./components/presets.js";
import { initPlots, updateGroundStatePlots, updateScanPlot, updateDynamicsPlot } from "./components/plots.js";

const statusText = document.getElementById("statusText");
const timingText = document.getElementById("timingText");

function setStatus(msg) {
  statusText.textContent = msg;
}

function setTiming(msg) {
  timingText.textContent = msg || "";
}

const initial = {
  model: "tfim",
  N: 24,
  J: 1.0,
  h: 0.6,
  chi_max: 32,
  cutoff: 1e-10,
  max_sweeps: 12,
  scan_h_min: 0.0,
  scan_h_max: 2.0,
  scan_points: 25,
  corr_max_r: 20,

  // TEBD defaults
  tebd_init_state: "+",
  tebd_dt: 0.05,
  tebd_steps: 120,
  tebd_measure_every: 2,
};

const controls = buildControls(document.getElementById("controls"), initial);

function buildPresetButtons() {
  const container = document.getElementById("presetButtons");
  container.innerHTML = "";
  for (const p of presets()) {
    const btn = document.createElement("button");
    btn.textContent = p.name;
    btn.addEventListener("click", () => controls.setParams(p.params));
    container.appendChild(btn);
  }
}

async function runGroundState() {
  try {
    setStatus("Running ground state (DMRG)...");
    setTiming("");

    const params = controls.getParams();
    const { data, ms } = await postJSON("/api/ground_state", params);

    setStatus("Done.");
    setTiming(`Request: ${ms.toFixed(0)} ms | Backend: ${data.backend_ms.toFixed(0)} ms (DMRG: ${data.dmrg_ms.toFixed(0)} ms)`);

    updateGroundStatePlots(data);
  } catch (err) {
    console.error(err);
    setStatus("Error.");
    setTiming(err.message);
    alert(err.message);
  }
}

async function runScan() {
  try {
    setStatus("Running scan...");
    setTiming("");

    const params = controls.getParams();
    const payload = {
      model: params.model,
      N: params.N,
      J: params.J,
      chi_max: params.chi_max,
      cutoff: params.cutoff,
      max_sweeps: params.max_sweeps,
      h_min: params.scan_h_min,
      h_max: params.scan_h_max,
      points: params.scan_points,
    };

    const { data, ms } = await postJSON("/api/scan", payload);

    setStatus("Done.");
    setTiming(`Request: ${ms.toFixed(0)} ms | Backend: ${data.backend_ms.toFixed(0)} ms`);

    updateScanPlot(data);
  } catch (err) {
    console.error(err);
    setStatus("Error.");
    setTiming(err.message);
    alert(err.message);
  }
}

async function runTEBD() {
  try {
    setStatus("Running TEBD quench...");
    setTiming("");

    const params = controls.getParams();
    const payload = {
      model: params.model,
      N: params.N,
      J: params.J,
      h: params.h,
      chi_max: params.chi_max,
      cutoff: params.cutoff,
      dt: params.tebd_dt,
      steps: params.tebd_steps,
      init_state: params.tebd_init_state,
      measure_every: params.tebd_measure_every,
    };

    const { data, ms } = await postJSON("/api/tebd", payload);

    setStatus("Done.");
    setTiming(`Request: ${ms.toFixed(0)} ms | Backend: ${data.backend_ms.toFixed(0)} ms`);

    updateDynamicsPlot(data);
  } catch (err) {
    console.error(err);
    setStatus("Error.");
    setTiming(err.message);
    alert(err.message);
  }
}

document.getElementById("runBtn").addEventListener("click", runGroundState);
document.getElementById("scanBtn").addEventListener("click", runScan);
document.getElementById("tebdBtn").addEventListener("click", runTEBD);

initPlots();
buildPresetButtons();
setStatus("Idle");