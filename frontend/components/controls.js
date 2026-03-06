export function buildControls(container, initial) {
  container.innerHTML = "";

  const controls = {};

  function addNumber(id, label, value, step=1, min=null, max=null) {
    const div = document.createElement("div");
    div.className = "control";
    div.innerHTML = `
      <label><span>${label}</span><span id="${id}_val">${value}</span></label>
      <input type="number" id="${id}" value="${value}" step="${step}" />
    `;
    container.appendChild(div);
    const input = div.querySelector(`#${id}`);
    const val = div.querySelector(`#${id}_val`);
    input.addEventListener("input", () => (val.textContent = input.value));
    controls[id] = input;
  }

  function addRange(id, label, value, step, min, max) {
    const div = document.createElement("div");
    div.className = "control";
    div.innerHTML = `
      <label><span>${label}</span><span id="${id}_val">${value}</span></label>
      <input type="range" id="${id}" value="${value}" step="${step}" min="${min}" max="${max}" />
    `;
    container.appendChild(div);
    const input = div.querySelector(`#${id}`);
    const val = div.querySelector(`#${id}_val`);
    input.addEventListener("input", () => (val.textContent = input.value));
    controls[id] = input;
  }

  function addSelect(id, label, options, value) {
    const div = document.createElement("div");
    div.className = "control";
    const opts = options.map(o => `<option value="${o.value}">${o.label}</option>`).join("");
    div.innerHTML = `
      <label><span>${label}</span><span></span></label>
      <select id="${id}">${opts}</select>
    `;
    container.appendChild(div);
    const sel = div.querySelector(`#${id}`);
    sel.value = value;
    controls[id] = sel;
  }

  // Model (only TFIM for now)
  addSelect("model", "Model", [
    {value: "tfim", label: "Transverse-Field Ising (TFIM)"},
  ], initial.model);

  // System + Hamiltonian
  addRange("N", "System size N", initial.N, 1, 6, 80);
  addRange("J", "Coupling J", initial.J, 0.05, 0.1, 3.0);
  addRange("h", "Field h", initial.h, 0.05, 0.0, 3.0);

  // MPS/DMRG controls
  addRange("chi_max", "Max bond dim χ", initial.chi_max, 1, 4, 256);
  addNumber("cutoff", "Truncation cutoff", initial.cutoff, 1e-12);
  addRange("max_sweeps", "DMRG sweeps", initial.max_sweeps, 1, 2, 50);

  // Scan controls
  addRange("scan_h_min", "Scan h min", initial.scan_h_min, 0.05, 0.0, 3.0);
  addRange("scan_h_max", "Scan h max", initial.scan_h_max, 0.05, 0.0, 3.0);
  addRange("scan_points", "Scan points", initial.scan_points, 1, 3, 80);

  // Correlator controls
  addRange("corr_max_r", "Max correlator r", initial.corr_max_r, 1, 2, 60);

  // --- TEBD controls ---
  addSelect("tebd_init_state", "TEBD init state", [
    {value: "+", label: "|+>^N (default)"},
    {value: "0", label: "|0>^N"},
    {value: "1", label: "|1>^N"},
    {value: "-", label: "|->^N"},
  ], initial.tebd_init_state);

  addNumber("tebd_dt", "TEBD dt", initial.tebd_dt, 0.01);
  addRange("tebd_steps", "TEBD steps", initial.tebd_steps, 1, 10, 600);
  addRange("tebd_measure_every", "Measure every", initial.tebd_measure_every, 1, 1, 50);

  return {
    getParams() {
      const p = {};
      for (const [k, el] of Object.entries(controls)) {
        if (el.tagName.toLowerCase() === "select") p[k] = el.value;
        else p[k] = Number(el.value);
      }
      return p;
    },
    setParams(next) {
      for (const [k, v] of Object.entries(next)) {
        if (controls[k]) {
          controls[k].value = v;
          const valSpan = container.querySelector(`#${k}_val`);
          if (valSpan) valSpan.textContent = String(v);
        }
      }
    }
  };
}