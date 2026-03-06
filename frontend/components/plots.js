function axisStyle(title) {
  return {
    title: {
      text: title,
      font: { size: 14, color: "#dfe7f3" },
      standoff: 10
    },
    tickfont: { size: 12, color: "#d7dfeb" },
    gridcolor: "rgba(255,255,255,0.08)",
    linecolor: "rgba(255,255,255,0.18)",
    zeroline: false,
    showline: true,
    ticks: "outside",
    automargin: true
  };
}

function baseLayout({ title, xTitle, yTitle, y2Title = null, height = 340 }) {
  return {
    height,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {
      family: "Inter, ui-sans-serif, system-ui, sans-serif",
      color: "#e8eef7",
      size: 13
    },
    margin: { l: 78, r: y2Title ? 80 : 26, t: 55, b: 66 },
    title: {
      text: title,
      x: 0.02,
      xanchor: "left",
      font: { size: 16, color: "#eef4ff" }
    },
    legend: {
      orientation: "h",
      x: 0,
      y: -0.22,
      font: { size: 12 }
    },
    hovermode: "closest",
    xaxis: axisStyle(xTitle),
    yaxis: axisStyle(yTitle),
    ...(y2Title
      ? {
          yaxis2: {
            ...axisStyle(y2Title),
            overlaying: "y",
            side: "right"
          }
        }
      : {})
  };
}

function emptyLayout({ title, xTitle, yTitle, y2Title = null, height = 340, message }) {
  return {
    ...baseLayout({ title, xTitle, yTitle, y2Title, height }),
    annotations: [
      {
        x: 0.5,
        y: 0.5,
        xref: "paper",
        yref: "paper",
        text: message,
        showarrow: false,
        font: {
          size: 14,
          color: "rgba(232,238,247,0.68)"
        },
        align: "center"
      }
    ]
  };
}

export function initPlots() {
  Plotly.newPlot(
    "plotEnergyMag",
    [],
    emptyLayout({
      title: "Ground-state energy and average spin polarization",
      xTitle: "Measured quantity",
      yTitle: "Numerical value",
      message: "Run a ground-state calculation to display the optimized state's energy and magnetization."
    }),
    { responsive: true, displaylogo: false }
  );

  Plotly.newPlot(
    "plotEntropy",
    [],
    emptyLayout({
      title: "Entanglement entropy across bond cuts",
      xTitle: "Bond cut position",
      yTitle: "Von Neumann entropy  S",
      message: "This plot will show how entanglement is distributed along the spin chain."
    }),
    { responsive: true, displaylogo: false }
  );

  Plotly.newPlot(
    "plotCorr",
    [],
    emptyLayout({
      title: "Two-point longitudinal spin correlation function",
      xTitle: "Site separation  r",
      yTitle: "Correlation  ⟨σᶻᵢ σᶻᵢ₊ᵣ⟩",
      message: "This panel reveals whether longitudinal spin order survives over long distances."
    }),
    { responsive: true, displaylogo: false }
  );

  Plotly.newPlot(
    "plotScan",
    [],
    emptyLayout({
      title: "Ground-state response as the transverse field is varied",
      xTitle: "Transverse field strength  h",
      yTitle: "Ground-state energy",
      y2Title: "Average longitudinal magnetization  ⟨σᶻ⟩",
      message: "Run a field scan to explore how the system crosses from ordered to field-dominated behavior."
    }),
    { responsive: true, displaylogo: false }
  );

  Plotly.newPlot(
    "plotDynamics",
    [],
    emptyLayout({
      title: "Real-time quench dynamics from TEBD evolution",
      xTitle: "Time  t",
      yTitle: "Average magnetization",
      y2Title: "Mid-chain entanglement entropy",
      height: 430,
      message: "Run a TEBD simulation to visualize oscillations, relaxation, and entanglement growth in time."
    }),
    { responsive: true, displaylogo: false }
  );
}

export function updateGroundStatePlots(result) {
  Plotly.react(
    "plotEnergyMag",
    [
      {
        type: "bar",
        x: [
          "Ground-state energy",
          "Average longitudinal magnetization ⟨σᶻ⟩",
          "Average transverse magnetization ⟨σˣ⟩"
        ],
        y: [result.energy, result.mz, result.mx],
        hovertemplate:
          "<b>%{x}</b><br>" +
          "Value = %{y:.6f}" +
          "<extra></extra>"
      }
    ],
    baseLayout({
      title: "Ground-state energy and average spin polarization",
      xTitle: "Measured quantity",
      yTitle: "Numerical value"
    }),
    { responsive: true, displaylogo: false }
  );

  Plotly.react(
    "plotEntropy",
    [
      {
        type: "scatter",
        mode: "lines+markers",
        x: result.entropy_cut_indices,
        y: result.entropy,
        name: "Entanglement entropy",
        hovertemplate:
          "<b>Bond cut %{x}</b><br>" +
          "S = %{y:.6f}" +
          "<extra></extra>"
      }
    ],
    baseLayout({
      title: "Entanglement entropy across bond cuts",
      xTitle: "Bond cut position",
      yTitle: "Von Neumann entropy  S"
    }),
    { responsive: true, displaylogo: false }
  );

  Plotly.react(
    "plotCorr",
    [
      {
        type: "scatter",
        mode: "lines+markers",
        x: result.corr_r,
        y: result.corr_zz,
        name: "Longitudinal spin correlation",
        hovertemplate:
          "<b>Separation r = %{x}</b><br>" +
          "⟨σᶻᵢ σᶻᵢ₊ᵣ⟩ = %{y:.6f}" +
          "<extra></extra>"
      }
    ],
    baseLayout({
      title: "Two-point longitudinal spin correlation function",
      xTitle: "Site separation  r",
      yTitle: "Correlation  ⟨σᶻᵢ σᶻᵢ₊ᵣ⟩"
    }),
    { responsive: true, displaylogo: false }
  );
}

export function updateScanPlot(scan) {
  Plotly.react(
    "plotScan",
    [
      {
        type: "scatter",
        mode: "lines+markers",
        x: scan.h_values,
        y: scan.energy_values,
        name: "Ground-state energy",
        hovertemplate:
          "<b>h = %{x:.4f}</b><br>" +
          "Energy = %{y:.6f}" +
          "<extra></extra>"
      },
      {
        type: "scatter",
        mode: "lines+markers",
        x: scan.h_values,
        y: scan.mz_values,
        yaxis: "y2",
        name: "Average longitudinal magnetization ⟨σᶻ⟩",
        hovertemplate:
          "<b>h = %{x:.4f}</b><br>" +
          "⟨σᶻ⟩ = %{y:.6f}" +
          "<extra></extra>"
      }
    ],
    baseLayout({
      title: "Ground-state response as the transverse field is varied",
      xTitle: "Transverse field strength  h",
      yTitle: "Ground-state energy",
      y2Title: "Average longitudinal magnetization  ⟨σᶻ⟩"
    }),
    { responsive: true, displaylogo: false }
  );
}

export function updateDynamicsPlot(dyn) {
  Plotly.react(
    "plotDynamics",
    [
      {
        type: "scatter",
        mode: "lines",
        x: dyn.times,
        y: dyn.mz,
        name: "Average longitudinal magnetization ⟨σᶻ⟩",
        hovertemplate:
          "<b>t = %{x:.4f}</b><br>" +
          "⟨σᶻ⟩ = %{y:.6f}" +
          "<extra></extra>"
      },
      {
        type: "scatter",
        mode: "lines",
        x: dyn.times,
        y: dyn.mx,
        name: "Average transverse magnetization ⟨σˣ⟩",
        hovertemplate:
          "<b>t = %{x:.4f}</b><br>" +
          "⟨σˣ⟩ = %{y:.6f}" +
          "<extra></extra>"
      },
      {
        type: "scatter",
        mode: "lines",
        x: dyn.times,
        y: dyn.entropy_mid,
        yaxis: "y2",
        name: "Mid-chain entanglement entropy",
        hovertemplate:
          "<b>t = %{x:.4f}</b><br>" +
          "Entropy = %{y:.6f}" +
          "<extra></extra>"
      }
    ],
    baseLayout({
      title: "Real-time quench dynamics from TEBD evolution",
      xTitle: "Time  t",
      yTitle: "Average magnetization",
      y2Title: "Mid-chain entanglement entropy",
      height: 430
    }),
    { responsive: true, displaylogo: false }
  );
}