export function presets() {
  return [
    {
      name: "Small demo (fast)",
      params: {
        N: 20, J: 1.0, h: 0.6, chi_max: 32, cutoff: 1e-10, max_sweeps: 10, corr_max_r: 15,
        tebd_init_state: "+", tebd_dt: 0.05, tebd_steps: 120, tebd_measure_every: 2
      }
    },
    {
      name: "Near critical (harder)",
      params: {
        N: 40, J: 1.0, h: 1.0, chi_max: 96, cutoff: 1e-12, max_sweeps: 20, corr_max_r: 25,
        tebd_init_state: "+", tebd_dt: 0.03, tebd_steps: 200, tebd_measure_every: 2
      }
    },
    {
      name: "Deep paramagnet",
      params: {
        N: 40, J: 1.0, h: 2.5, chi_max: 32, cutoff: 1e-10, max_sweeps: 12, corr_max_r: 25,
        tebd_init_state: "+", tebd_dt: 0.05, tebd_steps: 160, tebd_measure_every: 2
      }
    },
    {
      name: "Deep ferromagnet",
      params: {
        N: 40, J: 1.0, h: 0.2, chi_max: 32, cutoff: 1e-10, max_sweeps: 12, corr_max_r: 25,
        tebd_init_state: "+", tebd_dt: 0.05, tebd_steps: 160, tebd_measure_every: 2
      }
    }
  ];
}