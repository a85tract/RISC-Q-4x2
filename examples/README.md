# examples — co-simulation notebooks

Runnable notebooks that drive the **Verilated `PulseTableSoc`** under cocotb — the same host control
software (`riscq.lang`/`riscq.run` to compile and run on-core kernels, `riscq.cal`/`riscq.sim` for
the calibrations) that runs on the ZCU216 board, here against a Verilated SoC instead of real
hardware.

Start with [`amplitude_sweep.ipynb`](amplitude_sweep.ipynb) — a minimal drive demo — then the two
calibration notebooks, which run against a `TwoLevelModel` with **planted ground truth** instead of
a real qubit (co-sim analogues of the hardware `Calibration_X6Y3` and `single_qubit_autocalibrate_v2`
flows; specs [`05-simulation`](../specs/software/05-simulation.md),
[`06-calibrations`](../specs/software/06-calibrations.md)).

| Notebook | What it shows |
|---|---|
| [`amplitude_sweep.ipynb`](amplitude_sweep.ipynb) | the smallest end-to-end example — an on-core `@kernel` fires a train of gate pulses with a ramping amplitude; the notebook captures the raw gate **DAC waveform** off the SoC and plots the pulse train, the pulse shape, and the (linear) peak-vs-amplitude. No qubit model — just the drive signal the hardware emits |
| [`single_qubit_calibration.ipynb`](single_qubit_calibration.ipynb) | the `Calibration_X6Y3` flow, one experiment per cell — ReadoutCalibration → Separation → Fidelity → ReadoutFidelity → Frequency → Amplitude (coarse) → Amplitude (fine) → Phase — pulling a deliberately-detuned `Config` back to the planted `f_ge`/Rabi ground truth |
| [`single_qubit_autocalibrate.ipynb`](single_qubit_autocalibrate.ipynb) | the `single_qubit_autocalibrate_v2` flow via [`autocal.py`](autocal.py): spectroscopy → Ramsey (frequency) → Rabi (amplitude) → T1, with an `expts_to_run` gating dict, incremental frequency updates, and the automatic **Ramsey-after-Amplitude re-run** (AC-Stark) rule |

[`autocal.py`](autocal.py) is the plain host-python autocalibration **script** the second notebook
imports — the between-experiment control flow (ordering, gating, the conditional re-run) needs no
kernel DSL because it runs on the host.

Four notebooks target **real hardware** instead of the co-sim and are therefore *not* executed in CI:

| Notebook | What it shows |
|---|---|
| [`remote_pulse.ipynb`](remote_pulse.ipynb) | the ZCU216 quickstart — connect to the board server with `RemoteDriver`, upload/load a gateware bundle, and fire a pulse train from an on-core `@kernel` on a real DAC. Server setup: [docs/software/board-server.md](../docs/software/board-server.md) |
| [`readout_robs.ipynb`](readout_robs.ipynb) | fire **one** readout-drive pulse at 2.76 GHz on qubit 1's readout channel (1) on the `xm650-loopback` board and plot the SoC's readout-observation buffer (`robs`) — the raw ADC-rate trace the hardware streams (per-lane sum of the mapped ADCs) for the whole time the pulse is valid, i.e. the looped-back readout tone coming back off the DAC (folded to ~0.76 GHz at the 2 GS/s ADC). The minimal `robs` demo — one `rq.run` + one `rq.read_robs` |
| [`vna.ipynb`](vna.ipynb) | a wideband VNA on the `xm650-loopback` board — from-scratch on-core `@kernel`s sweep the readout carrier 0.1 → 7.9 GHz (the full 8 GS/s Nyquist zone, 781 points at 10 MHz) and plot the mean shot amplitude per frequency. Two implementations, both bounded by the 16 KB core RAM: one keeps **every** shot's raw IQ (6 MB → one `rq.rerun` per frequency, 781 reruns), the other **accumulates each shot's power** `re²+im²` on-core (phase-insensitive, 1 word/point, sweeping the frequency code on-core via a wrapping-Q16 accumulator → the whole sweep in one rerun); the notebook compares their data-collection time |
| [`iq_scatter.ipynb`](iq_scatter.ipynb) | single-shot IQ at a **fixed** 2.75 GHz on the `xm650-loopback` board — the readout drive (ch 1) plays a measurement tone, the demod carrier (ch 2, `demod_freq_to_code` = 4× the drive code, folded above its own Nyquist) returns one `(I, Q)` per shot. Records **10,000 shots** (10 reruns of 1,000 — the 16 KB core RAM caps one `out` buffer at ~2,000 words) and draws the IQ **scatter**; the blob's 1-σ spread is the readout noise floor |

Each notebook owns the sim lifecycle itself: `server.start(...)` brings up the cocotb bench,
`drv.sim.set_model(...)` plants the `TwoLevelModel`, the calibrations run, and a final cell calls
`server.stop(drv)`.

## Running

The notebooks import the editable-installed `riscq` package, so they run from any working directory.
Execute one headless (this is exactly what CI does):

```bash
jupyter nbconvert --to notebook --execute --inplace examples/amplitude_sweep.ipynb
jupyter nbconvert --to notebook --execute --inplace examples/single_qubit_calibration.ipynb
jupyter nbconvert --to notebook --execute --inplace examples/single_qubit_autocalibrate.ipynb
```

or open them interactively:

```bash
jupyter lab examples/
```

Each notebook takes roughly 2–3 minutes headless (deliberately small sweep point/shot counts — the
point is convergence to known truth, not fine sweeps).

## Requirements

- `riscq` installed (`pip install -e software`) with its deps (numpy, scipy, Pyro5), plus `qutip`
  (the `TwoLevelModel` physics), `matplotlib`, and `jupyter`.
- `verilator` and the riscv LLVM toolchain (`riscv64-unknown-elf-clang` + binutils) on `PATH` —
  the co-sim Verilates the SoC and cross-compiles the on-core kernels. The first run
  elaborates + Verilates the DUT (~1–2 min); later runs reuse the cached build under
  `software/build/sim-2q/`.
