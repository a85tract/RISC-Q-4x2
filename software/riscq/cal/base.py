"""Shared calibration infrastructure (spec 06 §1, spec 08 §6 batched cut-over).

Every calibration runs its whole sweep as ONE batched run on the core (riscq.cal.kernels): the kernel
COMPUTES the swept knob on-core from a scalar sweep descriptor (a Q16 or int pair — spec 09; the input
Arrays are gone), walking it on a FIXED-period grid whose idle head is the T1 relax reset (the model
has no auto-reset, spec-M4 B0), so every readout lands at the same time-referenced demod-LO phase. The
demod carrier's discrimination phase (measured by ReadoutCalibration) is baked into the readout tables;
firing the demod IS the readout (its `dur` is the integration window). This module holds the batch
composers:
  - readout_tables / demod_table — the ro-drive + demod-carrier table pair (phase baked in),
  - grid_period / batch_timeout — the host-computed fixed period + poll timeout,
  - sweep_q16 — the (x0q, dxq) Q16 sweep descriptor + the exact int x-axis the kernel realizes,
  - sweep_counts — a params-only counts-mode sweep returning self-normalised populations P = counts/shots,
  - acquire_shots — a per-prep raw-mode capture returning one prep state's IQ cluster,
  - sweep_iqsum — an iqsum-mode sweep returning per-point coherent IQ sums (the matched-pair VNA).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from riscq import run as rq
from riscq.lang import ParamTable
from riscq.map import LEAD, READOUT_LEAD, READOUT_MAX_WIN_LOG2
from riscq.pulses import Pulse, envelopes, golden, units

SEP = LEAD              # pulse-end → readout separation (co-sim: the readout datapath needs ~LEAD)
GRID = 1600            # readout grid period for the readout-only VNA sweep (batches)
RELAX = 1600           # per-point idle before the drive; ≫ T1 so the qubit relaxes back to |0>
GATE_ENV = envelopes.square(16)   # default gate envelope (4 batches at gateInterp 4)


def demod_table(n: int, phase: float = 0.0) -> ParamTable:
    """A square demod-carrier pulse on channel 2 — the readout demod is a plain drive channel fed to
    the carrier-triggered decoder, programmed/played with the generic init_pulse_params/set_freq/play.
    Its carrier freq is set separately via set_freq with an ADC-rate demod_freq_to_code code, so the
    table freq is unused (0). Played per shot; its `dur` (n batches) IS the readout integration window
    (n <= env_depth and the decoder's no-overflow cap 2^READOUT_MAX_WIN_LOG2). `phase` is the
    discrimination-carrier phase (rad) baked into the slot — the programmable discriminator knob
    (spec 08 §2.1): it rotates the decoder IQ so `sign(sumR)` separates |0>/|1> (a compiled-in
    write_slot("phase")); ReadoutCalibration measures it."""
    assert n <= (1 << READOUT_MAX_WIN_LOG2), \
        f"demod window {n} exceeds the decoder no-overflow cap {1 << READOUT_MAX_WIN_LOG2} batches"
    return ParamTable(2, 0.0, {"sq": Pulse(envelopes.square(n), amp=1.0, phase=phase)})


def readout_tables(cfg, q: int, phase: float | None = None, dur: int | None = None):
    """The (channel-1 readout drive, channel-2 demod carrier) table pair for qubit q's readout, plus
    the demod carrier code and window. The readout drive `ro["meas"]` covers the demod window (its
    rising edge is the projective co-sim model's window trigger, and it is the physical measurement
    tone on hardware); the demod carrier's phase is the discrimination knob, baked from
    readout/{q}/demod_phase (or the `phase` override) — see demod_table. `dur` overrides the config
    window (Fidelity compiles the tables at the longest candidate window, then retunes the slot's
    `dur` field per point — spec 08 §4). Returns (ro_table, demod_table, demod_code, dur)."""
    ro_freq = float(cfg[f"readout/{q}/freq"])
    dur = int(cfg.get(f"readout/{q}/dur", 40)) if dur is None else int(dur)
    ph = float(cfg.get(f"readout/{q}/demod_phase", 0.0)) if phase is None else float(phase)
    ro = ParamTable(1, ro_freq, {"meas": Pulse(envelopes.square(dur + 16), freq_hz=ro_freq, amp=0.5)})
    demod = demod_table(dur, ph)
    demod_code = units.demod_freq_to_code(ro_freq, _params(cfg))
    return ro, demod, demod_code, dur


def grid_period(relax: int, seq_batches: int, dur: int) -> int:
    """The fixed readout-grid period (batches) for a batched sweep (spec 08 §2.2): each shot fires on
    this grid, whose idle head is the T1 relax reset, so every readout lands at the same demod-LO
    phase. `seq_batches` is the longest point's drive prelude (earliest pulse start -> t_ro). Sized
    period >= relax + seq_batches + SEP + dur + READOUT_LEAD and rounded UP to a multiple of 8 (so the
    phase repeats). Host-computed, fail-loud."""
    need = int(relax) + int(seq_batches) + SEP + int(dur) + READOUT_LEAD
    period = -(-need // 8) * 8
    assert period % 8 == 0 and period >= need
    return period


def batch_timeout(nbatches: int) -> int:
    """poll_done timeout (host/sim cycles) for a batch of `nbatches` grid batches: 4 cycles/batch +
    boot slack (mirrors tests/test_batch)."""
    return int(nbatches) * 4 + 20_000_000


def sweep_q16(x0: int, x1: int, n: int) -> tuple[int, int, np.ndarray]:
    """(x0q, dxq) kernel params + the EXACT int x-axis the kernel realizes (host-mirrored
    integer arithmetic, not the float linspace). Asserts every point is a legal 16-bit code."""
    dxq = 0 if n <= 1 else round(((x1 - x0) << 16) / (n - 1))
    xs = (x0 * (1 << 16) + np.arange(n, dtype=np.int64) * dxq) >> 16
    assert np.all(np.abs(xs) < (1 << 15))
    return x0 << 16, dxq, xs.astype(int)


def gate_sigma(m, pulse: Pulse, carrier_hz: float, amp_code: int) -> float:
    """Σ over the pulse of the per-batch drive estimate amp_est = √(2·mean(sample²)) on the
    bit-exact DAC golden — exactly what TwoLevelModel integrates, so θ = rabi_rad_per_amp·gate_sigma
    is the qubit's rotation angle. Linear in amp_code (used to convert a fitted Rabi rate)."""
    lines = pulse.packed_lines(m, 0)   # gate channel
    w = golden.pulse_window(lines, int(amp_code), units._freq_code(carrier_hz, m.params),
                            0, 0, len(lines))
    return float(sum(math.sqrt(2 * np.mean(row.astype(float) ** 2)) for row in w))


def socmap(drv):
    """Derive the build's SocMap from the driver (the co-sim exposes its SocParams JSON)."""
    from riscq.map import SocMap, SocParams
    return SocMap(SocParams.from_json(drv.sim.get_params()))


@dataclass
class Result:
    """A calibration outcome (spec 06 §1): the raw sweep `data`, the `fit`, and a `proposal`
    (config path → value). `.apply()` writes the proposal into the calibration's Config; `.plot()`
    draws the sweep (matplotlib imported lazily)."""

    ok: bool
    data: dict
    fit: object
    proposal: dict = field(default_factory=dict)
    cfg: object = None
    label: str = ""

    def apply(self):
        if not self.ok:
            raise RuntimeError(f"{self.label}: fit failed, refusing to update config")
        for path, value in self.proposal.items():
            self.cfg[path] = value
        return self.cfg

    def plot(self, ax=None):
        import matplotlib.pyplot as plt          # lazy: headless CI must import cal without it
        if ax is None:
            _, ax = plt.subplots()
        x, y = self.data.get("x"), self.data.get("y")
        if x is not None and y is not None:
            ax.plot(x, y, "o-")
        ax.set_title(f"{self.label} ({'ok' if self.ok else 'FAILED'})")
        return ax


# ── gate-pulse construction from the Config tree ──

def qubit_freq(cfg, q: int) -> float:
    return float(cfg[f"qubit/{q}/freq"])


def x90_pulse(cfg, q: int, env=GATE_ENV, amp: float | None = None) -> Pulse:
    a = float(cfg[f"qubit/{q}/x90/amp"]) if amp is None else amp
    return Pulse(env, freq_hz=qubit_freq(cfg, q), amp=a, phase=float(cfg.get(f"qubit/{q}/x90/phase", 0.0)))


def x180_pulse(cfg, q: int, env=GATE_ENV) -> Pulse:
    """π pulse = 2× the X90 amplitude (rotation is linear in amp), clamped clip-safe."""
    return Pulse(env, freq_hz=qubit_freq(cfg, q), amp=min(2 * float(cfg[f"qubit/{q}/x90/amp"]), 0.99))


def readout_knobs(cfg, q: int) -> tuple[int, int]:
    """(demod code, window batches) for qubit q's readout from the config."""
    code = units.demod_freq_to_code(float(cfg[f"readout/{q}/freq"]), _params(cfg))
    return code, int(cfg.get(f"readout/{q}/dur", 40))


_PARAMS_CACHE = {}


def _params(cfg):
    return _PARAMS_CACHE["params"]


def bind_params(m):
    """Give the pulse-unit helpers the build's SocParams (set once per driver)."""
    _PARAMS_CACHE["params"] = m.params


# ── batched sweeps: one run is the whole sweep (spec 08 §2, §6; spec 09 computed knobs) ──

def sweep_counts(drv, m, prog, core, params, shots: int, timeout: int) -> np.ndarray:
    """Batched counts-mode sweep: one params-only run of `prog` (spec 09) — the swept knob is COMPUTED
    on-core from the scalar `params` (a Q16 or int pair, e.g. {"a0q": ..., "daq": ..., "prep": 1}),
    so there are no input Arrays. Returns the self-normalised populations P = out/shots in [0, 1]
    (no |0> reference / projection)."""
    out = rq.run(drv, m, {core: prog}, params={core: params}, results=["out"],
                 timeout=timeout)[core]["out"]
    return out.astype(float) / shots


def acquire_shots(drv, m, prog, core, prep: int, shots: int, timeout: int) -> np.ndarray:
    """Per-prep raw-IQ capture (spec 09): one RAW-mode RERUN of an already-`setup` single-point program
    with the `prep` scalar written for this run, reshaped to (shots, 2) real/imag. prep=0 → |0>,
    prep=1 → |1> cluster (the two prep states are two reruns of the resident image, no reload)."""
    out = rq.rerun(drv, m, {core: prog}, params={core: {"prep": int(prep)}}, results=["out"],
                   timeout=timeout)[core]["out"]
    return out.reshape(shots, 2).astype(float)


def sweep_iqsum(drv, m, prog, core, params, shots: int, timeout: int) -> np.ndarray:
    """Batched iqsum-mode sweep (spec 08 §6; spec 09 computed knob): one matched-pair VNA run (k_vna)
    over an on-core Q16 frequency sweep (`params` = {"c0q": ..., "dcq": ...}). The kernel retunes ro +
    demod as a matched pair and coherently sums `shots` per-point IQ integrals (>> sh headroom).
    Returns the per-point (real, imag) as `out` reshaped (npts, 2)."""
    out = rq.run(drv, m, {core: prog}, params={core: params}, results=["out"],
                 timeout=timeout)[core]["out"]
    return out.astype(float).reshape(-1, 2)
