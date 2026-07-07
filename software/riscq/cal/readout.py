"""Readout calibrations (spec 06 §2; batched per spec 08 §6): ReadoutCalibration (raw-IQ per prep
state → classifier + res-sign), ReadoutFidelity (confusion matrix), Separation (readout/demod
frequency), Fidelity (readout SNR diagnostic). The raw cals are a setup-once RAW program rerun once
per prep state (prep=0 → |0>, prep=1 → |1>, a runtime scalar — spec 09); Separation is the matched-pair
iqsum VNA (one computed Q16 run); Fidelity a setup-once program retuned per window with write_slot +
the two prep reruns (no recompile).
The classifier is supervised — the |0>/|1> prep states are captured as labelled clusters (no
unsupervised GMM needed): a linear discriminant along the cluster axis, which for two π-out-of-phase
readout tones is exactly the optimal separator."""

from __future__ import annotations

import math

import numpy as np

from riscq import run as rq
from riscq.cal import kernels
from riscq.cal.base import (GATE_ENV, RELAX, SEP, Result, acquire_shots, batch_timeout, bind_params,
                            grid_period, qubit_freq, readout_tables, socmap, sweep_iqsum, sweep_q16,
                            x180_pulse)
from riscq.lang import Array, ParamTable, compile_kernel
from riscq.map import READOUT_MAX_WIN_LOG2
from riscq.pulses import units


class Classifier:
    """Two labelled Gaussian IQ clusters (|0>, |1>). Classify along the cluster axis; separation is
    the cluster-distance SNR (mirrors qcal's GMM SNR for well-separated blobs)."""

    def __init__(self, iq0: np.ndarray, iq1: np.ndarray):
        self.iq0, self.iq1 = iq0, iq1
        self.m0, self.m1 = iq0.mean(0), iq1.mean(0)
        axis = self.m1 - self.m0
        self.axis = axis / (np.hypot(*axis) or 1.0)
        self.mid = 0.5 * (self.m0 + self.m1)
        p0, p1 = iq0 @ self.axis, iq1 @ self.axis
        self.thresh = 0.5 * (p0.mean() + p1.mean())
        spread = 0.5 * (p0.std() + p1.std()) or 1e-9
        self.separation = float(abs(p1.mean() - p0.mean()) / spread)   # SNR

    def classify(self, iq: np.ndarray) -> np.ndarray:
        """0 (|0>) / 1 (|1>) per point, by which side of the threshold along the cluster axis."""
        side = np.atleast_2d(iq) @ self.axis
        return (side > self.thresh).astype(int) if (self.m1 @ self.axis) > (self.m0 @ self.axis) \
            else (side < self.thresh).astype(int)

    def confusion(self) -> np.ndarray:
        """2×2 confusion: row = prepared state, col = classified state (normalised)."""
        c = np.zeros((2, 2))
        for prep, iq in ((0, self.iq0), (1, self.iq1)):
            pred = self.classify(iq)
            c[prep, 0] = np.mean(pred == 0)
            c[prep, 1] = np.mean(pred == 1)
        return c


def _rawiq_prog(m, cfg, q, env, shots, relax=RELAX, dur=None):
    """Batched raw-IQ program (spec 09): k_t1 in RAW mode at a fixed delay (d0=SEP, dd=0), one point —
    a plain |1>-prep readout when prep=1, |0> when prep=0. `prep` is a runtime scalar written per
    rerun, so the two prep states are two reruns of the one resident image. `dur` overrides the config
    window (Fidelity compiles at the longest candidate window, then retunes the demod slot's `dur`
    field per window). Returns (prog, period)."""
    ro, demod, code, win = readout_tables(cfg, q, dur=dur)
    x180 = x180_pulse(cfg, q, env)
    gate = ParamTable(0, qubit_freq(cfg, q), {"x180": x180})
    period = grid_period(relax, SEP + x180.dur_batches(m, gate.channel), win)
    prog = compile_kernel(kernels.k_t1, m, tables=dict(gate=gate, ro=ro, demod=demod),
                          out=Array(2 * shots), npts=1, shots=shots,
                          period=period, code=code, mode=kernels.RAW, d0=SEP, dd=0)
    return prog, period


def _clusters(drv, m, prog, q, period, shots):
    """setup once, then a prep=0 and a prep=1 rerun → the (|0>, |1>) IQ clusters."""
    timeout = batch_timeout(shots * period)
    rq.setup(drv, m, {q: prog})
    iq0 = acquire_shots(drv, m, prog, q, 0, shots, timeout)
    iq1 = acquire_shots(drv, m, prog, q, 1, shots, timeout)
    return iq0, iq1


class ReadoutCalibration:
    """Raw-IQ per prep state → a trained Classifier; fixes the res-sign convention by recording the
    demod phase that lands |0> on +real and which state is 'excited'."""

    def __init__(self, cfg, qubits, shots=16, env=GATE_ENV, relax=RELAX):
        self.cfg, self.q = cfg, _one(qubits)
        self.shots, self.env, self.relax = int(shots), env, int(relax)
        self.classifier = None
        self.data = {}
        self.fit = None

    def run(self, drv) -> Result:
        m = socmap(drv); bind_params(m)
        cfg, q = self.cfg, self.q
        prog, period = _rawiq_prog(m, cfg, q, self.env, self.shots, relax=self.relax)
        iq0, iq1 = _clusters(drv, m, prog, q, period, self.shots)   # prep=0 → |0>, prep=1 → |1>
        clf = Classifier(iq0, iq1)
        self.classifier = clf
        self.data = {"iq0": iq0, "iq1": iq1, "separation": clf.separation}
        demod_phase = -math.atan2(*clf.m0[::-1])        # rotate |0> mean onto +real
        proposal = {f"readout/{q}/demod_phase": float(demod_phase),
                    f"readout/{q}/res_sign": 1}          # |0>→+real→res=0, |1>→res=1
        ok = clf.separation > 1.0
        return Result(ok, self.data, clf, proposal, cfg, f"ReadoutCalibration q{q}")


class ReadoutFidelity:
    """Confusion matrix from the trained clusters (diagnostic; assignment fidelity = mean diagonal).
    Captures the |0>/|1> clusters in ONE batched RAW run (spec 08 §6), exactly like ReadoutCalibration,
    then reports the achieved assignment fidelity. Always ok — it is a measurement, not a fit."""

    def __init__(self, cfg, qubits, shots=16, env=GATE_ENV, relax=RELAX):
        self.cfg, self.q = cfg, _one(qubits)
        self.shots, self.env, self.relax = int(shots), env, int(relax)
        self.data = {}
        self.fit = None

    def run(self, drv) -> Result:
        m = socmap(drv); bind_params(m)
        cfg, q = self.cfg, self.q
        prog, period = _rawiq_prog(m, cfg, q, self.env, self.shots, relax=self.relax)
        iq0, iq1 = _clusters(drv, m, prog, q, period, self.shots)   # prep=0 → |0>, prep=1 → |1>
        clf = Classifier(iq0, iq1)
        conf = clf.confusion()
        fidelity = float(np.mean(np.diag(conf)))
        self.data = {"confusion": conf, "fidelity": fidelity, "separation": clf.separation}
        proposal = {f"readout/{q}/fidelity": fidelity}
        return Result(True, self.data, conf, proposal, cfg, f"ReadoutFidelity q{q}")


class Separation:
    """Readout frequency: the matched-pair VNA batched on-core in iqsum mode (spec 08 §2.2). One run
    sweeps the readout-drive DAC codes around the config center; k_vna retunes the readout drive and
    demod as a matched pair on-core (`set_freq(ro, codes[i])` + `set_freq(demod, 4*codes[i])` — the
    ADC-rate demod code is exactly 4× the DAC-rate drive code) and coherently sums `shots` per-point
    IQ integrals. The code of maximum |z| is the matched demod, i.e. the readout tone frequency
    (readout/q/freq = argmax |z|); a |0> VNA is enough because the |0>/|1> tones are π out of phase,
    so cluster separation ∝ |z|. (The demod DOES retune + re-play on-core — spec 08 §2.2 refuted the
    old "can't be re-tuned on-core" claim as insufficient-lead folklore, so the VNA batches in iqsum
    mode like every other cal.)"""

    def __init__(self, cfg, qubits, span_code=200, points=17, shots=16, relax=RELAX):
        self.cfg, self.q = cfg, _one(qubits)
        self.span_code, self.points = int(span_code), int(points)
        self.shots, self.relax = int(shots), int(relax)
        self.data = {}
        self.fit = None

    def run(self, drv) -> Result:
        m = socmap(drv); bind_params(m)
        cfg, q = self.cfg, self.q
        ro, demod, _, dur = readout_tables(cfg, q)
        c0 = units.freq_to_code(float(cfg[f"readout/{q}/freq"]), m.params)      # DAC-rate center code
        c0q, dcq, xs = sweep_q16(c0 - self.span_code, c0 + self.span_code, self.points)  # on-core sweep
        npts = self.points
        period = grid_period(self.relax, 0, dur)                               # no qubit prep; idle head
        sh = max(0, (self.shots - 1).bit_length())                            # >> headroom for the sum
        prog = compile_kernel(kernels.k_vna, m, tables=dict(ro=ro, demod=demod),
                              out=Array(2 * npts),
                              npts=npts, shots=self.shots, period=period, sh=sh)
        iq = sweep_iqsum(drv, m, prog, q, {"c0q": int(c0q), "dcq": int(dcq)}, self.shots,
                         batch_timeout(npts * self.shots * period))
        mag = np.hypot(iq[:, 0], iq[:, 1])
        best = int(np.argmax(mag))
        self.data = {"x": np.array(xs, float), "y": mag}
        freq = units.code_to_freq(int(xs[best]), m.params)
        proposal = {f"readout/{q}/freq": float(freq)}
        ok = mag[best] > 2 * np.median(mag)
        return Result(ok, self.data, None, proposal, cfg, f"Separation q{q}")


class Fidelity:
    """Readout SNR vs the integration window (spec 06's readout-amp lever is a no-op in the soft
    measurement model, so the other SNR lever — window duration — is calibrated instead): sweep
    `dur`, take the window that maximises cluster separation. The window is a demod-table slot field,
    so ONE raw-capture program is compiled at the LONGEST candidate window (rq.setup once) and each
    `dur` is a `write_slot("dur", d)` + `rerun` — two writes, no recompile (spec 08 §4)."""

    def __init__(self, cfg, qubits, durs=(16, 40), shots=12, env=GATE_ENV, relax=RELAX):
        self.cfg, self.q = cfg, _one(qubits)
        self.durs = [int(d) for d in durs]
        self.shots, self.env, self.relax = int(shots), env, int(relax)
        self.data = {}
        self.fit = None

    def run(self, drv) -> Result:
        m = socmap(drv); bind_params(m)
        cfg, q = self.cfg, self.q
        prog, period = _rawiq_prog(m, cfg, q, self.env, self.shots,           # compile at the longest
                                   relax=self.relax, dur=max(self.durs))       # window; retune per dur
        timeout = batch_timeout(self.shots * period)
        rq.setup(drv, m, {q: prog})
        seps = []
        for d in self.durs:
            assert d <= (1 << READOUT_MAX_WIN_LOG2), \
                f"demod window {d} over the decoder no-overflow cap {1 << READOUT_MAX_WIN_LOG2}"  # §4 host
            rq.write_slot(drv, m, q, prog, "demod", 0, "dur", d)              # retune the window (no recompile)
            iq0 = acquire_shots(drv, m, prog, q, 0, self.shots, timeout)   # |0>
            iq1 = acquire_shots(drv, m, prog, q, 1, self.shots, timeout)   # |1>
            seps.append(Classifier(iq0, iq1).separation)
        best = int(np.argmax(seps))
        self.data = {"x": np.array(self.durs, float), "y": np.array(seps)}
        proposal = {f"readout/{q}/dur": int(self.durs[best])}
        ok = seps[best] > 1.0
        return Result(ok, self.data, None, proposal, cfg, f"Fidelity q{q}")


def _one(qubits):
    return qubits[0] if isinstance(qubits, (list, tuple)) else int(qubits)
