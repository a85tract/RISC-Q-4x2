"""DRAG calibration (spec 14 F3; walkthrough stage 3.4, reference section "DRAG"):

  optimize_fast_drag — qcal's `optimize_FAST_DRAG` (optimization/pulse.py:19-110), a HOST-only
                       coordinate descent over the FAST_DRAG hyperparameters that minimises the
                       pulse spectrum's magnitude AT THE EF TRANSITION. No hardware: the envelope
                       and its FFT are the whole measurement.
  Leakage            — qcal's `Leakage` (calibration/leakage.py:70-207): a fixed n×X90 train read for
                       P(|2>) while ONE Config value is swept, taking the value that minimises it.

The two are the reference's pair: the FFT optimum is a first guess from the pulse's own spectrum,
`Leakage` then measures the leakage the qubit actually accumulates and refines the same knobs (the
virtual-Z phases first, then `N` and `weights/0`).

Both sweep knobs that are COMPILED IN — a virtual-Z pair is a kernel binding, an envelope kwarg
changes the envelope image — so `Leakage` recompiles per point rather than retuning a slot (spec 14
F3: per-point recompile accepted). The train is deep (the reference uses 101 X90s), which is only
trustworthy since the trains were paced (spec 14 F1)."""

from __future__ import annotations

import numpy as np

from riscq.cal import kernels
from riscq.cal.base import (GATE_CH, Result, batch_timeout, batches, gate_pulse, grid_period,
                            prep, qubit_freq, qubits_list, readout_tables, relax_batches, socmap,
                            sweep_levels, train_step, x90_vz, X90)
from riscq.cal.qubit import _classifiers
from riscq.lang import Array, compile_kernel
from riscq.pulses import envelopes, units

N_GRID = tuple(range(2, 11))                                   # qcal's default sweeps
W_GRID = (0.1, 0.3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
PAD = 1000                                                     # zero-padding each side, as qcal


def ef_spectral_weight(cfg, q, m, name="x90", kwargs=None) -> float:
    """qcal's leakage score (optimization/pulse.py:57-72): |FFT(envelope)| at the EF transition.

    The envelope is built on the gate channel's stored-sample grid, zero-padded `PAD` samples each
    side (which interpolates the spectrum finely enough to land on the transition), transformed, and
    read at the bin nearest the EF DETUNING f_EF − f_GE — qcal shifts its frequency axis by f_GE and
    looks up f_EF, which is the same bin. Lower is less leakage: that spectral component is what
    drives |1> → |2>."""
    path = f"qubit/{q}/{name}"
    ch = m.channel(GATE_CH)
    n = batches(cfg[f"{path}/dur"], m) * ch.samples_per_line
    rate = ch.samples_per_line * m.params.dsp_freq_hz
    env = envelopes.build(cfg[f"{path}/env"], n,
                          rate, **(cfg[f"{path}/kwargs"] if kwargs is None else kwargs))
    spec = np.abs(np.fft.fftshift(np.fft.fft(np.pad(env, (PAD, PAD)))))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(env) + 2 * PAD, 1.0 / rate))
    detune = float(cfg[f"qubit/{q}/EF/freq"]) - qubit_freq(cfg, q)
    return float(spec[int(np.abs(freqs - detune).argmin())])


def optimize_fast_drag(cfg, q, m, name="x90", n_grid=N_GRID, w_grid=W_GRID) -> dict:
    """qcal's `optimize_FAST_DRAG`, host-only: coordinate descent on the FAST_DRAG hyperparameters,
    scoring each candidate by `ef_spectral_weight`. `N` (the number of cosine terms) is swept first
    and fixed at its argmin, then EACH `weights` entry in turn — qcal's order exactly, and the order
    matters, since the weights shape the spectrum the chosen `N` produces.

    Returns the proposal `{qubit/{q}/{name}/kwargs: {...}}` — the whole kwargs dict, since that is the
    Config leaf (F0 gave it a write-back path into the qcal tree). Pure host arithmetic: no driver, no
    shots. Call it before `Leakage`, which refines the same knobs against the real qubit."""
    kw = dict(cfg[f"qubit/{q}/{name}/kwargs"])
    assert "N" in kw and "weights" in kw, \
        f"optimize_fast_drag needs FAST_DRAG kwargs (N, weights) at qubit/{q}/{name}, got {sorted(kw)}"

    def sweep(values, apply):
        """Score every candidate, then leave `kw` at the one with the least EF weight."""
        scores = []
        for v in values:
            apply(v)
            scores.append(ef_spectral_weight(cfg, q, m, name, kw))
        apply(values[int(np.argmin(scores))])

    def set_n(v):
        kw["N"] = int(v)

    def set_w(v, i):
        w = list(kw["weights"])
        w[i] = float(v)
        kw["weights"] = w

    sweep(list(n_grid), set_n)
    for i in range(len(kw["weights"])):
        sweep(list(w_grid), lambda v, i=i: set_w(v, i))
    return {f"qubit/{q}/{name}/kwargs": kw}


class Leakage:
    """qcal's `Leakage` (calibration/leakage.py:70-207): play an n×X90 train and read P(|2>) off the
    pre-trained 3-level `classifier`, once per value of ONE swept Config path, and take the value that
    MINIMISES the leaked population (`maximize=True` flips it — qcal's diagnostic direction).

    The reference sweeps the X90's virtual-Z phases first, then the FAST_DRAG `N` and `weights/0`;
    all of those are compiled in (a vz pair is a kernel binding, a kwarg changes the envelope image),
    so each point is its own compile + run — no slot retune is possible. `path` is any Config path and
    `values` the list written to it, so the vz pair is swept by passing whole `[p, p]` pairs.

    `n_gates` is the amplification: qcal uses 101 X90s, which only became trustworthy once the trains
    were paced (spec 14 F1). The train is RAW-mode — the hardware `res` bit cannot separate |1> from
    |2>, so the level comes from the host classifier, exactly as the EF cals do.

    `path` may carry a `{q}` field (`"qubit/{q}/x90/vz"`), which is filled per qubit — the swept
    VALUES are shared across qubits (as in qcal, given a scalar sweep) but each qubit is minimised on
    its own P(|2>) and gets its own proposal."""

    def __init__(self, cfg, qubits, classifier, path, values, n_gates=101, shots=32,
                 maximize=False):
        self.cfg, self.qubits = cfg, qubits_list(qubits)
        self.classifiers = _classifiers(classifier, self.qubits)
        self.path, self.values = str(path), list(values)
        self.n_gates, self.shots, self.maximize = int(n_gates), int(shots), bool(maximize)
        self.data, self.fit = {}, {}

    def _point(self, drv, m, cfg):
        """One swept value: compile the n-gate train for every qubit against THIS config and run it."""
        progs, params, timeout = {}, {}, 0
        for q in self.qubits:
            table, pg, _ = prep(cfg, q, m, "X90")
            pulse = gate_pulse(cfg, q, m)
            ro, demod, code, dur, ddly = readout_tables(cfg, q, m)
            step = train_step(pulse.dur_batches(m, GATE_CH))
            seq = (self.n_gates - 1) * step + pulse.dur_batches(m, GATE_CH)
            period = grid_period(relax_batches(cfg, m), seq, dur, ddly)
            progs[q] = compile_kernel(kernels.k_rabi, m,
                                      tables=dict(gate=table, ro=ro, demod=demod),
                                      out=Array(2 * self.shots), npts=1, shots=self.shots,
                                      period=period, ngates=self.n_gates, step=step, code=code,
                                      mode=kernels.RAW, ddly=ddly, prep_gate=X90, herald=0, hoff=0,
                                      **x90_vz(cfg, q))
            a = units._amp_code(float(cfg[f"qubit/{q}/x90/amp"]))
            params[q] = {"a0q": a << 16, "daq": 0, "prep": 1}
            timeout = max(timeout, batch_timeout(self.shots * period))
        return sweep_levels(drv, m, progs, params, 1, self.shots, timeout, self.classifiers, level=2)

    def run(self, drv) -> Result:
        m = socmap(drv)
        pops = {q: [] for q in self.qubits}
        for v in self.values:
            cfg = self.cfg.copy()
            for q in self.qubits:
                cfg[self.path.format(q=q)] = v
            out = self._point(drv, m, cfg)
            for q in self.qubits:
                pops[q].append(float(np.ravel(out[q])[0]))

        data, fit, proposal = {}, {}, {}
        pick = np.argmax if self.maximize else np.argmin
        for q in self.qubits:
            y = np.array(pops[q])
            data[q] = {"x": np.arange(len(self.values), dtype=float), "y": y,
                       "values": list(self.values)}
            fit[q] = None
            proposal[self.path.format(q=q)] = self.values[int(pick(y))]
        self.data, self.fit = data, fit
        return Result(True, data, fit, proposal, self.cfg,
                      f"Leakage {self.qubits} {self.path}")
