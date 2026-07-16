"""Readout calibrations (spec 06 §2; batched per spec 08 §6; corrected to qcal's statistics per
spec 13 §5):

  ReadoutCalibration — raw IQ per prep state → the trained Classifier + the demod discrimination phase
                       and res-sign. The classifier rides on the Result for the later steps to REUSE.
  Separation         — readout frequency: the matched-pair frequency sweep run at BOTH prep states
                       (k_vna RAW, two reruns), argmax of the two-state cluster SNR — qcal's statistic,
                       not the |0>-magnitude peak (§5: on a dispersive resonator they are different
                       frequencies).
  Fidelity           — readout DRIVE AMPLITUDE (`readout/{q}/amp`, qcal's knob): argmax of the
                       confusion diagonal ½[P(0|0) + P(1|1)] under the FIXED hardware discriminator.
  ReadoutFidelity    — the confusion matrix at the calibrated amplitude, straight from the `res` bit
                       under that same fixed discriminator (no recapture, no retraining).
  Window             — OURS, not qcal's: the demod integration window (`readout/{q}/demod/dur`) sweep,
                       scored exactly like Fidelity. `calibration_x6y3` does not run it.

Every cal takes `qubits: list` (a bare int is one qubit) and runs them SIMULTANEOUSLY (spec 13 §8):
one program per core, ONE setup + the reruns over all of them, then per-qubit analysis on the host.
The `Result.data`/`fit` are per-qubit dicts; `proposal` merges (the paths carry `q`).

Discrimination stays ON-CHIP (spec 13 §2): the `res` bit is the classifier that actually gates
feedback, so Fidelity/ReadoutFidelity/Window score IT (counts mode) rather than a host classifier
retrained per point. The host Classifier is a supervised linear discriminant over the two labelled
prep clusters (the boundary a 2-component GMM finds on labelled Gaussians); only its SNR is qcal's
(`_snr`).

Every knob is PHYSICAL (spec 13 §2): `span` in Hz, `amp_span` a normalized amplitude, `durs` in
seconds; the relax head comes from the Config (`reset/relax`). The |1> prep is qcal's `gate=` choice
(spec 13 §4): 'X90' plays the Config's X90 twice, 'X' plays its own X pulse once (base.prep)."""

from __future__ import annotations

import math

import numpy as np

from riscq import run as rq
from riscq.cal import kernels
from riscq.cal.base import (SEP, Result, acquire_shots, batch_timeout, batches, grid_period,
                            herald_offset, heralding, prep, qubits_list, readout_tables,
                            relax_batches, rerun_counts, res_sign, seconds, socmap, sweep_q16,
                            x90_vz)
from riscq.lang import Array, compile_kernel
from riscq.map import READOUT_MAX_WIN_LOG2
from riscq.pulses import units


def _snr(iq0: np.ndarray, iq1: np.ndarray) -> float:
    """qcal's two-cluster SNR (machine_learning/clustering.py:120, adopted verbatim per spec 13 §2 so
    that thresholds and argmaxes are comparable numbers): ‖Δmeans‖ / Σ(2·√cov), where `cov` is the
    SPHERICAL cluster variance its GMM fits — the mean of the per-axis variances. Note the
    denominator is 2σ₀ + 2σ₁, so SNR = 1 means the cluster means are 4σ apart (~2 % assignment
    error), and the number is ~4× smaller than a plain distance/σ SNR."""
    dist = float(np.hypot(*(iq1.mean(0) - iq0.mean(0))))
    err = sum(2.0 * math.sqrt(float(np.mean(iq.var(0)))) for iq in (iq0, iq1))
    return dist / (err or 1e-9)


class Classifier:
    """Two labelled Gaussian IQ clusters (|0>, |1>). Classify along the cluster axis (a supervised
    linear discriminant — for labelled clusters, the same boundary a 2-component GMM finds); the
    `separation` is qcal's cluster SNR (`_snr`)."""

    def __init__(self, iq0: np.ndarray, iq1: np.ndarray):
        self.iq0, self.iq1 = iq0, iq1
        self.m0, self.m1 = iq0.mean(0), iq1.mean(0)
        axis = self.m1 - self.m0
        self.axis = axis / (np.hypot(*axis) or 1.0)
        self.mid = 0.5 * (self.m0 + self.m1)
        p0, p1 = iq0 @ self.axis, iq1 @ self.axis
        self.thresh = 0.5 * (p0.mean() + p1.mean())
        self.separation = _snr(iq0, iq1)

    def classify(self, iq: np.ndarray) -> np.ndarray:
        """0 (|0>) / 1 (|1>) per point, by which side of the threshold along the cluster axis."""
        side = np.atleast_2d(iq) @ self.axis
        return (side > self.thresh).astype(int) if (self.m1 @ self.axis) > (self.m0 @ self.axis) \
            else (side < self.thresh).astype(int)

    def confusion(self) -> np.ndarray:
        """2×2 confusion: row = prepared state, col = classified state (normalised)."""
        c = np.zeros((2, 2))
        for state, iq in ((0, self.iq0), (1, self.iq1)):
            pred = self.classify(iq)
            c[state, 0] = np.mean(pred == 0)
            c[state, 1] = np.mean(pred == 1)
        return c


def _rawiq_prog(m, cfg, q, gate, shots):
    """Batched raw-IQ program (spec 09): k_t1 in RAW mode at a fixed delay (d0=SEP, dd=0), one point —
    a plain |1>-prep readout when prep=1, |0> when prep=0. `prep` is a runtime scalar written per
    rerun, so the two prep states are two reruns of the one resident image. `gate` ('X90' | 'X') is
    qcal's |1>-prep choice (spec 13 §4), folded into the kernel at compile time.

    The capture runs at demod phase ZERO (`phase=0.0`), NOT the config's stored phase: the proposal
    ReadoutCalibration derives from these clusters is an ABSOLUTE phase (spec 13 §5's rule rotates the
    |0>→|1> axis onto +real), so baking the current phase into the carrier would shift the measured
    axis by exactly that stale value and turn the proposal relative — invisible on the co-sim configs
    (stored phase 0), wrong on X6Y3 (−109.9°…+39.0°). Returns (prog, period)."""
    ro, demod, code, dur, ddly = readout_tables(cfg, q, m, phase=0.0)
    table, pg, plen = prep(cfg, q, m, gate)
    period = grid_period(relax_batches(cfg, m), SEP + plen, dur, ddly)
    prog = compile_kernel(kernels.k_t1, m, tables=dict(gate=table, ro=ro, demod=demod),
                          out=Array(2 * shots), npts=1, shots=shots, period=period, code=code,
                          mode=kernels.RAW, ddly=ddly, d0=SEP, dd=0, prep_gate=pg,
                          herald=0, hoff=0, **x90_vz(cfg, q))   # RAW readout capture: never heralded
    return prog, period


def _ro_amp_prog(m, cfg, q, gate, shots, npts, a0q, daq, win=None):
    """The COUNTS readout-amp program (k_ro_amp, spec 13 §5): prep (a runtime scalar) → a measurement
    at the swept readout-drive amplitude → the hardware-classified bit. npts=1 / daq=0 is the
    single-amplitude confusion program (ReadoutFidelity / Window). `win` (seconds) overrides the config
    demod window — Window compiles at the LONGEST candidate and retunes the slot's `dur` per point.
    `readout/herald` folds the same pre-prep herald read as the qubit cals' kernels (spec 13 §8 —
    qcal heralds EVERY circuit, confusion included); the callers decode with the matching flag.
    Returns (prog, period)."""
    ro, demod, code, dur, ddly = readout_tables(cfg, q, m, win=win)
    table, pg, plen = prep(cfg, q, m, gate)
    herald = heralding(cfg)
    period = grid_period(relax_batches(cfg, m), SEP + plen, dur, ddly, herald=herald)
    hoff = herald_offset(plen, ddly) if herald else 0
    prog = compile_kernel(kernels.k_ro_amp, m, tables=dict(gate=table, ro=ro, demod=demod),
                          out=Array(2 * npts if herald else npts), npts=npts, shots=shots,
                          period=period, code=code, ddly=ddly, prep_gate=pg, a0q=int(a0q),
                          daq=int(daq), herald=int(herald), hoff=hoff, **x90_vz(cfg, q))
    return prog, period


def _diagonal(drv, m, progs, shots, timeout, signs, herald=False):
    """The confusion diagonal under the HARDWARE discriminator across all cores: two COUNTS reruns of
    the resident programs (prep=0, prep=1) → ({q: P(1|0)}, {q: P(1|1)}, {q: ½[P(0|0) + P(1|1)]}). The
    discriminator is the `res` bit under the demod phase ReadoutCalibration fixed — NOT retrained per
    point, so this scoring cannot train on test (spec 13 §5). `herald` selects the interleaved
    (count, kept) decode (spec 13 §8) and must match what _ro_amp_prog compiled."""
    p0 = rerun_counts(drv, m, progs, {q: {"prep": 0} for q in progs}, shots, timeout, signs,
                      herald=herald)
    p1 = rerun_counts(drv, m, progs, {q: {"prep": 1} for q in progs}, shots, timeout, signs,
                      herald=herald)
    return p0, p1, {q: 0.5 * ((1.0 - p0[q]) + p1[q]) for q in progs}


class ReadoutCalibration:
    """Raw-IQ per prep state → a trained Classifier; fixes the discriminator by recording the demod
    phase that lands the |0>→|1> cluster AXIS on the real axis, and the res-sign convention. `gate`
    ('X90' | 'X') is qcal's |1>-prep choice (spec 13 §4). The trained classifier rides on the Result
    (`Result.fit[q]`, and `self.classifier[q]`) so the later steps reuse it instead of retraining.

    The proposed demod phase rotates m0 − m1 (NOT m0) onto +real: the hardware discriminator is
    sign(sumR) with a ZERO threshold, so what has to lie along the real axis is the |0>→|1> DIFFERENCE
    (|0> on the + side), with the cluster midpoint left on the imaginary axis. For a π-out-of-phase
    readout (m1 = −m0) the two rules are the same rotation; for a dispersive one they are not, and
    rotating m0 onto +real would leave BOTH clusters on the +real side — i.e. no discrimination.
    The proposal is ABSOLUTE, so the capture runs in the ZERO demod frame (_rawiq_prog) — re-running
    after apply() proposes the same phase again (a fixed point). The classifier and res_sign are
    trained/defined on that zero-frame capture, consistent with the phase being proposed against it."""

    def __init__(self, cfg, qubits, shots=16, gate="X90"):
        self.cfg, self.qubits = cfg, qubits_list(qubits)
        self.shots, self.gate = int(shots), gate
        self.classifier, self.data, self.fit = {}, {}, {}

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg = self.cfg
        progs, timeout = {}, 0
        for q in self.qubits:
            prog, period = _rawiq_prog(m, cfg, q, self.gate, self.shots)
            progs[q] = prog
            timeout = max(timeout, batch_timeout(self.shots * period))
        rq.setup(drv, m, progs)
        iq0 = acquire_shots(drv, m, progs, 0, self.shots, timeout)   # prep=0 → |0>
        iq1 = acquire_shots(drv, m, progs, 1, self.shots, timeout)   # prep=1 → |1>

        data, fit, proposal, oks = {}, {}, {}, {}
        for q in self.qubits:
            clf = Classifier(iq0[q], iq1[q])
            self.classifier[q] = clf
            data[q] = {"iq0": iq0[q], "iq1": iq1[q], "separation": clf.separation}
            demod_phase = -math.atan2(*(clf.m0 - clf.m1)[::-1])   # |0>→|1> axis onto +real (|0> on +real)
            proposal[f"readout/{q}/demod/phase"] = float(demod_phase)
            proposal[f"readout/{q}/res_sign"] = 1          # |0>→+real→res=0, |1>→res=1 (base.res_sign)
            fit[q] = clf
            oks[q] = clf.separation > 1.0                  # qcal SNR 1 = means 4σ apart (see _snr)
        self.data, self.fit = data, fit
        return Result(all(oks.values()), data, fit, proposal, cfg,
                      f"ReadoutCalibration {self.qubits}")


class Separation:
    """Readout frequency (qcal's statistic, spec 13 §5): the matched-pair frequency sweep (k_vna) in
    RAW mode, run TWICE — prep=0 and prep=1, two reruns of ONE resident program — so the host has both
    prep states' IQ clusters at every frequency. It trains a Classifier per frequency point and takes
    the argmax of the two-state cluster SNR.

    NOT the |0>-magnitude peak this cal used to take: on a dispersive resonator the |0> and |1>
    responses are Lorentzians split by 2χ, so max |S21| at |0> sits on the |0> dressed peak while max
    SEPARATION sits between the two — the whole point of a ±few-MHz sweep. (`data["mag0"]` keeps the
    |0> magnitude for the plot, and it is what the old cal would have argmax'd.)

    Sizing: the RAW capture is 2·points·shots words per prep and lives in the core's RAM — asserted
    against half of it, fail-loud (31 points × 32 shots = 1984 words ≈ 8 KB of 16 KB)."""

    def __init__(self, cfg, qubits, span=2.5e6, points=31, shots=32, gate="X90"):
        self.cfg, self.qubits = cfg, qubits_list(qubits)
        self.span, self.points = float(span), int(points)
        self.shots, self.gate = int(shots), gate
        self.data, self.fit = {}, {}

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg = self.cfg
        npts, shots = self.points, self.shots
        out_bytes = 4 * 2 * npts * shots
        assert out_bytes <= m.mem_bytes // 2, \
            (f"Separation's RAW capture is {out_bytes} B ({npts} points × {shots} shots × 2 words) — "
             f"over half the core's {m.mem_bytes} B RAM; cut points or shots")
        progs, meta, timeout = {}, {}, 0
        for q in self.qubits:
            ro, demod, _, dur, ddly = readout_tables(cfg, q, m)
            table, pg, plen = prep(cfg, q, m, self.gate)
            c0 = units._freq_code(float(cfg[f"readout/{q}/freq"]), m.params)   # DAC-rate center code (plain)
            span = abs(units._freq_code(self.span, m.params))                  # the span as a code offset
            c0q, dcq, xs = sweep_q16(c0 - span, c0 + span, npts)               # on-core sweep
            period = grid_period(relax_batches(cfg, m), SEP + plen, dur, ddly)
            progs[q] = compile_kernel(kernels.k_vna, m, tables=dict(gate=table, ro=ro, demod=demod),
                                      out=Array(2 * npts * shots), npts=npts, shots=shots, period=period,
                                      sh=0, ddly=ddly, mode=kernels.RAW, prep_gate=pg,
                                      c0q=int(c0q), dcq=int(dcq), **x90_vz(cfg, q))
            meta[q] = np.array(xs, float)
            timeout = max(timeout, batch_timeout(npts * shots * period))
        rq.setup(drv, m, progs)
        iq0 = acquire_shots(drv, m, progs, 0, npts * shots, timeout)
        iq1 = acquire_shots(drv, m, progs, 1, npts * shots, timeout)

        data, fit, proposal, oks = {}, {}, {}, {}
        for q in self.qubits:
            i0, i1 = iq0[q].reshape(npts, shots, 2), iq1[q].reshape(npts, shots, 2)
            seps = np.array([_snr(i0[i], i1[i]) for i in range(npts)])
            mag0 = np.hypot(i0[:, :, 0].mean(1), i0[:, :, 1].mean(1))   # what the OLD cal argmax'd
            best = int(np.argmax(seps))
            # DELTA-based physical Hz (spec 13 §2: codes never leave run()): the swept codes alias
            # (Nyquist fold) — X6Y3's 6.55 GHz readout in the DAC's 2nd Nyquist zone would come back
            # as its −1.44 GHz baseband alias through code_to_freq. f0 + code_to_freq(Δcode) stays in
            # the config's band, and is code-exact: re-deriving the code from the written-back Hz
            # reproduces xs[best] bit-for-bit (_freq_code folds mod 2^16 and Δcode is an integer).
            f0 = float(cfg[f"readout/{q}/freq"])
            c0 = units._freq_code(f0, m.params)
            freqs = f0 + units.code_to_freq(meta[q] - c0, m.params)
            data[q] = {"x": freqs, "y": seps, "mag0": mag0}
            proposal[f"readout/{q}/freq"] = float(freqs[best])
            fit[q] = Classifier(i0[best], i1[best])
            oks[q] = bool(seps[best] > 0.5)          # at least a 2σ split at the best frequency (see _snr)
        self.data, self.fit = data, fit
        return Result(all(oks.values()), data, fit, proposal, cfg, f"Separation {self.qubits}")


class Fidelity:
    """Readout assignment fidelity vs the readout DRIVE AMPLITUDE — qcal's knob and qcal's scoring
    (spec 13 §5). Sweeps `readout/{q}/amp` on-core over ±`amp_span` around the config value (a Q16 pair
    on the ro channel, k_ro_amp — the same machinery as k_rabi's gate-amp sweep) in COUNTS mode, one
    rerun per prep state, and scores each point with ½[P(0|0) + P(1|1)]: the confusion diagonal under
    the FIXED hardware discriminator, whose demod phase ReadoutCalibration measured and which is NOT
    retrained per point. argmax → `readout/{q}/amp`.

    (The demod-WINDOW sweep this class used to run — with a classifier retrained per point, i.e. scored
    on the very data it was trained on — is now `Window`, scored the same honest way.)"""

    def __init__(self, cfg, qubits, amp_span=0.005, points=11, shots=32, gate="X90"):
        self.cfg, self.qubits = cfg, qubits_list(qubits)
        self.amp_span, self.points = float(amp_span), int(points)
        self.shots, self.gate = int(shots), gate
        self.data, self.fit = {}, {}

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg = self.cfg
        progs, meta, signs, timeout = {}, {}, {}, 0
        for q in self.qubits:
            a = float(cfg[f"readout/{q}/amp"])
            # qcal sweeps exactly ±amp_span; clamp only to the legal [0, 1] range — an AMP_MIN floor
            # would silently truncate the lower half-span for X6Y3-class amps (q5: 0.0115).
            lo, hi = max(0.0, a - self.amp_span), min(1.0, a + self.amp_span)
            assert lo < hi, f"readout amp sweep [{lo}, {hi}] is empty (amp={a}, span={self.amp_span})"
            a0q, daq, xs = sweep_q16(units._amp_code(lo), units._amp_code(hi), self.points)
            prog, period = _ro_amp_prog(m, cfg, q, self.gate, self.shots, self.points, a0q, daq)
            progs[q] = prog
            meta[q] = np.array(xs, float) / units.AMP_SCALE            # the realized amps (host mirror)
            signs[q] = res_sign(cfg, q)
            timeout = max(timeout, batch_timeout(self.points * self.shots * period))
        rq.setup(drv, m, progs)
        p0, p1, fid = _diagonal(drv, m, progs, self.shots, timeout, signs, heralding(cfg))

        data, fit, proposal, oks = {}, {}, {}, {}
        for q in self.qubits:
            amps, fidq = meta[q], fid[q]
            best = int(np.argmax(fidq))
            data[q] = {"x": amps, "y": fidq, "p0": p0[q], "p1": p1[q]}
            proposal[f"readout/{q}/amp"] = float(amps[best])
            fit[q] = None
            oks[q] = bool(fidq[best] > 0.75)
        self.data, self.fit = data, fit
        return Result(all(oks.values()), data, fit, proposal, cfg, f"Fidelity {self.qubits}")


class ReadoutFidelity:
    """The confusion matrix at the calibrated readout amplitude, from the `res` bit under the FIXED
    hardware discriminator (qcal, spec 13 §5): two COUNTS reruns (prep=0, prep=1) of a single-point
    k_ro_amp program. Two words of output, no raw IQ, no retraining — unlike the old version (which
    recaptured the clusters, retrained a classifier on them and confused the same points), this one can
    actually detect the drift it exists to detect. Rows = prepared state, cols = measured state;
    assignment fidelity = the mean diagonal. Always ok — it is a measurement, not a fit."""

    def __init__(self, cfg, qubits, shots=64, gate="X90"):
        self.cfg, self.qubits = cfg, qubits_list(qubits)
        self.shots, self.gate = int(shots), gate
        self.data, self.fit = {}, {}

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg = self.cfg
        progs, signs, timeout = {}, {}, 0
        for q in self.qubits:
            a = units._amp_code(float(cfg[f"readout/{q}/amp"]))
            prog, period = _ro_amp_prog(m, cfg, q, self.gate, self.shots, 1, a << 16, 0)
            progs[q] = prog
            signs[q] = res_sign(cfg, q)
            timeout = max(timeout, batch_timeout(self.shots * period))
        rq.setup(drv, m, progs)
        p0, p1, fid = _diagonal(drv, m, progs, self.shots, timeout, signs, heralding(cfg))

        data, fit, proposal = {}, {}, {}
        for q in self.qubits:
            conf = np.array([[1.0 - p0[q][0], p0[q][0]], [1.0 - p1[q][0], p1[q][0]]])
            fidelity = float(fid[q][0])
            data[q] = {"confusion": conf, "fidelity": fidelity}
            fit[q] = conf
            proposal[f"readout/{q}/fidelity"] = fidelity
        self.data, self.fit = data, fit
        return Result(True, data, fit, proposal, cfg, f"ReadoutFidelity {self.qubits}")


class Window:
    """The demod INTEGRATION WINDOW (`readout/{q}/demod/dur`, seconds) — OURS, not qcal's (spec 13 §5),
    which is why `calibration_x6y3` does not run it — scored exactly like Fidelity: the confusion
    diagonal under the fixed hardware discriminator. The window is a demod-table slot field, so ONE
    COUNTS program is compiled at the LONGEST candidate window (rq.setup once) and each point is a
    `write_slot("dur", w)` + the two prep reruns — no recompile (spec 08 §4)."""

    def __init__(self, cfg, qubits, durs=(160e-9, 400e-9), shots=32, gate="X90"):
        self.cfg, self.qubits = cfg, qubits_list(qubits)
        self.durs = [float(d) for d in durs]
        self.shots, self.gate = int(shots), gate
        self.data, self.fit = {}, {}

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg = self.cfg
        wins = [batches(d, m) for d in self.durs]
        for w in wins:
            assert w <= (1 << READOUT_MAX_WIN_LOG2), \
                f"demod window {w} over the decoder no-overflow cap {1 << READOUT_MAX_WIN_LOG2}"
        progs, signs, timeout = {}, {}, 0
        for q in self.qubits:
            a = units._amp_code(float(cfg[f"readout/{q}/amp"]))
            prog, period = _ro_amp_prog(m, cfg, q, self.gate, self.shots, 1, a << 16, 0,
                                        win=max(self.durs))       # compile at the longest window
            progs[q] = prog
            signs[q] = res_sign(cfg, q)
            timeout = max(timeout, batch_timeout(self.shots * period))
        rq.setup(drv, m, progs)
        fids = {q: [] for q in self.qubits}
        for w in wins:
            for q in self.qubits:
                rq.write_slot(drv, m, q, progs[q], "demod", 0, "dur", w)      # retune (no recompile)
            diag = _diagonal(drv, m, progs, self.shots, timeout, signs, heralding(cfg))[2]
            for q in self.qubits:
                fids[q].append(float(diag[q]))

        data, fit, proposal, oks = {}, {}, {}, {}
        for q in self.qubits:
            best = int(np.argmax(fids[q]))
            data[q] = {"x": np.array(wins, float), "y": np.array(fids[q])}
            proposal[f"readout/{q}/demod/dur"] = seconds(wins[best], m)
            fit[q] = None
            oks[q] = bool(fids[q][best] > 0.75)
        self.data, self.fit = data, fit
        return Result(all(oks.values()), data, fit, proposal, cfg, f"Window {self.qubits}")
