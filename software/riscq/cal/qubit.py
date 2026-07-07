"""Single-qubit calibrations (spec 06 §2): Amplitude (Rabi), Frequency (Ramsey vs detuning), Phase,
T1, T2. Each runs its whole sweep as ONE batched counts-mode run on the core (riscq.cal.kernels —
k_rabi / k_ramsey / k_t1, spec 08 §2.1): the kernel COMPUTES the swept knob on-core from a scalar
sweep descriptor (a Q16 or int pair — spec 09; no input Arrays), accumulates the hardware-classified
shot per point on a fixed grid, and the host fits the self-normalised population P = counts/shots in
[0, 1] (no |0> reference / projection — counts are self-normalised, spec 08 §6). The host mirrors the
kernel's integer arithmetic to reconstruct the exact fit x-axis (sweep_q16 / the int pairs). Frequency
compiles ONCE and reruns one fringe per detuning (only the virtual-Z pair changes). The demod carrier's
discrimination phase (measured by ReadoutCalibration) is baked into the readout tables. The recovered
ground truth (rabi rate, detuning, T1/T2) drives the Config update."""

from __future__ import annotations

import math

import numpy as np

from riscq import run as rq
from riscq.cal import fits, kernels
from riscq.cal.base import (GATE_ENV, RELAX, SEP, Result, batch_timeout, bind_params, gate_sigma,
                            grid_period, qubit_freq, readout_tables, socmap, sweep_counts, sweep_q16,
                            x90_pulse, x180_pulse)
from riscq.lang import Array, ParamTable, compile_kernel
from riscq.pulses import units

TWO_PI = 2 * math.pi


def _one(qubits):
    return qubits[0] if isinstance(qubits, (list, tuple)) else int(qubits)


class Amplitude:
    """Rabi amplitude calibration. n_gates=1: cosine fit of the |1> population P vs the swept X90
    amplitude → the amplitude giving a π/2 rotation (and the recovered Rabi rate). n_gates>1:
    parabola vertex of the repeated-pulse population (amplifies the amplitude error) — in counts P
    the tuned amp is the MINIMUM |1> population (n·π/2 back to |0>), an UPWARD parabola (a > 0)."""

    def __init__(self, cfg, qubits, n_gates=1, amp_span=None, points=21, env=GATE_ENV,
                 target_angle=math.pi / 2, relax=RELAX, shots=160):
        self.cfg, self.q = cfg, _one(qubits)
        self.n_gates, self.points = int(n_gates), int(points)
        self.amp_span, self.env = amp_span, env
        self.target_angle, self.relax = float(target_angle), int(relax)
        self.shots = int(shots)
        self.data, self.fit, self.recovered_rabi = {}, None, None

    def run(self, drv) -> Result:
        m = socmap(drv); bind_params(m)
        cfg, q = self.cfg, self.q
        carrier = qubit_freq(cfg, q)
        x90 = x90_pulse(cfg, q, self.env)
        gate = ParamTable(0, carrier, {"x90": x90})
        ro, demod, code, dur = readout_tables(cfg, q)
        lo, hi = self.amp_span or (600, units.AMP_SCALE - 600)
        a0q, daq, xs = sweep_q16(int(lo), int(hi), self.points)   # on-core Q16 sweep + exact x-axis
        d = x90.dur_batches(m, gate.channel)
        npts = self.points
        period = grid_period(self.relax, self.n_gates * d, dur)
        prog = compile_kernel(kernels.k_rabi, m, tables=dict(gate=gate, ro=ro, demod=demod),
                              out=Array(npts), npts=npts, shots=self.shots, period=period,
                              ngates=self.n_gates, code=code, mode=kernels.COUNTS)
        P = sweep_counts(drv, m, prog, q, {"a0q": int(a0q), "daq": int(daq), "prep": 1}, self.shots,
                         batch_timeout(npts * self.shots * period))
        sig = np.array([gate_sigma(m, x90, carrier, int(a)) for a in xs]) * self.n_gates
        self.data = {"x": np.array(xs, float), "y": P, "sig": sig}

        ok, proposal = False, {}
        if self.n_gates == 1:
            fit = fits.fit_cosine(sig, P)                        # P = (1 − cos(rabi·sig))/2
            if fit.ok:
                self.recovered_rabi = float(TWO_PI * fit.value)
                g = sig[-1] / (int(xs[-1]) * self.n_gates)       # sig per amp-code (linear)
                a_star = (self.target_angle / self.recovered_rabi) / g
                proposal = {f"qubit/{q}/x90/amp": float(np.clip(a_star / units.AMP_SCALE, 0.0, 1.0)),
                            f"qubit/{q}/rabi": self.recovered_rabi}
                ok = 0 < a_star < units.AMP_SCALE
        else:
            fit = fits.fit_parabola(np.array(xs, float), P)      # |1> population MINIMISES at the tuned amp
            in_range = int(xs[0]) <= fit.value <= int(xs[-1])    # reject a wild extrapolated vertex
            if fit.ok and fit.params["a"] > 0 and in_range:      # upward vertex (min P), within the sweep
                proposal = {f"qubit/{q}/x90/amp": float(np.clip(fit.value / units.AMP_SCALE, 0.0, 1.0))}
                ok = True
        self.fit = fit
        return Result(ok, self.data, fit, proposal, cfg, f"Amplitude q{q} n_gates={self.n_gates}")


class Frequency:
    """Ramsey vs artificial detuning (spec 06). The drive carrier is the config's qubit frequency,
    off the true f_ge by δ, so the Ramsey fringe runs at (δ + applied). Fit each applied detuning's
    fringe, then a line through the sign(applied)-signed fringe frequencies (|applied| > |δ| so
    sign(δ+applied)=sign(applied)); its intercept at applied=0 is δ. Update: qubit/q/freq −= δ."""

    def __init__(self, cfg, qubits, detune_code=200, n_detune=4, points=14, t0=8, dt=4,
                 env=GATE_ENV, relax=RELAX, shots=96):
        self.cfg, self.q = cfg, _one(qubits)
        self.detune_code, self.n_detune = int(detune_code), int(n_detune)
        self.points, self.t0, self.dt = int(points), int(t0), int(dt)
        self.env, self.relax = env, int(relax)
        self.shots = int(shots)
        self.data, self.fit, self.recovered_detuning_code = {}, None, None

    def run(self, drv) -> Result:
        m = socmap(drv); bind_params(m)
        cfg, q = self.cfg, self.q
        carrier = qubit_freq(cfg, q)
        gate = ParamTable(0, carrier, {"x90": x90_pulse(cfg, q, self.env)})
        ro, demod, code, dur = readout_tables(cfg, q)
        d = gate.pulses["x90"].dur_batches(m, gate.channel)
        waits = [self.t0 + i * self.dt for i in range(self.points)]   # exact x-axis (host mirror)
        npts = self.points
        period = grid_period(self.relax, waits[-1] + 2 * d, dur)
        # compile ONCE: the waits (w0/dw) are constant across fringes, so bake them; only the virtual-Z
        # detuning pair (p0/dp) changes per fringe — leave them unbound and rewrite via the rerun params.
        prog = compile_kernel(kernels.k_ramsey, m, tables=dict(gate=gate, ro=ro, demod=demod),
                              out=Array(npts), npts=npts, shots=self.shots, period=period,
                              code=code, mode=kernels.COUNTS, w0=self.t0, dw=self.dt)
        rq.setup(drv, m, {q: prog})
        timeout = batch_timeout(npts * self.shots * period)
        D = self.detune_code
        d_codes = [dc for k in range(1, self.n_detune // 2 + 1) for dc in (-k * D, k * D)]

        applied, signed_obs, fringes = [], [], {}
        wf = np.array(waits, float)
        for dc in d_codes:                                  # one rerun per detuning — no reload
            out = rq.rerun(drv, m, {q: prog},
                           params={q: {"p0": 16 * dc * self.t0, "dp": 16 * dc * self.dt}},
                           results=["out"], timeout=timeout)[q]["out"]
            P = out.astype(float) / self.shots
            fit = fits.fit_damped_cosine(wf, P)
            fringes[dc] = (wf, P, fit)
            if fit.ok:
                applied.append(dc)
                signed_obs.append(math.copysign(4096 * fit.value, dc))   # cycles/batch → detuning code
        self.data = {"applied": np.array(applied, float), "obs": np.array(signed_obs), "fringes": fringes}

        line = fits.fit_linear(np.array(applied, float), np.array(signed_obs))
        self.fit = line
        ok, proposal = False, {}
        if line.ok and len(applied) >= 2:
            self.recovered_detuning_code = float(line.params["intercept"])
            delta_hz = units.code_to_freq(int(round(self.recovered_detuning_code)), m.params)
            proposal = {f"qubit/{q}/freq": carrier - delta_hz}
            ok = True
        return Result(ok, self.data, line, proposal, cfg, f"Frequency q{q}")


class Phase:
    """X90 axis calibration (spec 06). Two X90s separated by a swept virtual phase φ (wait 0) give
    the |1> population P = (1 + cos(φ − φ_err))/2 in counts mode (the ⟨σz⟩ = −cos(φ − φ_err) flips
    sign under P = (1 − ⟨σz⟩)/2, so fit_cosine's phase = −φ_err and the fold stays `old − fit.phase`).
    The fitted φ_err is the residual axis error, folded into qubit/q/x90/phase (≈0 with no planted
    error). `dphi_code` must span a full fringe (default 8192 × 13 pts ≈ 1.5 cycles) so the cosine
    fit locks — the pre-batch default (512) covered < 0.1 cycle and could not fit. The projective
    co-sim model plants no axis error, so this recovers ≈0; the absolute sign/π convention is pinned
    by the X6Y3 fine step (pass 3)."""

    def __init__(self, cfg, qubits, points=13, dphi_code=8192, env=GATE_ENV, relax=RELAX, shots=120):
        self.cfg, self.q = cfg, _one(qubits)
        self.points, self.dphi_code = int(points), int(dphi_code)
        self.env, self.relax = env, int(relax)
        self.shots = int(shots)
        self.data, self.fit = {}, None

    def run(self, drv) -> Result:
        m = socmap(drv); bind_params(m)
        cfg, q = self.cfg, self.q
        gate = ParamTable(0, qubit_freq(cfg, q), {"x90": x90_pulse(cfg, q, self.env)})
        ro, demod, code, dur = readout_tables(cfg, q)
        d = gate.pulses["x90"].dur_batches(m, gate.channel)
        period = grid_period(self.relax, 2 * d, dur)             # wait fixed at 0
        npts = self.points
        prog = compile_kernel(kernels.k_ramsey, m, tables=dict(gate=gate, ro=ro, demod=demod),
                              out=Array(npts), npts=npts, shots=self.shots, period=period,
                              code=code, mode=kernels.COUNTS)
        # wait 0 (w0=dw=0), virtual-Z phase computed on-core: φ_i = i·dphi (p0=0, dp=dphi_code)
        P = sweep_counts(drv, m, prog, q, {"w0": 0, "dw": 0, "p0": 0, "dp": self.dphi_code},
                         self.shots, batch_timeout(npts * self.shots * period))
        phis = [i * self.dphi_code for i in range(self.points)]          # realized φ codes (host mirror)
        x = np.array([p * math.pi / (1 << 15) for p in phis], float)     # φ in radians
        self.data = {"x": x, "y": P}
        fit = fits.fit_cosine(x, P)                                      # P = (1 + cos(φ + φ_err))/2
        self.fit = fit
        ok, proposal = False, {}
        if fit.ok:
            new_phase = float(cfg.get(f"qubit/{q}/x90/phase", 0.0)) - float(fit.params["phase"])
            proposal = {f"qubit/{q}/x90/phase": ((new_phase + math.pi) % TWO_PI) - math.pi}
            ok = True
        return Result(ok, self.data, fit, proposal, cfg, f"Phase q{q}")


class T1:
    """T1: prepare |1>, sweep the idle Δt before readout, exp-fit the |1> population
    P = A·exp(−Δt/T1) + C (A > 0: P decays 1 → 0 as the qubit relaxes to |0>)."""

    def __init__(self, cfg, qubits, points=9, t0=None, dt=None, env=GATE_ENV, relax=RELAX, shots=120):
        self.cfg, self.q = cfg, _one(qubits)
        self.points, self.t0, self.dt = int(points), t0, dt
        self.env, self.relax = env, int(relax)
        self.shots = int(shots)
        self.data, self.fit, self.recovered_t1 = {}, None, None

    def run(self, drv) -> Result:
        m = socmap(drv); bind_params(m)
        cfg, q = self.cfg, self.q
        x180 = x180_pulse(cfg, q, self.env)
        gate = ParamTable(0, qubit_freq(cfg, q), {"x180": x180})
        ro, demod, code, dur = readout_tables(cfg, q)
        t1_guess = float(cfg.get(f"qubit/{q}/T1", 120))
        t0 = self.t0 if self.t0 is not None else SEP
        dt = self.dt if self.dt is not None else max(8, int(3 * t1_guess / self.points))
        delays = [int(t0 + i * dt) for i in range(self.points)]   # exact x-axis (host mirror)
        npts = self.points
        period = grid_period(self.relax, delays[-1] + x180.dur_batches(m, gate.channel), dur)
        prog = compile_kernel(kernels.k_t1, m, tables=dict(gate=gate, ro=ro, demod=demod),
                              out=Array(npts), npts=npts, shots=self.shots, period=period,
                              code=code, mode=kernels.COUNTS)
        # prep |1>, sweep the idle delay computed on-core (d0/dd)
        P = sweep_counts(drv, m, prog, q, {"d0": int(t0), "dd": int(dt), "prep": 1}, self.shots,
                         batch_timeout(npts * self.shots * period))
        self.data = {"x": np.array(delays, float), "y": P}
        fit = fits.fit_exp_decay(np.array(delays, float), P)             # A·exp(−Δt/τ)+C, A>0 (P: 1→0)
        self.fit = fit
        ok, proposal = False, {}
        if fit.ok and fit.params["amp"] > 0:
            self.recovered_t1 = float(fit.value)
            proposal = {f"qubit/{q}/T1": self.recovered_t1}
            ok = True
        return Result(ok, self.data, fit, proposal, cfg, f"T1 q{q}")


class T2:
    """T2* (Ramsey): a single Ramsey delay-sweep at a small artificial detuning; the damped-cosine
    decay τ is T2*."""

    def __init__(self, cfg, qubits, detune_code=60, points=15, t0=8, dt=8, env=GATE_ENV, relax=RELAX,
                 shots=120):
        self.cfg, self.q = cfg, _one(qubits)
        self.detune_code, self.points = int(detune_code), int(points)
        self.t0, self.dt = int(t0), int(dt)
        self.env, self.relax = env, int(relax)
        self.shots = int(shots)
        self.data, self.fit, self.recovered_t2 = {}, None, None

    def run(self, drv) -> Result:
        m = socmap(drv); bind_params(m)
        cfg, q = self.cfg, self.q
        gate = ParamTable(0, qubit_freq(cfg, q), {"x90": x90_pulse(cfg, q, self.env)})
        ro, demod, code, dur = readout_tables(cfg, q)
        d = gate.pulses["x90"].dur_batches(m, gate.channel)
        waits = [self.t0 + i * self.dt for i in range(self.points)]   # exact x-axis (host mirror)
        npts = self.points
        period = grid_period(self.relax, waits[-1] + 2 * d, dur)
        prog = compile_kernel(kernels.k_ramsey, m, tables=dict(gate=gate, ro=ro, demod=demod),
                              out=Array(npts), npts=npts, shots=self.shots, period=period,
                              code=code, mode=kernels.COUNTS)
        # wait sweep computed on-core (w0/dw); the small detuning is applied as the virtual-Z pair
        dc = self.detune_code
        P = sweep_counts(drv, m, prog, q,
                         {"w0": self.t0, "dw": self.dt, "p0": 16 * dc * self.t0, "dp": 16 * dc * self.dt},
                         self.shots, batch_timeout(npts * self.shots * period))
        self.data = {"x": np.array(waits, float), "y": P}
        fit = fits.fit_damped_cosine(np.array(waits, float), P)
        self.fit = fit
        ok, proposal = False, {}
        if fit.ok and fit.params["tau"] > 0:
            self.recovered_t2 = float(fit.params["tau"])
            proposal = {f"qubit/{q}/T2": self.recovered_t2}
            ok = True
        return Result(ok, self.data, fit, proposal, cfg, f"T2 q{q}")
