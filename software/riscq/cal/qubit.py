"""Single-qubit calibrations (spec 06 §2): Amplitude (Rabi), Frequency (Ramsey vs detuning), Phase,
T1, T2. Each runs its whole sweep as ONE batched counts-mode run on the core (riscq.cal.kernels —
k_rabi / k_ramsey / k_t1 / k_phase, spec 08 §2.1): the kernel COMPUTES the swept knob on-core from a
scalar sweep descriptor (a Q16 or int pair — spec 09; no input Arrays), accumulates the
hardware-classified shot per point on a fixed grid, and the host fits the self-normalised population
P = counts/shots in [0, 1] (no |0> reference / projection — counts are self-normalised, spec 08 §6).
The host mirrors the kernel's integer arithmetic to reconstruct the exact fit x-axis (sweep_q16 /
_phase_sweep / the int pairs). Frequency compiles ONCE and reruns one fringe per detuning (only the
virtual-Z pair changes); Phase is the one cal that needs TWO runs — qcal's two sequences over the same
sweep, whose populations cross at the calibrated phase (spec 13 §6). The demod carrier's
discrimination phase (measured by ReadoutCalibration) is baked into the readout tables. The recovered
ground truth (rabi rate, detuning, X90 virtual-Z, T1/T2) drives the Config update.

Every cal takes `qubits: list` (a bare int is one qubit) and runs them SIMULTANEOUSLY (spec 13 §8):
one program per core, ONE run/setup+rerun over all of them, then per-qubit fits on the host. The
`Result.data`/`fit` are per-qubit dicts; `proposal` merges (the paths carry `q`). A single-qubit call
is just the one-key case.

Every knob is PHYSICAL (spec 13 §2): amplitudes are normalized [0, 1], detunings Hz, delays/waits
seconds; the amp/freq/phase codes and the batch grid are derived inside `run()` (riscq.cal.base
batches/seconds, riscq.pulses.units). The per-point relax head comes from the Config (`reset/relax`),
and the gate pulses are the Config's OWN (envelope + dur + amp + phase — base.gate_pulse); T1's |1>
prep is qcal's `gate=` choice (X90·X90 or the config's X — base.prep, spec 13 §4)."""

from __future__ import annotations

import math

import numpy as np

from riscq import run as rq
from riscq.cal import fits, kernels
from riscq.cal.base import (SEP, Result, batch_timeout, batches, gate_pulse, gate_sigma, grid_period,
                            herald_offset, heralding, population, population_heralded, prep,
                            qubit_freq, qubits_list, readout_tables, relax_batches, res_sign, seconds,
                            socmap, sweep_counts, sweep_q16, x90_vz)
from riscq.lang import Array, ParamTable, compile_kernel
from riscq.map import pack16
from riscq.pulses import units

TWO_PI = 2 * math.pi
PERIOD_FRAC = {"X90": 0.25, "X": 0.5}    # qcal: the fraction of a Rabi period the gate rotates
CODE_PER_CYCLE_PER_BATCH = 1 << 12       # a fringe in cycles/batch → a DAC-rate freq code (2^16 / 16)


class Amplitude:
    """Rabi amplitude calibration, on qcal's knobs (spec 13 §7). n_gates=1: cosine fit of the |1>
    population P vs the swept gate amplitude → the amplitude that rotates by the gate's own angle
    (2π·period_frac: π/2 for an X90, π for an X), and the recovered Rabi RATE. n_gates>1: parabola
    vertex of the repeated-pulse population (amplifies the amplitude error) — in counts P the tuned amp
    is the MINIMUM |1> population (n·π/2 back to |0>), an UPWARD parabola (a > 0) where qcal, fitting
    P(|0>), demands a < 0.

    `gate` ('X90' | 'X') picks the Config's OWN pulse (`qubit/{q}/{x90,x}`, base.prep) and the path the
    proposal writes. `amp_span` is a (lo, hi) pair of NORMALIZED amplitudes — or, with
    `relative_amp=True` (the notebook's fine pass), a pair of MULTIPLES of the current amplitude, which
    is qcal's `amplitudes = config[param] * amplitudes` (0.7–1.3× there). qcal's repetition guard is
    enforced: an amplified sweep has to land back on |0>, so n_gates is a multiple of 4 for X90 and of
    2 for X.

    The amplitude is recovered through the RABI RATE, not through qcal's period arithmetic: the fit
    x-axis is the pulse's drive integral σ (base.gate_sigma — the same Σ the model integrates), so
    `qubit/{q}/rabi` is a physical rad-per-drive rate and the proposed code is the one that makes
    rabi·σ the gate's angle. On the same sweep the two routes are the SAME number — σ is linear in the
    amp code, so target/(2π·f_σ·g) = period_frac/f_amp exactly (qcal's `amp = period_frac / f_fit`) —
    cross-checked host-side (test_cal_fits) and on the real sweep (test_cal)."""

    def __init__(self, cfg, qubits, gate="X90", n_gates=1, amp_span=(0.03, 0.97), points=21,
                 relative_amp=False, shots=160):
        assert gate in PERIOD_FRAC, f"gate must be 'X90' or 'X', got {gate!r}"
        if int(n_gates) > 1:                             # qcal single_qubit.py:154-158
            step = 4 if gate == "X90" else 2             # n·X90 back to |0> needs n % 4; n·X needs n % 2
            assert int(n_gates) % step == 0, \
                f"n_gates must be a multiple of {step} for gate {gate!r}, got {n_gates}"
        self.cfg, self.qubits = cfg, qubits_list(qubits)
        self.gate, self.n_gates, self.points = gate, int(n_gates), int(points)
        self.amp_span, self.relative_amp = amp_span, bool(relative_amp)
        self.target_angle = TWO_PI * PERIOD_FRAC[gate]
        self.shots = int(shots)
        self.data, self.fit, self.recovered_rabi = {}, {}, {}

    def _fit_single_gate(self, xs, sig, Pq):
        """The n_gates=1 cosine analysis → (fit, rabi rad/σ·drive, a_star amp code, ok). `ok` is
        qcal's in_range guard (single_qubit.py:273-279), the same rule the n_gates>1 vertex already
        obeys: the fitted gate amplitude must lie INSIDE the swept codes — a narrow sweep whose
        implied π/2 (or π) amp sits outside it is an extrapolation, not a measurement (the old
        `0 < a_star < AMP_SCALE` accepted it). Host-pure: unit-tested without a driver."""
        fq = fits.fit_cosine(sig, Pq)                        # P = (1 − cos(rabi·sig))/2
        if not fq.ok:
            return fq, 0.0, 0.0, False
        rabi = float(TWO_PI * fq.value)
        g = sig[-1] / (int(xs[-1]) * self.n_gates)           # sig per amp-code (linear)
        a_star = (self.target_angle / rabi) / g
        return fq, rabi, a_star, bool(int(xs[0]) <= a_star <= int(xs[-1]))

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg = self.cfg
        name = "x90" if self.gate == "X90" else "x"
        herald = heralding(cfg)
        progs, params, signs, meta = {}, {}, {}, {}
        timeout = 0
        for q in self.qubits:
            path = f"qubit/{q}/{name}/amp"
            carrier = qubit_freq(cfg, q)
            table, pg, _ = prep(cfg, q, m, self.gate)     # the Config's own gate pulses + the kernel fold
            pulse = table.pulses[name]
            ro, demod, code, dur, ddly = readout_tables(cfg, q, m)
            span = self.amp_span
            if self.relative_amp:                         # qcal: the sweep is a MULTIPLE of the current amp
                span = (span[0] * float(cfg[path]), span[1] * float(cfg[path]))
            lo, hi = units._amp_code(span[0]), units._amp_code(span[1])
            a0q, daq, xs = sweep_q16(lo, hi, self.points)             # on-core Q16 sweep + exact x-axis
            d = pulse.dur_batches(m, table.channel)
            seq = self.n_gates * d
            period = grid_period(relax_batches(cfg, m), seq, dur, ddly, herald=herald)
            hoff = herald_offset(seq, ddly) if herald else 0
            progs[q] = compile_kernel(kernels.k_rabi, m, tables=dict(gate=table, ro=ro, demod=demod),
                                      out=Array(2 * self.points if herald else self.points),
                                      npts=self.points, shots=self.shots,
                                      period=period, ngates=self.n_gates, code=code,
                                      mode=kernels.COUNTS, ddly=ddly, prep_gate=pg,
                                      herald=int(herald), hoff=hoff, **x90_vz(cfg, q))
            params[q] = {"a0q": int(a0q), "daq": int(daq), "prep": 1}
            signs[q] = res_sign(cfg, q)
            sig = np.array([gate_sigma(m, pulse, carrier, int(a)) for a in xs]) * self.n_gates
            meta[q] = (np.array(xs, float), sig, path)
            timeout = max(timeout, batch_timeout(self.points * self.shots * period))
        P = sweep_counts(drv, m, progs, params, self.shots, timeout, signs, herald=herald)

        data, fit, proposal, oks = {}, {}, {}, {}
        for q in self.qubits:
            xs, sig, path = meta[q]
            Pq = P[q]
            data[q] = {"x": xs, "y": Pq, "sig": sig}
            ok, prop = False, {}
            if self.n_gates == 1:
                fq, rabi, a_star, ok = self._fit_single_gate(xs, sig, Pq)
                if fq.ok:
                    self.recovered_rabi[q] = rabi
                    prop = {path: float(np.clip(a_star / units.AMP_SCALE, 0.0, 1.0)),
                            f"qubit/{q}/rabi": rabi}
            else:
                fq = fits.fit_parabola(xs, Pq)                       # |1> population MINIMISES at the tuned amp
                in_range = int(xs[0]) <= fq.value <= int(xs[-1])     # reject a wild extrapolated vertex
                if fq.ok and fq.params["a"] > 0 and in_range:        # upward vertex (min P), within the sweep
                    prop = {path: float(np.clip(fq.value / units.AMP_SCALE, 0.0, 1.0))}
                    ok = True
            fit[q], oks[q] = fq, ok
            proposal.update(prop)
        self.data, self.fit = data, fit
        return Result(all(oks.values()), data, fit, proposal, cfg,
                      f"Amplitude {self.qubits} {self.gate} n_gates={self.n_gates}")


class Frequency:
    """Ramsey vs artificial detuning — qcal's V-fit (spec 13 §7, single_qubit.py:360-736). The drive
    carrier is the config's qubit frequency, off the true f_ge by δ, so the fringe at an applied
    detuning runs at |δ + applied|. Fit each fringe (damped cosine), then fit the UNSIGNED fringe
    frequencies against the applied detuning to qcal's `a·|x − b| + c`: the V bottoms out at b = −δ —
    the applied detuning that CANCELS the config's error — so the corrected carrier is `carrier + b`,
    which is qcal's `new = old + b` verbatim (and identical to our `carrier − δ`, δ ≡ −b; the sign is
    pinned in co-sim against a planted detuning of EACH sign).

    The signed-line fit this replaces took the intercept of sign(applied)·|fringe| — only valid while
    |applied| > |δ| at EVERY detuning, since below that the fringe frequency folds. The V-fit is
    qcal's and needs no such assumption. Guards are qcal's, without its fallback (README principle 6):
    negative curvature (a < 0) fails, and a vertex outside the swept detunings fails.

    `detune` is the applied detuning STEP in Hz (the sweep is ±k·detune, k = 1..n_detune/2), `t0`/`dt`
    the Ramsey wait sweep in seconds. qcal's own signature (spec 13 §3) is the OPTIONAL pair:
    `detunings` — an explicit iterable of signed Hz (each is a host rerun, so any set is legal) that
    overrides the detune/n_detune ladder — and `t_max` (seconds), which makes the wait grid `points`
    steps from 0 to t_max, overriding t0/dt."""

    def __init__(self, cfg, qubits, detune=5e6, n_detune=4, points=14, t0=80e-9, dt=40e-9,
                 shots=96, detunings=None, t_max=None):
        self.cfg, self.qubits = cfg, qubits_list(qubits)
        self.detune, self.n_detune = float(detune), int(n_detune)
        self.points, self.t0, self.dt = int(points), float(t0), float(dt)
        self.shots = int(shots)
        self.detunings = None if detunings is None else [float(d) for d in detunings]
        self.t_max = None if t_max is None else float(t_max)
        self.data, self.fit, self.recovered_detuning_code = {}, {}, {}

    def _d_codes(self, params) -> list:
        """The applied detunings as DAC-rate codes: `detunings` (signed Hz, qcal's signature)
        verbatim when given, else the ±k·detune ladder shorthand."""
        if self.detunings is not None:
            return [units._freq_code(d, params) for d in self.detunings]
        D = units._freq_code(self.detune, params)
        return [dc for k in range(1, self.n_detune // 2 + 1) for dc in (-k * D, k * D)]

    def _wait_grid(self, m) -> tuple:
        """The Ramsey wait grid (t0, dt) in batches: `t_max` (seconds, qcal's signature) spreads
        `points` waits over [0, t_max]; else the explicit t0/dt shorthand."""
        if self.t_max is not None:
            return 0, batches(self.t_max / (self.points - 1), m)
        return batches(self.t0, m), batches(self.dt, m)

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg = self.cfg
        t0, dt = self._wait_grid(m)
        waits = [t0 + i * dt for i in range(self.points)]             # exact x-axis (host mirror)
        wf = np.array(waits, float)
        d_codes = self._d_codes(m.params)
        herald = heralding(cfg)
        progs, signs, carriers = {}, {}, {}
        timeout = 0
        for q in self.qubits:
            carrier = qubit_freq(cfg, q)
            gate = ParamTable(0, carrier, {"x90": gate_pulse(cfg, q, m)})
            ro, demod, code, dur, ddly = readout_tables(cfg, q, m)
            d = gate.pulses["x90"].dur_batches(m, gate.channel)
            seq = waits[-1] + 2 * d
            period = grid_period(relax_batches(cfg, m), seq, dur, ddly, herald=herald)
            hoff = herald_offset(seq, ddly) if herald else 0
            # compile ONCE: the waits (w0/dw) are constant across fringes, so bake them; only the
            # virtual-Z detuning pair (p0/dp) changes per fringe — leave it unbound, rewrite per rerun.
            progs[q] = compile_kernel(kernels.k_ramsey, m, tables=dict(gate=gate, ro=ro, demod=demod),
                                      out=Array(2 * self.points if herald else self.points),
                                      npts=self.points, shots=self.shots,
                                      period=period, code=code, mode=kernels.COUNTS, ddly=ddly,
                                      w0=t0, dw=dt, herald=int(herald), hoff=hoff, **x90_vz(cfg, q))
            signs[q], carriers[q] = res_sign(cfg, q), carrier
            timeout = max(timeout, batch_timeout(self.points * self.shots * period))
        rq.setup(drv, m, progs)

        acc = {q: {"applied": [], "obs": [], "fringes": {}} for q in self.qubits}
        for dc in d_codes:                                  # one rerun per detuning — no reload
            par = {q: {"p0": pack16(16 * dc * t0),          # phase pair, seated (spec 12); same all cores
                       "dp": pack16(16 * dc * dt)} for q in self.qubits}
            out = rq.rerun(drv, m, progs, params=par, results=["out"], timeout=timeout)
            for q in self.qubits:
                P = (population_heralded(out[q]["out"], signs[q]) if herald
                     else population(out[q]["out"], self.shots, signs[q]))
                fit = fits.fit_damped_cosine(wf, P)
                acc[q]["fringes"][dc] = (wf, P, fit)
                if fit.ok:                                  # cycles/batch → an UNSIGNED detuning code:
                    acc[q]["applied"].append(dc)            # the |δ + applied| qcal's V-fit takes
                    acc[q]["obs"].append(CODE_PER_CYCLE_PER_BATCH * fit.value)

        data, fit_out, proposal, oks = {}, {}, {}, {}
        for q in self.qubits:
            x, y = np.array(acc[q]["applied"], float), np.array(acc[q]["obs"], float)
            data[q] = {"applied": x, "obs": y, "fringes": acc[q]["fringes"]}
            v = fits.fit_absolute_value(x, y)               # qcal's a·|x − b| + c
            fit_out[q] = v
            ok, prop = False, {}
            if v.ok and len(x) >= 3 and v.params["a"] > 0 and x.min() <= v.value <= x.max():
                dcode = -float(v.value)                     # δ = −b: the carrier's error, in codes
                self.recovered_detuning_code[q] = dcode
                delta_hz = units.code_to_freq(dcode, m.params)
                prop = {f"qubit/{q}/freq": carriers[q] - delta_hz}     # == qcal's `old + b`
                ok = True
            oks[q] = ok
            proposal.update(prop)
        self.data, self.fit = data, fit_out
        return Result(all(oks.values()), data, fit_out, proposal, cfg, f"Frequency {self.qubits}")


def _phase_sweep(lo: float, hi: float, n: int) -> tuple[int, int, np.ndarray]:
    """(p0, dp) plain phase codes + the EXACT phase axis (rad) the kernel realizes — the host mirror
    of the kernel's integer accumulator, like sweep_q16 for the Q16 knobs. The codes are NOT wrapped
    into [-pi, pi): the fit x-axis must stay monotone (the hardware wraps them mod 2^16 by itself,
    which is the same angle)."""
    c0 = round(lo / math.pi * (1 << 15))
    c1 = round(hi / math.pi * (1 << 15))
    dc = 0 if n <= 1 else round((c1 - c0) / (n - 1))
    codes = c0 + dc * np.arange(n, dtype=np.int64)
    return c0, dc, codes * math.pi / (1 << 15)


class Phase:
    """X90 virtual-Z (Stark) phase calibration — qcal's line crossing (spec 13 §6,
    single_qubit.py:862-1088). An X90 that drives the qubit also ac-Stark-shifts it, so the pulse
    leaves a Z rotation behind; the config corrects it with the virtual-Z PAIR that brackets the pulse
    (`qubit/{q}/x90/vz` = [before, after]), and this calibration measures that pair.

    Two sequences (k_phase's compile-time `seq` fold) are run over the SAME phase sweep, three X90s
    each: `Y180_X90` = Rz(+pi/2) X90 X90 Rz(-pi/2) X90, and `X180_Y90` = X90 X90 Rz(+pi/2) X90. Their
    |1> populations are linear in phi with OPPOSITE slopes around the correct value, so the calibrated
    phase is where the two lines CROSS: phi = (b1 - b0)/(m0 - m1). qcal sweeps ONE phi and writes it to
    BOTH virtual-Z slots (single_qubit.py:1081) — which is exactly why the calibration collapses an
    asymmetric stored pair (X6Y3's q6) into a symmetric one — and so do we. The FAST_DRAG's own axis
    phase (`qubit/{q}/x90/phase`) is a different knob and is NOT touched.

    `span` (rad) is the sweep half-width around 0, or around the CURRENT vz[0] when
    `relative_phase=True` (the notebook's per-qubit pass: `phases + config[param]`). Guards are qcal's:
    a crossing outside the swept range fails (ok=False, no proposal — we do not update on a failed
    fit, README principle 6), and an underfit (reduced chi2 > 10, qcal's `FitLinear.error > 10`) falls
    back to the argmin of (P0 - P1)^2 over the grid. NOTE: with unweighted populations in [0, 1] the
    reduced chi2 is SSR/(N-2) << 1, so that threshold is as good as unreachable — in qcal too."""

    CHI2_MAX = 10.0             # qcal's underfitting guard (single_qubit.py:1067)

    def __init__(self, cfg, qubits, points=21, span=0.25, shots=120, relative_phase=False):
        self.cfg, self.qubits = cfg, qubits_list(qubits)
        self.points, self.span = int(points), float(span)
        self.shots, self.relative_phase = int(shots), bool(relative_phase)
        self.data, self.fit, self.recovered_vz, self.fallback = {}, {}, {}, {}

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg = self.cfg
        herald = heralding(cfg)
        axis, signs = {}, {}
        for q in self.qubits:
            centre = float(cfg.get(f"qubit/{q}/x90/vz", [0.0, 0.0])[0]) if self.relative_phase else 0.0
            axis[q] = _phase_sweep(centre - self.span, centre + self.span, self.points)
            signs[q] = res_sign(cfg, q)
        pops = {}                                              # seq -> {q: P}
        for seq in (kernels.Y180_X90, kernels.X180_Y90):       # one compile + run per qcal sequence
            progs, par = {}, {}
            timeout = 0
            for q in self.qubits:
                gate = ParamTable(0, qubit_freq(cfg, q), {"x90": gate_pulse(cfg, q, m)})
                ro, demod, code, dur, ddly = readout_tables(cfg, q, m)
                d = gate.pulses["x90"].dur_batches(m, gate.channel)
                period = grid_period(relax_batches(cfg, m), 3 * d, dur, ddly,   # three back-to-back X90s
                                     herald=herald)
                hoff = herald_offset(3 * d, ddly) if herald else 0
                progs[q] = compile_kernel(kernels.k_phase, m, tables=dict(gate=gate, ro=ro, demod=demod),
                                          out=Array(2 * self.points if herald else self.points),
                                          npts=self.points, shots=self.shots,
                                          period=period, code=code, ddly=ddly, seq=seq,
                                          hpi=pack16(units._phase_code(math.pi / 2)),
                                          herald=int(herald), hoff=hoff)
                p0, dp, _ = axis[q]
                par[q] = {"p0": pack16(p0), "dp": pack16(dp)}   # the swept virtual-Z, host-seated (spec 12)
                timeout = max(timeout, batch_timeout(self.points * self.shots * period))
            pops[seq] = sweep_counts(drv, m, progs, par, self.shots, timeout, signs, herald=herald)

        data, fit_out, proposal, oks = {}, {}, {}, {}
        for q in self.qubits:
            _, _, x = axis[q]
            p_y, p_x = pops[kernels.Y180_X90][q], pops[kernels.X180_Y90][q]   # Y180_X90, X180_Y90
            diff2 = (p_y - p_x) ** 2                            # qcal's sweep_results (the argmin metric)
            f_y, f_x = fits.fit_linear(x, p_y), fits.fit_linear(x, p_x)
            data[q] = {"x": x, "y": diff2, "p0": p_y, "p1": p_x}
            fit_out[q] = (f_y, f_x)
            self.fallback[q], self.recovered_vz[q] = False, math.nan
            ok, prop = False, {}
            if f_y.ok and f_x.ok:
                m0, b0 = f_y.params["slope"], f_y.params["intercept"]
                m1, b1 = f_x.params["slope"], f_x.params["intercept"]
                phi = (b1 - b0) / (m0 - m1) if m0 != m1 else math.nan
                if x[0] <= phi <= x[-1]:                       # in range (qcal's `in_range`)
                    if max(f_y.params["redchi"], f_x.params["redchi"]) > self.CHI2_MAX:
                        phi = float(x[int(np.argmin(diff2))])  # underfit → qcal's argmin fallback
                        self.fallback[q] = True
                    self.recovered_vz[q] = float(phi)
                    v = self.recovered_vz[q]
                    prop = {f"qubit/{q}/x90/vz": [v, v]}       # qcal: ONE crossing, BOTH slots
                    ok = True
            oks[q] = ok
            proposal.update(prop)
        self.data, self.fit = data, fit_out
        return Result(all(oks.values()), data, fit_out, proposal, cfg, f"Phase {self.qubits}")


class T1:
    """T1: prepare |1>, sweep the idle Δt before readout, exp-fit the |1> population
    P = A·exp(−Δt/T1) + C (A > 0: P decays 1 → 0 as the qubit relaxes to |0>). `t0`/`dt` are the
    delay sweep in seconds (default: start at the pulse→readout separation, step ~3·T1/points).
    `gate` ('X90' | 'X') is qcal's |1>-prep choice (spec 13 §4), folded into the kernel at compile
    time: two X90 plays, or one play of the config's own X pulse."""

    def __init__(self, cfg, qubits, points=9, t0=None, dt=None, shots=120, gate="X90"):
        self.cfg, self.qubits = cfg, qubits_list(qubits)
        self.points, self.t0, self.dt = int(points), t0, dt
        self.shots, self.gate = int(shots), gate
        self.data, self.fit, self.recovered_t1 = {}, {}, {}

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg = self.cfg
        herald = heralding(cfg)
        progs, params, signs, meta = {}, {}, {}, {}
        timeout = 0
        for q in self.qubits:
            gate, pg, plen = prep(cfg, q, m, self.gate)            # the |1> prep (table, fold, length)
            ro, demod, code, dur, ddly = readout_tables(cfg, q, m)
            t1_guess = batches(cfg[f"qubit/{q}/T1"], m)
            t0 = SEP if self.t0 is None else batches(self.t0, m)
            dt = max(8, int(3 * t1_guess / self.points)) if self.dt is None else batches(self.dt, m)
            delays = [t0 + i * dt for i in range(self.points)]     # exact x-axis (host mirror)
            seq = delays[-1] + plen
            period = grid_period(relax_batches(cfg, m), seq, dur, ddly, herald=herald)
            hoff = herald_offset(seq, ddly) if herald else 0
            progs[q] = compile_kernel(kernels.k_t1, m, tables=dict(gate=gate, ro=ro, demod=demod),
                                      out=Array(2 * self.points if herald else self.points),
                                      npts=self.points, shots=self.shots,
                                      period=period, code=code, mode=kernels.COUNTS, ddly=ddly,
                                      prep_gate=pg, herald=int(herald), hoff=hoff, **x90_vz(cfg, q))
            params[q] = {"d0": t0, "dd": dt, "prep": 1}            # prep |1>, sweep the idle delay (d0/dd)
            signs[q] = res_sign(cfg, q)
            meta[q] = np.array(delays, float)
            timeout = max(timeout, batch_timeout(self.points * self.shots * period))
        P = sweep_counts(drv, m, progs, params, self.shots, timeout, signs, herald=herald)

        data, fit_out, proposal, oks = {}, {}, {}, {}
        for q in self.qubits:
            delays = meta[q]
            fit = fits.fit_exp_decay(delays, P[q])                 # A·exp(−Δt/τ)+C, A>0 (P: 1→0)
            data[q] = {"x": delays, "y": P[q]}
            fit_out[q] = fit
            ok, prop = False, {}
            if fit.ok and fit.params["amp"] > 0:
                t1_s = seconds(fit.value, m)                       # batches → seconds
                self.recovered_t1[q] = t1_s
                prop = {f"qubit/{q}/T1": t1_s}
                ok = True
            oks[q] = ok
            proposal.update(prop)
        self.data, self.fit = data, fit_out
        return Result(all(oks.values()), data, fit_out, proposal, cfg, f"T1 {self.qubits}")


class T2:
    """T2* (Ramsey): a single Ramsey delay-sweep at a small artificial detuning (`detune`, Hz); the
    damped-cosine decay τ is T2*. `t0`/`dt` are the wait sweep in seconds."""

    def __init__(self, cfg, qubits, detune=1.5e6, points=15, t0=80e-9, dt=80e-9, shots=120):
        self.cfg, self.qubits = cfg, qubits_list(qubits)
        self.detune, self.points = float(detune), int(points)
        self.t0, self.dt = float(t0), float(dt)
        self.shots = int(shots)
        self.data, self.fit, self.recovered_t2 = {}, {}, {}

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg = self.cfg
        t0, dt = batches(self.t0, m), batches(self.dt, m)
        waits = [t0 + i * dt for i in range(self.points)]         # exact x-axis (host mirror)
        wf = np.array(waits, float)
        dc = units._freq_code(self.detune, m.params)
        herald = heralding(cfg)
        progs, params, signs = {}, {}, {}
        timeout = 0
        for q in self.qubits:
            gate = ParamTable(0, qubit_freq(cfg, q), {"x90": gate_pulse(cfg, q, m)})
            ro, demod, code, dur, ddly = readout_tables(cfg, q, m)
            d = gate.pulses["x90"].dur_batches(m, gate.channel)
            seq = waits[-1] + 2 * d
            period = grid_period(relax_batches(cfg, m), seq, dur, ddly, herald=herald)
            hoff = herald_offset(seq, ddly) if herald else 0
            progs[q] = compile_kernel(kernels.k_ramsey, m, tables=dict(gate=gate, ro=ro, demod=demod),
                                      out=Array(2 * self.points if herald else self.points),
                                      npts=self.points, shots=self.shots,
                                      period=period, code=code, mode=kernels.COUNTS, ddly=ddly,
                                      herald=int(herald), hoff=hoff, **x90_vz(cfg, q))
            # wait sweep computed on-core (w0/dw); the small detuning is applied as the virtual-Z pair
            params[q] = {"w0": t0, "dw": dt,                       # waits plain; phase pair seated (spec 12)
                         "p0": pack16(16 * dc * t0), "dp": pack16(16 * dc * dt)}
            signs[q] = res_sign(cfg, q)
            timeout = max(timeout, batch_timeout(self.points * self.shots * period))
        P = sweep_counts(drv, m, progs, params, self.shots, timeout, signs, herald=herald)

        data, fit_out, proposal, oks = {}, {}, {}, {}
        for q in self.qubits:
            fit = fits.fit_damped_cosine(wf, P[q])
            data[q] = {"x": wf, "y": P[q]}
            fit_out[q] = fit
            ok, prop = False, {}
            if fit.ok and fit.params["tau"] > 0:
                t2_s = seconds(fit.params["tau"], m)               # batches → seconds
                self.recovered_t2[q] = t2_s
                prop = {f"qubit/{q}/T2": t2_s}
                ok = True
            oks[q] = ok
            proposal.update(prop)
        self.data, self.fit = data, fit_out
        return Result(all(oks.values()), data, fit_out, proposal, cfg, f"T2 {self.qubits}")
