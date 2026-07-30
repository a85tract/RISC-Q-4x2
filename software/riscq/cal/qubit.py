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
from riscq.cal.base import (GATE_CH, SEP, Result, batch_timeout, batches, ef_pulse, ef_table, ef_vz,
                            gate_pulse, gate_sigma, grid_period, herald_offset, heralding, population,
                            population_heralded, prep, qubit_freq, qubits_list, readout_tables,
                            relax_batches, rerun_levels, res_sign, seconds, socmap, sweep_counts,
                            sweep_levels, sweep_q16, train_step, x90_vz)
from riscq.lang import Array, ParamTable, compile_kernel
from riscq.map import LEAD, pack16
from riscq.pulses import Pulse, units

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
            step = train_step(d)                       # the paced train grid (spec 14 F1)
            seq = (self.n_gates - 1) * step + d        # the last gate ends a clean SEP before t_ro
            period = grid_period(relax_batches(cfg, m), seq, dur, ddly, herald=herald)
            hoff = herald_offset(seq, ddly) if herald else 0
            progs[q] = compile_kernel(kernels.k_rabi, m, tables=dict(gate=table, ro=ro, demod=demod),
                                      out=Array(2 * self.points if herald else self.points),
                                      npts=self.points, shots=self.shots,
                                      period=period, ngates=self.n_gates, step=step, code=code,
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


def _line_crossing(x, p_y, p_x, chi2_max):
    """qcal Phase's two-line crossing (spec 13 §6), shared by the GE and EF Phase cals: linear-fit
    the two sequences' populations and cross them at phi = (b1 − b0)/(m0 − m1), with qcal's guards —
    a failed fit or an out-of-range crossing fails (no write, README principle 6); an underfit
    (reduced chi2 > `chi2_max`) falls back to the argmin of (p_y − p_x)² over the grid. Returns
    ((fit_y, fit_x), phi, fallback, ok), phi = nan when not ok."""
    f_y, f_x = fits.fit_linear(x, p_y), fits.fit_linear(x, p_x)
    if f_y.ok and f_x.ok:
        m0, b0 = f_y.params["slope"], f_y.params["intercept"]
        m1, b1 = f_x.params["slope"], f_x.params["intercept"]
        phi = (b1 - b0) / (m0 - m1) if m0 != m1 else math.nan
        if x[0] <= phi <= x[-1]:                       # in range (qcal's `in_range`)
            if max(f_y.params["redchi"], f_x.params["redchi"]) > chi2_max:
                return (f_y, f_x), float(x[int(np.argmin((p_y - p_x) ** 2))]), True, True
            return (f_y, f_x), float(phi), False, True
    return (f_y, f_x), math.nan, False, False


def _cosine_axis(x, y, centre):
    """qcal Phase(gate='X')'s cosine fit (spec 14 §3.3), shared by the GE and EF X-phase cals. The
    `X90 · X · X90` population is A·cos(2π·f·phi + ϑ) + C (fit_cosine canonicalises A ≥ 0), MINIMAL
    at (π − ϑ)/(2π·f) — the aligned axis, since the composite only returns to the prepared level
    there. The fringe runs at TWICE the swept axis (a π rotation's axis enters a Bloch rotation as
    2φ), so its period is π and the two solutions a period apart are the same gate
    (R_{φ+π}(π) = −R_φ(π)): take the one nearest the sweep `centre` so an already-calibrated qubit
    does not jump. qcal's guards follow the line-crossing ones — a failed fit or an out-of-range
    solution fails (no write) and falls back to the grid argmin. Returns (fit, phi, fallback, ok)."""
    f = fits.fit_cosine(x, y)
    phi, ok = 0.0, bool(f.ok and f.params["freq"] > 0)
    if ok:
        per = 1.0 / f.params["freq"]
        base = (math.pi - f.params["phase"]) / (2 * math.pi * f.params["freq"])
        phi = base - round((base - centre) / per) * per
    ok = ok and bool(x[0] <= phi <= x[-1])                  # qcal's in_range guard
    if not ok:                                              # ... and its argmin fallback
        return f, float(x[int(np.argmin(y))]), True, False
    return f, float(phi), False, True


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
    reduced chi2 is SSR/(N-2) << 1, so that threshold is as good as unreachable — in qcal too.

    `gate='X'` calibrates the OTHER pulse and the OTHER knob (qcal's second Phase mode, spec 14 §3.3):
    one circuit, `X90 · X · X90`, over the X's own AXIS phase (`qubit/{q}/x/phase`, the FAST_DRAG's
    `phase` — not a virtual-Z pair; the X6Y3 X carries none). The three pulses make a 2π rotation that
    returns to |0> only when the X sits on the X90s' axis, so P(1) is cosinusoidal in the swept axis
    and the calibrated phase is its MINIMUM (qcal fits the same cosine to P(0) − P(1) and takes the
    maximum). The default sweep is a full turn, as in the reference notebook."""

    CHI2_MAX = 10.0             # qcal's underfitting guard (single_qubit.py:1067)

    def __init__(self, cfg, qubits, gate="X90", points=21, span=None, shots=120,
                 relative_phase=False):
        assert gate in ("X90", "X"), f"Phase calibrates 'X90' or 'X', got {gate!r}"
        self.cfg, self.qubits, self.gate = cfg, qubits_list(qubits), gate
        self.points = int(points)
        self.span = float(math.pi if span is None and gate == "X" else 0.25 if span is None else span)
        self.shots, self.relative_phase = int(shots), bool(relative_phase)
        self.data, self.fit, self.recovered_vz, self.fallback = {}, {}, {}, {}

    def _centre(self, cfg, q) -> float:
        """The sweep's centre: qcal's `relative_phase` re-centres on the knob's CURRENT value."""
        if not self.relative_phase:
            return 0.0
        if self.gate == "X":
            return float(cfg.get(f"qubit/{q}/x/phase", 0.0))
        return float(cfg.get(f"qubit/{q}/x90/vz", [0.0, 0.0])[0])

    def run(self, drv) -> Result:
        if self.gate == "X":
            return self._run_x(drv)
        m = socmap(drv)
        cfg = self.cfg
        herald = heralding(cfg)
        axis, signs = {}, {}
        for q in self.qubits:
            centre = self._centre(cfg, q)
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
                                          hpi=pack16(units._phase_code(math.pi / 2)), vz0=0, vzsum=0,
                                          herald=int(herald), hoff=hoff)
                p0, dp, _ = axis[q]
                par[q] = {"p0": pack16(p0), "dp": pack16(dp)}   # the swept virtual-Z, host-seated (spec 12)
                timeout = max(timeout, batch_timeout(self.points * self.shots * period))
            pops[seq] = sweep_counts(drv, m, progs, par, self.shots, timeout, signs, herald=herald)

        data, fit_out, proposal, oks = {}, {}, {}, {}
        for q in self.qubits:
            _, _, x = axis[q]
            p_y, p_x = pops[kernels.Y180_X90][q], pops[kernels.X180_Y90][q]   # Y180_X90, X180_Y90
            data[q] = {"x": x, "y": (p_y - p_x) ** 2, "p0": p_y, "p1": p_x}   # qcal's argmin metric
            fit_out[q], phi, self.fallback[q], ok = _line_crossing(x, p_y, p_x, self.CHI2_MAX)
            self.recovered_vz[q] = phi
            oks[q] = ok
            if ok:
                proposal[f"qubit/{q}/x90/vz"] = [phi, phi]     # qcal: ONE crossing, BOTH slots
        self.data, self.fit = data, fit_out
        return Result(all(oks.values()), data, fit_out, proposal, cfg, f"Phase {self.qubits}")

    def _run_x(self, drv) -> Result:
        """gate='X': one X90 · X · X90 run, cosine-fitted; the X's axis phase is the fringe MINIMUM."""
        m = socmap(drv)
        cfg = self.cfg
        herald = heralding(cfg)
        progs, par, signs, axis = {}, {}, {}, {}
        timeout = 0
        for q in self.qubits:
            centre = self._centre(cfg, q)
            axis[q] = _phase_sweep(centre - self.span, centre + self.span, self.points)
            signs[q] = res_sign(cfg, q)
            x90 = gate_pulse(cfg, q, m)                 # keeps its own axis phase: the reference
            xp = gate_pulse(cfg, q, m, "x")
            # the swept phi REPLACES the X's stored axis (qcal writes the pulse's own phase kwarg),
            # so the slot is built at 0 and the on-core phase offset carries the whole axis
            table = ParamTable(0, qubit_freq(cfg, q),
                               {"x90": x90, "x": Pulse(xp.env, freq_hz=xp.freq_hz, amp=xp.amp)})
            ro, demod, code, dur, ddly = readout_tables(cfg, q, m)
            d = table.pulses["x90"].dur_batches(m, table.channel)
            xd = table.pulses["x"].dur_batches(m, table.channel)
            period = grid_period(relax_batches(cfg, m), 2 * d + xd, dur, ddly, herald=herald)
            hoff = herald_offset(2 * d + xd, ddly) if herald else 0
            progs[q] = compile_kernel(kernels.k_phase, m, tables=dict(gate=table, ro=ro, demod=demod),
                                      out=Array(2 * self.points if herald else self.points),
                                      npts=self.points, shots=self.shots, period=period, code=code,
                                      ddly=ddly, seq=kernels.X90_X_X90,
                                      hpi=pack16(units._phase_code(math.pi / 2)),
                                      herald=int(herald), hoff=hoff, **x90_vz(cfg, q))
            p0, dp, _ = axis[q]
            par[q] = {"p0": pack16(p0), "dp": pack16(dp)}
            timeout = max(timeout, batch_timeout(self.points * self.shots * period))
        P = sweep_counts(drv, m, progs, par, self.shots, timeout, signs, herald=herald)

        data, fit_out, proposal, oks = {}, {}, {}, {}
        for q in self.qubits:
            _, _, x = axis[q]
            data[q] = {"x": x, "y": P[q]}
            fit_out[q], phi, self.fallback[q], ok = _cosine_axis(x, P[q], self._centre(cfg, q))
            self.recovered_vz[q], oks[q] = phi, ok
            if ok:
                proposal[f"qubit/{q}/x/phase"] = float(phi)
        self.data, self.fit = data, fit_out
        return Result(all(oks.values()), data, fit_out, proposal, cfg, f"Phase {self.qubits} X")


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


# ── EF subspace (spec two-qubit/01 §4.1): the CZ-frequency prerequisite ──

def _classifiers(classifier, qubits) -> dict:
    """Normalize the EF cals' `classifier` argument to a {q: ClassifierN} dict — a bare ClassifierN is
    accepted for the single-qubit case (the pre-trained 3-level readout each qubit needs, spec 01 §5)."""
    if isinstance(classifier, dict):
        return classifier
    if len(qubits) != 1:
        raise ValueError("one classifier needs exactly one qubit — pass a {q: ClassifierN} dict")
    return {qubits[0]: classifier}


class EFAmplitude:
    """EF (|1>->|2>) amplitude calibration — qcal's `Amplitude(subspace='EF')` (spec 01 §4.1), the
    prerequisite that seeds f_02 for the CZ frequency and prepares |2> for 3-level readout. A GE pi
    populates |1> (k_ef_rabi's fixed prep, at the GE carrier), the gate carrier retunes to f_ef, and the
    swept EF drive rotates |1>->|2>; the host reads P(|2>) off the 3-level clusters (`classifier`) and
    fits it EXACTLY as the GE Rabi does — P(|2>) = (1 − cos(rabi_ef·σ))/2. `gate` picks the pulse and
    the target angle, mirroring the GE Amplitude's knob (spec 04 §2 / X4): 'X90' (default) calibrates
    `qubit/{q}/EF/x90/amp` toward π/2; 'X' the EF π `qubit/{q}/EF/x/amp` — the pulse the (5,6)/(6,7)
    sandwich CZ shelves with. n_gates=1 → a cosine whose period gives the EF Rabi RATE and the
    amplitude that rotates by the gate's angle; n_gates>1 → a parabola vertex (4·EF-X90 = 2·EF-X = 2π
    back to |1>, so P(|2>) MINIMISES at the tuned amp, an upward parabola). Also writes the recovered
    `qubit/{q}/EF/rabi`.

    `classifier` is core q's pre-trained ClassifierN (a bare one for a single qubit, else a
    {q: ClassifierN} dict): the EF readout needs |1> vs |2>, which the hardware res bit cannot separate.
    Mirrors Amplitude's knobs — `amp_span` (normalized, or MULTIPLES of the current EF amp when
    relative_amp) and qcal's repetition guard (n_gates % 4 for X90, % 2 for X). Documented deviation:
    the GE prep stays X90·X90 for BOTH gates (qcal preps the X-gate variant with a single GE X — the
    same π, ours keeps k_ef_rabi's one prep)."""

    def __init__(self, cfg, qubits, classifier, gate="X90", n_gates=1, amp_span=(0.03, 0.97),
                 points=21, relative_amp=False, shots=48):
        assert gate in PERIOD_FRAC, f"gate must be 'X90' or 'X', got {gate!r}"
        if int(n_gates) > 1:                          # 4·EF-X90 = 2·EF-X = 2π back to |1> (qcal's guard)
            step = 4 if gate == "X90" else 2
            assert int(n_gates) % step == 0, \
                f"n_gates must be a multiple of {step} for the EF {gate}, got {n_gates}"
        self.cfg, self.qubits = cfg, qubits_list(qubits)
        self.classifiers = _classifiers(classifier, self.qubits)
        self.gate, self.n_gates, self.points = gate, int(n_gates), int(points)
        self.amp_span, self.relative_amp = amp_span, bool(relative_amp)
        self.shots = int(shots)
        self.target_angle = TWO_PI * PERIOD_FRAC[gate]
        self.data, self.fit, self.recovered_rabi = {}, {}, {}

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg = self.cfg
        name = "x90" if self.gate == "X90" else "x"
        progs, params, meta = {}, {}, {}
        timeout = 0
        for q in self.qubits:
            path = f"qubit/{q}/EF/{name}/amp"
            table, ge_freq, ef_freq = ef_table(cfg, q, m, name)
            efp, f_ef = ef_pulse(cfg, q, m, name), float(cfg[f"qubit/{q}/EF/freq"])
            # classified host-side by ClassifierN — capture in the classifier's zero frame (spec 14
            # finding 7): the training clusters are taken at phase=0, so a config-frame capture would
            # arrive rotated by the stored demod phase relative to the classifier's means
            ro, demod, code, dur, ddly = readout_tables(cfg, q, m, phase=0.0)
            span = self.amp_span
            if self.relative_amp:                     # qcal: the sweep is a MULTIPLE of the current amp
                span = (span[0] * float(cfg[path]), span[1] * float(cfg[path]))
            lo, hi = units._amp_code(span[0]), units._amp_code(span[1])
            a0q, daq, xs = sweep_q16(lo, hi, self.points)
            ge = table.pulses["x90"].dur_batches(m, GATE_CH)
            ef = table.pulses["ef"].dur_batches(m, GATE_CH)
            step = train_step(ef)                          # the paced train grid (spec 14 F1)
            seq = SEP + (self.n_gates - 1) * step + ef + LEAD + 2 * ge   # earliest pulse → t_ro
            period = grid_period(relax_batches(cfg, m), seq, dur, ddly)
            progs[q] = compile_kernel(kernels.k_ef_rabi, m, tables=dict(gate=table, ro=ro, demod=demod),
                                      out=Array(2 * self.points * self.shots), npts=self.points,
                                      shots=self.shots, period=period, ngates=self.n_gates,
                                      step=step, code=code,
                                      ddly=ddly, ge_freq=ge_freq, ef_freq=ef_freq,
                                      # the bracket of the gate actually being played: the EF X90's
                                      # calibrated pair, or the EF X's absent (= [0, 0]) one
                                      **x90_vz(cfg, q), **ef_vz(cfg, q, name))
            params[q] = {"a0q": int(a0q), "daq": int(daq)}
            sig = np.array([gate_sigma(m, efp, f_ef, int(a)) for a in xs]) * self.n_gates
            meta[q] = (np.array(xs, float), sig, path)
            timeout = max(timeout, batch_timeout(self.points * self.shots * period))
        P = sweep_levels(drv, m, progs, params, self.points, self.shots, timeout, self.classifiers)

        data, fit, proposal, oks = {}, {}, {}, {}
        for q in self.qubits:
            xs, sig, path = meta[q]
            Pq = P[q]
            data[q] = {"x": xs, "y": Pq, "sig": sig}
            ok, prop = False, {}
            if self.n_gates == 1:
                fq = fits.fit_cosine(sig, Pq)                    # P(|2>) = (1 − cos(rabi·sig))/2
                if fq.ok:
                    rabi = float(TWO_PI * fq.value)
                    g = sig[-1] / (int(xs[-1]) * self.n_gates)   # sig per amp-code (linear)
                    a_star = (self.target_angle / rabi) / g
                    ok = int(xs[0]) <= a_star <= int(xs[-1])     # the fitted gate amp must be in the sweep
                    if ok:
                        self.recovered_rabi[q] = rabi
                        prop = {path: float(np.clip(a_star / units.AMP_SCALE, 0.0, 1.0)),
                                f"qubit/{q}/EF/rabi": rabi}
            else:
                fq = fits.fit_parabola(xs, Pq)                   # P(|2>) MINIMISES at the tuned amp
                in_range = int(xs[0]) <= fq.value <= int(xs[-1])
                if fq.ok and fq.params["a"] > 0 and in_range:    # upward vertex (min P), within the sweep
                    prop = {path: float(np.clip(fq.value / units.AMP_SCALE, 0.0, 1.0))}
                    ok = True
            fit[q], oks[q] = fq, ok
            proposal.update(prop)
        self.data, self.fit = data, fit
        return Result(all(oks.values()), data, fit, proposal, cfg,
                      f"EFAmplitude {self.qubits} {self.gate} n_gates={self.n_gates}")


class EFFrequency:
    """EF Ramsey vs artificial detuning — qcal's `Frequency(subspace='EF')` V-fit (spec 01 §4.1). GE pi
    prep, retune to f_ef, then two EF X90s around a swept wait with a per-wait virtual-Z detuning; the
    host reads P(|2>) off the 3-level clusters (`classifier`) and fits each fringe (damped cosine), then
    the UNSIGNED fringe frequencies |δ + applied| against the applied detuning to qcal's a·|x − b| + c —
    the V bottoms out where the applied detuning cancels the config's EF error (b = −δ). The corrected
    carrier `qubit/{q}/EF/freq` = carrier + b, identical to the GE Frequency lock-step (spec 13 §7).
    `detune`/`n_detune`/`t0`/`dt` are the GE Frequency knobs; the fringe frequency's sign is folded away
    (the V takes |·|), so no res-sign enters — the 3-level classifier reads P(|2>) directly."""

    def __init__(self, cfg, qubits, classifier, detune=5e6, n_detune=4, points=14, t0=80e-9, dt=40e-9,
                 shots=48):
        self.cfg, self.qubits = cfg, qubits_list(qubits)
        self.classifiers = _classifiers(classifier, self.qubits)
        self.detune, self.n_detune = float(detune), int(n_detune)
        self.points, self.t0, self.dt = int(points), float(t0), float(dt)
        self.shots = int(shots)
        self.data, self.fit, self.recovered_detuning_code = {}, {}, {}

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg = self.cfg
        t0, dt = batches(self.t0, m), batches(self.dt, m)
        waits = [t0 + i * dt for i in range(self.points)]            # exact x-axis (host mirror)
        wf = np.array(waits, float)
        D = units._freq_code(self.detune, m.params)
        d_codes = [dc for k in range(1, self.n_detune // 2 + 1) for dc in (-k * D, k * D)]
        progs, carriers = {}, {}
        timeout = 0
        for q in self.qubits:
            carrier = float(cfg[f"qubit/{q}/EF/freq"])
            table, ge_freq, ef_freq = ef_table(cfg, q, m)
            # classified host-side by ClassifierN — capture in the classifier's zero frame (spec 14
            # finding 7)
            ro, demod, code, dur, ddly = readout_tables(cfg, q, m, phase=0.0)
            ge = table.pulses["x90"].dur_batches(m, GATE_CH)
            ef = table.pulses["ef"].dur_batches(m, GATE_CH)
            seq = SEP + 2 * ef + waits[-1] + LEAD + 2 * ge          # earliest pulse → t_ro (longest wait)
            period = grid_period(relax_batches(cfg, m), seq, dur, ddly)
            # compile ONCE: the waits (w0/dw) are baked; only the virtual-Z detuning pair (p0/dp) changes
            # per fringe — leave it unbound, rewrite per rerun (as GE Frequency does).
            progs[q] = compile_kernel(kernels.k_ef_ramsey, m,
                                      tables=dict(gate=table, ro=ro, demod=demod),
                                      out=Array(2 * self.points * self.shots), npts=self.points,
                                      shots=self.shots, period=period, code=code, ddly=ddly,
                                      ge_freq=ge_freq, ef_freq=ef_freq, w0=t0, dw=dt,
                                      **x90_vz(cfg, q), **ef_vz(cfg, q))
            carriers[q] = carrier
            timeout = max(timeout, batch_timeout(self.points * self.shots * period))
        rq.setup(drv, m, progs)

        acc = {q: {"applied": [], "obs": [], "fringes": {}} for q in self.qubits}
        for dc in d_codes:                                          # one rerun per detuning — no reload
            par = {q: {"p0": pack16(16 * dc * t0),                  # phase pair, seated (spec 12)
                       "dp": pack16(16 * dc * dt)} for q in self.qubits}
            P = rerun_levels(drv, m, progs, par, self.points, self.shots, timeout, self.classifiers)
            for q in self.qubits:
                fit = fits.fit_damped_cosine(wf, P[q])
                acc[q]["fringes"][dc] = (wf, P[q], fit)
                if fit.ok:                                          # cycles/batch → an UNSIGNED code:
                    acc[q]["applied"].append(dc)                    # the |δ + applied| qcal's V-fit takes
                    acc[q]["obs"].append(CODE_PER_CYCLE_PER_BATCH * fit.value)

        data, fit_out, proposal, oks = {}, {}, {}, {}
        for q in self.qubits:
            x, y = np.array(acc[q]["applied"], float), np.array(acc[q]["obs"], float)
            data[q] = {"applied": x, "obs": y, "fringes": acc[q]["fringes"]}
            v = fits.fit_absolute_value(x, y)                       # qcal's a·|x − b| + c
            fit_out[q] = v
            ok, prop = False, {}
            if v.ok and len(x) >= 3 and v.params["a"] > 0 and x.min() <= v.value <= x.max():
                dcode = -float(v.value)                             # δ = −b: the EF carrier's error
                self.recovered_detuning_code[q] = dcode
                delta_hz = units.code_to_freq(dcode, m.params)
                prop = {f"qubit/{q}/EF/freq": carriers[q] - delta_hz}   # == qcal's `old + b`
                ok = True
            oks[q] = ok
            proposal.update(prop)
        self.data, self.fit = data, fit_out
        return Result(all(oks.values()), data, fit_out, proposal, cfg, f"EFFrequency {self.qubits}")


class EFPhase:
    """EF X90 virtual-Z pair — qcal's `Phase(subspace='EF')` (spec 04 §2 / X4, single_qubit.py:788-
    1088). Driving the EF X90 Stark-shifts the {|1>, |2>} frame exactly as the GE X90 shifts the GE
    one; qcal corrects it with the virtual-Z pair bracketing the pulse (`qubit/{q}/EF/x90/vz` — the
    X6Y3 config carries a calibrated pair on every qubit) and calibrates the pair with the SAME
    two-sequence line crossing as the GE Phase: qcal's EF circuits only PREPEND the GE π (X90·X90)
    that reaches the subspace, then play Y180_X90 / X180_Y90 on the EF X90 (k_ef_phase's `seq` fold,
    the swept phi as the pair — vz0 = vz1 = phi, written to BOTH slots at the crossing,
    single_qubit.py:1081). The host reads P(|2>) off the pre-trained 3-level `classifier` (the
    EFAmplitude pattern — the hardware res bit cannot tell |1> from |2>); the two populations are
    linear in phi with opposite slopes and cross at the calibrated phase (`_line_crossing`, the GE
    guards verbatim: in-range, chi2-argmin fallback).

    Knobs are the GE Phase's: `span` (rad) around 0, or around the CURRENT vz[0] with
    `relative_phase` (qcal's `phases + config[param]`); `classifier` is EFAmplitude's (a bare
    ClassifierN for one qubit, else {q: ClassifierN}). The sweep is `_phase_sweep`'s monotone axis —
    NOT the full-turn `_phi_sweep` of the Ramsey-peak cals: a line crossing needs a narrow linear
    window, and a full turn would wrap the lines. Writes `qubit/{q}/EF/x90/vz` = [phi, phi].

    THE EF BRACKET (spec 14 §3 finding 6). The two crossing sequences are EXEMPT from playing the
    stored `qubit/{q}/EF/x90/vz`: there the sweep IS the pair (qcal's sweep semantics — it REPLACES
    it, and `relative_phase` is how the sweep is centred on the current value instead), so the kernel
    binds evz0/evzsum = 0. The `gate='X'` circuit below is the opposite case and does play it.

    `gate='X'` is the EF twin of `Phase(gate='X')` (spec 14 §3.3): one circuit, EF-X90 · EF-X · EF-X90
    after the same GE π prep, over the EF X's own AXIS phase (`qubit/{q}/EF/x/phase` — not a
    virtual-Z pair). Its two EF X90s carry their calibrated bracket (`ef_vz`), so the swept phi is
    measured against the axis they actually sit on; the composite is a 2π rotation inside {|1>, |2>}
    that returns to |1> only on alignment, so P(|2>) is cosinusoidal in phi with its MINIMUM at the
    calibrated value (qcal fits the same cosine to P(1) − P(2) and takes the maximum). The default
    sweep is a full turn."""

    CHI2_MAX = 10.0             # qcal's underfitting guard (single_qubit.py:1067)

    def __init__(self, cfg, qubits, classifier, gate="X90", points=21, span=None, shots=48,
                 relative_phase=False):
        assert gate in ("X90", "X"), f"EFPhase calibrates 'X90' or 'X', got {gate!r}"
        self.cfg, self.qubits, self.gate = cfg, qubits_list(qubits), gate
        self.classifiers = _classifiers(classifier, self.qubits)
        self.points = int(points)
        self.span = float(math.pi if span is None and gate == "X" else 0.25 if span is None else span)
        self.shots, self.relative_phase = int(shots), bool(relative_phase)
        self.data, self.fit, self.recovered_vz, self.fallback = {}, {}, {}, {}

    def _centre(self, cfg, q) -> float:
        """The sweep's centre: qcal's `relative_phase` re-centres on the knob's CURRENT value."""
        if not self.relative_phase:
            return 0.0
        if self.gate == "X":
            return float(cfg.get(f"qubit/{q}/EF/x/phase", 0.0))
        return float(cfg.get(f"qubit/{q}/EF/x90/vz", [0.0, 0.0])[0])

    def run(self, drv) -> Result:
        if self.gate == "X":
            return self._run_x(drv)
        m = socmap(drv)
        cfg = self.cfg
        hpi = pack16(units._phase_code(math.pi / 2))
        axis = {}
        for q in self.qubits:
            centre = self._centre(cfg, q)
            axis[q] = _phase_sweep(centre - self.span, centre + self.span, self.points)
        pops = {}                                              # seq -> {q: P(|2>)}
        for seq in (kernels.Y180_X90, kernels.X180_Y90):       # one compile + run per qcal sequence
            progs, par = {}, {}
            timeout = 0
            for q in self.qubits:
                table, ge_freq, ef_freq = ef_table(cfg, q, m)
                # classified host-side by ClassifierN — capture in the classifier's zero frame
                # (spec 14 finding 7)
                ro, demod, code, dur, ddly = readout_tables(cfg, q, m, phase=0.0)
                ge = table.pulses["x90"].dur_batches(m, GATE_CH)
                ef = table.pulses["ef"].dur_batches(m, GATE_CH)
                seq_len = SEP + 3 * ef + LEAD + 2 * ge         # earliest pulse (GE prep) → t_ro
                period = grid_period(relax_batches(cfg, m), seq_len, dur, ddly)
                progs[q] = compile_kernel(kernels.k_ef_phase, m,
                                          tables=dict(gate=table, ro=ro, demod=demod),
                                          out=Array(2 * self.points * self.shots), npts=self.points,
                                          shots=self.shots, period=period, code=code, ddly=ddly,
                                          ge_freq=ge_freq, ef_freq=ef_freq, seq=seq, hpi=hpi,
                                          # the swept phi IS the EF pair here (qcal writes one
                                          # crossing to both slots), so the stored bracket is
                                          # REPLACED, not composed with — the GE Phase contract
                                          evz0=0, evzsum=0, **x90_vz(cfg, q))
                p0, dp, _ = axis[q]
                par[q] = {"p0": pack16(p0), "dp": pack16(dp)}  # the swept EF pair, host-seated
                timeout = max(timeout, batch_timeout(self.points * self.shots * period))
            pops[seq] = sweep_levels(drv, m, progs, par, self.points, self.shots, timeout,
                                     self.classifiers)

        data, fit_out, proposal, oks = {}, {}, {}, {}
        for q in self.qubits:
            _, _, x = axis[q]
            p_y, p_x = pops[kernels.Y180_X90][q], pops[kernels.X180_Y90][q]
            data[q] = {"x": x, "y": (p_y - p_x) ** 2, "p0": p_y, "p1": p_x}
            fit_out[q], phi, self.fallback[q], ok = _line_crossing(x, p_y, p_x, self.CHI2_MAX)
            self.recovered_vz[q] = phi
            oks[q] = ok
            if ok:
                proposal[f"qubit/{q}/EF/x90/vz"] = [phi, phi]  # qcal: ONE crossing, BOTH slots
        self.data, self.fit = data, fit_out
        return Result(all(oks.values()), data, fit_out, proposal, cfg, f"EFPhase {self.qubits}")

    def _run_x(self, drv) -> Result:
        """gate='X': one EF-X90 · EF-X · EF-X90 run; the EF X's axis phase is the P(|2>) MINIMUM."""
        m = socmap(drv)
        cfg = self.cfg
        progs, par, axis = {}, {}, {}
        timeout = 0
        for q in self.qubits:
            centre = self._centre(cfg, q)
            axis[q] = _phase_sweep(centre - self.span, centre + self.span, self.points)
            table, ge_freq, ef_freq = ef_table(cfg, q, m)      # "x90" GE prep + "ef" EF X90
            efx = ef_pulse(cfg, q, m, "x")
            # the swept phi REPLACES the EF X's stored axis (qcal writes the pulse's own phase kwarg),
            # so the slot is built at 0 and the on-core phase offset carries the whole axis
            table.pulses["efx"] = Pulse(efx.env, amp=efx.amp)
            # classified host-side by ClassifierN — capture in the classifier's zero frame (spec 14
            # finding 7)
            ro, demod, code, dur, ddly = readout_tables(cfg, q, m, phase=0.0)
            ge = table.pulses["x90"].dur_batches(m, GATE_CH)
            ef = table.pulses["ef"].dur_batches(m, GATE_CH)
            xd = table.pulses["efx"].dur_batches(m, GATE_CH)
            seq_len = SEP + 2 * ef + xd + LEAD + 2 * ge         # earliest pulse (GE prep) → t_ro
            period = grid_period(relax_batches(cfg, m), seq_len, dur, ddly)
            progs[q] = compile_kernel(kernels.k_ef_phase, m,
                                      tables=dict(gate=table, ro=ro, demod=demod),
                                      out=Array(2 * self.points * self.shots), npts=self.points,
                                      shots=self.shots, period=period, code=code, ddly=ddly,
                                      ge_freq=ge_freq, ef_freq=ef_freq, seq=kernels.X90_X_X90,
                                      hpi=pack16(units._phase_code(math.pi / 2)),
                                      # the two EF X90s keep their calibrated bracket; only the EF
                                      # X's axis is swept (its own pair is [0, 0])
                                      **x90_vz(cfg, q), **ef_vz(cfg, q))
            p0, dp, _ = axis[q]
            par[q] = {"p0": pack16(p0), "dp": pack16(dp)}       # the swept EF X axis, host-seated
            timeout = max(timeout, batch_timeout(self.points * self.shots * period))
        P = sweep_levels(drv, m, progs, par, self.points, self.shots, timeout, self.classifiers)

        data, fit_out, proposal, oks = {}, {}, {}, {}
        for q in self.qubits:
            _, _, x = axis[q]
            data[q] = {"x": x, "y": P[q]}
            fit_out[q], phi, self.fallback[q], ok = _cosine_axis(x, P[q], self._centre(cfg, q))
            self.recovered_vz[q], oks[q] = phi, ok
            if ok:
                proposal[f"qubit/{q}/EF/x/phase"] = float(phi)
        self.data, self.fit = data, fit_out
        return Result(all(oks.values()), data, fit_out, proposal, cfg, f"EFPhase {self.qubits} X")
