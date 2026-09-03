"""riscq.cal.rpe — robust phase estimation (spec 14 F5).

RPE amplifies a small coherent error by repeating a gate `d` times and reading the accumulated
phase out in two quadratures. Repeating over an exponential depth ladder gives an angle estimate
whose precision improves like 1/d rather than 1/sqrt(shots), which is what makes it the polish
step after the conventional (fit-a-curve) calibrations have converged.

The analysis half turns per-depth measurement counts into angles (`idle_angles` / `x90_angles` /
`cz_angles`) and angles into config corrections (`freq_error_hz`, `damped_update`); the pyRPE
branch-selection kernel it runs on is vendored under `riscq.cal._vendor.pyrpe`. The cal classes
(`RPEFrequency`, ...) drive the circuits that produce those counts and propose the config write.

The circuits reuse the existing kernels rather than adding RPE-specific ones wherever the shape
already matches: a repeated-idle Ramsey is `k_ramsey` with one point and the idle scaled by the
depth; an X90^d train is `k_rabi` at `ngates=d`; a CZ^d conditional Ramsey is `k_cz_cond`. Only
the interleaved X90 echo has no counterpart, and it gets the one new kernel (`k_rpe_echo`).

Conventions follow the qcal reference (`qcal/characterization/phase_estimation/`), with three
deliberate departures, each noted at its definition:

- the trusted-estimate uncertainty divides by the *actual* last-good depth rather than assuming
  the ladder is exactly 1, 2, 4, ...;
- a failed consistency check raises rather than silently wrapping to the last element of the
  ladder (the reference's negative-index bug);
- the ladder is additionally bounded by per-depth signal contrast, which is the only thing that
  catches a ladder run past the coherence time (see `_alive`).

Feedback is a damped clip update (spec 14 §4) — the reference's Kalman/CMA optimizer stack is
explicitly out of scope, and the walkthrough's own guidance is the damped clip.
"""

import math
from dataclasses import dataclass

import numpy as np

from riscq import run as rq
from riscq.cal import kernels, twoqubit
from riscq.cal._vendor.pyrpe import Q, RobustPhaseEstimation
from riscq.cal.base import (Result, batch_timeout, batches, gate_pulse, grid_period, herald_offset,
                            heralding, population, population_heralded, prep, qubit_freq,
                            qubits_list, readout_tables, relax_batches, res_sign, socmap,
                            sweep_counts, sweep_q16, train_step, x90_vz)
from riscq.lang import Array, ParamTable, compile_kernel
from riscq.map import pack16
from riscq.pulses import units

TWO_PI = 2.0 * np.pi

#: The three two-qubit state pairs an RPE CZ experiment measures. Each yields one accumulated
#: per-CZ angle; together they invert to the ZZ / IZ / ZI generator angles.
CZ_STATE_PAIRS = ((0, 1), (2, 3), (3, 1))

#: Ideal-CZ targets in the generator convention CZ = expm(-i/2 (θ_IZ·IZ + θ_ZI·ZI + θ_ZZ·ZZ)),
#: for which diag(1, 1, 1, -1) == CZ(π/2, π/2, -π/2).
CZ_TARGETS = {"ZZ": -np.pi / 2, "IZ": np.pi / 2, "ZI": np.pi / 2}

#: The X90 rotation angle RPE is calibrating towards.
X90_TARGET = np.pi / 2


def wrap(theta):
    """Wrap an angle (or array) into [-π, π) — the reference's `rectify_angle`."""
    return (theta + np.pi) % TWO_PI - np.pi


#: Minimum sqrt((2P_sin-1)^2 + (2P_cos-1)^2) at the shallowest depth for the run to mean anything.
#: Ideally the contrast is 1; it decays with depth as the state decoheres.
MIN_CONTRAST = 0.1


class RPEBranchError(RuntimeError):
    """No generation of the ladder can be trusted, so there is no angle to report.

    Two distinct failures raise this:

    - the consistency check rejected even the shallowest generation. The reference returns -1
      here, which it then uses as a *negative index*, silently trusting the deepest — least
      reliable — generation instead of failing;
    - the signal has no contrast. That one is invisible to the consistency check: a flat P = 1/2
      makes `arctan2(0, 0)` return 0 at every depth, which is perfectly self-consistent and
      reports as a converged angle of exactly zero — a dead qubit reads out as a perfect gate.
      Hence the explicit contrast floor.
    """


@dataclass
class Angles:
    """One RPE experiment's output.

    `estimates` maps each angle name to its per-generation ladder (one entry per depth);
    `errors` is the same ladder minus the ideal target. `last_good` indexes the deepest
    generation that survived the consistency check — every quantity a caller should act on is
    evaluated there, and everything past it is measured but discarded.
    """

    depths: tuple
    estimates: dict
    errors: dict
    last_good: int
    n_shots: int
    #: Per-depth signal contrast, ideally 1 and decaying towards 0 as the state decoheres. Worth
    #: plotting: it is what tells you whether the ladder was too deep for the qubit.
    contrast: np.ndarray = None
    #: The RAW per-experiment pyRPE ladders this angle set was inverted from, keyed by whatever
    #: labels the experiments (for `cz_angles`, the state pair), each
    #: {"ladder", "contrast", "last_good"}. Only the CZ estimator fills it, because only its
    #: inversion mixes three independent experiments: a composite angle that disagrees with the
    #: rest of the calibration says nothing about WHICH ladder moved, and the identities worth
    #: checking (`A(2,3) == A(3,1)` on a converged tree — spec 14 §3 finding 9) live on the raw
    #: ladders, not on the inverted angles.
    ladders: dict = None

    @property
    def trusted(self):
        """The trusted angle per name: the ladder evaluated at the last good generation."""
        return {name: float(ladder[self.last_good]) for name, ladder in self.estimates.items()}

    @property
    def trusted_error(self):
        """The trusted deviation from target per name — this is what a cal loop drives to zero."""
        return {name: float(ladder[self.last_good]) for name, ladder in self.errors.items()}

    @property
    def uncertainty(self):
        """Shot-noise-limited 1-sigma on the trusted angle: π / (2·L·sqrt(N_shots)).

        L is the last good *depth*. The reference hard-codes 2**last_good, which silently
        misreports the moment the ladder is not exactly 1, 2, 4, ...; indexing `depths` is the
        same number for a doubling ladder and correct for any other.
        """
        return np.pi / (2.0 * self.depths[self.last_good] * np.sqrt(self.n_shots))

    @property
    def last_good_depth(self):
        return self.depths[self.last_good]


def _ladder(cos_counts, sin_counts):
    """Run pyRPE over per-depth count pairs -> (per-generation angle ladder, last good index).

    `cos_counts` / `sin_counts` map depth -> (n_plus, n_minus). "Plus" is the outcome whose
    probability is (1 + cos(d·φ))/2 resp. (1 + sin(d·φ))/2; picking which measured bitstring that
    is belongs to the experiment, not here.
    """
    depths = sorted(cos_counts)
    if depths != sorted(sin_counts):
        raise ValueError("cos and sin count dicts must cover the same depths")

    q = Q()
    for d in depths:
        q.process_cos(d, np.asarray(cos_counts[d], dtype=int))
        q.process_sin(d, np.asarray(sin_counts[d], dtype=int))

    contrast = np.array([q.amplitude_N(d) for d in depths])
    if not contrast[0] > MIN_CONTRAST:
        raise RPEBranchError(
            f"no signal contrast at depth {depths[0]} ({contrast[0]:.3f} < {MIN_CONTRAST}); "
            "the qubit is not responding — an angle fitted to this would be meaningless"
        )

    analysis = RobustPhaseEstimation(q)
    last_good = analysis.check_unif_local(historical=True)
    if last_good < 0:
        raise RPEBranchError(
            f"no generation passed the consistency check (depths {depths}); "
            "the estimates disagree even at the shallowest rung"
        )
    return np.asarray(analysis.angle_estimates), min(last_good, _alive(contrast)), contrast


def _alive(contrast):
    """Index of the deepest rung whose signal is still above the contrast floor.

    The consistency check alone does not bound the ladder at the coherence time: once the state
    has decohered the two quadratures are both ~1/2, the extracted angle is arbitrary, and
    successive dead rungs can easily agree with each other to within their (by then very narrow)
    consistency windows — accepting a confidently wrong answer. Contrast decays with depth, so
    the trustworthy prefix is the run of rungs before it first drops through the floor.
    """
    below = np.flatnonzero(contrast <= MIN_CONTRAST)
    return int(below[0]) - 1 if below.size else len(contrast) - 1


def idle_angles(cos_counts, sin_counts, n_shots, depths=None):
    """1Q frequency RPE (the reference's gate='I'): phase accumulated over a bare idle.

    Circuit at depth d: prep (Y90 for cos, X90 for sin) -> idle · d -> Y(-90) -> measure.
    The estimated 'Z' angle is the phase per idle step; it is its own error (target 0), and
    `freq_error_hz` converts it to the detuning that produced it.
    """
    ladder, last_good, contrast = _ladder(cos_counts, sin_counts)
    z = wrap(ladder)
    return Angles(
        depths=tuple(depths or sorted(cos_counts)),
        estimates={"Z": z},
        errors={"Z": z},
        last_good=last_good,
        n_shots=n_shots,
        contrast=contrast,
    )


def x90_angles(direct_cos, direct_sin, interleaved_cos, interleaved_sin, n_shots, depths=None):
    """1Q amplitude + phase RPE (the reference's gate='X90').

    Two experiments run against the same X90. The **direct** one (X90 repeated d times) measures
    the rotation angle, so it sees amplitude error. The **interleaved** one repeats an echo block
    that cancels the rotation angle and amplifies the tilt of the rotation axis out of x̂, so it
    sees drive-phase error. The reference's linearized estimator recombines them into Cartesian
    axis components:

        ε      = θ_direct / (π/2) - 1                        (fractional over-rotation)
        θ_off  = sin(θ_int / 2) / (2 cos(π ε / 2))           (axis tilt toward ẑ)
        X      = (π/2)(1 + ε) cos(θ_off)   -> target π/2
        Z      = (π/2)(1 + ε) sin(θ_off)   -> target 0

    The ladders are truncated to their common length: the interleaved block spends four X90s per
    repetition, so its depth ladder is typically the shallower of the two.
    """
    direct, direct_k, direct_c = _ladder(direct_cos, direct_sin)
    interleaved, interleaved_k, interleaved_c = _ladder(interleaved_cos, interleaved_sin)

    n = min(len(direct), len(interleaved))
    direct, interleaved = wrap(direct[:n]), wrap(interleaved[:n])
    last_good = min(direct_k, interleaved_k, n - 1)
    contrast = np.minimum(direct_c[:n], interleaved_c[:n])

    epsilon = direct / X90_TARGET - 1.0
    tilt = np.sin(interleaved / 2.0) / (2.0 * np.cos(np.pi * epsilon / 2.0))
    magnitude = X90_TARGET * (1.0 + epsilon)
    x, z = magnitude * np.cos(tilt), magnitude * np.sin(tilt)

    return Angles(
        depths=tuple((depths or sorted(direct_cos))[:n]),
        estimates={"X": x, "Z": z},
        errors={"X": x - X90_TARGET, "Z": z},
        last_good=last_good,
        n_shots=n_shots,
        contrast=contrast,
    )


def x90_direct_angles(cos_counts, sin_counts, n_shots, depths=None):
    """The X90's rotation angle alone, from the direct (repeated-X90) experiment.

    This is what an amplitude calibration needs — the drive amplitude is linear in the rotation
    angle, so the correction is the ratio (pi/2)/angle. It carries no information about where the
    rotation axis points; recovering that needs the interleaved echo too (`x90_angles`).
    """
    ladder, last_good, contrast = _ladder(cos_counts, sin_counts)
    x = wrap(ladder)
    return Angles(
        depths=tuple(depths or sorted(cos_counts)),
        estimates={"X": x},
        errors={"X": x - X90_TARGET},
        last_good=last_good,
        n_shots=n_shots,
        contrast=contrast,
    )


def cz_angles(pair_counts, n_shots, depths=None):
    """CZ ZZ / IZ / ZI RPE (the reference's gate='CZ').

    `pair_counts` maps each of `CZ_STATE_PAIRS` to its (cos_counts, sin_counts). Each state pair
    is a Ramsey on one qubit with the other held in |0> or |1>, so it accumulates a different
    combination of the CZ's generator angles per repetition:

        A(0,1) = θ_IZ + θ_ZZ      (control |0>: ideally 0 — no phase on the target)
        A(2,3) = θ_IZ - θ_ZZ      (control |1>: ideally π)
        A(3,1) = θ_ZI - θ_ZZ      (target  |1>: ideally π)

    which inverts to ZZ = (A01 - A23)/2, IZ = (A01 + A23)/2, ZI = A31 + ZZ. Only the (0, 1)
    ladder is wrapped to [-π, π); the other two sit near π, where wrapping would straddle the cut.

    The three raw ladders come back on `Angles.ladders` as well as the inverted angles: the
    inversion is a 3->3 mix, so an angle that contradicts the rest of the calibration is only
    diagnosable against the ladder it came from (spec 14 §3 finding 9).
    """
    missing = set(CZ_STATE_PAIRS) - set(pair_counts)
    if missing:
        raise ValueError(f"missing counts for CZ state pair(s) {sorted(missing)}")

    per_pair, last_goods, contrasts = {}, [], []
    for pair in CZ_STATE_PAIRS:
        cos_counts, sin_counts = pair_counts[pair]
        ladder, k, c = _ladder(cos_counts, sin_counts)
        per_pair[pair] = {"ladder": wrap(ladder) if pair == (0, 1) else ladder,
                          "contrast": c, "last_good": k}
        last_goods.append(k)
        contrasts.append(c)

    n = min(len(d["ladder"]) for d in per_pair.values())
    a01, a23, a31 = (per_pair[p]["ladder"][:n] for p in CZ_STATE_PAIRS)
    contrast = np.min([c[:n] for c in contrasts], axis=0)

    zz = 0.5 * (a01 - a23)
    estimates = {"ZZ": zz, "IZ": 0.5 * (a01 + a23), "ZI": a31 + zz}
    errors = {name: ladder - CZ_TARGETS[name] for name, ladder in estimates.items()}

    return Angles(
        depths=tuple((depths or sorted(pair_counts[(0, 1)][0]))[:n]),
        estimates=estimates,
        errors=errors,
        last_good=min(*last_goods, n - 1),
        n_shots=n_shots,
        contrast=contrast,
        ladders=per_pair,
    )


def freq_error_hz(angle_z, idle_time):
    """Phase accumulated per idle step -> the detuning in Hz that produced it.

    A qubit detuned by Δf from its drive accumulates 2π·Δf·t of phase over an idle of length t,
    so Δf = θ / (2π·t). Subtract this from the config frequency. The estimate is unambiguous only
    while |Δf| < 1/(2·t) — a 100 ns idle resolves ±5 MHz, which is why the ladder starts short.
    """
    return angle_z / (TWO_PI * idle_time)


def vz_correction(x_angle, z_angle):
    """The virtual-Z pair shift that straightens a rotation axis tilted out of the drive plane.

    RPE reports the gate as exp(-i(X·sigma_x + Z·sigma_z)/2) — a rotation of Omega = hypot(X, Z)
    about an axis in the x-z plane. Its symmetric decomposition is Rz(beta)·Rx(X')·Rz(beta) with

        beta = atan2(Z·tan(Omega/2), Omega)

    (the same algebra `Phase`'s ac-Stark model uses, and beta -> (2/pi)·Z for a small tilt on a
    quarter turn — NOT Z/2), so the pulse leaves beta of Z on EACH side and adding beta to BOTH
    virtual-Z slots cancels it exactly. The frame the ladder measures already includes the config's
    current pair, so this is the SHIFT to apply, not the new pair.
    """
    omega = math.hypot(x_angle, z_angle)
    return math.atan2(z_angle * math.tan(omega / 2.0), omega)


def damped_update(old, correction, gain=0.5, max_step=None, multiplicative=False):
    """Apply one damped, clipped correction to a config value.

    This is the whole feedback rule (spec 14 §4): the reference wraps RPE in Kalman/CMA
    optimizers, but the walkthrough's own guidance is a damped clip update, so that is what we
    adopt. Damping keeps a wrong-branch estimate from throwing the parameter across the map;
    clipping bounds the damage when it does.
    """
    step = gain * correction
    if max_step is not None:
        step = float(np.clip(step, -max_step, max_step))
    return old * (1.0 + step) if multiplicative else old + step


class RPEFrequency:
    """1Q qubit frequency by RPE — qcal's `gate='I'` (spec 14 F5).

    A Ramsey whose idle is *repeated*: at depth d the qubit idles d * t_idle between the two X90s,
    so a carrier error delta writes 2*pi*delta*d*t_idle of phase. Each doubling of d halves the
    resolvable delta, which is what buys RPE its precision over the single-fringe V-fit that
    `Frequency` runs — and why RPE is the polish step, not the first pass: it needs a carrier
    already within 1/(2*t_idle) of the qubit for the depth-1 rung to be unambiguous.

    Runs the quadratures at each depth as reruns of one image (the closing X90's virtual-Z is the
    only thing that changes), feeds the counts to pyRPE, and proposes `qubit/{q}/freq` moved by a
    damped clip of the recovered detuning. `Frequency` should have run first.

    Two systematics set how deep and how slow the ladder has to be:

    - **finite pulses.** Every rung brackets its idle with the same two X90s, and the detuning
      writes phase during those too. So the measured angle is d*theta + phi_pulse rather than
      d*theta, and the estimate carries a fractional bias of about t_pulse/(d*t_idle) — the same
      at every depth in absolute terms, but divided by d. Keep `t_idle` well above the gate
      length and take the ladder deep: at t_idle = 2x the gate and depth 8 the bias is still ~5%.
      This is why the reference runs to depth 4096.
    - **the polish assumption.** A carrier far enough off to detune the drive *during* the pulse
      stops the X90 being an X90. Measured on the co-sim model, a 1.46 MHz error (0.12 cycles
      across an 80 ns pulse) collapsed the depth-1 contrast to 0.42 and biased the shallow rungs
      by ~0.9 rad. Run `Frequency` first; RPE refines, it does not acquire.
    """

    #: Each quadrature as (name, plus close, minus close) — the two *opposite* closing phases.
    #:
    #: pyRPE reads its signal as (C+ - C-)/(C+ + C-), which is only the intended cos/sin if the
    #: two counts are the two outcomes of a balanced measurement. Taking C- = shots - C+ off a
    #: single close is NOT balanced here: T1 decay in the gap between the last gate and the
    #: readout shrinks the measured |1> population, so the fringe is centred below 1/2 and the
    #: offset leaks straight into the angle. Measured on the co-sim model: the fringe sat at 0.37
    #: rather than 0.5, biasing the shallow rungs by ~0.05 rad.
    #:
    #: Closing at phi and at phi+pi instead gives P± = c ± a·cos(theta), so the ratio is
    #: (2a·cos(theta))/(2c) — the centre c divides out exactly and the amplitude a only scales
    #: both components, which arctan2 is blind to. That makes the measurement self-calibrating
    #: against readout asymmetry and decay, at the cost of two more reruns per depth.
    #:
    #: The sin close is placed so the ladder reports the CARRIER's error (positive when the
    #: carrier sits above the qubit) — the same sense as `Frequency`'s delta, so both cals
    #: subtract it. That sign is pinned in co-sim against a planted detuning of each sign.
    QUADRATURES = (("cos", 0.0, math.pi), ("sin", -math.pi / 2, math.pi / 2))

    def __init__(self, cfg, qubits, t_idle=100e-9, depths=(1, 2, 4, 8, 16), shots=256,
                 gain=1.0, max_step=None):
        self.cfg, self.qubits = cfg, qubits_list(qubits)
        self.t_idle, self.depths = float(t_idle), tuple(int(d) for d in depths)
        self.shots, self.gain = int(shots), float(gain)
        # Past 1/(2*t_idle) the depth-1 rung is already ambiguous, so a correction bigger than that
        # cannot have been measured — it can only be a wrong branch. Clip there by default.
        self.max_step = float(max_step) if max_step is not None else 0.5 / self.t_idle
        if min(self.depths) < 1:
            raise ValueError(f"depths must be >= 1, got {self.depths}")
        if sorted(self.depths) != list(self.depths):
            raise ValueError(f"depths must be ascending, got {self.depths}")
        self.data, self.fit, self.angles, self.recovered_detuning = {}, {}, {}, {}

    def _periods(self, m):
        """ONE grid period for the whole ladder, sized for the deepest rung.

        Every rung has to present the qubit with the same duty cycle. Size each rung's period
        from its own sequence and the shallow ones get a shorter relaxation head, so they start
        more often from a residual excitation and read back with less contrast — a depth-dependent
        state-prep error, which is exactly the systematic a ladder cannot absorb (it biases the
        shallow rungs, and those are what the branch selection leans on). Measured with per-depth
        periods on the co-sim model: contrast climbed 0.34 -> 1.0 from depth 1 to depth 8, i.e.
        backwards, and the shallow estimates were biased by ~0.3 rad.
        """
        cfg, herald = self.cfg, heralding(self.cfg)
        deepest = max(self.depths) * batches(self.t_idle, m)
        periods = {}
        for q in self.qubits:
            gate = ParamTable(0, qubit_freq(cfg, q), {"x90": gate_pulse(cfg, q, m)})
            _, _, _, dur, ddly = readout_tables(cfg, q, m)
            d = gate.pulses["x90"].dur_batches(m, gate.channel)
            periods[q] = grid_period(relax_batches(cfg, m), deepest + 2 * d, dur, ddly,
                                     herald=herald)
        return periods

    def _programs(self, drv, m, depth, periods):
        """Compile + load one image per depth: the idle is baked, the quadrature stays runtime."""
        cfg, herald = self.cfg, heralding(self.cfg)
        wait = depth * batches(self.t_idle, m)
        progs, signs, timeout = {}, {}, 0
        for q in self.qubits:
            gate = ParamTable(0, qubit_freq(cfg, q), {"x90": gate_pulse(cfg, q, m)})
            ro, demod, code, dur, ddly = readout_tables(cfg, q, m)
            d = gate.pulses["x90"].dur_batches(m, gate.channel)
            progs[q] = compile_kernel(
                kernels.k_ramsey, m, tables=dict(gate=gate, ro=ro, demod=demod),
                out=Array(2 if herald else 1), npts=1, shots=self.shots, period=periods[q],
                code=code, mode=kernels.COUNTS, ddly=ddly, w0=wait, dw=0, dp=0,
                herald=int(herald),
                hoff=herald_offset(wait + 2 * d, ddly) if herald else 0, **x90_vz(cfg, q))
            signs[q] = res_sign(cfg, q)
            timeout = max(timeout, batch_timeout(self.shots * periods[q]))
        rq.setup(drv, m, progs)
        return progs, signs, timeout

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg, herald = self.cfg, heralding(self.cfg)
        counts = {q: {"cos": {}, "sin": {}} for q in self.qubits}
        periods = self._periods(m)

        for depth in self.depths:
            progs, signs, timeout = self._programs(drv, m, depth, periods)

            def _populations(phase, progs=progs, signs=signs, timeout=timeout):
                par = {q: {"p0": pack16(units._phase_code(phase))} for q in self.qubits}
                out = rq.rerun(drv, m, progs, params=par, results=["out"], timeout=timeout)
                return {q: float(np.atleast_1d(
                    population_heralded(out[q]["out"], signs[q]) if herald
                    else population(out[q]["out"], self.shots, signs[q]))[0]) for q in self.qubits}

            for name, plus, minus in self.QUADRATURES:
                p_plus, p_minus = _populations(plus), _populations(minus)
                for q in self.qubits:
                    counts[q][name][depth] = (int(round(p_plus[q] * self.shots)),
                                              int(round(p_minus[q] * self.shots)))

        data, fit_out, proposal, oks = {}, {}, {}, {}
        for q in self.qubits:
            carrier = qubit_freq(cfg, q)
            data[q] = {"depths": self.depths, "counts": counts[q], "carrier": carrier}
            ok, prop = False, {}
            try:
                angles = idle_angles(counts[q]["cos"], counts[q]["sin"], self.shots, self.depths)
            except RPEBranchError as exc:
                data[q]["error"] = str(exc)
            else:
                self.angles[q] = fit_out[q] = angles
                data[q]["contrast"] = angles.contrast
                # The Ramsey phase runs at (carrier - f_qubit), so the rate this recovers IS the
                # carrier's error — the same quantity `Frequency` calls delta — and the corrected
                # carrier SUBTRACTS it, exactly as `Frequency`'s `carrier - delta_hz` does. Pinned
                # in co-sim against a planted detuning of each sign: get it backwards and the
                # "correction" doubles the error for one sign while looking perfect for the other.
                detuning = freq_error_hz(angles.trusted["Z"], self.t_idle)
                self.recovered_detuning[q] = detuning
                prop = {f"qubit/{q}/freq": damped_update(carrier, -detuning, gain=self.gain,
                                                         max_step=self.max_step)}
                ok = True
            oks[q] = ok
            proposal.update(prop)

        self.data, self.fit = data, fit_out
        return Result(all(oks.values()), data, fit_out, proposal, cfg, f"RPEFrequency {self.qubits}")


def _x90_train(drv, m, cfg, qubits, shots, ngates, periods):
    """Compile + run ONE `ngates`-long X90 train across all qubits -> {q: P(|1>)}.

    The DIRECT half of both X90 experiments — `RPEAmplitude` reads the rotation angle straight off
    it, and `RPEPhase` needs the same angle to linearize its tilt. `k_rabi` at npts = 1, so it
    inherits the F1 pacing contract: each gate is pushed only once the gate TRAIN_AHEAD before it
    has started, on the fixed `train_step` grid.
    """
    herald = heralding(cfg)
    progs, params, signs, timeout = {}, {}, {}, 0
    for q in qubits:
        table, pg, _ = prep(cfg, q, m, "X90")
        ro, demod, code, dur, ddly = readout_tables(cfg, q, m)
        d = table.pulses["x90"].dur_batches(m, table.channel)
        step = train_step(d)                                      # the paced train grid (spec 14 F1)
        progs[q] = compile_kernel(
            kernels.k_rabi, m, tables=dict(gate=table, ro=ro, demod=demod),
            out=Array(2 if herald else 1), npts=1, shots=shots, period=periods[q],
            ngates=ngates, step=step, code=code, mode=kernels.COUNTS, ddly=ddly, prep_gate=pg,
            herald=int(herald), hoff=herald_offset((ngates - 1) * step + d, ddly) if herald else 0,
            **x90_vz(cfg, q))
        a0q, daq, _ = sweep_q16(units._amp_code(float(cfg[f"qubit/{q}/x90/amp"])),
                                units._amp_code(float(cfg[f"qubit/{q}/x90/amp"])), 1)
        params[q] = {"a0q": int(a0q), "daq": int(daq), "prep": 1}
        signs[q] = res_sign(cfg, q)
        timeout = max(timeout, batch_timeout(shots * periods[q]))
    P = sweep_counts(drv, m, progs, params, shots, timeout, signs, herald=herald)
    return {q: float(np.atleast_1d(P[q])[0]) for q in qubits}


class RPEAmplitude:
    """1Q X90 amplitude by RPE — the direct half of qcal's `gate='X90'` (spec 14 F5).

    Repeating the X90 amplifies its amplitude error: a gate that rotates by Theta instead of pi/2
    lands the state at d*Theta after d gates, so the deviation grows with depth while the readout
    noise does not. The recovered per-gate Theta corrects the amplitude by the ratio (pi/2)/Theta,
    since the rotation angle is linear in drive amplitude.

    Each rung measures FOUR trains — d, d+1, d+2, d+3 gates. Two gates is a pi rotation, so the
    d+2 train reads the exact opposite outcome to the d train and the pair is balanced; d+1 and
    d+3 are the same trick a quarter turn along, giving the sin quadrature. That balance is what
    keeps a fringe not centred on 1/2 (readout asymmetry, T1 decay before the readout) from
    leaking into the angle — see `RPEFrequency.QUADRATURES` for the measurement of that effect.

    The trains are `k_rabi`'s, so they inherit the F1 pacing contract: each gate is pushed only
    once the gate TRAIN_AHEAD before it has started, on a fixed `train_step` grid. `Amplitude`
    should have run first — RPE refines a working X90, it does not find one.
    """

    #: Extra gates appended to a rung to read the four balanced outcomes, as
    #: (quadrature, plus train, minus train) offsets from the rung's depth.
    TRAINS = (("cos", 2, 0), ("sin", 1, 3))

    def __init__(self, cfg, qubits, depths=(1, 2, 4, 8), shots=256, gain=1.0, max_step=0.25):
        self.cfg, self.qubits = cfg, qubits_list(qubits)
        self.depths = tuple(int(d) for d in depths)
        self.shots, self.gain, self.max_step = int(shots), float(gain), float(max_step)
        if min(self.depths) < 1:
            raise ValueError(f"depths must be >= 1, got {self.depths}")
        if sorted(self.depths) != list(self.depths):
            raise ValueError(f"depths must be ascending, got {self.depths}")
        self.data, self.fit, self.angles, self.recovered_angle = {}, {}, {}, {}

    def _period(self, m, q):
        """One grid period for the whole ladder, sized for the longest train (see `RPEFrequency`)."""
        cfg = self.cfg
        table, _, _ = prep(cfg, q, m, "X90")
        _, _, _, dur, ddly = readout_tables(cfg, q, m)
        d = table.pulses["x90"].dur_batches(m, table.channel)
        longest = max(self.depths) + max(off for _, plus, minus in self.TRAINS
                                         for off in (plus, minus))
        seq = (longest - 1) * train_step(d) + d
        return grid_period(relax_batches(cfg, m), seq, dur, ddly, herald=heralding(cfg))

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg = self.cfg
        periods = {q: self._period(m, q) for q in self.qubits}
        counts = {q: {"cos": {}, "sin": {}} for q in self.qubits}
        cache = {}

        for depth in self.depths:
            for name, plus, minus in self.TRAINS:
                for off in (plus, minus):
                    if depth + off not in cache:
                        cache[depth + off] = _x90_train(drv, m, cfg, self.qubits, self.shots,
                                                        depth + off, periods)
                for q in self.qubits:
                    counts[q][name][depth] = (int(round(cache[depth + plus][q] * self.shots)),
                                              int(round(cache[depth + minus][q] * self.shots)))

        data, fit_out, proposal, oks = {}, {}, {}, {}
        for q in self.qubits:
            path = f"qubit/{q}/x90/amp"
            amp = float(cfg[path])
            data[q] = {"depths": self.depths, "counts": counts[q], "amp": amp}
            ok, prop = False, {}
            try:
                angles = x90_direct_angles(counts[q]["cos"], counts[q]["sin"], self.shots,
                                           self.depths)
            except RPEBranchError as exc:
                data[q]["error"] = str(exc)
            else:
                self.angles[q] = fit_out[q] = angles
                data[q]["contrast"] = angles.contrast
                theta = angles.trusted["X"]
                self.recovered_angle[q] = theta
                if theta > 0:                       # a non-positive rotation angle is not a gate
                    prop = {path: float(np.clip(
                        damped_update(amp, X90_TARGET / theta - 1.0, gain=self.gain,
                                      max_step=self.max_step, multiplicative=True), 0.0, 1.0))}
                    ok = True
            oks[q] = ok
            proposal.update(prop)

        self.data, self.fit = data, fit_out
        return Result(all(oks.values()), data, fit_out, proposal, cfg, f"RPEAmplitude {self.qubits}")


class RPEPhase:
    """1Q X90 drive PHASE (axis) by RPE — the interleaved half of qcal's `gate='X90'`, its
    `loss_angle='Z'` (spec 14 F5).

    Two experiments run against the same X90 and are recombined by `x90_angles`:

    - the DIRECT train (`k_rabi`, d gates) gives the rotation ANGLE, i.e. the amplitude error —
      the same measurement `RPEAmplitude` makes, needed here only to linearize the tilt;
    - the INTERLEAVED echo (`k_rpe_echo`, the block `Z90 X90 X90 Z90 Z90 X90 X90 Z90` repeated d
      times) cancels the rotation angle and amplifies the tilt of the rotation AXIS out of the
      drive plane, which is the drive phase relative to the virtual-Z frame.

    The recovered Z angle is the z-component of the X90's generator; `vz_correction` turns it into
    the shift to add to BOTH slots of `qubit/{q}/x90/vz` — the pair `Phase` calibrates, written
    the way `Phase` writes it (one value in both slots, qcal single_qubit.py:1081). RPE polishes
    it: run `Amplitude`, then `Phase`, then this.

    What this canNOT see is a uniform axis offset — the pulse's own `phase` on every X90 alike. It
    conjugates the whole sequence by an Rz, which a z-basis-in/z-basis-out experiment is blind to
    by construction; it is a frame convention, not an error. What tilts the axis measurably is the
    frame ADVANCE between gates being wrong for the phase the pulse actually leaves behind — an
    ac-Stark shift, or simply a mis-set vz pair.
    """

    #: The direct train's balanced closes as (quadrature, plus, minus) EXTRA gates — RPEAmplitude's.
    TRAINS = RPEAmplitude.TRAINS

    #: The interleaved block's balanced closes as (quadrature, plus, minus) trailing X90s. Two
    #: trailing X90s are an X180, which maps (x, y, z) -> (x, -y, -z) and so reads exactly the
    #: opposite z-basis outcome — the same "a close a further pi around" trick the direct train
    #: gets for free from its d+2 rung. Reading both is what keeps a fringe not centred on 1/2 out
    #: of the angle (see `RPEFrequency.QUADRATURES` for the measurement of that effect).
    TAILS = (("cos", 2, 0), ("sin", 3, 1))

    def __init__(self, cfg, qubits, depths=(1, 2, 4, 8), shots=256, gain=1.0, max_step=0.25):
        self.cfg, self.qubits = cfg, qubits_list(qubits)
        self.depths = tuple(int(d) for d in depths)
        self.shots, self.gain, self.max_step = int(shots), float(gain), float(max_step)
        if min(self.depths) < 1:
            raise ValueError(f"depths must be >= 1, got {self.depths}")
        if sorted(self.depths) != list(self.depths):
            raise ValueError(f"depths must be ascending, got {self.depths}")
        self.data, self.fit, self.angles, self.recovered_tilt = {}, {}, {}, {}

    def _period(self, m, q):
        """ONE grid period for BOTH ladders, sized for the longest sequence either of them runs.

        The interleaved rung spends four X90s per repetition, so it sets the length; the direct
        trains then sit on the same grid. Per-experiment (or per-rung) periods would give the
        shorter sequences a longer relax head and a different duty cycle, which biases the shallow
        rungs — the systematic `RPEFrequency._periods` measured.
        """
        cfg = self.cfg
        table, _, _ = prep(cfg, q, m, "X90")
        _, _, _, dur, ddly = readout_tables(cfg, q, m)
        d = table.pulses["x90"].dur_batches(m, table.channel)
        longest = 4 * max(self.depths) + max(t for _, plus, minus in self.TAILS
                                             for t in (plus, minus))
        seq = (longest - 1) * train_step(d) + d
        return grid_period(relax_batches(cfg, m), seq, dur, ddly, herald=heralding(cfg))

    def _echo(self, drv, m, depth, tail, periods):
        """Compile + run ONE interleaved rung (`depth` blocks + `tail` closing X90s) -> {q: P(|1>)}."""
        cfg, herald = self.cfg, heralding(self.cfg)
        progs, signs, timeout = {}, {}, 0
        for q in self.qubits:
            gate = ParamTable(0, qubit_freq(cfg, q), {"x90": gate_pulse(cfg, q, m)})
            ro, demod, code, dur, ddly = readout_tables(cfg, q, m)
            d = gate.pulses["x90"].dur_batches(m, gate.channel)
            step = train_step(d)                                  # the paced train grid (spec 14 F1)
            seq = (4 * depth + tail - 1) * step + d
            progs[q] = compile_kernel(
                kernels.k_rpe_echo, m, tables=dict(gate=gate, ro=ro, demod=demod),
                out=Array(2 if herald else 1), shots=self.shots, period=periods[q],
                depth=depth, tail=tail, step=step, code=code, ddly=ddly,
                hpi=pack16(units._phase_code(math.pi / 2)), herald=int(herald),
                hoff=herald_offset(seq, ddly) if herald else 0, **x90_vz(cfg, q))
            signs[q] = res_sign(cfg, q)
            timeout = max(timeout, batch_timeout(self.shots * periods[q]))
        P = sweep_counts(drv, m, progs, {q: {} for q in progs}, self.shots, timeout, signs,
                         herald=herald)
        return {q: float(np.atleast_1d(P[q])[0]) for q in self.qubits}

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg = self.cfg
        periods = {q: self._period(m, q) for q in self.qubits}
        direct = {q: {"cos": {}, "sin": {}} for q in self.qubits}
        echo = {q: {"cos": {}, "sin": {}} for q in self.qubits}
        trains = {}

        for depth in self.depths:
            for name, plus, minus in self.TRAINS:
                for off in (plus, minus):
                    if depth + off not in trains:
                        trains[depth + off] = _x90_train(drv, m, cfg, self.qubits, self.shots,
                                                         depth + off, periods)
                for q in self.qubits:
                    direct[q][name][depth] = (int(round(trains[depth + plus][q] * self.shots)),
                                              int(round(trains[depth + minus][q] * self.shots)))
            for name, plus, minus in self.TAILS:
                p_plus = self._echo(drv, m, depth, plus, periods)
                p_minus = self._echo(drv, m, depth, minus, periods)
                for q in self.qubits:
                    echo[q][name][depth] = (int(round(p_plus[q] * self.shots)),
                                            int(round(p_minus[q] * self.shots)))

        data, fit_out, proposal, oks = {}, {}, {}, {}
        for q in self.qubits:
            path = f"qubit/{q}/x90/vz"
            pair = [float(v) for v in cfg.get(path, [0.0, 0.0])]
            data[q] = {"depths": self.depths, "counts": direct[q], "echo": echo[q], "vz": pair}
            ok, prop = False, {}
            try:
                angles = x90_angles(direct[q]["cos"], direct[q]["sin"], echo[q]["cos"],
                                    echo[q]["sin"], self.shots, self.depths)
            except RPEBranchError as exc:
                data[q]["error"] = str(exc)
            else:
                self.angles[q] = fit_out[q] = angles
                data[q]["contrast"] = angles.contrast
                shift = vz_correction(angles.trusted["X"], angles.trusted["Z"])
                self.recovered_tilt[q] = shift
                # the measured frame ALREADY carries the config's pair, so the correction is a
                # shift; qcal writes one phase into both slots, which collapses an asymmetric
                # stored pair — keep its SUM (the only part a repeated gate feels) and symmetrize
                phi = damped_update(0.5 * (pair[0] + pair[1]), shift, gain=self.gain,
                                    max_step=self.max_step)
                prop = {path: [phi, phi]}
                ok = True
            oks[q] = ok
            proposal.update(prop)

        self.data, self.fit = data, fit_out
        return Result(all(oks.values()), data, fit_out, proposal, cfg, f"RPEPhase {self.qubits}")


class CZRPE:
    """CZ ZZ / ZI / IZ generator angles by RPE — qcal's `gate='CZ'` (spec 14 F5, finding 5).

    Three conditional-Ramsey ladders, one per state pair (`CZ_STATE_PAIRS`), each `k_cz_cond` at
    ngates = depth: (0, 1) / (2, 3) Ramsey the TARGET with the control prepped |0> / |1>; (3, 1)
    Ramseys the CONTROL with the target prepped |1> — the role swap `_cz_cond_progs(ramsey=...)`
    compiles, each line still playing its own CZ tone. `cz_angles` inverts the three per-CZ
    accumulated angles into the generator angles.

    Every rung reads BOTH quadratures at BOTH close signs (`quad` 0..3: ±Y90 / ±X90) — the
    balanced pairs of `RPEFrequency.QUADRATURES`, which divide the fringe centre out of the angle
    and make the Ramsey qubit's own marginal a valid pyRPE input. (The reference instead
    post-selects joint two-qubit bitstrings, which removes partner-prep errors but still assumes a
    centred fringe — the T1 offset F5 finding 1 measured goes straight into its angle. A partner
    prep error here only mixes in the other branch's fringe, a second-order bias for a polish
    step.) All depths share ONE grid, sized for the deepest rung (finding 2).

    The proposal SHIFTS the pair's local virtual-Z entries by the damped IZ / ZI errors: the
    measured frame already folds the config's entries per CZ, so the error is the residual shift
    to ADD (the reference notebook's `+= trusted_error` on `pulse/2`/`pulse/3`). The ZZ error —
    the conditional-phase residual — is REPORTED (`angles`, `data`), never written: its knobs
    (`CZ/freq`, the amps, the relative phase) need a measured slope, which is qcal's
    LinearResponse/optimizer stack, out of scope by spec 14 §4. Drive it from the notebook with
    `damped_update` and a hand-measured gain, or re-run `CZFrequency`/`CZAmplitude`.

    **The gap bias, and why this does not reproduce `LocalPhases`' write** (spec 14 §3 finding 9 —
    the two-qubit twin of finding 4's `t_pulse/(d·t_idle)` bias). `k_cz_cond` brackets its CZ train
    with the prep→train and train→close LEAD gaps and the prep X90; those enter a rung ONCE while
    the tone enters it `d` times. So the accumulated phase is `d·θ + φ_gap` where the RPE model
    wants `d·Θ`, and the per-CZ angle this reports is `θ + φ_gap/d` — with
    `φ_gap = 2π·δ·(2·LEAD + xd)` for a residual carrier detuning δ on the Ramsey qubit. `LocalPhases`
    measures at d = 1 and therefore ABSORBS `φ_gap` into the vz it writes; this class fits a slope
    and excludes it. The two are pinned to the same number only as δ → 0 or d → ∞ — so **do not
    compose them**: a tree that has just been through `LocalPhases` will read `IZ`/`ZI` here split by
    `2π·δ·gap·(1 − 1/d_last)`, per qubit, with δ's sign. Measured on the real kernel
    (`test_cz_rpe_ladder_drifts_with_the_lead_gap`): planting ±73.2 kHz on the two cores moved
    `IZ`/`ZI` 5.5 rad apart at depths (1, 2, 4) with the true local phases at 0, and both offsets
    vanished on resonance. The remedy is finding 4's: run `RPEFrequency` first and take the ladder
    DEEP. The gap is ~1.05× the tone on X6Y3 (419 ns vs a 400 ns CZ) but ~10× on the co-sim twin,
    which is why the twin exaggerates this by an order of magnitude.

    Run the CZ chain first (`CZFrequency` → `RelativePhase` → `CZAmplitude` → `LocalPhases`, the
    reference DAG's order): RPE polishes a working CZ, it does not find one.
    """

    #: The balanced closes per quadrature as (name, plus quad, minus quad) — `k_cz_cond` close
    #: codes 0..3 = Y90, X90, −Y90, −X90. On the Y90-prep fringe the kernel's P(|1>) counts read
    #: (1 − sin(Θ − φ_close))/2, so +Y90 is the cos-plus outcome and −X90 the sin-plus. The sign
    #: chain is pinned in co-sim against planted local phases of both signs, like every landed
    #: RPE class.
    QUADRATURES = (("cos", 0, 2), ("sin", 3, 1))

    #: State pair -> (Ramsey on the pair's control?, partner `prep` scalar).
    RUNGS = {(0, 1): (False, 0), (2, 3): (False, 1), (3, 1): (True, 1)}

    def __init__(self, cfg, pair, depths=(1, 2, 4), shots=128, gain=0.5, max_step=0.25):
        self.cfg, self.pair = cfg, (int(pair[0]), int(pair[1]))
        self.depths = tuple(int(d) for d in depths)
        self.shots, self.gain, self.max_step = int(shots), float(gain), float(max_step)
        if min(self.depths) < 1:
            raise ValueError(f"depths must be >= 1, got {self.depths}")
        if sorted(self.depths) != list(self.depths):
            raise ValueError(f"depths must be ascending, got {self.depths}")
        self.data, self.fit, self.angles = {}, {}, {}

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg, pair = self.cfg, self.pair
        ctrl, tgt = pair
        period = twoqubit._cz_cond_period(cfg, m, pair, max(self.depths))
        fcz = twoqubit._cz_freq_word(cfg, pair, m)               # a dead sweep pinned at CZ/freq
        counts = {sp: {"cos": {}, "sin": {}} for sp in CZ_STATE_PAIRS}

        for depth in self.depths:
            for ramsey_ctrl in (False, True):
                ramsey = ctrl if ramsey_ctrl else tgt
                other = tgt if ramsey_ctrl else ctrl
                progs, _, signs, timeout = twoqubit._cz_cond_progs(
                    cfg, m, pair, "freq", fcz, 0, 1, depth, self.shots,
                    ramsey=ramsey, period=period)
                rq.setup(drv, m, progs)
                for sp in CZ_STATE_PAIRS:
                    rc, prep_v = self.RUNGS[sp]
                    if rc != ramsey_ctrl:
                        continue
                    for name, plus, minus in self.QUADRATURES:
                        n = {}
                        for quad in (plus, minus):
                            par = {other: {"prep": prep_v}, ramsey: {"quad": quad}}
                            out = rq.rerun(drv, m, progs, params=par, results=["out"],
                                           timeout=timeout)
                            p1 = population(out[ramsey]["out"], self.shots, signs[ramsey])
                            n[quad] = int(round(float(np.atleast_1d(p1)[0]) * self.shots))
                        counts[sp][name][depth] = (n[plus], n[minus])

        pk = twoqubit.pair_key(pair)
        data = {pair: {"depths": self.depths, "counts": counts}}
        ok, prop, fit_out = False, {}, {}
        try:
            angles = cz_angles({sp: (counts[sp]["cos"], counts[sp]["sin"])
                                for sp in CZ_STATE_PAIRS}, self.shots, self.depths)
        except RPEBranchError as exc:
            data[pair]["error"] = str(exc)
        else:
            self.angles[pair] = fit_out[pair] = angles
            data[pair]["contrast"] = angles.contrast
            # the raw per-state-pair ladders the inversion mixed, so a composite angle that
            # disagrees with LocalPhases can be traced to the rung it came from (finding 9)
            data[pair]["ladders"] = angles.ladders
            err = angles.trusted_error
            data[pair]["zz_error"] = err["ZZ"]                    # reported, not written (docstring)
            prop = {f"two_qubit/{pk}/CZ/pulse": twoqubit._cz_local_set(
                cfg, pair,
                damped_update(twoqubit._local_phase(cfg, pair, ctrl), err["ZI"],
                              gain=self.gain, max_step=self.max_step),
                damped_update(twoqubit._local_phase(cfg, pair, tgt), err["IZ"],
                              gain=self.gain, max_step=self.max_step))}
            ok = True
        self.data, self.fit = data, fit_out
        return Result(ok, data, fit_out, prop, cfg, f"CZRPE {pair}")
