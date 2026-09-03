"""spec 14 F5 — the host half of the RPE gate: pyRPE analysis recovers planted angles.

Every test here synthesizes noiseless (or shot-noisy) counts from a *planted* angle using the
model RPE assumes — P_cos = (1 + cos(d·phi))/2, P_sin = (1 + sin(d·phi))/2 — pushes them through
the real `riscq.cal.rpe` estimators, and asserts the planted angle comes back. That covers the
branch selection, the consistency check, and each estimator's inversion algebra.

The physics that produces those counts is the kernels' job and is gated separately in co-sim.
"""

import math

import numpy as np
import pytest

from riscq import run as rq
from riscq.cal import Config, twoqubit
from riscq.cal.base import GATE_ENV, gate_sigma
from riscq.cal.rpe import (CZ_STATE_PAIRS, CZ_TARGETS, X90_TARGET, Angles, CZRPE, RPEBranchError,
                           RPEAmplitude, RPEFrequency, RPEPhase, cz_angles, damped_update,
                           freq_error_hz, idle_angles, vz_correction, wrap, x90_angles)
from riscq.map import pack16
from riscq.pulses import Pulse, units
from tests.probe import Probe

DEPTHS = (1, 2, 4, 8, 16, 32, 64)


def _counts(phi, depths=DEPTHS, shots=2048, rng=None):
    """Synthesize the (cos, sin) count pairs an ideal RPE experiment would return for `phi`."""
    cos, sin = {}, {}
    for d in depths:
        p_cos = (1.0 + np.cos(d * phi)) / 2.0
        p_sin = (1.0 + np.sin(d * phi)) / 2.0
        if rng is None:
            n_cos, n_sin = round(p_cos * shots), round(p_sin * shots)
        else:
            n_cos, n_sin = rng.binomial(shots, p_cos), rng.binomial(shots, p_sin)
        cos[d] = (n_cos, shots - n_cos)
        sin[d] = (n_sin, shots - n_sin)
    return cos, sin


# ── the idle / frequency estimator (qcal's gate='I') ─────────────────────────────────────────

@pytest.mark.parametrize("planted", [0.0, 0.013, -0.021, 0.4, -0.4])
def test_idle_recovers_the_planted_phase_per_step(planted):
    """The 'Z' angle is the phase accumulated per idle step, and is its own error (target 0)."""
    cos, sin = _counts(planted)
    got = idle_angles(cos, sin, shots := 2048, DEPTHS)
    assert got.trusted["Z"] == pytest.approx(planted, abs=1e-4)
    assert got.trusted_error["Z"] == got.trusted["Z"]
    assert got.n_shots == shots
    # the deepest generation is trusted on noiseless counts, and its uncertainty is the tightest
    assert got.last_good == len(DEPTHS) - 1
    assert got.last_good_depth == DEPTHS[-1]
    assert got.uncertainty == pytest.approx(np.pi / (2 * DEPTHS[-1] * np.sqrt(shots)))


def test_idle_beats_the_single_depth_resolution():
    """The point of the ladder: precision is set by the deepest rung, not by the shot count.

    At depth 1 a 3e-4 phase moves P by 1.5e-4 — far under 2048-shot noise. Amplified 64x it is
    resolvable, and the estimate lands within shot noise of the planted value.
    """
    planted = 3e-4
    cos, sin = _counts(planted, rng=np.random.default_rng(7))
    got = idle_angles(cos, sin, 2048, DEPTHS)
    depth_1_resolution = np.pi / (2 * DEPTHS[0] * np.sqrt(2048))
    assert got.uncertainty == pytest.approx(depth_1_resolution / DEPTHS[-1])
    assert abs(got.trusted["Z"] - planted) < 3 * got.uncertainty


def test_idle_angle_converts_to_a_detuning_in_hz():
    """A qubit detuned by df accumulates 2*pi*df*t over an idle of t, so df = theta/(2*pi*t)."""
    t_idle, detuning = 100e-9, 1.5e5
    planted = 2 * np.pi * detuning * t_idle
    cos, sin = _counts(planted)
    got = idle_angles(cos, sin, 2048, DEPTHS)
    assert freq_error_hz(got.trusted["Z"], t_idle) == pytest.approx(detuning, rel=1e-3)


def test_idle_unwraps_past_the_depth_1_window():
    """Branch selection is the whole trick: each generation resolves the 2*pi/d ambiguity of the
    next using the previous estimate, so a phase that wraps many times at depth 64 still lands."""
    planted = 1.1  # 64 * 1.1 = 70.4 rad — more than 11 full turns at the deepest rung
    cos, sin = _counts(planted)
    assert idle_angles(cos, sin, 2048, DEPTHS).trusted["Z"] == pytest.approx(planted, abs=1e-4)


def test_flat_signal_is_an_error_not_a_converged_zero():
    """A dead qubit must raise rather than read out as a perfectly calibrated one.

    P = 1/2 everywhere makes arctan2(0, 0) return 0 at every depth, which the consistency check
    happily accepts as a converged angle of exactly zero. Only the contrast floor catches it.
    """
    dead = {d: (1024, 1024) for d in DEPTHS}
    with pytest.raises(RPEBranchError, match="contrast"):
        idle_angles(dead, dead, 2048, DEPTHS)


def test_decohered_deep_rungs_are_not_trusted():
    """When the signal dies partway up the ladder the trusted generation must stop there.

    This is the case the consistency check does *not* cover: dead rungs return an arbitrary angle
    with a very narrow consistency window, and successive dead rungs readily agree with each
    other — so the ladder converges confidently on a wrong answer. The contrast floor is what
    bounds the ladder at the coherence time.
    """
    cos, sin = _counts(0.05)
    for d in (16, 32, 64):  # decohered: both quadratures sit at P = 1/2
        cos[d] = sin[d] = (1030, 2048 - 1030)
    got = idle_angles(cos, sin, 2048, DEPTHS)
    assert got.last_good_depth == 8
    assert got.trusted["Z"] == pytest.approx(0.05, abs=1e-3)
    assert got.contrast[-1] < 0.05 < got.contrast[0]


def test_mismatched_depth_ladders_are_rejected():
    cos, sin = _counts(0.1)
    with pytest.raises(ValueError, match="same depths"):
        idle_angles(cos, {d: v for d, v in sin.items() if d != 64}, 2048)


# ── the X90 estimator: amplitude (X angle) + drive phase (Z angle) ────────────────────────────

def _x90_counts(x_angle, z_angle, depths=DEPTHS, shots=2048):
    """Invert the linearized estimator to get the direct/interleaved angles that produce (X, Z).

    The direct experiment sees the rotation *magnitude*; the interleaved echo sees the axis tilt.
    """
    magnitude = np.hypot(x_angle, z_angle)
    tilt = np.arctan2(z_angle, x_angle)
    epsilon = magnitude / X90_TARGET - 1.0
    interleaved = 2.0 * np.arcsin(2.0 * tilt * np.cos(np.pi * epsilon / 2.0))
    return _counts(magnitude, depths, shots), _counts(interleaved, depths, shots)


@pytest.mark.parametrize("x_err,z_err", [(0.0, 0.0), (0.02, 0.0), (0.0, 0.03), (-0.05, 0.04)])
def test_x90_recovers_the_planted_rotation_and_axis(x_err, z_err):
    """X is the rotation angle (amplitude error), Z the axis tilt out of x-hat (phase error)."""
    x_angle, z_angle = X90_TARGET + x_err, z_err
    (dcos, dsin), (icos, isin) = _x90_counts(x_angle, z_angle)
    got = x90_angles(dcos, dsin, icos, isin, 2048, DEPTHS)
    assert got.trusted["X"] == pytest.approx(x_angle, abs=1e-3)
    assert got.trusted["Z"] == pytest.approx(z_angle, abs=1e-3)
    assert got.trusted_error["X"] == pytest.approx(x_err, abs=1e-3)
    assert got.trusted_error["Z"] == pytest.approx(z_err, abs=1e-3)


def test_x90_truncates_to_the_shallower_ladder():
    """The interleaved block spends four X90s per repetition, so its ladder is the shorter one;
    the recombination has to run on the common prefix rather than off the end of an array."""
    shallow = DEPTHS[:4]
    (dcos, dsin), _ = _x90_counts(X90_TARGET + 0.02, 0.01)
    _, (icos, isin) = _x90_counts(X90_TARGET + 0.02, 0.01, depths=shallow)
    got = x90_angles(dcos, dsin, icos, isin, 2048, DEPTHS)
    assert got.depths == shallow
    assert len(got.estimates["X"]) == len(shallow)
    assert got.trusted["X"] == pytest.approx(X90_TARGET + 0.02, abs=1e-3)


def _rot(x, z):
    """exp(-i(x·X + z·Z)/2) as a 2x2 — the gate RPE's (X, Z) angles describe."""
    omega = np.hypot(x, z)
    axis = np.array([[z, x], [x, -z]], dtype=complex) / (omega if omega else 1.0)
    return np.cos(omega / 2) * np.eye(2) - 1j * np.sin(omega / 2) * axis


@pytest.mark.parametrize("tilt", [0.0, 0.05, -0.05, 0.3])
def test_vz_correction_straightens_the_rotation_axis(tilt):
    """Adding the correction to BOTH virtual-Z slots must leave a pure x-rotation.

    The frame convention is that a virtual-Z of `a` inserts Rz(-a), so the corrected gate is
    Rz(-beta)·G·Rz(-beta). Note beta is NOT tilt/2: a z-error accrued during a quarter turn
    splits as (2/pi)·tilt per side.
    """
    beta = vz_correction(X90_TARGET, tilt)
    corrected = _rot(0.0, -beta) @ _rot(X90_TARGET, tilt) @ _rot(0.0, -beta)
    assert corrected[0, 0].imag == pytest.approx(0.0, abs=1e-12)   # no Z left in the generator
    assert corrected[0, 1].real == pytest.approx(0.0, abs=1e-12)   # and none in Y either
    assert beta == pytest.approx((2 / np.pi) * tilt, rel=0.05)


# ── the CZ estimator: ZZ / IZ / ZI from the three state pairs ─────────────────────────────────

def _cz_counts(zz, iz, zi, depths=DEPTHS, shots=2048):
    """Per-CZ accumulated angle for each state pair, in the CZ = expm(-i/2(...)) convention."""
    per_pair = {(0, 1): iz + zz, (2, 3): iz - zz, (3, 1): zi - zz}
    return {pair: _counts(per_pair[pair], depths, shots) for pair in CZ_STATE_PAIRS}


def test_cz_recovers_the_ideal_targets():
    """Sanity anchor: an ideal CZ = diag(1,1,1,-1) accumulates 0, pi, pi on the three state pairs
    and must invert back to exactly the (-pi/2, pi/2, pi/2) targets, i.e. zero error."""
    ideal = _cz_counts(zz=CZ_TARGETS["ZZ"], iz=CZ_TARGETS["IZ"], zi=CZ_TARGETS["ZI"])
    got = cz_angles(ideal, 2048, DEPTHS)
    for name, target in CZ_TARGETS.items():
        assert got.trusted[name] == pytest.approx(target, abs=1e-3)
        assert got.trusted_error[name] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("dzz,diz,dzi", [(0.05, 0.0, 0.0), (0.0, 0.03, 0.0), (0.0, 0.0, -0.04),
                                         (-0.06, 0.02, 0.05)])
def test_cz_recovers_planted_generator_errors(dzz, diz, dzi):
    """Each of ZZ / IZ / ZI must move independently — the inversion mixes all three state pairs,
    so a sign or factor slip shows up as crosstalk between the recovered errors."""
    planted = {"zz": CZ_TARGETS["ZZ"] + dzz, "iz": CZ_TARGETS["IZ"] + diz,
               "zi": CZ_TARGETS["ZI"] + dzi}
    got = cz_angles(_cz_counts(**planted), 2048, DEPTHS)
    assert got.trusted_error["ZZ"] == pytest.approx(dzz, abs=2e-3)
    assert got.trusted_error["IZ"] == pytest.approx(diz, abs=2e-3)
    assert got.trusted_error["ZI"] == pytest.approx(dzi, abs=2e-3)


def test_cz_missing_state_pair_is_rejected():
    counts = _cz_counts(zz=-np.pi / 2, iz=np.pi / 2, zi=np.pi / 2)
    del counts[(3, 1)]
    with pytest.raises(ValueError, match=r"state pair\(s\) \[\(3, 1\)\]"):
        cz_angles(counts, 2048, DEPTHS)


# ── the feedback rule (spec 14 §4: damped clip updates, no optimizer stack) ───────────────────

def test_damped_update_is_damped_and_clipped():
    assert damped_update(5.0, -1.0, gain=0.5) == pytest.approx(4.5)
    assert damped_update(5.0, -10.0, gain=1.0, max_step=0.5) == pytest.approx(4.5)
    assert damped_update(5.0, +10.0, gain=1.0, max_step=0.5) == pytest.approx(5.5)


def test_damped_update_multiplicative_for_amplitude():
    """Amplitude is linear in rotation angle, so its correction is a fractional one."""
    assert damped_update(0.1, -0.04, gain=1.0, multiplicative=True) == pytest.approx(0.096)


def test_wrap_is_the_references_rectify_angle():
    assert wrap(0.0) == pytest.approx(0.0)
    assert wrap(3 * np.pi) == pytest.approx(-np.pi)
    assert wrap(np.array([2 * np.pi + 0.1, -0.1])) == pytest.approx([0.1, -0.1])


def test_angles_dataclass_reports_the_trusted_generation():
    a = Angles(depths=(1, 2, 4), estimates={"Z": np.array([0.5, 0.4, 0.41])},
               errors={"Z": np.array([0.5, 0.4, 0.41])}, last_good=1, n_shots=100)
    assert a.trusted == {"Z": pytest.approx(0.4)}
    assert a.last_good_depth == 2
    assert a.uncertainty == pytest.approx(np.pi / (2 * 2 * 10.0))


# ── the CZRPE class end-to-end host-pure: real compiles, analytic fake driver ─────────────────

def _analytic_cz_driver(monkeypatch, a_phys, vz_of, shots):
    """Stub the driver layer with the analytic conditional Ramsey `k_cz_cond` implements.

    `_cz_cond_progs` still runs FOR REAL, through a recording wrapper — the quad-0..3 / role-swap
    / shared-period compile gate; only `setup`/`rerun` are faked. Each rerun returns the Ramsey
    core's P(|1>) = (1 − sin(Θ − φ_close))/2 at Θ = d·(`a_phys`[state pair] − `vz_of`[Ramsey core]),
    i.e. the PHYSICAL per-CZ angle minus that core's own config virtual-Z word — the kernel's frame
    convention (a frame word SUBTRACTS from the accrued angle), pinned on RTL by
    `test_cz_rpe_reads_planted_local_phases_through_the_real_kernel`.

    Returns (driver, the recorded compile state).
    """
    from riscq import run as rqrun
    from riscq.cal import twoqubit
    from tests.test_twoqubit import _SIM2Q

    phi_close = {0: np.pi / 2, 1: 0.0, 2: -np.pi / 2, 3: np.pi}   # quad 0..3 = ±Y90 / ±X90
    state = {"ngates": None, "ramseys": set(), "periods": set()}
    real_progs = twoqubit._cz_cond_progs

    def recording_progs(cfg_, m_, pair_, knob, x0, dx, points, ngates, shots_, **kw):
        state["ngates"] = ngates
        state["ramseys"].add(kw.get("ramsey"))
        state["periods"].add(kw.get("period"))
        return real_progs(cfg_, m_, pair_, knob, x0, dx, points, ngates, shots_, **kw)

    def fake_rerun(drv, m_, progs, params=None, results=None, timeout=0):
        ramsey = next(q for q, p in params.items() if "quad" in p)
        other = next(q for q, p in params.items() if "prep" in p)
        sp = (3, 1) if ramsey == 0 else ((2, 3) if params[other]["prep"] else (0, 1))
        theta = state["ngates"] * (a_phys[sp] - vz_of[ramsey])
        p1 = (1.0 - np.sin(theta - phi_close[params[ramsey]["quad"]])) / 2.0
        return {ramsey: {"out": np.array([round(p1 * shots)])},
                other: {"out": np.array([0])}}

    monkeypatch.setattr(twoqubit, "_cz_cond_progs", recording_progs)
    monkeypatch.setattr(rqrun, "setup", lambda *a, **k: None)
    monkeypatch.setattr(rqrun, "rerun", fake_rerun)

    class _Drv:                                       # socmap(drv) reads drv.sim.get_params()
        class sim:
            @staticmethod
            def get_params():
                return _SIM2Q.read_text()

    return _Drv(), state


@pytest.mark.parametrize("sign", [1, -1])
def test_cz_rpe_class_recovers_planted_generator_errors(monkeypatch, sign):
    """spec 14 F5 finding 5 — `CZRPE` end-to-end on the analytic fake driver, both signs.

    A_phys per state pair is composed from planted generator angles and the config carries planted
    (wrong) local vz entries. Pins the class's rung/close wiring, the balanced-pair count assembly,
    `cz_angles`' inversion, and the proposal arithmetic: at gain 1 the written local vz entries
    must land exactly on the physical deviations the plant hid.
    """
    from tests.test_twoqubit import _drive_cfg

    dzz, diz, dzi = 0.06 * sign, 0.12 * sign, -0.10 * sign     # planted generator errors
    zi_c, iz_c = -0.15 * sign, 0.20 * sign                     # planted (wrong) config vz entries
    thzz, thiz, thzi = -np.pi / 2 + dzz, np.pi / 2 + diz, np.pi / 2 + dzi
    a_phys = {(0, 1): thiz + thzz, (2, 3): thiz - thzz, (3, 1): thzi - thzz}
    shots = 4096

    cfg = _drive_cfg()
    cfg["two_qubit/(0, 1)/CZ/pulse"][2]["kwargs"]["phase"] = zi_c
    cfg["two_qubit/(0, 1)/CZ/pulse"][3]["kwargs"]["phase"] = iz_c

    drv, state = _analytic_cz_driver(monkeypatch, a_phys, {0: zi_c, 1: iz_c}, shots)
    cal = CZRPE(cfg, (0, 1), depths=(1, 2, 4, 8), shots=shots, gain=1.0, max_step=1.0)
    r = cal.run(drv)

    assert r.ok
    assert state["ramseys"] == {0, 1}, "both role assignments must compile (the (3, 1) swap)"
    assert len(state["periods"]) == 1 and None not in state["periods"], \
        "every depth must share ONE explicit grid period"
    a = cal.angles[(0, 1)]
    assert a.trusted["ZZ"] == pytest.approx(thzz, abs=5e-3)
    assert a.trusted["IZ"] == pytest.approx(thiz - iz_c, abs=5e-3)
    assert a.trusted["ZI"] == pytest.approx(thzi - zi_c, abs=5e-3)
    assert r.data[(0, 1)]["zz_error"] == pytest.approx(dzz, abs=5e-3)
    pl = r.proposal["two_qubit/(0, 1)/CZ/pulse"]
    assert pl[2]["kwargs"]["phase"] == pytest.approx(dzi, abs=5e-3)    # control's ZI entry
    assert pl[3]["kwargs"]["phase"] == pytest.approx(diz, abs=5e-3)    # target's IZ entry
    assert cfg["two_qubit/(0, 1)/CZ/pulse"][2]["kwargs"]["phase"] == zi_c   # original untouched


@pytest.mark.parametrize("theta_zz", [-np.pi / 2, -1.2])
@pytest.mark.parametrize("raw", [(0.4, -0.9), (-2.8, 2.6)])
def test_local_phases_write_pins_cz_rpe_to_pi_over_two(monkeypatch, theta_zz, raw):
    """spec 14 §3 finding 9 — the LocalPhases → CZRPE analysis chain, host-pure, end to end.

    The two cals are pinned to the same number by construction: at ngates = 1 `k_cz_local` and
    `k_cz_cond` emit the same circuit, and the φ `LocalPhases` sweeps occupies exactly the frame
    slot the `iz`/`zi` word fills. So this runs the whole chain on ONE plant — raw local phases
    θ_ZI/θ_IZ and a conditional phase θ_ZZ — and checks the identities that must hold afterwards:

        A(2,3) = A(3,1)   and   IZ = ZI = +π/2,   whatever θ_ZZ is.

    Both are ZZ-FREE, which is what makes them a defect detector rather than a quality metric:
    `_branch_correction` writes ψ₀ + δ/2 = θ_raw − π/2 (its δ = −2·θ_ZZ − π), so the effective
    local phase lands at exactly +π/2 for both qubits however badly the conditional phase is
    calibrated. θ_ZZ is therefore parameterized both at the ideal −π/2 and deliberately off it; the
    raw phases are parameterized once well inside (−π, π] and once straddling its wrap.

    The fringes are synthesized the way `test_twoqubit.test_local_phases_fringe_and_mean` does —
    `fringe(peak) = 0.5 + 0.4·cos(φ − peak)`, which already encodes the RTL-pinned sign (P peaks at
    φ = +ψ, the accrued angle) — and the CZ counts by the shared analytic driver. A converged
    `LocalPhases` must therefore leave `CZRPE` nothing to write.
    """
    from riscq.cal.twoqubit import _branch_correction, _cz_local_set, _fringe_peak, _phi_sweep
    from tests.test_twoqubit import _drive_cfg

    theta_zi_raw, theta_iz_raw = raw
    shots = 4096
    _, _, phi = _phi_sweep(24)           # the full-turn, endpoint-exclusive axis LocalPhases fits

    def fringe(peak):
        return 0.5 + 0.4 * np.cos(phi - peak)

    # LocalPhases, once per qubit: the active qubit accrues ψ_s = θ_raw ± θ_ZZ over the CZ (the
    # spectator's Z_s flips the ZZ term), `_fringe_peak` reads each branch peak back, and
    # `_branch_correction` combines them into the vz the class writes.
    vz = {}
    for q, theta_raw in ((0, theta_zi_raw), (1, theta_iz_raw)):
        off0, _ = _fringe_peak(phi, fringe(theta_raw + theta_zz))     # spectator |0>
        off1, _ = _fringe_peak(phi, fringe(theta_raw - theta_zz))     # spectator |1>
        vz[q] = _branch_correction(off0, off1)
        assert wrap(vz[q] - (theta_raw - np.pi / 2)) == pytest.approx(0.0, abs=2e-3), \
            "the branch combination is not ZZ-free"

    cfg = _drive_cfg()
    cfg["two_qubit/(0, 1)/CZ/pulse"] = _cz_local_set(cfg, (0, 1), vz[0], vz[1])   # the proposal

    # CZRPE against the SAME plant: each ladder's physical angle is the raw accrued phase; the
    # driver subtracts the config vz the write above just landed.
    a_phys = {(0, 1): theta_iz_raw + theta_zz, (2, 3): theta_iz_raw - theta_zz,
              (3, 1): theta_zi_raw - theta_zz}
    drv, _ = _analytic_cz_driver(monkeypatch, a_phys, vz, shots)
    cal = CZRPE(cfg, (0, 1), depths=(1, 2, 4, 8), shots=shots, gain=1.0, max_step=1.0)
    r = cal.run(drv)
    assert r.ok

    a = cal.angles[(0, 1)]
    ladders = r.data[(0, 1)]["ladders"]
    assert ladders[(3, 1)]["ladder"] == pytest.approx(ladders[(2, 3)]["ladder"], abs=5e-3), \
        "A(2,3) = A(3,1) is ZZ-free and must hold rung by rung on a converged tree"
    assert a.trusted["ZZ"] == pytest.approx(theta_zz, abs=5e-3)
    assert a.trusted["IZ"] == pytest.approx(np.pi / 2, abs=5e-3)
    assert a.trusted["ZI"] == pytest.approx(np.pi / 2, abs=5e-3)

    pl = r.proposal["two_qubit/(0, 1)/CZ/pulse"]           # nothing left to correct
    assert pl[2]["kwargs"]["phase"] == pytest.approx(vz[0], abs=5e-3)
    assert pl[3]["kwargs"]["phase"] == pytest.approx(vz[1], abs=5e-3)


# ── the co-sim half of the F5 gate: the rung CIRCUITS, on the converters ─────────────────────────
#
# The estimators above already recover a planted angle from planted counts, through the real
# `riscq.cal.rpe` inversion — branch selection, the consistency check, the contrast floor and each
# class's proposal arithmetic. What that cannot say is whether the class ASKS THE HARDWARE FOR THE
# RIGHT CIRCUIT, and that is a statement about emitted pulses: how long the idle a depth-d rung
# brackets really is, how many X90s a d-gate train really plays, and where each one's rotation axis
# sits. So the co-sim half is **L1** (specs/software-test-refactor/01 §3): model OFF, one shot, the
# gate DAC captured and read against the circuit each class documents.
#
# What this replaces: six end-to-end runs (`RPEFrequency`/`RPEAmplitude`/`RPEPhase` × two planted
# signs) that each drove a 4-rung ladder at 64–96 projective shots per quadrature to recover an
# angle the host-pure tests above already recover exactly. Their remaining content — that a planted
# detuning/amplitude/axis error comes back with the right SIGN — is the estimators' arithmetic (L0,
# above) composed with the physics that a detuned carrier ramps the drive axis and a bigger
# amplitude turns further (L2, test_cal.py::test_frequency_recovers_detuning and
# test_batch.py::test_counts_rabi). One end-to-end RPE ladder remains as an L3 anchor (02 §4).

from tests.test_cal import F_GE, _cfg, _s  # noqa: E402

# Armed before a `rq.rerun`, a capture pays for the core's boot + preamble (~2 000 batches) and then
# the shot's grid period; armed before a `rq.run` it pays for the image load too. `dac_capture_get`
# BLOCKS until the armed window is full, so these are sized, not generous (01 §7).
NCAP_RERUN = 2800
NCAP_RUN = 13000


def _windows(drv, handle):
    """[(absolute batch start, length, samples)] of every active window of a captured DAC."""
    t0, cap = drv.sim.dac_capture_get(handle)
    on = np.flatnonzero(np.abs(cap).sum(axis=1) > 0)
    if not on.size:
        return []
    runs = np.split(on, np.flatnonzero(np.diff(on) > 1) + 1)
    return [(int(t0 + r[0]), int(r[-1] - r[0] + 1), cap[r[0]:r[-1] + 1]) for r in runs]


def _axis(win, code):
    """A window's carrier phase in the ABSOLUTE-time frame of `code`.

    The NCO is time-referenced — a window is a slice of one free-running carrier — so the difference
    between two windows of the same carrier is exactly the difference of the `set_phase_offset`
    words that produced them, which is what an RPE circuit's 'axis' is."""
    start, _, samples = win
    x = np.asarray(samples, float).reshape(-1)
    n = start * 16 + np.arange(len(x))
    z = complex(np.sum(x * np.exp(-1j * np.pi * code * n / (1 << 15))))
    return math.atan2(z.imag, z.real)


def _relative_axes(wins, code):
    """Each window's axis relative to the first, wrapped into (−π, π]."""
    a0 = _axis(wins[0], code)
    return [float(wrap(_axis(w, code) - a0)) for w in wins]


def _gate_len(cfg, m, q=0):
    from riscq.cal.base import GATE_CH, gate_pulse
    return gate_pulse(cfg, q, m).dur_batches(m, GATE_CH)


@pytest.mark.cosim
@pytest.mark.parametrize("depth", [1, 3])
def test_rpe_frequency_rung_brackets_a_depth_scaled_idle(cosim, depth):
    """L1 (spec 14 F5) — `RPEFrequency`'s rung is a Ramsey whose idle is REPEATED `depth` times,
    and the two X90s that bracket it must actually land `depth · t_idle` apart on the gate DAC.

    That length is the whole experiment: the class converts the recovered angle to Hz by dividing
    by `t_idle` (`freq_error_hz`), so an idle that does not scale with the depth reports a detuning
    scaled by the same factor — and every rung would still be self-consistent, which is exactly the
    failure the estimators cannot see. It is also the systematic the class's own docstring bounds
    (the finite-pulse bias `t_pulse/(d·t_idle)`), stated here as a measurement rather than a model.

    The closing quadrature is the second knob: the class reads its two quadratures by rerunning one
    image with a different `p0`, a virtual-Z that must rotate the CLOSING X90's axis by exactly that
    angle and leave the opening one alone. Both come off the same single captured shot.

    Compiled and loaded through the class's own `_periods`/`_programs`, so the depth → wait mapping,
    the shared grid period and the herald fold are the production ones."""
    drv, m = cosim
    drv.sim.set_model({"kind": "zero"})
    cfg = _cfg(m, F_GE, relax=8)                     # L1: the relax head is not the subject
    idle = 32                                        # batches, ≫ the 4-batch gate
    cal = RPEFrequency(cfg, 0, t_idle=_s(idle, m), depths=(depth,), shots=1)
    progs, _, timeout = cal._programs(drv, m, depth, cal._periods(m))
    close = cal.QUADRATURES[1][1]                    # the class's own sin-PLUS close, −π/2
    d = _gate_len(cfg, m)

    h = drv.sim.dac_capture_arm(m.gate_dac(0), NCAP_RERUN)
    rq.rerun(drv, m, progs, params={0: {"p0": pack16(units._phase_code(close))}},
             results=["out"], timeout=timeout)
    wins = _windows(drv, h)
    axes = _relative_axes(wins, units._freq_code(F_GE, m.params)) if wins else []
    print(f"\n[rpe-freq d={depth}] windows={[(s, n) for s, n, _ in wins]} "
          f"gap={wins[1][0] - wins[0][0] if len(wins) > 1 else None} "
          f"want={depth * idle + d}  Δaxis={np.round(axes, 4).tolist()} want={close:+.4f}")
    assert len(wins) == 2, f"a Ramsey rung plays TWO X90s, the gate DAC shows {len(wins)}"
    assert [n for _, n, _ in wins] == [d, d], "the bracketing pulses are not the config's X90"
    assert wins[1][0] - wins[0][0] == depth * idle + d, \
        "the idle between the two X90s does not scale with the rung depth"
    assert axes[1] == pytest.approx(close, abs=0.02), \
        "the quadrature virtual-Z did not rotate the CLOSING X90's axis by the angle it asked for"


@pytest.mark.cosim
def test_rpe_amplitude_rung_is_a_paced_single_axis_train(cosim):
    """L1 (spec 14 F5) — `RPEAmplitude`'s rung is the DIRECT train: `depth + off` X90s, all on ONE
    axis, walking the paced `train_step` grid.

    Two properties, and the estimator depends on both. **The count**: the rung amplifies the
    amplitude error by exactly the number of gates played, so a train that dropped one (the depth-4
    param-queue trap the pacing exists to avoid — an unpaced train silently plays 4 and drops the
    rest, spec 14 F1) reports an angle short by that fraction at every depth, consistently. **The
    single axis**: the balanced closes `TRAINS` reads are the d+1/d+2/d+3 trains, which are the d
    train plus more of the SAME gate — two extra X90s are a π only if no frame advance crept in
    between. That is what distinguishes this rung from the interleaved echo's Z90 bracket, and it is
    the difference the recombination in `x90_angles` rests on.

    Run through the class's own `_period` + `_x90_train`, at the `depth + plus` of its cos pair."""
    from riscq.cal.base import train_step
    from riscq.cal.rpe import _x90_train

    drv, m = cosim
    drv.sim.set_model({"kind": "zero"})
    cfg = _cfg(m, F_GE, relax=8)
    cal = RPEAmplitude(cfg, 0, depths=(1, 2), shots=1)
    depth, (_, plus, _minus) = 2, cal.TRAINS[0]      # the cos pair: the d+2 train against the d one
    ngates = depth + plus
    d = _gate_len(cfg, m)
    step = train_step(d)

    h = drv.sim.dac_capture_arm(m.gate_dac(0), NCAP_RUN)
    _x90_train(drv, m, cfg, [0], cal.shots, ngates, {0: cal._period(m, 0)})
    wins = _windows(drv, h)
    axes = _relative_axes(wins, units._freq_code(F_GE, m.params)) if wins else []
    starts = [s for s, _, _ in wins]
    print(f"\n[rpe-amp n={ngates}] windows={[(s, n) for s, n, _ in wins]} "
          f"steps={np.diff(starts).tolist()} want={step}  Δaxis={np.round(axes, 4).tolist()}")
    assert len(wins) == ngates, \
        f"a depth-{depth} + {plus} rung must play {ngates} X90s, the gate DAC shows {len(wins)}"
    assert [n for _, n, _ in wins] == [d] * ngates, "a train gate is not the config's X90"
    assert np.all(np.diff(starts) == step), \
        f"the train does not walk the paced train_step grid: starts {starts}"
    assert np.allclose(axes, 0.0, atol=0.02), \
        f"the direct train is not single-axis — a frame advance crept between gates: {axes}"


@pytest.mark.cosim
def test_rpe_phase_echo_plays_the_z90_bracket(cosim):
    """L1 (spec 14 F5) — `RPEPhase`'s INTERLEAVED rung, the one circuit RPE needed a new kernel for.

    `k_rpe_echo` emits 2·depth halves of `Z90 · X90 · X90 · Z90` and then `tail` closing X90s. Every
    Z90 is a frame advance of π/2 and no pulse, so the whole circuit is visible on the DAC as
    4·depth + tail X90 windows whose AXES step by π per half-block:

        half-block b plays its two X90s at (2b + 1)·π/2, and the tail — after 2·depth halves have
        advanced the frame by 4·depth·(π/2) ≡ 0 — plays back at the axis it started from.

    That pattern is the experiment. It is what echoes the rotation ANGLE away (the two X90 pairs sit
    a full π apart in the frame, so an amplitude error cancels) while letting the AXIS tilt
    accumulate — i.e. the only reason the recovered Z angle means drive phase rather than amplitude.
    A missing Z90, a Z90 of the wrong sign, or a frame that does not return would each still produce
    a plausible ladder.

    Compiled and run through the class's own `_period` + `_echo`, so the depth/tail/step/hpi
    bindings are the production ones."""
    from riscq.cal.base import train_step

    drv, m = cosim
    drv.sim.set_model({"kind": "zero"})
    cfg = _cfg(m, F_GE, relax=8)
    cal = RPEPhase(cfg, 0, depths=(1,), shots=1)
    depth, (_, tail, _minus) = 1, cal.TAILS[0]       # the cos pair's PLUS tail: two closing X90s
    d = _gate_len(cfg, m)
    step = train_step(d)

    h = drv.sim.dac_capture_arm(m.gate_dac(0), NCAP_RUN)
    cal._echo(drv, m, depth, tail, {0: cal._period(m, 0)})
    wins = _windows(drv, h)
    axes = _relative_axes(wins, units._freq_code(F_GE, m.params)) if wins else []
    # the frame at play j, minus the frame at play 0 (= π/2), wrapped
    want = [float(wrap((2 * (j // 2) + 1) * np.pi / 2 - np.pi / 2)) for j in range(4 * depth)] + \
           [float(wrap(4 * depth * np.pi / 2 - np.pi / 2))] * tail
    starts = [s for s, _, _ in wins]
    print(f"\n[rpe-echo d={depth} tail={tail}] windows={len(wins)} want={4 * depth + tail}"
          f"\n  steps={np.diff(starts).tolist()} want={step}"
          f"\n  Δaxis={np.round(axes, 4).tolist()}\n  want ={np.round(want, 4).tolist()}")
    assert len(wins) == 4 * depth + tail, \
        f"a depth-{depth} echo + {tail} closes is {4 * depth + tail} X90s, the DAC shows {len(wins)}"
    assert [n for _, n, _ in wins] == [d] * len(wins), "an echo gate is not the config's X90"
    assert np.all(np.diff(starts) == step), \
        f"the echo does not walk the paced train_step grid: starts {starts}"
    assert [float(wrap(g - w)) for g, w in zip(axes, want)] == pytest.approx([0.0] * len(want),
                                                                            abs=0.02), \
        "the echo's X90 axes are not the Z90 bracket the block documents"


# ── CZRPE: the frame conventions on the real kernel ──────────────────────────────────────────────

# The two cores' GE carriers, distinct so the two readouts are frequency-multiplexed. BOTH are
# multiples of 2048 in DAC code (50 MHz = 2048, 100 MHz = 4096), which is what makes the L2 target
# below exact: `TwoLevelModel` takes the drive axis by demodulating the gate DAC over ONE batch's 16
# samples, and the counter-rotating 2ω term of that demod closes exactly when 2·code·16 is a whole
# number of turns — i.e. when the code is a multiple of 2048. At an off-lattice carrier (75 MHz =
# code 3072, measured) the recovered axis carries a residual that depends on the pulse's batch
# parity, and a two-X90 circuit lands up to 0.1 rad away from the textbook Bloch vector.
CZ_F = {0: F_GE, 1: 100e6}
CZ_CODE = {0: 2048, 1: 1024}       # the two cores' demod codes
# The CZ drive's length, batches. Sized, not arbitrary: `k_cz_cond` resumes at the end of the CZ
# train (`wait_until(t)`) and must then issue set_start + set_freq + the quadrature's
# set_phase_offset + the close's own set_start/fire before `t_close − LEAD`, i.e. within
# `ngates · czd` batches. Measured on this build at ngates = 1: czd = 20 is NOT enough — the
# `quad == 2` close, one branch deeper in the if/elif chain than `quad == 0/1` and one taken jump
# more than the `else`, misses its lead and DROPS, silently, leaving the Ramsey qubit unclosed.
# czd = 40 already clears it; 60 leaves margin. (X6Y3's CZ is 400 ns = 40 batches on this scaling,
# so the real gate is not near the edge — but a co-sim config that shortens it is.)
CZ_DUR = 60


def _cz_cfg(m, cz_amp, zi=0.0, iz=0.0):
    """A two-qubit-drive-form CZ Config on the 2-core sim-2q build, with each qubit's local virtual-Z
    planted. `cz_amp = 0` makes the CZ a pure FRAME operation — no conditional physics at all — which
    is what isolates the sign conventions; a real amp makes the tone visible on the gate DACs."""
    cfg = Config()
    for q in (0, 1):
        cfg[f"qubit/{q}/freq"] = CZ_F[q]
        cfg[f"qubit/{q}/x90/amp"] = 0.5
        cfg[f"readout/{q}/freq"] = float(units.demod_code_to_freq(CZ_CODE[q], m.params))
        cfg[f"readout/{q}/amp"] = 0.5
        cfg[f"readout/{q}/dur"] = _s(56, m)
        cfg[f"readout/{q}/demod/dur"] = _s(40, m)
    cfg["reset/relax"] = _s(8, m)
    cfg["two_qubit/(0, 1)/CZ/freq"] = 25e6
    cfg["two_qubit/(0, 1)/CZ/pulse"] = [
        {"channel": "Q0", "time": _s(CZ_DUR, m), "kwargs": {"amp": cz_amp, "phase": 0.0}, "env": "square"},
        {"channel": "Q1", "time": _s(CZ_DUR, m), "kwargs": {"amp": cz_amp, "phase": 0.0}, "env": "square"},
        {"channel": "Q0", "env": "virtualz", "kwargs": {"phase": zi}},
        {"channel": "Q1", "env": "virtualz", "kwargs": {"phase": iz}},
    ]
    return cfg


def _cz_close_bloch(vz, ngates, quad):
    """Where a `k_cz_cond` shot leaves the Ramsey qubit when the CZ carries NO physics.

    Y90 takes |0> to +x̂. The CZ train then advances that core's frame by its own local virtual-Z
    once per gate, so the closing X90's axis sits at ψ = ngates·vz ± π/2 (quad 0 closes at +π/2 —
    a Y90 — quad 2 at −π/2, its balanced partner). Rodrigues for a π/2 rotation of +x̂ about an
    xy-plane axis at ψ gives, exactly,

        b = (cos²ψ,  sin ψ · cos ψ,  −sin ψ)

    so the two quads read opposite ⟨σz⟩ (the balance) off the SAME equatorial vector, and it is the
    y-component — odd in vz — that carries the sign of the frame word."""
    psi = ngates * vz + (np.pi / 2 if quad == 0 else -np.pi / 2)
    return [math.cos(psi) ** 2, math.sin(psi) * math.cos(psi), -math.sin(psi)]


@pytest.mark.cosim
@pytest.mark.parametrize("ramsey", [1, 0])
def test_cz_rpe_reads_the_planted_local_phase_through_the_real_kernel(cosim, ramsey):
    """L2 (spec 14 F5 finding 5) — the CZRPE frame chain on the REAL kernel, both roles and both
    signs of the planted local phase.

    The CZ drive amplitude is ZERO and the two cores run independent two-level models, so the gate
    has no conditional physics whatever: every rung is pure frame arithmetic, and the Ramsey core's
    state must be exactly what its OWN planted config virtual-Z puts there. `ramsey = 1` is the
    (0, 1)/(2, 3) rungs — the target Ramseys, reading its `iz` entry, planted POSITIVE; `ramsey = 0`
    is the (3, 1) rung, the role swap onto the physical control, reading `zi`, planted NEGATIVE. So
    between the two parameters this pins: each core's own vz binding, the (3, 1) role swap (prep,
    close and readout all move), the balanced quad 0 / quad 2 closes, and the sign both ways.

    Read off `model_state()` rather than through counts: the accrued angle lives in the EQUATORIAL
    Bloch phase, and ⟨σz⟩ — all a readout can see — is even in it. The counts version had to run a
    two-rung ladder at 64 shots per quadrature per state pair, plus a `ReadoutCalibration` to make
    the `res` bit mean anything, and could then only place the angle to 0.15 rad; this places the
    whole Bloch vector to 0.02 off one shot per point.

    Only the Ramsey core's image is loaded. With the CZ amp at zero the partner contributes no
    physics at all — the old test's own expectation was that (0, 1) and (2, 3) read the SAME angle —
    and a second `k_cz_cond` image costs another ~13 k simulated batches, which puts the test over
    the budget for nothing. That the partner's `prep = 1` really reaches |1> is the plain two-X90 π,
    owned at L2 by test_cal.py::test_prep_gate_x90_and_x_agree."""
    zi_p, iz_p = -0.15, 0.20                       # planted local phases: negative on q0, positive on q1
    _, m = cosim
    ngates = 1
    cfg = _cz_cfg(m, cz_amp=0.0, zi=zi_p, iz=iz_p)
    vz = {0: zi_p, 1: iz_p}[ramsey]
    progs, _, _, _ = twoqubit._cz_cond_progs(
        cfg, m, (0, 1), "freq", twoqubit._cz_freq_word(cfg, (0, 1), m), 0, 1, ngates, 1,
        ramsey=ramsey)
    # the Rabi rate that makes amp 0.5 an exact X90 at this core's carrier — so the Y90 prep and the
    # close are textbook gates and the target below is the ideal one
    spec = {"kind": "twolevel", "core": ramsey, "f_ge": CZ_F[ramsey],
            "readout_code": CZ_CODE[ramsey], "readout_amp": 20000.0, "noise_scale": 0.0,
            "collapse": False,
            "rabi_rad_per_amp": float((np.pi / 2) / gate_sigma(
                m, Pulse(GATE_ENV, freq_hz=CZ_F[ramsey], amp=0.5), CZ_F[ramsey],
                units._amp_code(0.5)))}
    p = Probe(cosim, {ramsey: progs[ramsey]})

    for quad in (0, 2):                            # the balanced pair: +Y90 and −Y90 closes
        b = p.state(spec, {ramsey: {"quad": quad}})["bloch"]
        want = _cz_close_bloch(vz, ngates, quad)
        print(f"\n[cz-rpe ramsey=q{ramsey} vz={vz:+.2f} quad={quad}] "
              f"bloch={np.round(b, 4).tolist()} want={np.round(want, 4).tolist()}")
        assert b == pytest.approx(want, abs=0.02), \
            f"the Ramsey core landed at {b}, not the {vz:+.2f} frame word's {want}"


@pytest.mark.cosim
@pytest.mark.parametrize("ngates", [1, 2])
def test_cz_rpe_rung_carries_one_lead_gap_whatever_the_depth(cosim, ngates):
    """L1 (spec 14 §3 finding 9) — the LEAD-gap mechanism, measured where it lives: on the DAC.

    `k_cz_cond` brackets its CZ train with the prep→train and train→close retune gaps and the prep
    X90 itself. The class's whole caveat is that those enter a depth-`d` rung ONCE while the tone
    enters it `d` times, so a residual carrier detuning δ on the Ramsey qubit writes

        A_d = −2π·δ·(czd + gap/d),   gap = 2·LEAD + xd

    — an angle that DRIFTS with depth away from the depth-1 value `LocalPhases` pins, which is the
    candidate mechanism for `IZ != ZI` on a converged tree. That is a claim about the shot's
    geometry, and every term of it is directly observable: the Ramsey core's gate DAC plays the prep
    X90, then the `d`-long CZ train, then the close, and the two retune gaps between them are
    exactly LEAD. Parameterized over the depth, so `d·czd` scaling with the rung while `2·LEAD + xd`
    does not is asserted rather than inferred.

    The counts version instead planted ±73 kHz on the two cores and ran a three-rung CZRPE ladder at
    64 shots × 4 quadratures × 3 state pairs (plus a `ReadoutCalibration` to make `res` mean
    something) to watch the ladder drift — measuring the timing through the phase it produces. The
    other half of that inference, that a detuned carrier ramps the drive axis at 2π·δ per batch, is
    L2 in test_cal.py::test_frequency_recovers_detuning."""
    from riscq.cal.base import GATE_CH, batch_timeout, gate_pulse
    from riscq.map import LEAD

    drv, m = cosim
    drv.sim.set_model({"kind": "zero"})
    cfg = _cz_cfg(m, cz_amp=0.35)                  # a REAL CZ tone: the train is visible on the DAC
    ramsey, czd = 1, twoqubit._cz_dur_batches(cfg, (0, 1), m)
    xd = gate_pulse(cfg, 0, m).dur_batches(m, GATE_CH)
    progs, _, _, _ = twoqubit._cz_cond_progs(
        cfg, m, (0, 1), "freq", twoqubit._cz_freq_word(cfg, (0, 1), m), 0, 1, ngates, 1,
        ramsey=ramsey)
    period = twoqubit._cz_cond_period(cfg, m, (0, 1), ngates)
    progs = {ramsey: progs[ramsey]}                # the geometry is one core's; the partner's image
    rq.setup(drv, m, progs)                        # would cost another ~13 k batches for nothing

    h = drv.sim.dac_capture_arm(m.gate_dac(ramsey), NCAP_RERUN)
    rq.rerun(drv, m, progs, params={ramsey: {"quad": 0}}, results=["out"],
             timeout=batch_timeout(period))
    wins = _windows(drv, h)
    lens = [n for _, n, _ in wins]
    starts = [s for s, _, _ in wins]
    gaps = [starts[i + 1] - (starts[i] + lens[i]) for i in range(len(wins) - 1)] if wins else []
    print(f"\n[lead-gap d={ngates}] windows={list(zip(starts, lens))} gaps={gaps} want=[{LEAD}, {LEAD}]"
          f"\n  prep→close={starts[-1] - starts[0] if wins else None} "
          f"want={ngates * czd + 2 * LEAD + xd}  (czd={czd} xd={xd})")
    assert len(wins) == 3, \
        f"a drive-form rung plays prep, CZ train and close on the gate DAC; the DAC shows {lens}"
    assert lens == [xd, ngates * czd, xd], \
        f"the rung's windows are {lens}, not [X90, {ngates}·CZ, X90]"
    assert gaps == [LEAD, LEAD], \
        f"the f_GE↔f_CZ retune gaps are {gaps}, not one LEAD on each side of the train"
    assert starts[-1] - starts[0] == ngates * czd + 2 * LEAD + xd, \
        "the Ramsey interval is not d·czd + (2·LEAD + xd) — the gap term the RPE model omits"
