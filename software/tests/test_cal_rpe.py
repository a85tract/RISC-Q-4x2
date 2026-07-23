"""spec 14 F5 — the host half of the RPE gate: pyRPE analysis recovers planted angles.

Every test here synthesizes noiseless (or shot-noisy) counts from a *planted* angle using the
model RPE assumes — P_cos = (1 + cos(d·phi))/2, P_sin = (1 + sin(d·phi))/2 — pushes them through
the real `riscq.cal.rpe` estimators, and asserts the planted angle comes back. That covers the
branch selection, the consistency check, and each estimator's inversion algebra.

The physics that produces those counts is the kernels' job and is gated separately in co-sim.
"""

import numpy as np
import pytest

from riscq.cal import Config, ReadoutCalibration
from riscq.cal.base import GATE_ENV, batches, gate_sigma
from riscq.cal.rpe import (CZ_STATE_PAIRS, CZ_TARGETS, X90_TARGET, Angles, CZRPE, RPEBranchError,
                           RPEAmplitude, RPEFrequency, RPEPhase, cz_angles, damped_update,
                           freq_error_hz, idle_angles, vz_correction, wrap, x90_angles)
from riscq.pulses import Pulse, units

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

@pytest.mark.parametrize("sign", [1, -1])
def test_cz_rpe_class_recovers_planted_generator_errors(monkeypatch, sign):
    """spec 14 F5 finding 5 — `CZRPE` end-to-end on the analytic fake driver, both signs.

    The driver layer is stubbed (setup / rerun); `_cz_cond_progs` runs FOR REAL through a
    recording wrapper — the quad-0..3 / role-swap / shared-period compile gate. Each rerun
    returns the Ramsey core's P(|1>) = (1 − sin(Θ − φ_close))/2 at Θ = d·(A_phys − vz_cfg), with
    A_phys per state pair composed from planted generator angles — the physics `k_cz_cond`
    implements (its close at φ on the Y90-prep fringe). Pins the class's rung/close wiring, the
    balanced-pair count assembly, `cz_angles`' inversion, and the proposal arithmetic: at gain 1
    the written local vz entries must land exactly on the physical deviations the plant hid.
    """
    from riscq import run as rqrun
    from riscq.cal import twoqubit
    from tests.test_twoqubit import _SIM2Q, _drive_cfg

    dzz, diz, dzi = 0.06 * sign, 0.12 * sign, -0.10 * sign     # planted generator errors
    zi_c, iz_c = -0.15 * sign, 0.20 * sign                     # planted (wrong) config vz entries
    thzz, thiz, thzi = -np.pi / 2 + dzz, np.pi / 2 + diz, np.pi / 2 + dzi
    a_phys = {(0, 1): thiz + thzz, (2, 3): thiz - thzz, (3, 1): thzi - thzz}
    phi_close = {0: np.pi / 2, 1: 0.0, 2: -np.pi / 2, 3: np.pi}
    shots = 4096

    cfg = _drive_cfg()
    cfg["two_qubit/(0, 1)/CZ/pulse"][2]["kwargs"]["phase"] = zi_c
    cfg["two_qubit/(0, 1)/CZ/pulse"][3]["kwargs"]["phase"] = iz_c

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
        vz = zi_c if ramsey == 0 else iz_c                     # the Ramsey core's OWN config entry
        theta = state["ngates"] * (a_phys[sp] - vz)
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

    cal = CZRPE(cfg, (0, 1), depths=(1, 2, 4, 8), shots=shots, gain=1.0, max_step=1.0)
    r = cal.run(_Drv())

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


# ── the co-sim half of the F5 gate: a planted detuning recovered at light depths ──────────────

# The co-sim tests reuse test_cal's harness: its planted 2-level model, its physical-units Config,
# and its session-scoped `demod_phase` (the discrimination phase every counts-mode cal bakes in —
# measured once per session, so importing the fixture here shares that one measurement).
from tests.test_cal import F_GE, _cfg, _model, _s, _true_x90_amp  # noqa: E402
from tests.test_cal import demod_phase  # noqa: E402,F401  (a fixture: used by injection)


@pytest.mark.cosim
@pytest.mark.parametrize("d0_code", [24, -24])
def test_rpe_frequency_recovers_a_planted_detuning(cosim, demod_phase, d0_code):
    """spec 14 F5 co-sim gate — RPEFrequency on the 2-level model, both signs.

    The config carrier is planted off the model's f_ge by delta, so each idle step writes
    2*pi*delta*t_idle of phase and the ladder amplifies it. As with `Frequency`, the sign is the
    part a magnitude-only check cannot catch: get the quadrature convention or the update
    arithmetic backwards and the "correction" doubles the error for one sign while looking
    perfect for the other. So plant it both ways and require the carrier to move toward f_ge.

    The planted detuning is deliberately SMALL. RPE is the polish step: it assumes the gate is
    already a good X90, and a carrier far enough off to detune the drive during the pulse breaks
    that assumption. Measured here at a 1.46 MHz plant (0.12 cycles across the 80 ns pulse, so a
    42-degree axis tilt): contrast collapsed to 0.42 at depth 1 and the shallow rungs were biased
    by ~0.9 rad, which dragged the branch selection. 0.59 MHz (17 degrees) is comfortable.
    """
    drv, m = cosim
    rabi = float((np.pi / 2) / gate_sigma(m, Pulse(GATE_ENV, freq_hz=F_GE, amp=0.5),
                                          F_GE, units._amp_code(0.5)))
    delta = units.code_to_freq(d0_code, m.params)          # the carrier error we are planting
    drive = units.code_to_freq(units._freq_code(F_GE, m.params) + d0_code, m.params)
    drv.sim.set_model(_model(rabi, t1=400, t2=3000, noise=300.0, seed=3, collapse=True))
    cfg = _cfg(m, drive, x90_amp=0.5, relax=800)
    cfg["readout/0/demod/phase"] = demod_phase

    # size the idle so the depth-1 rung sees ~0.5 rad — comfortably inside the +-pi it can resolve
    t_idle = _s(batches(0.5 / (2 * np.pi * abs(delta)), m), m)
    cal = RPEFrequency(cfg, 0, t_idle=t_idle, depths=(1, 2, 4, 8), shots=96)
    r = cal.run(drv)

    a = cal.angles[0]
    print(f"\n[rpe-freq d0={d0_code:+d}] t_idle={t_idle * 1e9:.1f} ns  planted delta={delta:+.4g} Hz"
          f"\n  Pcos={[round(r.data[0]['counts']['cos'][d][0] / cal.shots, 3) for d in cal.depths]}"
          f"  Psin={[round(r.data[0]['counts']['sin'][d][0] / cal.shots, 3) for d in cal.depths]}"
          f"\n  angles={np.round(a.estimates['Z'], 4).tolist()}"
          f"  contrast={np.round(a.contrast, 3).tolist()}"
          f"\n  last good depth={a.last_good_depth}  recovered={cal.recovered_detuning[0]:+.4g} Hz")
    assert r.ok
    assert a.contrast[0] > 0.6, "the depth-1 rung has no contrast — the X90 is not a good gate here"
    # the ladder reports the CARRIER's error (Frequency's delta), so it is positive when the
    # carrier sits above the qubit — this is what pins the quadrature convention
    assert np.sign(a.estimates["Z"][0]) == np.sign(delta), "quadrature convention inverted"
    assert abs(cal.recovered_detuning[0]) == pytest.approx(abs(delta), rel=0.15), \
        "recovered detuning magnitude wrong"
    r.apply()
    before, after = abs(drive - F_GE), abs(cfg["qubit/0/freq"] - F_GE)
    print(f"[rpe-freq d0={d0_code:+d}] |freq-f_ge| before={before:.4g} after={after:.4g}")
    assert after < 0.2 * before, "config frequency did not move toward f_ge"


@pytest.mark.cosim
@pytest.mark.parametrize("err", [0.06, -0.06])
def test_rpe_amplitude_recovers_a_planted_amplitude_error(cosim, demod_phase, err):
    """spec 14 F5 — RPEAmplitude on the 2-level model, the error planted both ways.

    The config's X90 amplitude is planted off the one that actually rotates by pi/2, so the gate
    over- or under-rotates by `err` and the repeated train amplifies it. The correction is a
    ratio, so an inverted sign multiplies the error instead of dividing it out — plant both ways.
    """
    drv, m = cosim
    rabi = float((np.pi / 2) / gate_sigma(m, Pulse(GATE_ENV, freq_hz=F_GE, amp=0.5),
                                          F_GE, units._amp_code(0.5)))
    drv.sim.set_model(_model(rabi, t1=400, t2=3000, noise=300.0, seed=3, collapse=True))
    true_amp = _true_x90_amp(m, rabi)
    cfg = _cfg(m, F_GE, x90_amp=true_amp * (1 + err), relax=800)
    cfg["readout/0/demod/phase"] = demod_phase

    cal = RPEAmplitude(cfg, 0, depths=(1, 2, 4, 8), shots=96)
    r = cal.run(drv)

    a = cal.angles[0]
    got = r.proposal["qubit/0/x90/amp"]
    print(f"\n[rpe-amp err={err:+.3f}] true_amp={true_amp:.5f} planted={true_amp * (1 + err):.5f}"
          f"\n  angles={np.round(a.estimates['X'], 4).tolist()} (target {X90_TARGET:.4f})"
          f"  contrast={np.round(a.contrast, 3).tolist()}"
          f"\n  last good depth={a.last_good_depth}  recovered angle={cal.recovered_angle[0]:.4f}"
          f"  -> amp={got:.5f}")
    assert r.ok
    assert a.contrast[0] > 0.6, "the depth-1 rung has no contrast"
    # the rotation angle must come back on the correct side of pi/2
    assert np.sign(cal.recovered_angle[0] - X90_TARGET) == np.sign(err), "amplitude error inverted"
    assert got == pytest.approx(true_amp, rel=0.03), "amplitude not corrected onto the true value"


@pytest.mark.cosim
@pytest.mark.parametrize("planted", [0.2, -0.2])
def test_rpe_phase_recovers_a_planted_axis_error(cosim, demod_phase, planted):
    """spec 14 F5 — RPEPhase on the 2-level model, the axis error planted both ways.

    The plant is a WRONG virtual-Z pair. `qubit/0/x90/vz` = [v, v] makes every kernel advance the
    gate frame by 2v across each X90, so the gate actually played is Rz(-v)·Rx(pi/2)·Rz(-v): a
    rotation about an axis tilted out of the drive plane by v. That is the only kind of axis error
    this experiment can see — a uniform pulse `phase` conjugates the whole sequence by an Rz, which
    a z-in/z-out measurement is blind to by construction (it is a frame convention, not an error).
    So the interleaved echo must recover a correction of -v and propose the pair back onto ~0.

    The sign is the part a magnitude check cannot catch: get the interleaved close assignment
    backwards and the "correction" doubles the tilt for one sign while looking perfect for the
    other. The depth-4 rung spends 19 X90s on the paced train grid, so T1 has to cover ~650
    batches of sequence for the ladder to stay alive that far.
    """
    drv, m = cosim
    rabi = float((np.pi / 2) / gate_sigma(m, Pulse(GATE_ENV, freq_hz=F_GE, amp=0.5),
                                          F_GE, units._amp_code(0.5)))
    drv.sim.set_model(_model(rabi, t1=1000, t2=3000, noise=300.0, seed=3, collapse=True))
    cfg = _cfg(m, F_GE, x90_amp=_true_x90_amp(m, rabi), relax=2500)
    cfg["readout/0/demod/phase"] = demod_phase
    cfg["qubit/0/x90/vz"] = [planted, planted]        # the axis error, played by every kernel

    cal = RPEPhase(cfg, 0, depths=(1, 2, 4), shots=64)
    r = cal.run(drv)

    a = cal.angles[0]
    vz = r.proposal["qubit/0/x90/vz"]
    echo = {name: [tuple(round(n / cal.shots, 3) for n in r.data[0]["echo"][name][d])
                   for d in cal.depths] for name in ("cos", "sin")}
    print(f"\n[rpe-phase v={planted:+.3f}] echo P+/P- cos={echo['cos']}  sin={echo['sin']}"
          f"\n  X ladder={np.round(a.estimates['X'], 4).tolist()} (target {X90_TARGET:.4f})"
          f"\n  Z ladder={np.round(a.estimates['Z'], 4).tolist()}"
          f"  contrast={np.round(a.contrast, 3).tolist()}"
          f"\n  last good depth={a.last_good_depth}  correction={cal.recovered_tilt[0]:+.4f}"
          f"  -> vz={vz[0]:+.4f}")
    assert r.ok
    assert a.contrast[0] > 0.6, "the depth-1 rung has no contrast — the X90 is not a good gate here"
    assert vz[0] == vz[1], "qcal writes ONE phase into BOTH virtual-Z slots"
    # the correction must UNDO the plant: right magnitude and the opposite sign
    assert np.sign(cal.recovered_tilt[0]) == -np.sign(planted), "axis convention inverted"
    assert cal.recovered_tilt[0] == pytest.approx(-planted, abs=0.08), "tilt magnitude wrong"
    assert abs(vz[0]) < 0.3 * abs(planted), "the virtual-Z pair did not move back onto zero"


@pytest.mark.cosim
@pytest.mark.parametrize("planted", [(0.2, -0.15), (-0.2, 0.15)])
def test_cz_rpe_reads_planted_local_phases_through_the_real_kernel(cosim, planted):
    """spec 14 F5 finding 5 — the CZRPE sign chain on the REAL kernel, both signs.

    The CZ drive amp is ZERO and the two cores run independent 2-level models, so the gate has NO
    conditional physics: every rung reads pure frame arithmetic, and each state pair's ladder must
    return the Ramsey core's own planted config vz, negated — A(0,1) = A(2,3) = −iz, A(3,1) = −zi.
    That pins, on real RTL: the quad 2/3 balanced closes (the −Y90/−X90 phase words) and their
    plus/minus pairing, the per-core vz binds, and the (3, 1) role swap (prep, close AND readout
    on the physical control) — every convention the analytic fake cannot arbitrate. The T1-shifted
    fringe centre (measured ~0.36 here) must divide out of the angles — F5 finding 1's remedy on
    the real readout chain.

    Asserted on the per-pair WRAPPED ladders, not the class composite: `cz_angles` deliberately
    does not wrap the (2, 3)/(3, 1) ladders (they sit near π for any real CZ), and a zero-amp CZ
    parks them at ~0 ≡ 2π, exactly on that convention's cut — a test artifact, not a class bug.
    The composite inversion + proposal arithmetic (and the conditional-π physics, via the
    test_models conditionality R) are host-gated — the Q4 precedent: co-sim pins conventions, the
    model pins physics.
    """
    iz_p, zi_p = planted
    drv, m = cosim
    code = {0: 2048, 1: 1024}                        # distinct demod codes → freq-multiplexed
    f = {0: F_GE, 1: 75e6}
    # per-core Rabi rate such that amp 0.5 is an exact X90 at that core's carrier
    rabi = {q: float((np.pi / 2) / gate_sigma(m, Pulse(GATE_ENV, freq_hz=f[q], amp=0.5), f[q],
                                              units._amp_code(0.5))) for q in (0, 1)}
    cfg = Config()
    for q in (0, 1):
        cfg[f"qubit/{q}/freq"] = f[q]
        cfg[f"qubit/{q}/x90/amp"] = 0.5
        cfg[f"readout/{q}/freq"] = float(units.demod_code_to_freq(code[q], m.params))
        cfg[f"readout/{q}/amp"] = 0.5
        cfg[f"readout/{q}/dur"] = _s(56, m)
        cfg[f"readout/{q}/demod/dur"] = _s(40, m)
    cfg["reset/relax"] = _s(800, m)
    cfg["two_qubit/(0, 1)/CZ/freq"] = 25e6
    cfg["two_qubit/(0, 1)/CZ/pulse"] = [
        {"channel": "Q0", "time": _s(20, m), "kwargs": {"amp": 0.0, "phase": 0.0}, "env": "square"},
        {"channel": "Q1", "time": _s(20, m), "kwargs": {"amp": 0.0, "phase": 0.0}, "env": "square"},
        {"channel": "Q0", "env": "virtualz", "kwargs": {"phase": zi_p}},
        {"channel": "Q1", "env": "virtualz", "kwargs": {"phase": iz_p}},
    ]
    # the two readout tones SUM on the shared ADC (amps halved), each core's demod its own
    sub = [{"kind": "twolevel", "core": q, "rabi_rad_per_amp": rabi[q], "readout_code": code[q],
            "readout_amp": 14000.0, "f_ge": f[q], "t1": 400, "t2": 3000, "noise_scale": 300.0,
            "noise_seed": 5 + q, "collapse": True} for q in (0, 1)]
    # bake each core's demod phase + res sign on the readout cals' t1/relax budget (relax ≫ T1 ≫
    # SEP, test_cal's RO_T1/RO_RELAX rationale) — the phase is a readout-chain property, not the
    # dynamics' — then restore the tighter RPE grid
    drv.sim.set_model({"kind": "multi", "models": [dict(s, t1=600) for s in sub]})
    cfg["reset/relax"] = _s(3200, m)
    rc = ReadoutCalibration(cfg, [0, 1], shots=24).run(drv)
    assert rc.ok, "readout clusters did not separate on the two-core model"
    rc.apply()
    cfg["reset/relax"] = _s(800, m)
    drv.sim.set_model({"kind": "multi", "models": sub})

    cal = CZRPE(cfg, (0, 1), depths=(1, 2), shots=64)
    r = cal.run(drv)
    assert r.ok

    from riscq.cal.rpe import _ladder
    counts = r.data[(0, 1)]["counts"]
    expected = {(0, 1): -iz_p, (2, 3): -iz_p, (3, 1): -zi_p}
    for sp, want in expected.items():
        ladder, k, contrast = _ladder(counts[sp]["cos"], counts[sp]["sin"])
        got = float(wrap(ladder)[k])
        print(f"\n[cz-rpe iz={iz_p:+.2f} zi={zi_p:+.2f}] pair {sp}: "
              f"ladder={np.round(wrap(ladder), 4).tolist()} trusted={got:+.4f} "
              f"(want {want:+.4f})  contrast={np.round(contrast, 3).tolist()}")
        assert contrast[0] > 0.5, f"state pair {sp}: no depth-1 contrast"
        assert got == pytest.approx(want, abs=0.15), f"state pair {sp}: sign chain wrong"
