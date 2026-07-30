"""Two-qubit CZ calibration — Q0 host-pure tests (specs/two-qubit/01): the config-frequency seed,
the `two_qubit` schema round-trip, the (i, j) key convention, the coupler-core role lookup, and the
joint-readout shot-index zip. The cross-core alignment on the real 3-core co-sim is test_twoqubit_cosim."""

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from riscq.cal import (JAZZ, ClassifierN, Config, CZAmpFreqSweep, CZAmplitude, CZFrequency, CZSweep,
                       EFAmplitude, EFPhase, LocalPhases, RelativePhase, calc_cz_frequency,
                       coupler_core, cz_coupler_form, cz_drive_table, cz_sandwich, cz_table,
                       joint_populations, pair_key)
from riscq.cal.base import GATE_CH, _levels_pop, ef_pulse, ef_table
from riscq.cal.twoqubit import (_branch_correction, _cz_amp, _cz_cond_progs, _cz_drive_indices,
                                _cz_entry, _cz_local_set, _cz_pulse_set, _cz_rel_phase_set,
                                _cz_vz_entry, _fringe_peak, _local_phase_code, _mean_offset,
                                _sandwich_binds, _signed_fft_freq)
from riscq.map import LEAD, SocMap, SocParams, pack16
from tests.responder import counts
from riscq.pulses import envelopes, units

_SIM2Q = Path(__file__).resolve().parents[1] / "configs" / "sim-2q.json"
_SIM2Q1C = Path(__file__).resolve().parents[1] / "configs" / "sim-2q1c.json"
_X6Y3 = Path(__file__).resolve().parents[2] / "examples" / "cal-config-x6y3.yaml"


def test_pair_key_matches_qcal_tuple_repr():
    assert pair_key((0, 1)) == "(0, 1)"
    assert pair_key((3, 12)) == "(3, 12)"
    assert pair_key((0, 1)) == str((0, 1))          # interchangeable with qcal's str(qp) key


def test_calc_cz_frequency_02_and_20():
    """qcal calculate_parametric_cz_frequency (cz.py:37-77): CZ/freq = |f_state − f_11|, with
    f_11 = f_GE(i)+f_GE(j), f_02 = f_GE(j)+f_EF(j), f_20 = f_GE(i)+f_EF(i)."""
    cfg = Config()
    cfg["qubit/0/freq"] = 5.0e9
    cfg["qubit/1/freq"] = 5.2e9
    cfg["qubit/0/EF/freq"] = 4.7e9      # f_GE + anharmonicity (−300 MHz)
    cfg["qubit/1/EF/freq"] = 4.9e9

    calc_cz_frequency(cfg, [(0, 1)], state="02")
    # f_11 = 10.2e9; f_02 = 5.2e9 + 4.9e9 = 10.1e9; |10.1 − 10.2| = 100 MHz
    assert cfg["two_qubit/(0, 1)/CZ/freq"] == pytest.approx(100e6)

    calc_cz_frequency(cfg, [(0, 1)], state="20")
    # f_20 = 5.0e9 + 4.7e9 = 9.7e9; |9.7 − 10.2| = 500 MHz
    assert cfg["two_qubit/(0, 1)/CZ/freq"] == pytest.approx(500e6)


def test_calc_cz_frequency_multiple_pairs_and_guard():
    cfg = Config()
    for q, f in ((0, 5.0e9), (1, 5.2e9), (2, 5.4e9)):
        cfg[f"qubit/{q}/freq"] = f
        cfg[f"qubit/{q}/EF/freq"] = f - 0.3e9
    calc_cz_frequency(cfg, [(0, 1), (1, 2)], state="02")
    assert cfg["two_qubit/(0, 1)/CZ/freq"] == pytest.approx(abs((5.2e9 + 4.9e9) - (5.0e9 + 5.2e9)))
    assert cfg["two_qubit/(1, 2)/CZ/freq"] == pytest.approx(abs((5.4e9 + 5.1e9) - (5.2e9 + 5.4e9)))
    with pytest.raises(AssertionError):
        calc_cz_frequency(cfg, [(0, 1)], state="11")


def test_two_qubit_schema_round_trips_through_yaml(tmp_path):
    """The coupler-drive CZ layout (spec 01 §2): the (i, j) key, the pulse list, the coupler core, and
    the CZ freq survive a YAML save/load — a plain slash-path tree, no special loader."""
    cfg = Config()
    cfg["two_qubit/(0, 1)/core"] = 2
    cfg["two_qubit/(0, 1)/CZ/freq"] = 213.0e6
    cfg["two_qubit/(0, 1)/CZ/pulse"] = [
        {"channel": "C0_1", "time": 200.0e-9, "kwargs": {"amp": 0.35, "phase": 0.0},
         "env": {"env_func": "square", "ramp_fraction": 0.1}},
        {"channel": "Q0", "env": "virtualz", "kwargs": {"phase": 0.0}},
        {"channel": "Q1", "env": "virtualz", "kwargs": {"phase": 0.0}},
    ]
    cfg["two_qubit/(0, 1)/ZZ11"] = 0.0

    path = tmp_path / "twoq.yaml"
    cfg.save(path)
    back = Config.load(path)
    assert back["two_qubit/(0, 1)/CZ/freq"] == pytest.approx(213.0e6)
    assert coupler_core(back, (0, 1)) == 2
    assert back["two_qubit/(0, 1)/CZ/pulse"][0]["channel"] == "C0_1"
    assert back["two_qubit/(0, 1)/CZ/pulse"][1]["env"] == "virtualz"
    assert back["two_qubit/(0, 1)/ZZ11"] == 0.0
    assert back.to_dict() == cfg.to_dict()


def test_joint_populations_zips_by_shot_index():
    """The two-qubit readout zip (spec 01 §5): shot k on control and shot k on target are the same
    repetition. A deterministic per-shot pattern must land in the right joint bin."""
    control = np.array([0, 0, 1, 1, 1, 0])
    target = np.array([0, 1, 0, 1, 1, 0])
    p = joint_populations({0: control, 1: target}, order=(0, 1))
    # bins: 00 (shots 0,5)=2, 01 (shot 1)=1, 10 (shot 2)=1, 11 (shots 3,4)=2  →  /6
    assert np.allclose(p, np.array([2, 1, 1, 2]) / 6)
    assert p.sum() == pytest.approx(1.0)
    # order matters: swapping control/target swaps the 01 and 10 bins
    p_swapped = joint_populations({0: control, 1: target}, order=(1, 0))
    assert np.allclose(p_swapped, np.array([2, 1, 1, 2]) / 6)   # symmetric counts here, but...
    ctrl2, tgt2 = np.array([1, 0]), np.array([0, 0])
    # control=[1,0], target=[0,0]: shot0 → 10, shot1 → 00  ⇒  [P00, P01, P10, P11] = [.5, 0, .5, 0]
    assert np.allclose(joint_populations({0: ctrl2, 1: tgt2}, (0, 1)), [0.5, 0, 0.5, 0])
    # order (1, 0): control=[0,0], target=[1,0]: shot0 → 01, shot1 → 00  ⇒  [.5, .5, 0, 0]
    assert np.allclose(joint_populations({0: ctrl2, 1: tgt2}, (1, 0)), [0.5, 0.5, 0, 0])


def test_joint_populations_length_mismatch_is_loud():
    """A desynced pair of shot streams is an alignment failure, not a silent truncation."""
    with pytest.raises(ValueError, match="desynced"):
        joint_populations({0: np.zeros(10), 1: np.zeros(9)}, order=(0, 1))


# ── EF subspace (spec two-qubit/01 §4.1): host-pure ──

def test_ef_table_is_baseband_with_both_carriers():
    """ef_table puts the GE prep X90 and the EF X90 in ONE channel-0 table, both BASEBAND (freq_hz
    None) so the kernel drives each segment at its own runtime carrier, and returns the SEATED GE/EF
    carrier words (spec 01 §4.1 — one NCO per channel, retuned between segments)."""
    m = SocMap(SocParams.load(_SIM2Q1C))
    cfg = Config()
    cfg["qubit/0/freq"] = 150e6
    cfg["qubit/0/x90/amp"] = 0.5
    cfg["qubit/0/EF/freq"] = 50e6
    cfg["qubit/0/EF/x90/amp"] = 0.4
    table, ge_freq, ef_freq = ef_table(cfg, 0, m)
    assert list(table.pulses) == ["x90", "ef"]
    assert table.pulses["x90"].freq_hz is None and table.pulses["ef"].freq_hz is None
    assert table.pulses["ef"].amp == 0.4
    assert ge_freq == units.freq_to_code(150e6, m.params)
    assert ef_freq == units.freq_to_code(50e6, m.params)


def test_levels_pop_reads_the_target_population():
    """_levels_pop classifies RAW IQ into levels with a 3-level ClassifierN and counts the target level
    (spec 01 §4.1): a point whose shots sit on the |2> centroid reads P(2)=1, on |1> reads P(2)=0 — the
    {|1>, |2>} discrimination the hardware res bit cannot do."""
    means = np.array([[10.0, 0.0], [-5.0, 8.66], [-5.0, -8.66]])   # 3 clusters ~120° apart
    rng = np.random.default_rng(0)
    clf = ClassifierN([means[k] + 0.2 * rng.standard_normal((40, 2)) for k in range(3)])
    npts, shots = 2, 16
    pt0 = means[1] + 0.2 * rng.standard_normal((shots, 2))         # all near |1>
    pt1 = means[2] + 0.2 * rng.standard_normal((shots, 2))         # all near |2>
    out = np.concatenate([pt0, pt1]).reshape(-1)                   # point-major, flat (2·npts·shots)
    assert np.array_equal(_levels_pop(out, npts, shots, clf, 2), [0.0, 1.0])
    assert np.array_equal(_levels_pop(out, npts, shots, clf, 1), [1.0, 0.0])


def test_ef_amplitude_guards_and_classifier_arg():
    """The EF X90 repetition guard (4·EF-X90 = 2π) and the classifier-arg normalization: a bare
    ClassifierN is one qubit, several qubits need a {q: ClassifierN} dict."""
    clf = ClassifierN([np.zeros((4, 2)), np.ones((4, 2)), 2 * np.ones((4, 2))])
    cfg = Config()
    with pytest.raises(AssertionError, match="multiple of 4"):
        EFAmplitude(cfg, 0, clf, n_gates=2)
    assert EFAmplitude(cfg, 0, clf, n_gates=4).classifiers == {0: clf}     # bare → one qubit
    with pytest.raises(ValueError, match="one classifier needs exactly one qubit"):
        EFAmplitude(cfg, [0, 1], clf)                                       # two qubits need a dict
    assert EFAmplitude(cfg, [0, 1], {0: clf, 1: clf}).classifiers == {0: clf, 1: clf}


# ── JAZZ fit (spec two-qubit/01 §4.3): host-pure ──

def test_signed_fft_freq_resolves_the_sign():
    """The complex quadrature FFT (qcal est_freq_fft) returns a SIGNED frequency: I − jQ of a fringe
    running the other way lands on a negative bin, which a real cosine fit could never tell apart."""
    t = np.linspace(0, 4e-6, 40)
    zp = np.exp(1j * 2 * np.pi * 2e6 * t)
    zn = np.exp(-1j * 2 * np.pi * 2e6 * t)
    assert _signed_fft_freq(t, zp) == pytest.approx(2e6, rel=0.05)
    assert _signed_fft_freq(t, zn) == pytest.approx(-2e6, rel=0.05)


def test_jazz_recovers_zz_from_synthetic_fringes():
    """JAZZ's fit: ZZ11 = f(control=1) − f(control=0), each control state's fringe frequency measured
    from its I (damped-cosine magnitude) and its complex I − jQ (sign). Two synthetic fringes at
    +1.0 MHz and +2.0 MHz → ZZ = 1.0 MHz; a control state running the other way is signed negative."""
    t = np.linspace(0, 4e-6, 40)
    rng = np.random.default_rng(0)

    def fringe(f):                                  # I = cos, Q = −sin ⇒ I − jQ = e^{+j2πft} (recovers +f)
        env = 0.45 * np.exp(-t / 3e-6)              # a decaying, slightly noisy fringe (as in co-sim)
        n = lambda: 0.01 * rng.standard_normal(len(t))
        return (0.5 + env * np.cos(2 * np.pi * f * t) + n(),
                0.5 - env * np.sin(2 * np.pi * f * t) + n())

    cal = JAZZ(None, (0, 1))
    f0, ok0 = cal._signed_freq(t, *fringe(1.0e6))
    f1, ok1 = cal._signed_freq(t, *fringe(2.0e6))
    assert ok0 and ok1
    assert f0 == pytest.approx(1.0e6, abs=5e4) and f1 == pytest.approx(2.0e6, abs=5e4)
    assert (f1 - f0) == pytest.approx(1.0e6, abs=1e5)         # the ZZ
    fneg, _ = cal._signed_freq(t, *fringe(-1.2e6))            # a fringe running the other way
    assert fneg == pytest.approx(-1.2e6, abs=5e4)


# ── CZ resonance & conditionality (spec two-qubit/01 §4.4-4.5): host-pure ──

def _cz_config():
    """A minimal two-qubit Config with a coupler-drive CZ entry (spec 01 §2)."""
    cfg = Config()
    for q in (0, 1):
        cfg[f"qubit/{q}/freq"] = 50e6
        cfg[f"qubit/{q}/x90/amp"] = 0.5
    cfg["two_qubit/(0, 1)/core"] = 2
    cfg["two_qubit/(0, 1)/CZ/freq"] = 25e6
    cfg["two_qubit/(0, 1)/CZ/pulse"] = [
        {"channel": "C0_1", "time": 200e-9, "kwargs": {"amp": 0.35, "phase": 0.0},
         "env": {"env_func": "cosine_square", "ramp_fraction": 0.1}},
        {"channel": "Q0", "env": "virtualz", "kwargs": {"phase": 0.3}},   # a ZI correction
        {"channel": "Q1", "env": "virtualz", "kwargs": {"phase": -0.2}},  # an IZ correction
    ]
    return cfg


def test_cz_table_and_config_accessors():
    """cz_table builds the coupler-drive slot BASEBAND at the config amp/env with the CZ/freq carrier;
    the local-phase accessors read each QUBIT's virtual-Z entry (channel-matched — the control's is
    the ZI, the target's the IZ); _cz_pulse_set updates the physical drive in a fresh list (the
    proposal payload, since it lives in a list leaf). The coupler form: one drive, `core` present."""
    m = SocMap(SocParams.load(_SIM2Q1C))
    cfg = _cz_config()
    assert cz_coupler_form(cfg, (0, 1))                          # spec 04 §4.1: `core` ⇒ coupler form
    assert _cz_drive_indices(cfg["two_qubit/(0, 1)/CZ/pulse"]) == [0]
    tbl = cz_table(cfg, (0, 1), m, dur_batches=20)
    assert list(tbl.pulses) == ["cz"]
    assert tbl.pulses["cz"].freq_hz is None and tbl.pulses["cz"].amp == 0.35
    assert _cz_amp(cfg, (0, 1)) == 0.35
    assert _local_phase_code(cfg, (0, 1), 0) == pack16(units._phase_code(0.3))    # ZI (qubit 0, 'Q0')
    assert _local_phase_code(cfg, (0, 1), 1) == pack16(units._phase_code(-0.2))   # IZ (qubit 1, 'Q1')
    assert _local_phase_code(cfg, (0, 1), 5) == 0                                 # absent → 0
    pulses = _cz_pulse_set(cfg, (0, 1), "amp", 0.42)
    assert pulses[0]["kwargs"]["amp"] == 0.42
    assert cfg["two_qubit/(0, 1)/CZ/pulse"][0]["kwargs"]["amp"] == 0.35           # original untouched
    assert _cz_pulse_set(cfg, (0, 1), "time", 1.5e-7)[0]["time"] == 1.5e-7


def test_cz_resonance_dip_fit_recovers_fcz():
    """CZSweep('freq')'s analysis (spec 01 §4.4): the CONTROL P(1) dip = 1 − the off-resonant Rabi
    transfer Ω²/(Ω²+Δ²)·sin²(√(Ω²+Δ²)·N/2) (the closed form test_models pins on the model), fit by a
    parabola, locates f_CZ — replacing qcal's argmax-mean with a proper fit. The vertex lands within a
    sweep step of the planted resonance."""
    from riscq.cal import fits
    m = SocMap(SocParams.load(_SIM2Q1C))
    f_cz_code, N, A = 2048, 40, 10000.0
    Om = (math.pi / (N * A)) * A                                  # resonant rate rad/batch
    codes = np.arange(f_cz_code - 40, f_cz_code + 41, 8)
    freqs = np.array([units.code_to_freq(int(c), m.params) for c in codes])

    def p11(code):                                               # control P(1) = 1 − transfer to |02>
        dlt = (code - f_cz_code) * 16 * math.pi / (1 << 15)      # demod axis-ramp per batch (the detuning)
        g = math.hypot(Om, dlt)
        return 1.0 - (Om ** 2 / g ** 2) * math.sin(g * N / 2) ** 2

    P1 = np.array([p11(int(c)) for c in codes])
    fit = fits.fit_parabola(freqs, P1)
    assert fit.ok and fit.params["a"] > 0, "a P(1) dip is an UPWARD parabola (min at resonance)"
    assert abs(fit.value - units.code_to_freq(f_cz_code, m.params)) < abs(units.code_to_freq(8, m.params))


def test_cz_conditionality_argmax_and_amplitude_vertex():
    """The conditionality analyses (spec 01 §4.5): CZFrequency writes the argmax of R over the freq
    sweep (parabola-refined at the peak), CZAmplitude the parabola VERTEX of R over the amp window. Both
    exercised on synthetic R curves peaked at a known value (the model's R physics is
    test_models.test_twoqubit_cz_conditionality_R_peaks_at_the_cz)."""
    from riscq.cal import fits
    x = np.linspace(-1.0, 1.0, 21)
    R = 1.0 - 0.6 * (x - 0.2) ** 2                                # a peak at x = 0.2
    assert x[int(np.argmax(R))] == pytest.approx(0.2, abs=0.06)   # CZFrequency's argmax
    v = fits.fit_parabola(x, R)                                   # CZAmplitude's vertex (downward, a<0)
    assert v.ok and v.params["a"] < 0 and v.value == pytest.approx(0.2, abs=1e-6)


def test_cz_classes_construct_and_carry_defaults():
    """The CZ cal classes take (cfg, pair) with qcal-faithful defaults (the coupler-drive layout is the
    default, spec 01 §2 — no per-call params= boilerplate) and the (i, j) tuple key."""
    cfg = _cz_config()
    assert CZSweep(cfg, (0, 1), knob="dur").pair == (0, 1)
    assert CZFrequency(cfg, (0, 1)).ngates == 1
    assert CZAmplitude(cfg, (0, 1)).n_gates == (1, 3, 5, 7, 9)
    assert LocalPhases(cfg, (0, 1)).pair == (0, 1)
    with pytest.raises(AssertionError, match="freq/dur/amp"):
        CZSweep(cfg, (0, 1), knob="phase")


# ── Local phases (spec two-qubit/01 §4.6): host-pure ──

def test_local_phases_fringe_and_mean():
    """LocalPhases' analysis (spec 01 §4.6; branch combination per spec 04 §3/X1, qcal
    cz.py:2013-2051 parity): the ACTIVE Ramsey fringe peak is the first-harmonic phase of P vs the
    full-turn φ sweep (robust to the cosine sign); the correction removes the conditional π from the
    spectator-|1> branch BEFORE the shorter-arc midpoint. On ideal-CZ branches (|1> peak = |0> peak
    + π + δ for a small residual conditionality error δ) it recovers the |0>-branch local phase
    + δ/2 — NO π/2 term — stable for δ of EITHER sign and under noise (the raw midpoint sat at
    local ± π/2 with a noise-unstable sign: _mean_offset's wrap-boundary degeneracy)."""
    phi = np.linspace(-math.pi, math.pi, 24, endpoint=False)
    rng = np.random.default_rng(7)

    def fringe(peak):                                            # a noisy Ramsey P peaking at `peak`
        return 0.5 + 0.4 * np.cos(phi - peak) + 0.01 * rng.standard_normal(phi.size)

    def wrap(x):
        return (x + math.pi) % (2 * math.pi) - math.pi

    a, ca = _fringe_peak(phi, fringe(0.3))
    assert a == pytest.approx(0.3, abs=0.05) and ca > 0.15      # the contrast the run gates on
    assert _mean_offset(0.3, 0.42) == pytest.approx(0.36)       # the plain shorter-arc midpoint

    for peak0 in (0.3, -2.9):               # -2.9: the corrected pair straddles the ±π wrap
        for delta in (0.12, -0.12):         # the residual conditional-phase error, EITHER sign
            o0, _ = _fringe_peak(phi, fringe(peak0))
            o1, _ = _fringe_peak(phi, fringe(peak0 + math.pi + delta))
            corr = _branch_correction(o0, o1)
            assert wrap(corr - (peak0 + delta / 2)) == pytest.approx(0.0, abs=0.05)

    assert math.isnan(_branch_correction(0.3, math.nan))        # a failed branch stays a nan


def test_layout_accessors_on_the_two_qubit_drive_form():
    """(X0 gate) the layout-aware walk on the real X6Y3 config (spec 04 §4.2): nothing is found by
    index — drives by the find_pulse_index walk (string references shift every position), virtual-Z
    entries by their `channel` key — on a plain pair, a spectator-carrying pair, and the EF-X
    sandwich pair."""
    cfg = Config.from_qcal(_X6Y3)

    # (0, 1) — the plain two-qubit-drive layout: drives at 0/1, vz Q0/Q1/Q2 behind them.
    pl = cfg["two_qubit/(0, 1)/CZ/pulse"]
    assert not cz_coupler_form(cfg, (0, 1))                      # no `core` ⇒ two-qubit-drive form
    assert _cz_drive_indices(pl) == [0, 1]
    assert _cz_entry(cfg, (0, 1))["channel"] == "Q0.qdrv"        # control drive, phase 0
    tgt = _cz_entry(cfg, (0, 1), drive=1)
    assert tgt["channel"] == "Q1.qdrv" and tgt["kwargs"]["phase"] != 0.0   # the RELATIVE phase
    assert _cz_vz_entry(pl, 0) is pl[2] and _cz_vz_entry(pl, 1) is pl[3]   # ZI / IZ by channel
    assert _cz_vz_entry(pl, 2) is pl[4]                                     # the spectator (Q2)
    assert _local_phase_code(cfg, (0, 1), 0) == pack16(units._phase_code(pl[2]["kwargs"]["phase"]))
    assert _local_phase_code(cfg, (0, 1), 1) == pack16(units._phase_code(pl[3]["kwargs"]["phase"]))

    # (5, 6) — the EF-X shelving sandwich: string references at 0/3 shift drives to 1/2, vz to 4/5.
    pl = cfg["two_qubit/(5, 6)/CZ/pulse"]
    assert isinstance(pl[0], str) and isinstance(pl[3], str)
    assert _cz_drive_indices(pl) == [1, 2]
    assert _cz_entry(cfg, (5, 6))["channel"] == "Q5.qdrv"
    assert _cz_entry(cfg, (5, 6), drive=1)["channel"] == "Q6.qdrv"
    assert _cz_vz_entry(pl, 5) is pl[4] and _cz_vz_entry(pl, 6) is pl[5]
    assert _cz_vz_entry(pl, 7) is None and _local_phase_code(cfg, (5, 6), 7) == 0

    # updates land on BOTH drive lines (qcal calibrates them jointly) and never on strings or vz
    pulses = _cz_pulse_set(cfg, (5, 6), "amp", 0.42)
    assert [pulses[i]["kwargs"]["amp"] for i in (1, 2)] == [0.42, 0.42]
    assert pulses[0] == pl[0] and pulses[3] == pl[3]             # the sandwich survives verbatim
    local = _cz_local_set(cfg, (5, 6), 0.11, -0.22)
    assert local[4]["kwargs"]["phase"] == 0.11 and local[5]["kwargs"]["phase"] == -0.22
    assert local[0] == pl[0] and local[1] == pl[1]               # drives and strings untouched


def test_cz_table_envelope_kwargs_reach_the_build():
    """(X0 gate) cz_table's envelope kwargs = the entry's `kwargs` minus amp/phase (spec 04 §3):
    X6Y3 carries `ramp_fraction` beside amp/phase, and the old drop was invisible only because
    cosine_square's default equals the config value — plant a NON-default one and check it lands.
    The legacy `{env_func, ...}` dict form still supplies (and merges under) shape kwargs."""
    m = SocMap(SocParams.load(_SIM2Q1C))
    ch = m.channel(GATE_CH)
    n = 20 * ch.samples_per_line
    rate = ch.samples_per_line * m.params.dsp_freq_hz

    cfg = Config.from_qcal(_X6Y3)
    pl = copy.deepcopy(cfg["two_qubit/(0, 1)/CZ/pulse"])
    pl[0]["kwargs"]["ramp_fraction"] = 0.5
    cfg["two_qubit/(0, 1)/CZ/pulse"] = pl
    tbl = cz_table(cfg, (0, 1), m, dur_batches=20)
    assert np.array_equal(tbl.pulses["cz"].env, envelopes.build("cosine_square", n, rate,
                                                                ramp_fraction=0.5))
    assert not np.array_equal(tbl.pulses["cz"].env, envelopes.build("cosine_square", n, rate,
                                                                    ramp_fraction=0.25))
    assert tbl.pulses["cz"].amp == pl[0]["kwargs"]["amp"]        # amp/phase stay slot params
    assert tbl.pulses["cz"].phase == 0.0

    tbl2 = cz_table(_cz_config(), (0, 1), m, dur_batches=20)     # dict-env form: ramp_fraction 0.1
    assert np.array_equal(tbl2.pulses["cz"].env, envelopes.build("cosine_square", n, rate,
                                                                 ramp_fraction=0.1))


def test_cz_local_set_writes_both_frames():
    """LocalPhases writes the control ZI (pulse/1) and target IZ (pulse/2) virtual-Z phases into a fresh
    pulse list (the proposal payload) without touching the original config."""
    cfg = _cz_config()
    pulses = _cz_local_set(cfg, (0, 1), 0.11, -0.22)
    assert pulses[1]["kwargs"]["phase"] == 0.11 and pulses[2]["kwargs"]["phase"] == -0.22
    assert cfg["two_qubit/(0, 1)/CZ/pulse"][1]["kwargs"]["phase"] == 0.3   # original untouched


# ── Two-qubit-drive form (spec 04 §4.1-4.4 / X2): host-pure ──

def _drive_cfg():
    """A minimal two-qubit-drive Config (spec 04 §1): NO coupler core — both CZ lines on the pair's
    own gate channels at the shared in-band `CZ/freq`, the TARGET line carrying the relative phase.
    Distinct GE carriers so the f_GE → f_CZ retune is visible per core."""
    cfg = Config()
    for q, f in ((0, 50e6), (1, 75e6)):
        cfg[f"qubit/{q}/freq"] = f
        cfg[f"qubit/{q}/x90/amp"] = 0.5
        cfg[f"readout/{q}/freq"] = 10e6
        cfg[f"readout/{q}/amp"] = 0.5
        cfg[f"readout/{q}/dur"] = 56e-8
        cfg[f"readout/{q}/demod/dur"] = 40e-8
    cfg["reset/relax"] = 8e-6
    cfg["two_qubit/(0, 1)/CZ/freq"] = 25e6
    cfg["two_qubit/(0, 1)/CZ/pulse"] = [
        {"channel": "Q0", "time": 30e-8, "kwargs": {"amp": 0.35, "phase": 0.0}, "env": "square"},
        {"channel": "Q1", "time": 30e-8, "kwargs": {"amp": 0.35, "phase": 0.267}, "env": "square"},
        {"channel": "Q0", "env": "virtualz", "kwargs": {"phase": 0.0}},
        {"channel": "Q1", "env": "virtualz", "kwargs": {"phase": 0.0}},
    ]
    return cfg


def test_calc_cz_frequency_drive_form():
    """The drive-form seed (spec 04 §4.4): CZ/freq = (f_11 + f_state)/4 — half the two-photon
    midpoint — with the '20' mirror; the parametric arithmetic is untouched; a bad form is loud."""
    cfg = Config()
    cfg["qubit/0/freq"] = 5.0e9
    cfg["qubit/1/freq"] = 5.2e9
    cfg["qubit/0/EF/freq"] = 4.7e9
    cfg["qubit/1/EF/freq"] = 4.9e9

    calc_cz_frequency(cfg, [(0, 1)], state="02", form="drive")
    assert cfg["two_qubit/(0, 1)/CZ/freq"] == pytest.approx((10.2e9 + 10.1e9) / 4)   # 5.075 GHz
    calc_cz_frequency(cfg, [(0, 1)], state="20", form="drive")
    assert cfg["two_qubit/(0, 1)/CZ/freq"] == pytest.approx((10.2e9 + 9.7e9) / 4)    # 4.975 GHz
    calc_cz_frequency(cfg, [(0, 1)], state="02")                                     # parametric default
    assert cfg["two_qubit/(0, 1)/CZ/freq"] == pytest.approx(100e6)
    with pytest.raises(AssertionError, match="parametric"):
        calc_cz_frequency(cfg, [(0, 1)], form="coupler")


def test_drive_seed_lands_near_x6y3_calibrated():
    """(X2 gate) the drive-form seed reproduces the six PLAIN X6Y3 pairs' calibrated `CZ/freq` to
    within ±100 MHz (spec 04 §1: −92…+61 MHz on state='02'); the EF-sandwich pairs (5,6)/(6,7) sit
    in the shelved manifold and are excluded (X4)."""
    cfg = Config.from_qcal(_X6Y3)
    plain = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (7, 0)]
    cal = {p: float(cfg[f"two_qubit/{pair_key(p)}/CZ/freq"]) for p in plain}
    calc_cz_frequency(cfg, plain, state="02", form="drive")
    for p in plain:
        seed = float(cfg[f"two_qubit/{pair_key(p)}/CZ/freq"])
        assert abs(seed - cal[p]) < 100e6, \
            f"pair {p}: drive seed {seed / 1e9:.4f} GHz vs calibrated {cal[p] / 1e9:.4f} GHz"


def test_cz_drive_table_roles():
    """(X2 gate) `cz_drive_table` builds each qubit core's drive-form gate table (spec 04 §4.1): GE
    'x90' at the core's OWN carrier + a BASEBAND 'cz' slot from ITS line of the pair — control
    drive-0 (phase 0), target drive-1 (the relative phase lands on the TARGET slot). A plain pair
    gets NO 'ef' slot (the sandwich resolution is test_cz_sandwich_resolves_the_x6y3_pair)."""
    m = SocMap(SocParams.load(_SIM2Q))
    cfg = _drive_cfg()
    assert not cz_coupler_form(cfg, (0, 1))                      # no `core` ⇒ two-qubit-drive form
    assert cz_sandwich(cfg, (0, 1)) is None                      # no string refs ⇒ plain
    tc = cz_drive_table(cfg, (0, 1), 0, 0, m, 30)                # control core, drive line 0
    tt = cz_drive_table(cfg, (0, 1), 1, 1, m, 30)                # target core, drive line 1
    assert list(tc.pulses) == ["x90", "cz"] == list(tt.pulses)
    assert tc.slot_of("cz") == 1 == tt.slot_of("cz")             # the host-paced write_slot target
    assert tc.freq_hz == 50e6 and tt.freq_hz == 75e6             # each core's OWN GE carrier
    assert tc.pulses["cz"].freq_hz is None and tt.pulses["cz"].freq_hz is None   # runtime retune
    assert tc.pulses["cz"].amp == 0.35 == tt.pulses["cz"].amp    # equal lines (calibrated jointly)
    assert tc.pulses["cz"].phase == 0.0 and tt.pulses["cz"].phase == 0.267       # relative phase
    assert cz_drive_table(cfg, (0, 1), 0, 0, m, 30, amp=0.5).pulses["cz"].amp == 0.5


def test_cz_cond_progs_drive_form_lockstep():
    """(X2 gate) `_cz_cond_progs` on a drive-form pair compiles TWO programs (no coupler core) with
    the swept pair bound LOCKSTEP on both — the x0/dx literals land in BOTH cores' generated C (the
    coupler form binds a dead 0 sweep on the qubit cores) — and each core's cz slot carries its own
    line's phase. Also the compile gate for the k_cz_cond DRIVE_FORM branches (both roles)."""
    m = SocMap(SocParams.load(_SIM2Q))
    cfg = _drive_cfg()
    x0, dx = 1024 << 16, 65536
    progs, tables, signs, timeout = _cz_cond_progs(cfg, m, (0, 1), "freq", x0, dx,
                                                   points=5, ngates=3, shots=8)
    assert sorted(progs) == [0, 1]                               # 2 cores — no coupler program
    for q in (0, 1):
        assert str(x0) in progs[q].c_source, f"core {q} missing the lockstep x0"
        assert str(dx) in progs[q].c_source, f"core {q} missing the lockstep dx"
    assert tables[0].pulses["cz"].phase == 0.0 and tables[1].pulses["cz"].phase == 0.267

    # the coupler form still compiles 3 programs, the sweep on the coupler alone
    cfgc = _cz_config()
    cfgc["reset/relax"] = 8e-6
    for q in (0, 1):
        cfgc[f"readout/{q}/freq"] = 10e6
        cfgc[f"readout/{q}/amp"] = 0.5
        cfgc[f"readout/{q}/dur"] = 56e-8
        cfgc[f"readout/{q}/demod/dur"] = 40e-8
    progs_c, _, _, _ = _cz_cond_progs(cfgc, m, (0, 1), "freq", x0, dx, points=5, ngates=1, shots=8)
    assert sorted(progs_c) == [0, 1, 2]
    for q in (0, 1):
        assert str(x0) not in progs_c[q].c_source                # qubit cores: dead 0 sweep
    assert str(x0) in progs_c[2].c_source                        # the COUPLER carries it


def test_drive_form_pop_and_local_kernels_compile():
    """(X2 gate) the DRIVE_FORM branches of `k_cz_pop` (all three knobs) and `k_cz_local` (both
    roles) compile against the drive tables on the 2-core build — the dead COUPLER branches fold
    away, so no coupler table is ever needed. (`k_cz_cond`'s compile gate is the lockstep test.)"""
    from riscq.cal import kernels
    from riscq.cal.base import readout_tables, x90_vz
    from riscq.lang import Array, compile_kernel
    m = SocMap(SocParams.load(_SIM2Q))
    cfg = _drive_cfg()
    fcz = units.freq_to_code(25e6, m.params)
    for q, drive in ((0, 0), (1, 1)):
        gate = cz_drive_table(cfg, (0, 1), q, drive, m, 30)
        ro, demod, code, dur, ddly = readout_tables(cfg, q, m)
        tables = dict(gate=gate, ro=ro, demod=demod)
        for knob in (kernels.FREQ, kernels.DUR, kernels.AMP):
            compile_kernel(kernels.k_cz_pop, m, tables=tables, out=Array(3), npts=3, shots=2,
                           period=800, code=code, ddly=ddly, role=kernels.CONTROL, knob=knob,
                           form=kernels.DRIVE_FORM, xd=4, czmax=30, fcz=fcz, fef=0, sw=0, tail=0,
                           x0=int(fcz), dx=0, **x90_vz(cfg, q))
        for role in (kernels.ACTIVE, kernels.SPECTATOR):
            compile_kernel(kernels.k_cz_local, m, tables=tables, out=Array(3), npts=3, shots=2,
                           period=800, code=code, ddly=ddly, role=role, form=kernels.DRIVE_FORM,
                           hpi=pack16(units._phase_code(math.pi / 2)), xd=4, czd=30, fcz=fcz,
                           fef=0, sw=0, tail=0, p0=0, dp=0, sp=0, **x90_vz(cfg, q))


def test_relative_phase_recovers_the_peak(responder):
    """(X2 gate) `RelativePhase` end-to-end host-pure on synthetic branch data: the driver layer is
    replaced by the shared `Responder` (specs/software-test-refactor/01 §2.2), the four tomography
    branches generated per rerun from the CURRENTLY WRITTEN target cz-slot phase with a conditional
    phase θ(φ) peaking (θ=π) at a planted φ* — so R = |sin(θ/2)| maximises there. The class's
    slot-write pacing, the four `_cond_R` reruns, the parabola/argmax refine and the target
    `kwargs/phase` write-back all run for real; a coupler pair asserts out."""
    cfg = _drive_cfg()
    phi_star, shots = 0.7, 400
    r = responder(_SIM2Q)

    @r.answer
    def _(progs, params):
        core, table, slot, field, value = r.slot_writes[-1]
        assert (core, table, slot, field) == (1, "gate", 1, "phase")   # the TARGET core's cz slot
        phi = value * math.pi / (1 << 15)                              # plain phase code → rad
        prep = params.get(0, {}).get("prep", 0)
        quad = params.get(1, {}).get("quad", 0)
        theta = math.pi * math.cos((phi - phi_star) / 2) if prep else 0.0
        p0 = (1 + (math.cos(theta) if quad == 0 else math.sin(theta))) / 2    # target P(0)
        return {0: {"out": np.array([0])},
                1: {"out": counts([1 - p0], shots)}}                   # the kernel counts |1>s

    cal = RelativePhase(cfg, (0, 1), points=21, shots=shots)
    res = cal.run(r.drv)
    R = res.data[(0, 1)]["R"]
    assert res.ok and R.max() > 0.9
    written = res.proposal["two_qubit/(0, 1)/CZ/pulse"][1]["kwargs"]["phase"]
    assert written == pytest.approx(phi_star, abs=0.2)             # within the argmax half-step
    assert cfg["two_qubit/(0, 1)/CZ/pulse"][1]["kwargs"]["phase"] == 0.267   # original untouched

    with pytest.raises(AssertionError, match="two-qubit-drive"):
        RelativePhase(_cz_config(), (0, 1)).run(r.drv)


def _spect_cfg():
    """`_drive_cfg()` + a third qubit (the ring spectator) and its channel-matched vz entry in the
    pair's CZ pulse list (the X6Y3 layout: every pair carries 1-2 spectator corrections)."""
    cfg = _drive_cfg()
    cfg["qubit/2/freq"] = 60e6
    cfg["qubit/2/x90/amp"] = 0.5
    cfg["readout/2/freq"] = 12e6
    cfg["readout/2/amp"] = 0.5
    cfg["readout/2/dur"] = 56e-8
    cfg["readout/2/demod/dur"] = 40e-8
    pl = copy.deepcopy(cfg["two_qubit/(0, 1)/CZ/pulse"])
    pl.append({"channel": "Q2", "env": "virtualz", "kwargs": {"phase": 0.0}})
    cfg["two_qubit/(0, 1)/CZ/pulse"] = pl
    return cfg


def test_spectator_phase_recovers_planted_phase(responder):
    """(X3 gate) `SpectatorPhase` end-to-end host-pure on synthetic fringes: the driver layer is
    replaced by the shared `Responder` and each conditional-branch rerun returns the spectator Ramsey
    fringe P(1) = (1 + cos(ψ_c + φ))/2 for a planted spectator kick ψ_c = ψ + c·δ (a small spectator
    conditionality δ — NO conditional π, the spectator is outside the gate). The class's REAL 3-core
    compile (spectator = the COUPLER_FORM ACTIVE Ramsey at czd+LEAD, the pair = DRIVE_FORM SPECTATOR
    roles, `sp` live only on the conditional core), the two `sp` reruns, the wrap-aware branch mean
    and the channel-matched write-back all run for real; the recovered −ψ − δ/2 lands in the
    SPECTATOR's entry of the pair's pulse list."""
    from riscq.cal import SpectatorPhase
    from riscq.cal.twoqubit import _phi_sweep
    cfg = _spect_cfg()
    psi, delta, points, shots = 0.9, 0.1, 24, 200
    r = responder(_SIM2Q1C)

    _, _, phi_ax = _phi_sweep(points)                            # the class's own φ axis

    @r.answer
    def _(progs, params):
        sp = params[0]["sp"]                                     # conditional defaults to pair[0]
        P1 = (1 + np.cos(psi + sp * delta + phi_ax)) / 2         # fringe peaks at φ = −ψ_c
        out = {q: {"out": np.zeros(points)} for q in progs}
        out[2] = {"out": np.rint(P1 * shots)}
        return out

    cal = SpectatorPhase(cfg, (0, 1), spectator=2, points=points, shots=shots)
    res = cal.run(r.drv)
    assert res.ok
    compiled = {q for setup in r.setups for q in setup}
    assert sorted(compiled) == [0, 1, 2], "expected a real 3-core compile through setup"
    pulses = res.proposal["two_qubit/(0, 1)/CZ/pulse"]
    got = _cz_vz_entry(pulses, 2)["kwargs"]["phase"]
    want = -(psi + delta / 2)                                    # the two branch peaks' midpoint
    assert got == pytest.approx(want, abs=0.03)
    # the pair's own vz entries and the original config stay untouched
    assert _cz_vz_entry(pulses, 0)["kwargs"]["phase"] == 0.0
    assert _cz_vz_entry(pulses, 1)["kwargs"]["phase"] == 0.0
    assert _cz_vz_entry(cfg["two_qubit/(0, 1)/CZ/pulse"], 2)["kwargs"]["phase"] == 0.0


def test_spectator_phase_guards():
    """SpectatorPhase's loud edges: a spectator inside the pair, a conditional outside it, a
    coupler-form pair (unsupported — spec 04 §4.5 scopes the drive form), and a write-back for a
    qubit with no vz entry."""
    from riscq.cal import SpectatorPhase
    from riscq.cal.twoqubit import _cz_spectator_set
    cfg = _spect_cfg()
    with pytest.raises(AssertionError, match="LocalPhases"):
        SpectatorPhase(cfg, (0, 1), spectator=1)                 # spectator is in the pair
    with pytest.raises(AssertionError, match="conditional"):
        SpectatorPhase(cfg, (0, 1), spectator=2, conditional=2)  # conditional is not in the pair
    with pytest.raises(AssertionError, match="two-qubit-drive"):
        SpectatorPhase(_cz_config(), (0, 1), spectator=3)        # coupler form: not wired
    with pytest.raises(ValueError, match="virtualz"):
        _cz_spectator_set(cfg, (0, 1), 5, 0.1)                   # qubit 5 has no entry
    assert SpectatorPhase(cfg, (0, 1), spectator=2).conditional == 0   # defaults to the control


def test_phi_sweep_is_a_real_full_turn():
    """(X3 fix) the class-level φ sweep must actually SWEEP: ±π wrap to the SAME phase code, so the
    old inclusive −π→+π span collapsed to a zero step (a flat axis — LocalPhases/SpectatorPhase
    could never see a fringe). `_phi_sweep` is endpoint-exclusive: a nonzero uniform step covering
    one full turn without the duplicate endpoint."""
    from riscq.cal.twoqubit import _phi_sweep
    assert units._phase_code(math.pi) == units._phase_code(-math.pi)   # the wrap that bit
    for points in (15, 24):
        c0, dc, phi = _phi_sweep(points)
        assert dc > 0 and len(phi) == points
        assert phi[0] == pytest.approx(-math.pi, abs=1e-4)
        assert points * dc == pytest.approx(1 << 16, abs=points / 2)   # one full turn, exclusive
        assert phi[-1] < math.pi - 1e-3                                # no duplicate ±π sample
    assert _phi_sweep(1)[1] == 0


def test_cz_rel_phase_set_targets_the_second_drive():
    """`_cz_rel_phase_set` writes ONLY the target drive line's `kwargs/phase` (qcal's pulse/{idx+1}
    param) into a fresh list; a single-drive (coupler) list is a loud error."""
    cfg = _drive_cfg()
    pulses = _cz_rel_phase_set(cfg, (0, 1), 0.5)
    assert pulses[1]["kwargs"]["phase"] == 0.5
    assert pulses[0]["kwargs"]["phase"] == 0.0                   # control line untouched
    assert cfg["two_qubit/(0, 1)/CZ/pulse"][1]["kwargs"]["phase"] == 0.267   # original untouched
    with pytest.raises(ValueError, match="coupler form"):
        _cz_rel_phase_set(_cz_config(), (0, 1), 0.5)


# ── CZ 2D amp x freq seed landscape (spec 14 F4): host-pure ──

_AF_N = 60                    # CZ tone length in batches (the planted pseudo-qubit product's grid)
_AF_AMP = 0.35                # the planted 2pi-round-trip amp — _drive_cfg()'s own CZ amp
_AF_DELTA_PER_HZ = 6.0 / 1.5e6   # planted detuning ramp: 6 rad of axis walk at 1.5 MHz off


def _cz_uv(amp, freq, f_star):
    """The planted (|11>, |02>) amplitudes after one CZ tone: `_AF_N` batches, each rotating the
    {|11>, |02>} pseudo-qubit by theta/N about an axis advancing by delta/N per batch — exactly
    `TwoQubitModel`'s drive-form activation (demod-then-rotate per batch, the axis ramping with the
    carrier detuning), reproduced to 1e-3 across amp AND detuning. theta = 2*pi at `_AF_AMP` on
    resonance, where the round trip closes and stamps the conditional pi."""
    theta = 2 * math.pi * amp / _AF_AMP
    delta = _AF_DELTA_PER_HZ * (freq - f_star)
    a, b = 1 + 0j, 0j
    c, s = math.cos(theta / (2 * _AF_N)), math.sin(theta / (2 * _AF_N))
    for t in range(_AF_N):
        e = np.exp(1j * delta * t / _AF_N)
        a, b = c * a - 1j * s * b / e, -1j * s * e * a + c * b
    return a, b


def _cz_branch_p0(u, v, quad):
    """The tomography branch's target P(read 0) for a CZ that left |11> with amplitude u and |02>
    with v. The control-|1> row holds (|0> + u|1>)/sqrt(2) after the Y90 prep + the CZ, and the
    close is Y90 (quad 0) or X90 (quad 1) — |1 - u|^2/4 and |1 - i*u|^2/4 — while the |02> leg
    parks |v|^2/2 of the target in |2>, which our ONE-BIT discriminator reads as a coin flip
    (+|v|^2/4; the |2> ambiguity of spec 01 §4.5, which qcal's 3-level classifier resolves and we
    do not). The control-|0> branch is the same with u = 1, v = 0 (the CZ cannot touch |01>)."""
    return abs(1 - (1j * u if quad else u)) ** 2 / 4 + abs(v) ** 2 / 4


def test_cz_amp_freq_sweep_seeds_the_argmax(responder):
    """(F4 gate) `CZAmpFreqSweep` end-to-end host-pure on the exact drive-form CZ physics: the
    driver layer is replaced by the shared `Responder` and the four tomography branches are
    generated per rerun from the {|11>, |02>} pseudo-qubit amplitude at the CURRENTLY WRITTEN
    lockstep amp and each swept carrier (`_cz_uv` + `_cz_branch_p0`, the per-batch ramping-axis
    product the model runs, read through our 1-bit discriminator). The class's ONE compile, the
    both-lines `write_slot('amp')` pacing, the four `_cond_R` reruns per amp row, the 2D argmax and
    the CZ/freq + CZ/pulse write-back all run for real."""
    from riscq.cal.base import sweep_q16
    cfg = _drive_cfg()
    points, shots = 9, 400
    m = SocMap(SocParams.load(_SIM2Q))
    span = 3e6
    lo = units._freq_code(25e6 - span, m.params)
    hi = units._freq_code(25e6 + span, m.params)
    _, _, xs = sweep_q16(lo, hi, points)
    fax = np.array([units.code_to_freq(int(x), m.params) for x in xs])
    f_star = float(fax[points // 2])                             # plant on a grid point
    r = responder(_SIM2Q)

    @r.answer
    def _(progs, params):
        prep = params.get(0, {}).get("prep", 0)
        quad = params.get(1, {}).get("quad", 0)
        amps = [r.slot(core, "gate", 1, "amp") for core in (0, 1)]   # each core's OWN cz slot
        assert amps[0] == amps[1], "the two CZ lines must be written LOCKSTEP"
        amp = int(amps[1]) / units.AMP_SCALE
        p0 = np.array([_cz_branch_p0(*(_cz_uv(amp, f, f_star) if prep else (1.0, 0.0)),
                                     quad) for f in fax])
        return {0: {"out": np.zeros(points, int)},
                1: {"out": np.rint((1 - p0) * shots).astype(int)}}   # the kernel counts |1>s

    cal = CZAmpFreqSweep(cfg, (0, 1), span=span, points=points, shots=shots)
    res = cal.run(r.drv)
    d = res.data[(0, 1)]
    assert len(r.setups) == 1                                    # ONE compile: the amp is host-side
    assert d["R"].shape == (7, points)                           # the default 7-amp x freq grid
    assert d["amps"] == pytest.approx(np.linspace(0.5 * _AF_AMP, 1.5 * _AF_AMP, 7))
    assert d["freqs"] == pytest.approx(fax)
    assert len(r.slot_writes) == 2 * 7                           # both lines, once per amp row
    assert all((t, s, f) == ("gate", 1, "amp")                   # each core's OWN cz slot
               for (_, t, s, f, _) in r.slot_writes)
    ka, kf = np.unravel_index(int(np.argmax(d["R"])), d["R"].shape)
    assert (ka, kf) == (3, points // 2)                          # the planted (amp*, f_CZ*)
    assert d["R"][ka, kf] > 0.95 and res.ok
    assert res.proposal["two_qubit/(0, 1)/CZ/freq"] == pytest.approx(f_star)
    written = res.proposal["two_qubit/(0, 1)/CZ/pulse"]
    assert [p["kwargs"]["amp"] for p in written[:2]] == pytest.approx([_AF_AMP, _AF_AMP])
    assert written[1]["kwargs"]["phase"] == 0.267                # the relative phase is untouched
    assert cfg["two_qubit/(0, 1)/CZ/freq"] == 25e6               # original config untouched

    # a landscape with no conditional response anywhere: refuse, write nothing (CZFrequency's gate)
    dead = CZAmpFreqSweep(cfg, (0, 1), amps=[0.02, 0.03], span=span, points=points,
                          shots=shots).run(r.drv)
    assert dead.data[(0, 1)]["R"].max() < 0.5
    assert not dead.ok and dead.proposal == {}


# ── EF-sandwich CZ playback (spec 04 §1 / X4): host-pure ──


def _sandwich_cfg():
    """`_drive_cfg()` rebuilt as an EF-sandwich pair (the X6Y3 (5,6)/(6,7) layout on the 2-core
    sim-2q labels): qubit 1's EF keys + the SAME drive/vz list bracketed by the two identical
    `single_qubit/1/EF/X/pulse` string references."""
    cfg = _drive_cfg()
    cfg["qubit/1/EF/freq"] = 125e6
    cfg["qubit/1/EF/x/amp"] = 0.6
    pl = cfg["two_qubit/(0, 1)/CZ/pulse"]
    cfg["two_qubit/(0, 1)/CZ/pulse"] = (["single_qubit/1/EF/X/pulse"] + pl[:2]
                                        + ["single_qubit/1/EF/X/pulse"] + pl[2:])
    return cfg


def test_cz_sandwich_resolves_the_x6y3_pair():
    """(X4 gate) sandwich table building on the REAL X6Y3 pair (5, 6): the string references resolve
    through the config into an 'ef' gate-table slot — qubit 6's OWN EF X (FAST_DRAG envelope, config
    amp, baseband) — with the drives' amps/relative phase untouched and NO NotImplementedError; the
    bindings put sw/fef on the SHELF core (6) only and the (LEAD + EF X) tail on both."""
    m = SocMap(SocParams.load(_SIM2Q))
    cfg = Config.from_qcal(_X6Y3)
    assert cz_sandwich(cfg, (0, 1)) is None                      # plain pair: no references
    assert cz_sandwich(cfg, (5, 6)) == 6 and cz_sandwich(cfg, (6, 7)) == 6
    t5 = cz_drive_table(cfg, (5, 6), 5, 0, m, 20)                # control core, drive line 0
    t6 = cz_drive_table(cfg, (5, 6), 6, 1, m, 20)                # target core (the shelf), line 1
    assert list(t5.pulses) == ["x90", "cz", "ef"] == list(t6.pulses)
    assert t5.slot_of("cz") == 1 == t6.slot_of("cz")             # the write_slot target unmoved
    pl = cfg["two_qubit/(5, 6)/CZ/pulse"]
    assert t5.pulses["cz"].amp == pl[1]["kwargs"]["amp"] == t6.pulses["cz"].amp
    assert t5.pulses["cz"].phase == 0.0 and t6.pulses["cz"].phase == pl[2]["kwargs"]["phase"]
    efx = ef_pulse(cfg, 6, m, "x")                               # qubit 6's OWN EF X (FAST_DRAG)
    for t in (t5, t6):                                           # on BOTH cores (the partner pads)
        assert t.pulses["ef"].amp == float(cfg["qubit/6/EF/x/amp"])
        assert np.array_equal(t.pulses["ef"].env, efx.env)
        assert t.pulses["ef"].freq_hz is None                    # baseband: retuned to f_EF at runtime
    # (the fef SEAT happens at class run time against the driver's own map — the sim map's 1.6 GS/s
    # cannot carry the chip's GHz EF carrier; the binds numerics are the synthetic-pair test's)


def test_cz_sandwich_rejects_unsupported_layouts():
    """`cz_sandwich` supports exactly the X6Y3 sandwich; every other string-reference layout is a
    loud error, never a silent mis-play (X4 scope): a lone reference, a non-EF-X path, a non-member
    qubit, references that do not bracket the drives, and missing EF calibration keys."""
    base = _sandwich_cfg()["two_qubit/(0, 1)/CZ/pulse"]

    def with_pulses(pl):
        c = _sandwich_cfg()
        c["two_qubit/(0, 1)/CZ/pulse"] = pl
        return c

    with pytest.raises(ValueError, match="2 identical"):
        cz_sandwich(with_pulses(base[:3] + base[4:]), (0, 1))    # the post reference dropped
    pl = copy.deepcopy(base)
    pl[0] = pl[3] = "single_qubit/1/GE/X/pulse"
    with pytest.raises(ValueError, match="not an EF X"):
        cz_sandwich(with_pulses(pl), (0, 1))
    pl = copy.deepcopy(base)
    pl[0] = pl[3] = "single_qubit/5/EF/X/pulse"
    with pytest.raises(ValueError, match="not a member"):
        cz_sandwich(with_pulses(pl), (0, 1))
    pl = [base[1], base[0], base[3], base[2]] + base[4:]         # refs BETWEEN the drives
    with pytest.raises(ValueError, match="bracket"):
        cz_sandwich(with_pulses(pl), (0, 1))
    c = _drive_cfg()                                             # refs but no EF calibration
    c["two_qubit/(0, 1)/CZ/pulse"] = copy.deepcopy(base)
    with pytest.raises(ValueError, match="missing qubit/1/EF"):
        cz_sandwich(c, (0, 1))


def test_cz_sandwich_progs_compile_lockstep():
    """(X4 gate) the whole conditionality machinery compiles through the sandwich fold on a
    synthetic pair: `_sandwich_binds` puts sw/fef on the SHELF core only and the (LEAD + EF X) tail
    on both, and `_cz_cond_progs` builds BOTH cores' programs (the shelf's ef branches fold in, the
    partner's fold away against the SAME padded 3-slot table — the X3 unequal-table grid finding),
    with the sweep still bound lockstep."""
    m = SocMap(SocParams.load(_SIM2Q))
    cfg = _sandwich_cfg()
    shelf, binds, tail = _sandwich_binds(cfg, (0, 1), m)
    efd = ef_pulse(cfg, 1, m, "x").dur_batches(m, GATE_CH)
    assert shelf == 1 and tail == LEAD + efd
    assert binds[1] == {"sw": 1, "fef": units.freq_to_code(125e6, m.params)}
    assert binds[0] == {"sw": 0, "fef": 0}
    x0, dx = 1024 << 16, 65536
    progs, tables, signs, timeout = _cz_cond_progs(cfg, m, (0, 1), "freq", x0, dx,
                                                   points=3, ngates=1, shots=4)
    assert sorted(progs) == [0, 1]
    assert list(tables[0].pulses) == ["x90", "cz", "ef"] == list(tables[1].pulses)
    for q in (0, 1):
        assert str(x0) in progs[q].c_source, f"core {q} missing the lockstep x0"


# ── EFPhase / EF-X amplitude (spec 04 §2 / X4): host-pure ──


_EF_MEANS = np.array([[10.0, 0.0], [-5.0, 8.66], [-5.0, -8.66]])   # |0>/|1>/|2> IQ centroids


def _ef_clf(seed=3):
    """A 3-level ClassifierN on well-separated synthetic clusters (the test_levels_pop geometry)."""
    rng = np.random.default_rng(seed)
    return ClassifierN([_EF_MEANS[k] + 0.1 * rng.standard_normal((30, 2)) for k in range(3)])


def _levels_iq(P, shots):
    """A RAW `out` array whose per-point P(|2>) is exactly `P`: round(p·shots) shots on the |2>
    centroid, the rest on |1> (the kernels' point-major 2·npts·shots cursor layout)."""
    iq = np.zeros((len(P), shots, 2))
    for i, p in enumerate(np.clip(P, 0.0, 1.0)):
        n2 = int(round(float(p) * shots))
        iq[i, :n2] = _EF_MEANS[2]
        iq[i, n2:] = _EF_MEANS[1]
    return iq.reshape(-1)


def test_ef_phase_recovers_planted_vz(responder):
    """(X4 gate) `EFPhase` end-to-end host-pure on planted lines (the GE Phase golden probe, on the
    3-level decode): the driver layer is replaced by the shared `Responder` and each sequence's
    P(|2>) is linear in the swept phi with opposite slopes crossing at a planted phi* — the class's
    REAL two-seq k_ef_phase compile, the ClassifierN decode, the `_line_crossing` analysis and the
    `qubit/{q}/EF/x90/vz` = [phi*, phi*] write-back all run for real. The relative_phase pass
    re-centres the sweep on the STORED vz[0] (qcal's `phases + config[param]`)."""
    from riscq.cal.qubit import _phase_sweep
    cfg = _drive_cfg()
    cfg["qubit/0/EF/freq"] = 40e6
    cfg["qubit/0/EF/x90/amp"] = 0.4
    points, shots, span = 15, 100, 0.25    # RAW out = 2·npts·shots words: sized for the 16KB core RAM
    state = {"x": None, "phi_star": 0.1, "runs": 0}
    r = responder(_SIM2Q)

    @r.answer
    def _(progs, params):
        slope = 1.0 if state["runs"] == 0 else -1.0              # Y180_X90 first, then X180_Y90
        state["runs"] += 1
        P = 0.5 + slope * (state["x"] - state["phi_star"])
        return {0: {"out": _levels_iq(P, shots)}}

    state["x"] = _phase_sweep(-span, span, points)[2]            # the class's own axis
    cal = EFPhase(cfg, 0, _ef_clf(), points=points, span=span, shots=shots)
    res = cal.run(r.drv)
    assert res.ok and state["runs"] == 2 and not cal.fallback[0]
    assert cal.recovered_vz[0] == pytest.approx(0.1, abs=0.01)
    assert res.proposal["qubit/0/EF/x90/vz"] == pytest.approx([0.1, 0.1], abs=0.01)

    cfg["qubit/0/EF/x90/vz"] = [0.3, 0.25]                       # stored pair (X6Y3: asymmetric)
    state.update(x=_phase_sweep(0.3 - span, 0.3 + span, points)[2], phi_star=0.38, runs=0)
    res2 = EFPhase(cfg, 0, _ef_clf(), points=points, span=span, shots=shots,
                   relative_phase=True).run(r.drv)
    assert res2.ok
    assert res2.proposal["qubit/0/EF/x90/vz"] == pytest.approx([0.38, 0.38], abs=0.01)


def test_ef_amplitude_gate_x_knob(responder):
    """(X4 gate) EFAmplitude's `gate` knob — qcal `Amplitude(subspace='EF', gate='X')` (spec 04
    §2): the repetition guard flips to qcal's multiple-of-2 (pairs of EF π's return to |1>), the
    write path moves to `qubit/{q}/EF/x/amp`, and the n_gates=1 cosine fit recovers a planted π
    amplitude — P(|2>) generated from a planted EF Rabi rate over the ACTUAL swept codes (the driver
    layer is replaced by the shared `Responder`), maximal at the π amp."""
    from riscq.cal.base import gate_sigma
    clf = _ef_clf(5)
    cfg = _drive_cfg()
    cfg["qubit/0/EF/freq"] = 40e6
    cfg["qubit/0/EF/x/amp"] = 0.5
    cfg["qubit/0/EF/x90/amp"] = 0.4
    with pytest.raises(AssertionError, match="multiple of 2"):
        EFAmplitude(cfg, 0, clf, gate="X", n_gates=3)
    with pytest.raises(AssertionError, match="multiple of 4"):
        EFAmplitude(cfg, 0, clf, gate="X90", n_gates=2)
    with pytest.raises(AssertionError, match="gate must be"):
        EFAmplitude(cfg, 0, clf, gate="EFX")
    assert EFAmplitude(cfg, 0, clf, gate="X", n_gates=2).target_angle == pytest.approx(math.pi)

    m = SocMap(SocParams.load(_SIM2Q))
    efp = ef_pulse(cfg, 0, m, "x")
    a_star = 0.5
    rabi = math.pi / gate_sigma(m, efp, 40e6, units._amp_code(a_star))   # π EXACTLY at a* = 0.5
    points, shots = 15, 100
    r = responder(_SIM2Q)

    @r.answer
    def _(progs, params):
        a0q, daq = params[0]["a0q"], params[0]["daq"]            # the kernel's own Q16 walk
        P = [(1 - math.cos(rabi * gate_sigma(m, efp, 40e6, (a0q + i * daq) >> 16))) / 2
             for i in range(points)]
        return {0: {"out": _levels_iq(np.array(P), shots)}}

    cal = EFAmplitude(cfg, 0, clf, gate="X", n_gates=1, amp_span=(0.05, 0.95), points=points,
                      shots=shots)
    res = cal.run(r.drv)
    assert res.ok
    assert res.proposal["qubit/0/EF/x/amp"] == pytest.approx(a_star, abs=0.02)
    assert res.proposal["qubit/0/EF/rabi"] == pytest.approx(rabi, rel=0.05)
    assert "qubit/0/EF/x90/amp" not in res.proposal              # the X90 path is untouched


# ── spec 14 findings 6 + 7: the EF bracket, and the frame the 3-level classifier reads in ──


def _ef_bracket_cfg():
    """`_drive_cfg` plus everything an EF cal compiles against — including the X6Y3 q2 EF pair and a
    NON-ZERO stored demod phase (X6Y3's are −109.9°…+39.0°), so both findings are observable."""
    cfg = _drive_cfg()
    cfg["qubit/0/EF/freq"] = 40e6
    cfg["qubit/0/EF/x90/amp"] = 0.4
    cfg["qubit/0/EF/x/amp"] = 0.5
    cfg["qubit/0/EF/x90/vz"] = [-0.16289759, -0.16289759]
    cfg["readout/0/demod/phase"] = -1.918                       # the config frame the res bit uses
    return cfg


def _ef_srcs(r, cal, points, shots, p=None):
    """Run `cal` against the shared `Responder` and return the generated C of every program it
    compiled in THIS run (one entry per program the class sets up)."""
    P = np.full(points, 0.5) if p is None else p
    r.answer(lambda progs, params: {0: {"out": _levels_iq(P, shots)}})
    seen = len(r.sources)
    cal.run(r.drv)
    return r.sources[seen:]


def test_ef_kernels_play_the_calibrated_ef_vz_bracket(responder):
    """(spec 14 finding 6) qcal's EF X90 is virtualz(vz0) · FAST_DRAG · virtualz(vz1), so every EF
    gate in its EF Amplitude/Frequency/Phase circuits plays the calibrated pair — the pair EFPhase
    writes to `qubit/{q}/EF/x90/vz`. The EF kernels used to play the train in a fresh 0 frame and
    never consume it. Assert on the C the classes actually compile: the seated bracket words
    (`ef_vz`) are emitted where the gate carries the pair, and NOT where qcal has none to play —
    the bare EF X, and the two crossing sequences whose sweep REPLACES the pair."""
    from riscq.cal import EFFrequency
    from riscq.cal.base import ef_vz
    cfg = _ef_bracket_cfg()
    b = ef_vz(cfg, 0)
    points, shots = 7, 16
    r = responder(_SIM2Q)

    # EF Rabi (EFAmplitude, gate='X90'): every gate of the train fires at frame + evz0 and steps the
    # frame by evzsum, exactly as the GE k_rabi train does
    src, = _ef_srcs(r, EFAmplitude(cfg, 0, _ef_clf(), n_gates=4, points=points,
                                   shots=shots), points, shots)
    assert str(b["evz0"]) in src and str(b["evzsum"]) in src

    # ... and gate='X' binds the pair of the gate ACTUALLY played — the bare EF X has none
    src, = _ef_srcs(r, EFAmplitude(cfg, 0, _ef_clf(), gate="X", n_gates=2, points=points,
                                   shots=shots), points, shots)
    assert str(b["evz0"]) not in src and str(b["evzsum"]) not in src

    # EF Ramsey (EFFrequency): the swept detuning is the Rz BETWEEN the two EF X90s, so it COMPOSES
    # with each gate's bracket — the 2nd fires at evzsum + phi + evz0
    fringe = 0.5 + 0.4 * np.cos(np.arange(points) * 0.7)
    srcs = _ef_srcs(r, EFFrequency(cfg, 0, _ef_clf(), points=points, shots=shots),
                    points, shots, p=fringe)
    assert all(str(b["evz0"]) in s and str(b["evzsum"]) in s for s in srcs)

    # EFPhase's two crossing sequences are EXEMPT: there the swept phi IS the pair (qcal writes one
    # crossing to both slots), so the stored bracket must not be composed on top of it
    for s in _ef_srcs(r, EFPhase(cfg, 0, _ef_clf(), points=points, shots=shots),
                      points, shots):
        assert str(b["evz0"]) not in s and str(b["evzsum"]) not in s

    # EFPhase(gate='X') is the opposite case: the two EF X90s keep their bracket and only the EF X's
    # own axis is swept
    src, = _ef_srcs(r, EFPhase(cfg, 0, _ef_clf(), gate="X", points=points, shots=shots),
                    points, shots)
    assert str(b["evz0"]) in src and str(b["evzsum"]) in src


def test_ef_cals_capture_in_the_classifiers_zero_demod_frame(responder, monkeypatch):
    """(spec 14 finding 7) `ClassifierN`'s training captures are deliberately zero-frame
    (`_rawiq_prog`/`_ef_prep_prog` bake `phase=0.0`), so every consumer that classifies host-side
    must capture in that same frame or its IQ clouds arrive rotated by the stored demod phase
    relative to the classifier's means — 0 on the co-sim configs, −109.9°…+39.0° on X6Y3.

    The res-bit cals are the deliberate opposite: there the stored phase IS the hardware
    discrimination knob, so they must keep passing the config frame (`phase=None`)."""
    from riscq.cal import Amplitude, EFFrequency, Phase
    from riscq.cal import base as cal_base
    from riscq.cal import qubit as cal_qubit
    cfg = _ef_bracket_cfg()
    points, shots = 7, 16
    seen = []
    real = cal_base.readout_tables

    def recorder(cfg_, q, m_, phase=None, win=None):
        seen.append(phase)
        return real(cfg_, q, m_, phase=phase, win=win)

    monkeypatch.setattr(cal_qubit, "readout_tables", recorder)
    r = responder(_SIM2Q)

    fringe = 0.5 + 0.4 * np.cos(np.arange(points) * 0.7)
    for cal, p in ((EFAmplitude(cfg, 0, _ef_clf(), points=points, shots=shots), None),
                   (EFFrequency(cfg, 0, _ef_clf(), points=points, shots=shots), fringe),
                   (EFPhase(cfg, 0, _ef_clf(), points=points, shots=shots), None),
                   (EFPhase(cfg, 0, _ef_clf(), gate="X", points=points, shots=shots), None)):
        seen.clear()
        _ef_srcs(r, cal, points, shots, p=p)
        assert seen and set(seen) == {0.0}, f"{type(cal).__name__} captured at {set(seen)}"

    # the res-bit consumers must NOT move
    r.answer(lambda progs, params: {0: {"out": np.zeros(points, dtype=int)}})
    for cal in (Amplitude(cfg, 0, points=points, shots=shots),
                Phase(cfg, 0, points=points, shots=shots)):
        seen.clear()
        cal.run(r.drv)
        assert seen and set(seen) == {None}, f"{type(cal).__name__} captured at {set(seen)}"
