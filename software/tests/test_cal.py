"""M4b acceptance: the calibration suite against TwoLevelModel ground truth (spec 07 M4 / spec 08 B5).

Every calibration batches its whole sweep into ONE run (spec 08): the host preloads the swept knobs
as point columns, the kernel walks them on a fixed grid whose idle head is the T1 relax reset, and
the host fits the self-normalised counts population P = counts/shots (no |0> reference / projection —
counts are self-normalised) against the PROJECTIVE TwoLevelModel (collapse=True), recovering the
planted Rabi rate (<1%), detuning, and T1/T2. Readout cals keep host-side analysis on raw IQ clusters.
The Calibration_X6Y3 sequence runs top-to-bottom and moves a deliberately-detuned Config toward the
model's ground truth.
"""

import math

import numpy as np
import pytest

from riscq.cal import (Amplitude, Classifier, Config, Fidelity, Frequency, Phase,
                       ReadoutCalibration, ReadoutFidelity, Separation, T1, T2, calibration_x6y3)
from riscq.cal.base import bind_params, gate_sigma, GATE_ENV
from riscq.pulses import Pulse, units

F_GE = 50e6                              # planted qubit frequency (freq_code 2048)


# ── host unit test: the readout classifier ──

def test_classifier_separates_and_confuses():
    rng = np.random.default_rng(0)
    iq0 = rng.normal([1000, 0], 120, (40, 2))     # |0> cluster on +real
    iq1 = rng.normal([-1000, 0], 120, (40, 2))    # |1> cluster on −real (π flip)
    clf = Classifier(iq0, iq1)
    assert clf.separation > 5                      # well separated
    conf = clf.confusion()
    assert conf[0, 0] > 0.95 and conf[1, 1] > 0.95   # near-perfect assignment
    assert np.allclose(conf.sum(1), 1.0)


# ── cosim helpers ──

def _readout_freq(m):
    """The physical readout frequency whose demod code = the model tone code 2048."""
    return units.demod_code_to_freq(2048, m.params)


def _cfg(m, qfreq, x90_amp=0.5, dur=40):
    bind_params(m)
    c = Config()
    c["qubit/0/freq"] = float(qfreq)
    c["qubit/0/x90/amp"] = float(x90_amp)
    c["qubit/0/T1"] = 120
    c["readout/0/freq"] = float(_readout_freq(m))
    c["readout/0/dur"] = int(dur)
    return c


def _sig_max(m, carrier):
    x90 = Pulse(GATE_ENV, freq_hz=carrier, amp=0.5)
    return gate_sigma(m, x90, carrier, units.AMP_SCALE - 600)


def _true_x90_amp(m, rabi):
    """The X90 amplitude (float) that rotates by π/2 for the planted rabi: θ = rabi·G·amp_code."""
    g = _sig_max(m, F_GE) / (units.AMP_SCALE - 600)     # sig per amp code (linear)
    return (math.pi / 2) / (rabi * g) / units.AMP_SCALE


def _model(rabi, f_ge=F_GE, t1=300, t2=450, noise=0.0, seed=0, collapse=False):
    return dict(kind="twolevel", core=0, rabi_rad_per_amp=float(rabi), readout_code=2048,
                readout_amp=20000.0, readout_phase=0.0, f_ge=float(f_ge),
                t1=int(t1), t2=int(t2), noise_scale=float(noise), noise_seed=int(seed),
                collapse=bool(collapse))


def _rabi_pi(m):
    """The Rabi rate that makes the x180 amp (0.99) a π rotation."""
    return float(math.pi / gate_sigma(m, Pulse(GATE_ENV, freq_hz=F_GE, amp=0.99), F_GE,
                                      units._amp_code(0.99)))


@pytest.fixture(scope="session")
def demod_phase(cosim):
    """The real demod discrimination phase, measured ONCE by ReadoutCalibration on the projective
    model (spec 08 §2.1): it captures |0>/|1> clusters and proposes the demod-carrier phase that
    lands |0> on +real (sign(sumR) then discriminates). All counts-mode tests bake it into their
    readout tables (cfg["readout/0/demod_phase"]) — the real compile-time write_slot("phase") path.
    The pipeline angle is model-rabi/T-independent (readout_phase pinned 0), so one measurement
    serves every counts cal."""
    drv, m = cosim
    bind_params(m)
    drv.sim.set_model(_model(_rabi_pi(m), t1=2000, t2=3000, noise=300.0, seed=7, collapse=True))
    cfg = _cfg(m, F_GE)
    r = ReadoutCalibration(cfg, 0, shots=24).run(drv)
    drv.sim.set_model({"kind": "zero"})
    assert r.ok, f"ReadoutCalibration could not separate clusters (sep={r.data['separation']:.2f})"
    return float(r.proposal["readout/0/demod_phase"])


@pytest.fixture(autouse=True)
def _zero_after(request):
    yield
    if request.config.getoption("--cosim"):
        request.getfixturevalue("cosim")[0].sim.set_model({"kind": "zero"})


# ── Amplitude: recover the planted Rabi rate within 1% ──

@pytest.mark.cosim
def test_amplitude_recovers_rabi(cosim, demod_phase):
    drv, m = cosim
    bind_params(m)
    rabi = float(4 * math.pi / _sig_max(m, F_GE))          # ~2 Rabi periods (counts sub-1% budget)
    drv.sim.set_model(_model(rabi, t1=200, t2=2000, noise=300.0, seed=1, collapse=True))
    cfg = _cfg(m, F_GE)
    cfg["readout/0/demod_phase"] = demod_phase
    r = Amplitude(cfg, 0, n_gates=1).run(drv)
    recovered = r.proposal["qubit/0/rabi"]
    print(f"\n[amplitude] recovered={recovered:.6e} planted={rabi:.6e} ratio={recovered/rabi:.4f}")
    assert r.ok
    assert abs(recovered / rabi - 1) < 0.01, f"recovered {recovered} vs planted {rabi}"


# ── Frequency: recover a planted detuning ──

@pytest.mark.cosim
def test_frequency_recovers_detuning(cosim, demod_phase):
    drv, m = cosim
    bind_params(m)
    rabi = float((math.pi / 2) / gate_sigma(m, Pulse(GATE_ENV, freq_hz=F_GE, amp=0.5),
                                            F_GE, units._amp_code(0.5)))
    d0_code = 60
    drive = units.code_to_freq(units._freq_code(F_GE, m.params) + d0_code, m.params)   # f_ge + δ0
    drv.sim.set_model(_model(rabi, t1=200, t2=3000, noise=300.0, seed=3, collapse=True))
    cfg = _cfg(m, drive, x90_amp=0.5)
    cfg["readout/0/demod_phase"] = demod_phase
    freq = Frequency(cfg, 0, detune_code=200, n_detune=4, dt=4, points=14)
    r = freq.run(drv)
    print(f"\n[frequency] applied={r.data['applied']} signed_obs={r.data['obs']} "
          f"recovered_code={freq.recovered_detuning_code} planted={d0_code}")
    assert r.ok
    assert abs(freq.recovered_detuning_code - d0_code) < 25, "detuning code not recovered"
    r.apply()
    err_before = abs(drive - F_GE)
    err_after = abs(cfg["qubit/0/freq"] - F_GE)
    print(f"[frequency] |freq−f_ge| before={err_before:.3g} after={err_after:.3g}")
    assert err_after < 0.3 * err_before, "config frequency did not move toward f_ge"


# ── T1 / T2: recover planted decay constants ──

@pytest.mark.cosim
def test_t1_recovers_decay(cosim, demod_phase):
    drv, m = cosim
    bind_params(m)
    t1 = 120
    drv.sim.set_model(_model(_rabi_pi(m), t1=t1, t2=2000, noise=300.0, seed=4, collapse=True))
    cfg = _cfg(m, F_GE, x90_amp=0.495)                            # x180 = 2×0.495 ≈ 0.99
    cfg["readout/0/demod_phase"] = demod_phase
    r = T1(cfg, 0, points=9).run(drv)
    print(f"\n[t1] fit ok={r.fit.ok} tau={r.fit.value} amp={r.fit.params.get('amp')} "
          f"delays={r.data['x'].tolist()} P={np.round(r.data['y'],3).tolist()}")
    assert r.ok
    assert 0.8 * t1 < r.proposal["qubit/0/T1"] < 1.2 * t1


@pytest.mark.cosim
def test_t2_recovers_decay(cosim, demod_phase):
    drv, m = cosim
    bind_params(m)
    t2 = 200
    rabi = float((math.pi / 2) / gate_sigma(m, Pulse(GATE_ENV, freq_hz=F_GE, amp=0.5),
                                            F_GE, units._amp_code(0.5)))
    drv.sim.set_model(_model(rabi, t1=400, t2=t2, noise=300.0, seed=5, collapse=True))
    cfg = _cfg(m, F_GE, x90_amp=0.5)
    cfg["readout/0/demod_phase"] = demod_phase
    r = T2(cfg, 0, detune_code=70, points=15, dt=16).run(drv)     # dt=16 → covers > t2 (decay visible)
    print(f"\n[t2] fit ok={r.fit.ok} freq={r.fit.value} tau={r.fit.params.get('tau')} "
          f"waits={r.data['x'].tolist()} P={np.round(r.data['y'],3).tolist()}")
    assert r.ok
    assert 0.8 * t2 < r.proposal["qubit/0/T2"] < 1.25 * t2


# ── Phase: counts-mode axis calibration recovers ≈0 (the model plants no axis error) ──

@pytest.mark.cosim
def test_phase_recovers_zero(cosim, demod_phase):
    drv, m = cosim
    bind_params(m)
    rabi = float((math.pi / 2) / gate_sigma(m, Pulse(GATE_ENV, freq_hz=F_GE, amp=0.5),
                                            F_GE, units._amp_code(0.5)))
    drv.sim.set_model(_model(rabi, t1=200, t2=3000, noise=300.0, seed=6, collapse=True))
    cfg = _cfg(m, F_GE, x90_amp=0.5)
    cfg["readout/0/demod_phase"] = demod_phase
    r = Phase(cfg, 0, points=13).run(drv)
    applied = r.proposal["qubit/0/x90/phase"]
    print(f"\n[phase] fit ok={r.fit.ok} fit_phase={r.fit.params.get('phase'):.3f} "
          f"applied_x90_phase={applied:.3f} P={np.round(r.data['y'],3).tolist()}")
    assert r.ok
    assert abs(applied) < 0.25, f"Phase folded a spurious axis error: {applied:.3f} rad"


# ── acquire_shots: ReadoutCalibration captures bimodal |0>/|1> clusters (raw mode) ──

@pytest.mark.cosim
def test_acquire_shots_chunks(cosim):
    drv, m = cosim
    bind_params(m)
    drv.sim.set_model(_model(_rabi_pi(m), t1=2000, t2=3000, noise=400.0, seed=2, collapse=True))
    cfg = _cfg(m, F_GE, x90_amp=0.495)
    r = ReadoutCalibration(cfg, 0, shots=16).run(drv)   # per-prep RAW reruns: |0>, |1>
    assert r.data["iq0"].shape == (16, 2) and r.data["iq1"].shape == (16, 2)
    assert r.ok and r.data["separation"] > 1.0


# ── readout freq / fidelity / window cals on the batched path (spec 08 B5 pass 2) ──

@pytest.mark.cosim
def test_separation_finds_peak(cosim):
    """Separation batches the matched-pair VNA (iqsum) and lands readout/q/freq on the model's tone.
    The projective model synthesizes the ADC tone at readout_code=2048; the demod matches at the DAC
    code whose 4x equals it, so the |z| peak (and the recovered freq) sit on the readout tone. iqsum
    magnitude needs no baked demod phase."""
    drv, m = cosim
    bind_params(m)
    drv.sim.set_model(_model(0.0, noise=300.0, seed=8, collapse=True))
    cfg = _cfg(m, F_GE)                                     # readout/0/freq = demod_code_to_freq(2048)
    r = Separation(cfg, 0, span_code=200, points=7, shots=8, relax=200).run(drv)
    tone = units.demod_code_to_freq(2048, m.params)
    mag = r.data["y"]
    print(f"\n[separation] ok={r.ok} recovered={r.proposal['readout/0/freq']:.6e} tone={tone:.6e} "
          f"argmax={int(np.argmax(mag))} mag={np.round(mag).astype(int).tolist()}")
    assert r.ok
    assert abs(r.proposal["readout/0/freq"] - tone) < abs(units.code_to_freq(60, m.params)), \
        "Separation did not land on the readout tone"


@pytest.mark.cosim
def test_readout_fidelity(cosim):
    """ReadoutFidelity captures the |0>/|1> clusters in one batched RAW run (no baked demod phase —
    the classifier is rotation-invariant) and reports a high assignment fidelity for the well
    separated projective clusters."""
    drv, m = cosim
    bind_params(m)
    drv.sim.set_model(_model(_rabi_pi(m), t1=2000, t2=3000, noise=300.0, seed=9, collapse=True))
    cfg = _cfg(m, F_GE, x90_amp=0.495)                     # x180 = 0.99 = pi
    r = ReadoutFidelity(cfg, 0, shots=16, relax=4000).run(drv)   # relax >> SEP so |1> resets between shots
    fid = r.proposal["readout/0/fidelity"]
    print(f"\n[readout-fidelity] ok={r.ok} fidelity={fid:.3f} sep={r.data['separation']:.2f}\n"
          f"{np.round(r.data['confusion'], 3)}")
    assert r.ok
    assert fid > 0.8, f"readout fidelity too low: {fid:.3f}"


@pytest.mark.cosim
def test_fidelity_picks_window(cosim):
    """Fidelity retunes the demod window via write_slot + rerun (no recompile, spec 08 §4) and picks
    the window that separates best. It compiles ONE program at the longest candidate window, then
    sweeps `dur` with two host writes each; the picked window is the argmax of cluster separation."""
    drv, m = cosim
    bind_params(m)
    drv.sim.set_model(_model(_rabi_pi(m), t1=2000, t2=3000, noise=15000.0, seed=10, collapse=True))
    cfg = _cfg(m, F_GE, x90_amp=0.495)
    durs = (16, 64)
    r = Fidelity(cfg, 0, durs=durs, shots=12, relax=4000).run(drv)
    seps = r.data["y"]
    best = int(np.argmax(seps))
    print(f"\n[fidelity] ok={r.ok} durs={r.data['x'].astype(int).tolist()} "
          f"seps={np.round(seps, 2).tolist()} picked={r.proposal['readout/0/dur']}")
    assert r.ok
    assert r.proposal["readout/0/dur"] == durs[best] and durs[best] in durs
    assert seps[best] >= seps[0], "picked window separates worse than the shortest"


# ── §7 cost accounting: a batched cal is O(1) client seam ops in one run ──

@pytest.mark.cosim
def test_cost_accounting_amplitude(cosim):
    """spec 08 §7: a batched Amplitude cal costs O(1) client seam ops in ONE run (a few block writes
    in, one poll, one block read out) — not the ~50k of the old one-run-per-point host loop. The op
    count is independent of npts×shots (block writes/reads move O(n) bytes in O(1) ops), so this tiny
    5×8 sweep counts the same as a full 21×160 one. CountingDriver wraps the 4 seam ops (spec 08 B3)."""
    from test_rerun import CountingDriver
    drv, m = cosim
    bind_params(m)
    drv.sim.set_model(_model(_rabi_pi(m), t1=200, t2=2000, noise=300.0, seed=12, collapse=True))
    cfg = _cfg(m, F_GE)
    cfg["readout/0/demod_phase"] = math.pi / 2          # any phase; the op count is physics-independent
    cd = CountingDriver(drv)
    Amplitude(cfg, 0, n_gates=1, points=5, shots=8).run(cd)
    print(f"\n[cost] one batched Amplitude cal = {cd.ops} client seam ops in 1 run "
          f"(setup+rerun; the rerun batch alone is 10 seam ops — test_rerun.py, B3)")
    assert cd.ops < 60, f"batched Amplitude cost {cd.ops} ops (expected O(1), a few dozen), not ~50k"


# ── the Calibration_X6Y3 sequence improves a deliberately-detuned Config ──

@pytest.mark.cosim
def test_x6y3_improves_detuned_config(cosim):
    drv, m = cosim
    bind_params(m)
    rabi = float(3 * math.pi / _sig_max(m, F_GE))
    # projective model (spec 08 §2.4): t1=300 relaxes to |0> in the grid's idle head (relax=1600 ≫ t1)
    # yet |1> survives the SEP before readout, so the whole X6Y3 chain — readout-cluster cals AND the
    # counts-mode qubit cals — runs on ONE model. High-SNR clusters (noise=300 vs readout_amp=20000).
    drv.sim.set_model(_model(rabi, t1=300, t2=450, noise=300.0, seed=11, collapse=True))

    # deliberately-detuned config: qubit freq off by d0, X90 amp wrong (0.5 vs the true π/2 amp)
    d0_code = 60
    drive = units.code_to_freq(units._freq_code(F_GE, m.params) + d0_code, m.params)
    cfg = _cfg(m, drive, x90_amp=0.5)
    true_x90_amp = _true_x90_amp(m, rabi)

    freq_before = abs(cfg["qubit/0/freq"] - F_GE)
    amp_before = abs(cfg["qubit/0/x90/amp"] - true_x90_amp)
    results = calibration_x6y3(cfg, 0, drv, verbose=True)
    assert all(r.label for r in results) and len(results) == 8
    freq_after = abs(cfg["qubit/0/freq"] - F_GE)
    amp_after = abs(cfg["qubit/0/x90/amp"] - true_x90_amp)
    print(f"\n[x6y3] freq err {freq_before:.3g}→{freq_after:.3g}  "
          f"x90amp {cfg['qubit/0/x90/amp']:.4f} (true≈{true_x90_amp:.4f}) "
          f"err {amp_before:.4f}→{amp_after:.4f}")
    assert freq_after < 0.4 * freq_before, "X6Y3 did not improve the qubit frequency"
    assert amp_after < 0.5 * amp_before, "X6Y3 did not improve the X90 amplitude"

