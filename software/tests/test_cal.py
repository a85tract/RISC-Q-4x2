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
from pathlib import Path

import numpy as np
import pytest

from riscq.cal import (Amplitude, Classifier, Config, Fidelity, Frequency, Phase,
                       Punchout, ReadoutCalibration, ReadoutFidelity, Separation, T1, T2, Window,
                       calibration_x6y3)
from riscq.cal import fits
from riscq.cal.base import batch_timeout
from riscq.cal.readout import _ro_amp_prog
from riscq import run as rq
from riscq.cal.base import batches, gate_sigma, socmap, x90_vz, GATE_ENV
from riscq.map import SocMap, SocParams, pack16
from riscq.pulses import Pulse, units

F_GE = 50e6                              # planted qubit frequency (freq_code 2048)
RELAX = 1600                             # co-sim relax head (batches) — the Config carries it in SECONDS
QCAL_COSIM_YAML = Path(__file__).with_name("qcal_cosim.yaml")   # the co-sim-scaled qcal tree (spec 13 Q6)
RO_RELAX, RO_T1 = 3200, 600              # the readout cals' budget: relax ≫ T1 ≫ SEP (= LEAD = 96),
#                                          so the |1> prep both SURVIVES to the window and RESETS after it


# ── host unit test: the readout classifier ──

def test_classifier_separates_and_confuses():
    rng = np.random.default_rng(0)
    iq0 = rng.normal([1000, 0], 120, (40, 2))     # |0> cluster on +real
    iq1 = rng.normal([-1000, 0], 120, (40, 2))    # |1> cluster on −real (π flip)
    clf = Classifier(iq0, iq1)
    # qcal's SNR (spec 13 §2): ‖Δmeans‖ / (2σ₀ + 2σ₁) = 2000 / (4 · 120) ≈ 4.2 — a quarter of the
    # plain distance/σ number this used to report (same clusters, qcal's denominator).
    assert clf.separation == pytest.approx(2000 / (4 * 120), rel=0.1)
    conf = clf.confusion()
    assert conf[0, 0] > 0.95 and conf[1, 1] > 0.95   # near-perfect assignment
    assert np.allclose(conf.sum(1), 1.0)


# ── host unit tests: qcal's Amplitude guard + the X90 frame bracket (spec 13 Q4) ──

def test_amplitude_enforces_qcals_n_gates_guard():
    """qcal single_qubit.py:154-158: an amplified sweep has to land back on |0>, so a repeated-X90
    train is a multiple of 4 (each X90 is a quarter period) and a repeated-X train a multiple of 2.
    n_gates=1 is unconstrained."""
    cfg = Config()
    with pytest.raises(AssertionError):
        Amplitude(cfg, 0, n_gates=2)                   # X90: 2 · π/2 = π, does NOT return to |0>
    with pytest.raises(AssertionError):
        Amplitude(cfg, 0, gate="X", n_gates=3)         # X: 3 · π = π
    with pytest.raises(AssertionError):
        Amplitude(cfg, 0, gate="Y90")                  # qcal has exactly two native gates here
    Amplitude(cfg, 0, n_gates=4)                       # both fine
    Amplitude(cfg, 0, gate="X", n_gates=2)


def test_x90_frame_bracket_words():
    """spec 13 Q4 — the bracket every X90 play now carries: `set_phase_offset(frame + vz0); play;
    frame += vz0 + vz1`. The kernels bind the pair as two SEATED phase words (vz0 and the frame step
    vz0 + vz1); the pair is carried as TWO values because X6Y3's q6 pair is asymmetric. A config with
    no pair — every co-sim one — binds 0/0, so the bracket writes the phase offset init_pulse_params
    already left at 0: a no-op."""
    cfg = Config()
    assert x90_vz(cfg, 0) == {"vz0": 0, "vzsum": 0}
    cfg["qubit/0/x90/vz"] = [0.0429, 0.0080]                       # X6Y3 q6 (asymmetric)
    c0, c1 = units._phase_code(0.0429), units._phase_code(0.0080)
    assert x90_vz(cfg, 0) == {"vz0": pack16(c0), "vzsum": pack16(c0 + c1)}


def test_amplitude_n1_rejects_a_fit_outside_the_swept_span():
    """qcal's in_range guard for the n_gates=1 amplitude fit (single_qubit.py:273-279): the fitted
    π/2 amp code must lie INSIDE the swept codes, exactly as the n_gates>1 vertex already must. The
    old `0 < a_star < AMP_SCALE` accepted any on-scale extrapolation: a clean cosine over a narrow
    high-amp span implies a π/2 amp far BELOW the sweep — a good fit, but not a measurement."""
    rng = np.random.default_rng(2)
    sigma_per_code = 3.7e-4                             # gate_sigma is linear in the amp code
    a_true = 2000.0                                     # the implied π/2 amp code
    rabi = (math.pi / 2) / (a_true * sigma_per_code)

    def sweep(x0, x1):
        xs = np.linspace(x0, x1, 21)
        sig = sigma_per_code * xs
        return xs, sig, (1 - np.cos(rabi * sig)) / 2 + rng.normal(0, 0.01, xs.size)

    amp = Amplitude(Config(), 0)                        # gate X90, n_gates=1 (target π/2)
    fq, _, a_star, ok = amp._fit_single_gate(*sweep(600, 19000))    # a_true inside the sweep
    assert fq.ok and ok and abs(a_star - a_true) < 100
    fq, _, a_star, ok = amp._fit_single_gate(*sweep(8000, 19000))   # a_true OUTSIDE the sweep
    assert fq.ok and not ok, "a fit whose π/2 amp lies outside the sweep must be rejected"
    assert 0 < a_star < units.AMP_SCALE, "the old on-scale guard would have accepted it"


def test_frequency_qcal_signature_maps_to_the_shorthand():
    """spec 13 §3's promised signature: `detunings` (signed Hz, each a host rerun) overrides the
    ±k·detune ladder, `t_max` (seconds) spreads `points` waits over [0, t_max]. The Hz values are
    code-exact multiples (code_to_freq of an int) so the ladder (±k·code(d)) and the explicit list
    (code(±k·d)) land on identical codes — at arbitrary Hz the two round independently by ±1 LSB."""
    m = SocMap(SocParams.load(Path(__file__).resolve().parents[1] / "configs" / "sim-2q.json"))
    cfg = Config()
    d = units.code_to_freq(200, m.params)
    ladder = Frequency(cfg, 0, detune=d, n_detune=4)
    explicit = Frequency(cfg, 0, detunings=(-2 * d, -d, d, 2 * d))
    assert sorted(explicit._d_codes(m.params)) == sorted(ladder._d_codes(m.params))
    t0, dt = Frequency(cfg, 0, t_max=1e-6, points=30)._wait_grid(m)
    assert t0 == 0 and dt == round(1e-6 / 29 * m.params.dsp_freq_hz) > 0
    assert Frequency(cfg, 0, t0=80e-9, dt=40e-9)._wait_grid(m) == \
        (batches(80e-9, m), batches(40e-9, m))          # the shorthand stays the shorthand


def test_socmap_reads_the_board_driver():
    """The cal suite must run on hardware: RemoteDriver has no `.sim` (spec 10 §6) — its SocParams
    JSON is on `.board`. socmap takes either seam."""
    text = (Path(__file__).resolve().parents[1] / "configs" / "sim-2q.json").read_text()

    class _Src:
        def __init__(self, t): self._t = t
        def get_params(self): return self._t

    class _Board:
        def __init__(self, t): self.board = _Src(t)

    class _Sim:
        def __init__(self, t): self.sim = _Src(t)

    assert socmap(_Board(text)).params.name == "sim-2q"
    assert socmap(_Sim(text)).params.qubit_num == 2


# ── cosim helpers ──

def _readout_freq(m):
    """The physical readout frequency whose demod code = the model tone code 2048."""
    return units.demod_code_to_freq(2048, m.params)


def _s(n_batches, m):
    """batches → seconds: the co-sim's own short times, expressed in the Config's physical units
    (spec 13 §2). The kernels see exactly the batch counts the pre-units tests used."""
    return units.ns(n_batches, m.params) * 1e-9


def _cfg(m, qfreq, x90_amp=0.5, dur=40, drive=None, relax=RELAX):
    """The co-sim Config, in PHYSICAL units (spec 13 §3). `dur` is the demod WINDOW (batches here,
    seconds in the tree) and `drive` the readout-drive length — the projective model only emits its
    tone while the drive is on, so the drive must cover the window (default: window + 16)."""
    c = Config()
    c["qubit/0/freq"] = float(qfreq)
    c["qubit/0/x90/amp"] = float(x90_amp)
    c["qubit/0/T1"] = _s(120, m)
    c["readout/0/freq"] = float(_readout_freq(m))
    c["readout/0/amp"] = 0.5
    c["readout/0/dur"] = _s(dur + 16 if drive is None else drive, m)
    c["readout/0/demod/dur"] = _s(dur, m)
    c["reset/relax"] = _s(relax, m)
    return c


def _sig_max(m, carrier):
    x90 = Pulse(GATE_ENV, freq_hz=carrier, amp=0.5)
    return gate_sigma(m, x90, carrier, units.AMP_SCALE - 600)


def _true_x90_amp(m, rabi):
    """The X90 amplitude (float) that rotates by π/2 for the planted rabi: θ = rabi·G·amp_code."""
    g = _sig_max(m, F_GE) / (units.AMP_SCALE - 600)     # sig per amp code (linear)
    return (math.pi / 2) / (rabi * g) / units.AMP_SCALE


def _model(rabi, f_ge=F_GE, t1=300, t2=450, noise=0.0, seed=0, collapse=False, **kw):
    """The planted TwoLevelModel. `kw` carries the optional dispersive readout (f_r/kappa/chi, spec
    13 Q2 — off by default, so every non-dispersive test sees exactly the old flat tone) and any
    readout_amp override."""
    return dict(kind="twolevel", core=0, rabi_rad_per_amp=float(rabi), readout_code=2048,
                readout_phase=0.0, f_ge=float(f_ge),
                t1=int(t1), t2=int(t2), noise_scale=float(noise), noise_seed=int(seed),
                collapse=bool(collapse), **{"readout_amp": 20000.0, **kw})


def _dispersive(m, chi_code=60, kappa_code=170):
    """A dispersive readout planted on the config's readout frequency (spec 13 Q2): the resonator sits
    at f_r = readout/0/freq with linewidth κ, pulled to f_r ± χ by |0>/|1>. With 2χ/κ ≈ 0.7 the |0>
    magnitude peaks at f_r + χ while the two-state SEPARATION peaks at f_r — two DIFFERENT frequencies,
    which is what makes Separation's acceptance gate non-vacuous."""
    return dict(f_r=_readout_freq(m), chi=units.code_to_freq(chi_code, m.params),
                kappa=units.code_to_freq(kappa_code, m.params))


def _calibrate(drv, m, cfg, spec, shots=16):
    """Bake the demod discrimination phase into `cfg` (what every counts-mode readout needs), measured
    by ReadoutCalibration on a LOW-NOISE copy of `spec` — the phase is a property of the readout chain,
    not of its noise — then restore `spec`. Returns the (clean) trained classifier."""
    drv.sim.set_model({**spec, "noise_scale": 200.0})
    r = ReadoutCalibration(cfg, 0, shots=shots).run(drv)
    assert r.ok, f"ReadoutCalibration could not separate clusters (sep={r.data[0]['separation']:.2f})"
    cfg["readout/0/demod/phase"] = float(r.proposal["readout/0/demod/phase"])
    drv.sim.set_model(spec)
    return r.fit[0]


def _rabi_pi(m):
    """The Rabi rate that makes a π rotation out of the default |1> prep — X90·X90 at x90_amp=0.495
    (spec 13 §4), i.e. the same drive integral as one GATE_ENV pulse at amp 0.99."""
    return float(math.pi / gate_sigma(m, Pulse(GATE_ENV, freq_hz=F_GE, amp=0.99), F_GE,
                                      units._amp_code(0.99)))


@pytest.fixture(scope="session")
def demod_phase(cosim):
    """The real demod discrimination phase, measured ONCE by ReadoutCalibration on the projective
    model (spec 08 §2.1): it captures |0>/|1> clusters and proposes the demod-carrier phase that
    lands |0> on +real (sign(sumR) then discriminates). All counts-mode tests bake it into their
    readout tables (cfg["readout/0/demod/phase"]) — the real compile-time write_slot("phase") path.
    The pipeline angle is model-rabi/T-independent (readout_phase pinned 0), so one measurement
    serves every counts cal."""
    drv, m = cosim
    # The readout budget the qcal cluster SNR needs (its 2σ₀ + 2σ₁ denominator charges for every
    # un-prepped shot in the |1> cluster, where the old distance/σ number barely noticed): the relax head
    # must RESET the qubit (3200 ≈ 5·T1) and T1 must let |1> SURVIVE the pulse-end → readout separation
    # (600 ≫ SEP = LEAD = 96). t1 = 2000 with a 1600 relax reset only half the shots; t1 = 300 with the
    # same relax let a third of |1> decay before the window.
    drv.sim.set_model(_model(_rabi_pi(m), t1=RO_T1, t2=3000, noise=300.0, seed=7, collapse=True))
    cfg = _cfg(m, F_GE, relax=RO_RELAX)
    r = ReadoutCalibration(cfg, 0, shots=24).run(drv)
    drv.sim.set_model({"kind": "zero"})
    assert r.ok, f"ReadoutCalibration could not separate clusters (sep={r.data[0]['separation']:.2f})"
    return float(r.proposal["readout/0/demod/phase"])


@pytest.fixture(autouse=True)
def _zero_after(request):
    yield
    if request.config.getoption("--cosim"):
        request.getfixturevalue("cosim")[0].sim.set_model({"kind": "zero"})


# ── Amplitude: recover the planted Rabi rate within 1% ──

@pytest.mark.cosim
def test_amplitude_recovers_rabi(cosim, demod_phase):
    drv, m = cosim
    rabi = float(4 * math.pi / _sig_max(m, F_GE))          # ~2 Rabi periods (counts sub-1% budget)
    drv.sim.set_model(_model(rabi, t1=200, t2=2000, noise=300.0, seed=1, collapse=True))
    cfg = _cfg(m, F_GE)
    cfg["readout/0/demod/phase"] = demod_phase
    r = Amplitude(cfg, 0, n_gates=1).run(drv)
    recovered = r.proposal["qubit/0/rabi"]
    # spec 13 Q4 — THE cross-check, on the real sweep: we take the amplitude through a physical Rabi
    # RATE (fit P against the drive integral σ), qcal reads it off the period of the same cosine in
    # the amplitude axis (amp = period_frac / f_fit; 0.25 for an X90). σ is linear in the amp code, so
    # the two routes must land on the same amplitude — ours additionally recovers the rate.
    fq = fits.fit_cosine(r.data[0]["x"], r.data[0]["y"])   # the cosine in the AMPLITUDE-CODE axis
    a_qcal = (0.25 / fq.value) / units.AMP_SCALE
    ours = r.proposal["qubit/0/x90/amp"]
    print(f"\n[amplitude] recovered={recovered:.6e} planted={rabi:.6e} ratio={recovered/rabi:.4f}\n"
          f"  x90 amp: rate route={ours:.5f}  qcal 0.25/f_fit={a_qcal:.5f}  "
          f"ratio={ours/a_qcal:.5f}")
    assert r.ok
    assert abs(recovered / rabi - 1) < 0.01, f"recovered {recovered} vs planted {rabi}"
    assert abs(ours / a_qcal - 1) < 0.01, f"our {ours} vs qcal's period arithmetic {a_qcal}"


@pytest.mark.cosim
def test_amplitude_fine_pass_improves_the_coarse(cosim, demod_phase):
    """spec 13 Q4 — the notebook's two-step amplitude cal, on qcal's knobs: a coarse n_gates=1 cosine
    sweep, then a FINE pass that repeats the gate 4× (qcal's multiple-of-4 guard: 4 · X90 = 2π, back
    to |0>) over `relative_amp` 0.7–1.3× whatever the coarse step just wrote. Four gates amplify an
    amplitude error 4×, so the parabola's minimum pins it far more sharply than the cosine did — the
    fine pass must beat the coarse one against the model's true π/2 amplitude."""
    drv, m = cosim
    rabi = float(4 * math.pi / _sig_max(m, F_GE))
    drv.sim.set_model(_model(rabi, t1=200, t2=2000, noise=300.0, seed=17, collapse=True))
    cfg = _cfg(m, F_GE)
    cfg["readout/0/demod/phase"] = demod_phase
    true_amp = _true_x90_amp(m, rabi)

    coarse = Amplitude(cfg, 0, n_gates=1, points=13, shots=64).run(drv)
    assert coarse.ok
    coarse.apply()                                          # the fine pass sweeps RELATIVE to this
    err_coarse = abs(cfg["qubit/0/x90/amp"] - true_amp)
    fine = Amplitude(cfg, 0, n_gates=4, amp_span=(0.7, 1.3), relative_amp=True, points=9,
                     shots=64).run(drv)
    assert fine.ok
    fine.apply()
    err_fine = abs(cfg["qubit/0/x90/amp"] - true_amp)
    print(f"\n[amp-fine] true={true_amp:.5f}  coarse={coarse.proposal['qubit/0/x90/amp']:.5f} "
          f"(err {err_coarse:.5f})  fine={fine.proposal['qubit/0/x90/amp']:.5f} (err {err_fine:.5f})\n"
          f"  fine P={np.round(fine.data[0]['y'], 3).tolist()}")
    assert fine.fit[0].params["a"] > 0, "the 4-gate |1> population must MINIMISE at the tuned amp"
    assert err_fine < err_coarse, f"the fine pass did not improve: {err_coarse:.5f} → {err_fine:.5f}"


# ── Frequency: qcal's V-fit recovers a planted detuning of EITHER sign (the sign lock-step) ──

@pytest.mark.cosim
@pytest.mark.parametrize("d0_code", [60, -60])
def test_frequency_recovers_detuning(cosim, demod_phase, d0_code):
    """spec 13 Q4 — the update sign, PINNED. The fringe frequency is |δ + applied|, so its magnitude
    alone cannot tell a carrier that is too high from one that is too low: only the position of the
    V's vertex can (b = −δ), and only if the whole chain — the on-core virtual-Z ramp's sign, the
    model's axis ramp, and the proposal's arithmetic — agrees. Get any of them backwards and the
    "correction" DOUBLES the error for one of the two signs while looking perfect for the other. So
    plant the detuning BOTH ways and require the config to move toward f_ge each time.

    (The V-fit itself is the other half of Q4: the old signed line needed |applied| > |δ| at every
    detuning to keep sign(δ + applied) = sign(applied). qcal's a·|x − b| + c does not.)

    Sized for the co-sim clock: 12 waits × 96 shots × 4 detunings, on a 4·T1 relax head (t1 = 200, so
    ~2 % of the shots start with a residual excitation — an offset the cosine fit absorbs) instead of
    the 8·T1 the other cals use, which is what buys the second sign its runtime."""
    drv, m = cosim
    rabi = float((math.pi / 2) / gate_sigma(m, Pulse(GATE_ENV, freq_hz=F_GE, amp=0.5),
                                            F_GE, units._amp_code(0.5)))
    drive = units.code_to_freq(units._freq_code(F_GE, m.params) + d0_code, m.params)   # f_ge + δ0
    drv.sim.set_model(_model(rabi, t1=200, t2=3000, noise=300.0, seed=3, collapse=True))
    cfg = _cfg(m, drive, x90_amp=0.5, relax=800)
    cfg["readout/0/demod/phase"] = demod_phase
    freq = Frequency(cfg, 0, detune=units.code_to_freq(200, m.params), n_detune=4,
                     t0=_s(8, m), dt=_s(4, m), points=12, shots=96)
    r = freq.run(drv)
    print(f"\n[frequency δ={d0_code:+d}] applied={r.data[0]['applied']} "
          f"|fringe|={np.round(r.data[0]['obs'])}\n"
          f"  V-fit a={r.fit[0].params['a']:.3f} b={r.fit[0].value:+.1f} (want {-d0_code:+d}) → "
          f"recovered δ={freq.recovered_detuning_code[0]:+.1f} (planted {d0_code:+d})")
    assert r.ok
    assert r.fit[0].params["a"] > 0, "the V must open upward (qcal's negative-curvature guard)"
    assert abs(freq.recovered_detuning_code[0] - d0_code) < 25, "detuning code not recovered"
    r.apply()
    err_before = abs(drive - F_GE)
    err_after = abs(cfg["qubit/0/freq"] - F_GE)
    print(f"[frequency δ={d0_code:+d}] |freq−f_ge| before={err_before:.3g} after={err_after:.3g}")
    assert err_after < 0.3 * err_before, "config frequency did not move toward f_ge"


# ── T1 / T2: recover planted decay constants ──

@pytest.mark.cosim
def test_t1_recovers_decay(cosim, demod_phase):
    drv, m = cosim
    t1 = 120                                                      # batches (the model's own unit)
    drv.sim.set_model(_model(_rabi_pi(m), t1=t1, t2=2000, noise=300.0, seed=4, collapse=True))
    cfg = _cfg(m, F_GE, x90_amp=0.495)                            # X90·X90 prep = π (spec 13 §4)
    cfg["readout/0/demod/phase"] = demod_phase
    r = T1(cfg, 0, points=9).run(drv)
    print(f"\n[t1] fit ok={r.fit[0].ok} tau={r.fit[0].value} amp={r.fit[0].params.get('amp')} "
          f"delays={r.data[0]['x'].tolist()} P={np.round(r.data[0]['y'],3).tolist()}")
    assert r.ok
    t1_s = _s(t1, m)                                              # the proposal is in SECONDS
    assert 0.8 * t1_s < r.proposal["qubit/0/T1"] < 1.2 * t1_s


@pytest.mark.cosim
def test_t2_recovers_decay(cosim, demod_phase):
    drv, m = cosim
    t2 = 200
    rabi = float((math.pi / 2) / gate_sigma(m, Pulse(GATE_ENV, freq_hz=F_GE, amp=0.5),
                                            F_GE, units._amp_code(0.5)))
    drv.sim.set_model(_model(rabi, t1=400, t2=t2, noise=300.0, seed=5, collapse=True))
    cfg = _cfg(m, F_GE, x90_amp=0.5)
    cfg["readout/0/demod/phase"] = demod_phase
    r = T2(cfg, 0, detune=units.code_to_freq(70, m.params), points=15,
           t0=_s(8, m), dt=_s(16, m)).run(drv)                    # dt=16 batches → covers > t2
    print(f"\n[t2] fit ok={r.fit[0].ok} freq={r.fit[0].value} tau={r.fit[0].params.get('tau')} "
          f"waits={r.data[0]['x'].tolist()} P={np.round(r.data[0]['y'],3).tolist()}")
    assert r.ok
    t2_s = _s(t2, m)                                              # the proposal is in SECONDS
    assert 0.8 * t2_s < r.proposal["qubit/0/T2"] < 1.25 * t2_s


# ── Phase: qcal's line crossing recovers a PLANTED ac-Stark phase (spec 13 Q3) ──

def _stark_vz(eps):
    """The virtual-Z the calibration must recover for a planted per-X90 Stark rotation `eps`.

    The X90 is exp(−i[θ·(n̂·σ) + ε·Z]/2) with θ = π/2 (the drive and the Stark accrue TOGETHER — the
    model interleaves them batch by batch, as a real detuning-during-drive does). Its symmetric
    decomposition is Rz(β)·R_n(θ')·Rz(β) with β = atan((ε/Ω)·tan(Ω/2)), Ω = √(θ² + ε²) — so the pulse
    leaves β of Z on EACH side, and the virtual-Z pair that cancels it is φ = β on each side (the frame
    chases the phase the qubit accrued). β → (2/π)·ε ≈ 0.64·ε for small ε, NOT ε/2: a Z error accrued
    DURING a π/2 rotation does not split as ε/2 per side (spec 13 §6 says ε/2; that is the lumped-error
    approximation, and its sign is the opposite of what our frame convention gives)."""
    omega = math.hypot(math.pi / 2, eps)
    return math.atan2(eps * math.tan(omega / 2), omega)


@pytest.mark.cosim
def test_phase_recovers_the_stark_shift(cosim, demod_phase):
    """spec 13 Q3 — THE gate the old Phase could not pose. Plant an ac-Stark drive phase in the model
    (`stark_rad_per_sigma`: every driven batch also rotates the qubit about z, so an X90 leaves ε of Z
    behind) and qcal's two sequences must find it: their |1> populations are lines of opposite slope in
    the swept virtual-Z, and the crossing is the phase that cancels the error. With no Stark planted
    the answer is 0 and ANY sequence 'passes' — which is why the old cosine-Ramsey Phase could only
    concede that it 'recovers ≈0'."""
    drv, m = cosim
    x90 = Pulse(GATE_ENV, freq_hz=F_GE, amp=0.5)
    sigma = gate_sigma(m, x90, F_GE, units._amp_code(0.5))   # the X90's drive integral
    rabi = float((math.pi / 2) / sigma)                      # calibrated X90 (θ = π/2)
    eps = 0.3                                                # the planted Z rotation per X90 (rad)
    # t1/relax are the SNR levers here: the |1> population decays over the pulse→readout SEP, which
    # flattens both lines (and the crossing's error goes as 1/slope), while the relax head has to stay
    # ≫ T1 to reset. t1 = 300 / relax = 1000 is the cheapest pair that keeps the slopes at ±0.4.
    drv.sim.set_model(_model(rabi, t1=300, t2=3000, noise=300.0, seed=6, collapse=True,
                             stark_rad_per_sigma=eps / sigma))
    cfg = _cfg(m, F_GE, x90_amp=0.5, relax=1000)
    cfg["readout/0/demod/phase"] = demod_phase
    cal = Phase(cfg, 0, points=7, span=0.3, shots=96)
    r = cal.run(drv)
    want = _stark_vz(eps)
    vz = r.proposal["qubit/0/x90/vz"]
    print(f"\n[phase] planted eps={eps:.3f} → vz should be {want:+.4f} (ε/2 = {eps/2:+.4f})\n"
          f"  recovered vz={vz}  slopes={r.fit[0][0].value:+.2f}/{r.fit[0][1].value:+.2f} "
          f"fallback={cal.fallback[0]}\n"
          f"  P(Y180_X90)={np.round(r.data[0]['p0'], 3).tolist()}\n"
          f"  P(X180_Y90)={np.round(r.data[0]['p1'], 3).tolist()}")
    assert r.ok and not cal.fallback[0]
    assert vz[0] == vz[1], "qcal writes ONE crossing to BOTH virtual-Z slots"
    # Tolerance = 2σ of the crossing under binomial noise: with slopes of ±0.42 and 7×96 shots per
    # sequence, σ ≈ 2.1·0.5/(√N·|m0 − m1|) ≈ 0.05 rad (and σ ∝ √runtime, so buying it down costs
    # minutes). It still excludes 0 (what the old cosine Phase "recovered"), a flipped sign, and a
    # frame contract that advanced by φ per gate instead of 2φ (which would land at 2·want).
    assert abs(vz[0] - want) < 0.10, f"recovered {vz[0]:+.4f} rad, planted Stark wants {want:+.4f}"
    assert "qubit/0/x90/phase" not in r.proposal, "the FAST_DRAG's own axis phase is not the knob"


@pytest.mark.cosim
@pytest.mark.parametrize("planted", [0.0, 0.5])
def test_phase_x_gate_recovers_the_planted_axis(cosim, demod_phase, planted):
    """spec 14 F1 — `Phase(gate='X')`, qcal's second Phase mode. Its circuit is X90 · X · X90, a 2π
    rotation that returns to |0> only when the X's own AXIS matches the X90s'. Plant that axis by
    giving the X90s a phase of `planted`: the X's calibrated axis must follow them, and the proposal
    lands on `qubit/0/x/phase` (an axis phase — NOT the virtual-Z pair the X90 mode writes). The X is
    the X6Y3 shape (double LENGTH, same amplitude, spec 13 §4).

    The fringe runs at 2φ, so it has period π and the recovered axis is only defined mod π (the two
    solutions are the same gate) — which is what is checked. Like every counts-mode cal here it needs
    the measured `demod_phase`: without it the discriminator's polarity is whatever the grid period
    leaves the demod LO at, and the recovered axis lands half a fringe out."""
    drv, m = cosim
    x90 = Pulse(GATE_ENV, freq_hz=F_GE, amp=0.5)
    sigma = gate_sigma(m, x90, F_GE, units._amp_code(0.5))
    drv.sim.set_model(_model(float((math.pi / 2) / sigma), t1=400, t2=3000, noise=300.0, seed=3,
                             collapse=True))
    cfg = _cfg(m, F_GE, x90_amp=0.5, relax=1000)
    cfg["readout/0/demod/phase"] = demod_phase    # the measured discrimination phase (counts mode)
    cfg["qubit/0/x90/phase"] = planted            # the reference axis both X90s sit on
    cfg["qubit/0/x/env"] = "square"               # the X: double LENGTH, same amp → π (spec 13 §4)
    cfg["qubit/0/x/dur"] = _s(8, m)
    cfg["qubit/0/x/amp"] = 0.5
    cfg["qubit/0/x/phase"] = 0.0                  # deliberately off the X90s when planted != 0
    cal = Phase(cfg, 0, gate="X", points=13, shots=96)
    r = cal.run(drv)
    got = r.proposal.get("qubit/0/x/phase")
    print(f"\n[phase-X] planted axis={planted:+.3f} recovered={got} "
          f"fallback={cal.fallback.get(0)}\n  P(1)={np.round(r.data[0]['y'], 3).tolist()}")
    assert r.ok and "qubit/0/x90/vz" not in r.proposal   # the X mode writes the axis, not the pair
    assert abs(math.remainder(got - planted, math.pi)) < 0.25, \
        f"recovered {got:+.4f} rad, the X90s sit at {planted:+.4f} (mod π)"


# ── acquire_shots: ReadoutCalibration captures bimodal |0>/|1> clusters (raw mode) ──

@pytest.mark.cosim
def test_acquire_shots_chunks(cosim):
    drv, m = cosim
    drv.sim.set_model(_model(_rabi_pi(m), t1=RO_T1, t2=3000, noise=400.0, seed=2, collapse=True))
    cfg = _cfg(m, F_GE, x90_amp=0.495, relax=RO_RELAX)
    r = ReadoutCalibration(cfg, 0, shots=16).run(drv)   # per-prep RAW reruns: |0>, |1>
    assert r.data[0]["iq0"].shape == (16, 2) and r.data[0]["iq1"].shape == (16, 2)
    assert r.ok and r.data[0]["separation"] > 1.0       # qcal SNR: means ≥ 4σ apart (spec 13 §2)
    assert isinstance(r.fit[0], Classifier)             # the trained classifier rides on the Result


@pytest.mark.cosim
def test_readout_calibration_phase_proposal_is_a_fixed_point(cosim):
    """The demod-phase proposal is ABSOLUTE (rotate the |0>→|1> cluster axis onto +real, spec 13 §5),
    so the RAW capture must run in the ZERO demod frame: baking the config's CURRENT phase into the
    capture carrier rotates the measured axis by exactly that stale value and the 'absolute' proposal
    comes out relative — invisible on co-sim configs (stored phase 0), wrong on X6Y3 (−109.9°…+39.0°).
    Plant a stale nonzero phase, calibrate, apply, re-calibrate: the second run must propose the SAME
    phase (a fixed point). The old capture-at-stored-phase left each proposal ~1 rad off the last."""
    drv, m = cosim
    drv.sim.set_model(_model(_rabi_pi(m), t1=RO_T1, t2=3000, noise=300.0, seed=7, collapse=True))
    cfg = _cfg(m, F_GE, x90_amp=0.495, relax=RO_RELAX)
    cfg["readout/0/demod/phase"] = 1.0                  # any stale nonzero phase (rad)
    r1 = ReadoutCalibration(cfg, 0, shots=24).run(drv)
    assert r1.ok
    r1.apply()
    r2 = ReadoutCalibration(cfg, 0, shots=24).run(drv)
    assert r2.ok
    p1, p2 = r1.proposal["readout/0/demod/phase"], r2.proposal["readout/0/demod/phase"]
    print(f"\n[rc-phase] stale=1.0 first={p1:+.3f} second={p2:+.3f}")
    assert abs(math.remainder(p2 - p1, 2 * math.pi)) < 0.2, \
        "the demod-phase proposal must be a fixed point (absolute, not relative to the stored phase)"


# ── the |1> prep: qcal's two gates reach the same population (spec 13 Q1) ──

@pytest.mark.cosim
def test_prep_gate_x90_and_x_agree(cosim):
    """spec 13 Q1: `gate='X90'` plays the config's X90 TWICE (one play + one bare fire — B0's
    startTime auto-advance makes the train contiguous); `gate='X'` plays the config's OWN X pulse
    once, here (as on X6Y3) a double-LENGTH, same-amplitude pulse — NOT the deleted "π = 2× the X90
    amp" synthesis. Both integrate to the same drive, so both prep |1>: their clusters classify the
    same under ONE fixed classifier (the X90 run's), to within shot noise."""
    drv, m = cosim
    drv.sim.set_model(_model(_rabi_pi(m), t1=800, t2=3000, noise=300.0, seed=13, collapse=True))
    cfg = _cfg(m, F_GE, x90_amp=0.495, relax=3200)   # X90 = π/2 (4 batches @ 0.495); relax = 4·T1
    cfg["qubit/0/x/env"] = "square"                  # the X: double LENGTH, same amp → the same π
    cfg["qubit/0/x/dur"] = _s(8, m)
    cfg["qubit/0/x/amp"] = 0.495
    r90 = ReadoutCalibration(cfg, 0, shots=24, gate="X90").run(drv)
    rx = ReadoutCalibration(cfg, 0, shots=24, gate="X").run(drv)
    assert r90.ok and rx.ok
    clf = r90.fit[0]                                  # the FIXED classifier (trained on the X90 preps)
    p90 = float(clf.classify(r90.data[0]["iq1"]).mean())
    px = float(clf.classify(rx.data[0]["iq1"]).mean())
    print(f"\n[prep-gate] P(|1>): X90·X90={p90:.3f} X={px:.3f}  "
          f"sep={r90.data[0]['separation']:.1f}/{rx.data[0]['separation']:.1f}")
    assert p90 > 0.75 and px > 0.75, f"a prep did not reach |1>: X90·X90={p90:.3f} X={px:.3f}"
    assert abs(p90 - px) < 0.15, f"the two preps disagree: X90·X90={p90:.3f} vs X={px:.3f}"


# ── the readout cals, on qcal's statistics (spec 13 Q2) ──

@pytest.mark.cosim
def test_separation_picks_max_separation_not_the_magnitude_peak(cosim):
    """spec 13 Q2 — THE regression that catches the old Separation. On a dispersive readout the |0>
    response peaks at f_r + χ while the two-state separation peaks at f_r, so the |0>-magnitude argmax
    (what the old |0>-only VNA took) and the cluster-SNR argmax (qcal's statistic, what we take now)
    are DIFFERENT grid points. Separation runs the matched-pair sweep at both prep states (k_vna RAW,
    two reruns of one resident program) and must pick the latter.

    Soft readout (collapse=False): the clusters are then the readout noise around each state's
    response — Gaussian, as a real experiment's are. (Projective collapse would make the |1> cluster a
    MIXTURE of both tones through the co-sim's small T1/relax budget, and that spread is ∝ ‖Δmeans‖, so
    it divides straight back out of the SNR — an artifact of the 1600-batch reset, not physics.)"""
    drv, m = cosim
    disp = _dispersive(m, chi_code=60, kappa_code=170)
    drv.sim.set_model(_model(_rabi_pi(m), t1=300, t2=3000, noise=300.0, seed=14,
                             readout_amp=4000.0, **disp))
    cfg = _cfg(m, F_GE, x90_amp=0.495)                     # X90·X90 = π; readout/0/freq = f_r
    r = Separation(cfg, 0, span=units.code_to_freq(120, m.params), points=5, shots=64).run(drv)
    sep, mag0 = r.data[0]["y"], r.data[0]["mag0"]
    c_r = units._freq_code(_readout_freq(m), m.params)
    # data["x"] is physical Hz around the stored freq (codes stay inside run(), spec 13 §2) — map it
    # back to the swept codes for the grid-point assertions below.
    xs = np.round((r.data[0]["x"] - _readout_freq(m))
                  / units.code_to_freq(1, m.params)).astype(int) + c_r
    best, peak = int(np.argmax(sep)), int(np.argmax(mag0))
    print(f"\n[separation] ok={r.ok} codes={xs.tolist()} (f_r={c_r}, |0> peak={c_r + 60})\n"
          f"  cluster SNR   ={np.round(sep, 3).tolist()} -> argmax {best} (code {xs[best]})\n"
          f"  |0> magnitude ={np.round(mag0).astype(int).tolist()} -> argmax {peak} (code {xs[peak]})")
    assert r.ok
    assert xs[peak] == c_r + 60, "the |0> magnitude does not peak at the |0> dressed resonance"
    assert best != peak, "max separation and the |0>-magnitude peak coincide — the gate is vacuous"
    assert xs[best] == c_r, "Separation did not pick the max-separation frequency"


@pytest.mark.cosim
def test_separation_proposes_physical_hz_not_the_alias(cosim):
    """The 16-bit sweep codes alias (Nyquist fold): on X6Y3 the 6.55 GHz readout, synthesized in the
    DAC's 2nd Nyquist zone, folds to a code whose code_to_freq is ≈ −1.44 GHz — which is what the old
    proposal wrote back into the tree of record. The proposal must be DELTA-based physical Hz
    (f0 + code_to_freq(best − c0)): store the readout freq as the out-of-band alias f − fs — the
    IDENTICAL hardware code bit-for-bit, so the run itself is unchanged — and the proposal must come
    back in that same band, within the swept span of the stored value."""
    drv, m = cosim
    disp = _dispersive(m, chi_code=60, kappa_code=170)
    drv.sim.set_model(_model(_rabi_pi(m), t1=300, t2=3000, noise=300.0, seed=14,
                             readout_amp=4000.0, **disp))
    cfg = _cfg(m, F_GE, x90_amp=0.495)
    f_alias = _readout_freq(m) - units.sample_rate(m.params)     # same code, the other Nyquist band
    assert units._freq_code(f_alias, m.params) == units._freq_code(_readout_freq(m), m.params)
    cfg["readout/0/freq"] = float(f_alias)
    span = units.code_to_freq(120, m.params)
    r = Separation(cfg, 0, span=span, points=5, shots=64).run(drv)
    prop = r.proposal["readout/0/freq"]
    print(f"\n[separation-alias] stored={f_alias:.6g} proposed={prop:.6g} span={span:.4g}")
    assert r.ok
    assert abs(prop - f_alias) <= span, \
        "the proposal must stay delta-based in the stored band, not jump to the baseband alias"
    assert np.all(np.abs(r.data[0]["x"] - f_alias) <= span)      # the x-axis is in-band Hz too


@pytest.mark.cosim
def test_fidelity_picks_readout_amp(cosim):
    """spec 13 Q2 — Fidelity sweeps qcal's knob (the readout DRIVE amplitude, on-core via k_ro_amp) and
    scores the confusion diagonal ½[P(0|0) + P(1|1)] under the FIXED hardware discriminator (the demod
    phase ReadoutCalibration measured; never retrained per point). The dispersive resonator's answer is
    proportional to the drive, so the diagonal responds to the amplitude and the argmax lands in the
    sweep — with the old flat tone (amplitude a model constant) this knob was a no-op."""
    drv, m = cosim
    spec = _model(_rabi_pi(m), t1=RO_T1, t2=3000, noise=5600.0, seed=15, collapse=True,
                  readout_amp=4000.0, **_dispersive(m))
    cfg = _cfg(m, F_GE, x90_amp=0.495, relax=RO_RELAX)      # readout/0/amp = 0.5 (the sweep centre)
    _calibrate(drv, m, cfg, spec)                           # bake the discrimination phase
    r = Fidelity(cfg, 0, amp_span=0.45, points=5, shots=24).run(drv)
    amps, fid = r.data[0]["x"], r.data[0]["y"]
    print(f"\n[fidelity] ok={r.ok} amps={np.round(amps, 3).tolist()}\n"
          f"  diagonal={np.round(fid, 3).tolist()} P(1|0)={np.round(r.data[0]['p0'], 3).tolist()} "
          f"P(1|1)={np.round(r.data[0]['p1'], 3).tolist()} -> picked={r.proposal['readout/0/amp']:.3f}")
    assert r.ok
    assert fid[-1] - fid[0] > 0.1, "the confusion diagonal does not respond to the readout amplitude"
    assert amps[0] <= r.proposal["readout/0/amp"] <= amps[-1]


@pytest.mark.cosim
def test_fidelity_sweeps_the_full_span_at_tiny_amp(cosim):
    """qcal's Fidelity sweeps exactly ±amp_span around the config amp; the old AMP_MIN = 0.01 floor
    silently truncated the lower half-span for X6Y3-class readout amps (q5: 0.0115 lost 54 % of it).
    At amp 0.012, span 0.005, the first swept point must be 0.007 — not the clamped 0.01. (Only the
    realized sweep axis is asserted; the populations at these near-noise amps are not the point.)"""
    drv, m = cosim
    drv.sim.set_model(_model(_rabi_pi(m), t1=RO_T1, t2=3000, noise=300.0, seed=20, collapse=True))
    cfg = _cfg(m, F_GE, x90_amp=0.495, relax=RO_RELAX)
    cfg["readout/0/amp"] = 0.012
    r = Fidelity(cfg, 0, amp_span=0.005, points=3, shots=16).run(drv)
    xs = r.data[0]["x"]
    print(f"\n[fid-span] amps={np.round(xs, 5).tolist()}")
    assert xs[0] == pytest.approx(0.007, abs=1e-4), "the lower half-span was clamped away"


@pytest.mark.cosim
def test_readout_fidelity_matches_the_host_classifier(cosim):
    """spec 13 Q2 — ReadoutFidelity's confusion comes from the `res` bit under the FIXED discriminator
    (two COUNTS reruns; no raw IQ, no retraining). It must agree, within shot noise, with the confusion
    of a host classifier trained on the same readout's clusters — i.e. the on-chip discriminator really
    is the classifier we think it is. The old version could not show this: it retrained on the very
    points it then confused."""
    drv, m = cosim
    spec = _model(_rabi_pi(m), t1=RO_T1, t2=3000, noise=5600.0, seed=16, collapse=True,
                  readout_amp=4000.0, **_dispersive(m))
    cfg = _cfg(m, F_GE, x90_amp=0.495, relax=RO_RELAX)
    _calibrate(drv, m, cfg, spec)                              # bake the discrimination phase
    host = ReadoutCalibration(cfg, 0, shots=48).run(drv).fit[0]  # the host classifier, same readout
    r = ReadoutFidelity(cfg, 0, shots=48).run(drv)             # the res bit under that discriminator
    conf, hconf = r.data[0]["confusion"], host.confusion()
    print(f"\n[readout-fidelity] fidelity={r.data[0]['fidelity']:.3f} "
          f"(host {np.mean(np.diag(hconf)):.3f})\n  res bit:\n{np.round(conf, 3)}\n"
          f"  host classifier:\n{np.round(hconf, 3)}")
    assert r.ok
    assert 0.7 < r.data[0]["fidelity"] < 1.0, "a saturated confusion would test nothing"
    assert np.allclose(np.diag(conf), np.diag(hconf), atol=0.15)   # ~2σ of two 48-shot estimates


@pytest.mark.cosim
def test_window_picks_the_longer_integration(cosim, demod_phase):
    """The demod-window sweep — OURS, not qcal's (spec 13 §5), so it is out of the X6Y3 chain — retunes
    the window via write_slot + rerun (no recompile, spec 08 §4) and is scored exactly like Fidelity:
    the confusion diagonal under the fixed discriminator (not a classifier retrained per window, as the
    old Fidelity did). A longer integration collects more SNR, so it must win."""
    drv, m = cosim
    drv.sim.set_model(_model(_rabi_pi(m), t1=RO_T1, t2=3000, noise=7000.0, seed=10, collapse=True,
                             readout_amp=1000.0))                   # the plain (flat-tone) readout
    cfg = _cfg(m, F_GE, x90_amp=0.495, dur=64, drive=80, relax=RO_RELAX)   # drive covers the window
    cfg["readout/0/demod/phase"] = demod_phase
    durs = (16, 64)                                                 # candidate windows (batches)
    r = Window(cfg, 0, durs=[_s(d, m) for d in durs], shots=24).run(drv)
    fid = r.data[0]["y"]
    print(f"\n[window] ok={r.ok} durs={r.data[0]['x'].astype(int).tolist()} "
          f"diagonal={np.round(fid, 3).tolist()} picked={r.proposal['readout/0/demod/dur']:.3e} s")
    assert r.ok
    assert fid[1] > fid[0] + 0.05, "the longer integration window did not read out better"
    assert r.proposal["readout/0/demod/dur"] == pytest.approx(_s(durs[1], m))


# ── multi-qubit simultaneous: both cores calibrated in ONE run (spec 13 Q5 / §8) ──

# ── the readout TIMING knobs (spec 14 F2) ──

@pytest.mark.cosim
@pytest.mark.parametrize("knob", ["demod/delay"])
def test_readout_timing_knob_moves_the_readout(cosim, demod_phase, knob):
    """(F2 gate) The demod DELAY — when the integration window opens after the drive starts, qcal's
    `demod/delay` — swept through `Window`'s machinery: compile once at the longest candidate, then
    retune per point. Unlike the two durations it is NOT a table field (the kernel adds it to the
    demod's play time), so it is compiled as a per-run param instead, which is what this exercises.

    Gated on the large, unambiguous effect: the projective model emits its readout tone only while the
    DRIVE is on, so a delay that opens the window past the END of the drive integrates silence and
    discrimination collapses. Differences between the two prompt points are within binomial noise at
    these shot counts (~0.07 per point), so only the collapse is asserted.

    The readout DRIVE length (`readout/{q}/dur`) shares this machinery but gets no scored gate: this
    model latches the shot's level at the drive's RISING EDGE and then emits the tone for the whole
    window, so the drive's LENGTH is unobservable here (a sweep over it returns the same three numbers
    it returns for any other candidates). `test_readout_drive_length_reaches_the_dac` gates that knob
    where it IS observable — on the converter."""
    drv, m = cosim
    q = 0
    drv.sim.set_model(_model(_rabi_pi(m), t1=RO_T1, t2=3000, noise=300.0, seed=11, collapse=True))
    cfg = _cfg(m, F_GE, relax=RO_RELAX)
    cfg[f"readout/{q}/demod/phase"] = demod_phase
    cfg[f"readout/{q}/demod/dur"] = _s(40, m)
    cfg[f"readout/{q}/dur"] = _s(56, m)
    durs = [_s(n, m) for n in (0, 24, 96)]         # 96 opens the window past the 56-batch drive
    bad = 2
    r = Window(cfg, q, durs=durs, shots=48, knob=knob).run(drv)
    y = r.data[q]["y"]
    good = [k for k in range(len(durs)) if k != bad]
    print(f"\n[{knob}] fidelity={np.round(y, 3).tolist()} -> {r.proposal[f'readout/{q}/{knob}']:.2e}")
    assert y[bad] < min(y[k] for k in good) - 0.2, \
        f"{knob}: the starved point did not collapse the readout: {y}"
    assert int(np.argmax(y)) in good, f"{knob} picked the starved point: {y}"
    assert r.proposal[f"readout/{q}/{knob}"] == pytest.approx(durs[int(np.argmax(y))], rel=0.05)


@pytest.mark.cosim
def test_punchout_maps_frequency_against_drive_power(cosim):
    """(F2 gate) The punchout map — walkthrough stage 1.2, the one readout stage we had no tool for.
    One k_vna program per qubit, then a `write_slot("ro", 0, "amp")` + a |0> rerun per amplitude, so
    the map is (amps × points) of |S21| at the |0> resonator.

    On this model the resonator is LINEAR: its response scales with the drive and its dressed peak
    does NOT walk with power (real punchout needs a nonlinear cavity, which the model does not have).
    So what is gated is what this model can actually show, and it is exactly what would be broken by a
    mis-wired amp loop: every row peaks at the same dressed frequency (f_r + χ), and the rows scale
    with the drive amplitude in the right order."""
    drv, m = cosim
    q = 0
    disp = _dispersive(m, chi_code=60, kappa_code=170)
    drv.sim.set_model(_model(_rabi_pi(m), t1=RO_T1, t2=3000, noise=60.0, seed=4, collapse=False,
                             **disp))
    cfg = _cfg(m, F_GE, relax=RO_RELAX)
    amps = [0.1, 0.3, 0.6]
    r = Punchout(cfg, q, amps=amps, span=2.5e6, points=21, shots=16).run(drv)
    mag, freqs = r.data[q]["mag"], r.data[q]["x"]
    peaks = [int(np.argmax(row)) for row in mag]
    c_r = units._freq_code(_readout_freq(m), m.params)
    print(f"\n[punchout] amps={amps}\n  peak idx per row={peaks} (f_r code={c_r}, |0> peak={c_r + 60})"
          f"\n  row max={np.round([row.max() for row in mag], 1).tolist()}")
    assert mag.shape == (len(amps), 21)
    assert len(set(peaks)) == 1, f"the dressed peak moved with power on a LINEAR resonator: {peaks}"
    rowmax = np.array([row.max() for row in mag])
    assert np.all(np.diff(rowmax) > 0), f"the response did not grow with drive amplitude: {rowmax}"
    # ... and it grows about linearly, which is the amp knob actually reaching the drive
    assert rowmax[2] / rowmax[0] == pytest.approx(amps[2] / amps[0], rel=0.25)


@pytest.mark.cosim
def test_readout_drive_length_reaches_the_dac(cosim):
    """(F2 gate) The readout DRIVE length knob, gated where it is observable: the converter. `Window`
    compiles the drive at the longest candidate and retunes the slot per point, so the readout DAC's
    active window must be exactly the batches written — the proof the knob is applied even though the
    projective model (which latches on the drive's rising edge) cannot see it."""
    drv, m = cosim
    q = 0
    drv.sim.set_model({"kind": "zero"})               # the readout DAC carries only the core's drive
    cfg = _cfg(m, F_GE, relax=400)
    cfg[f"readout/{q}/dur"] = _s(56, m)
    cfg[f"readout/{q}/demod/dur"] = _s(40, m)
    a = units._amp_code(float(cfg[f"readout/{q}/amp"]))
    prog, period = _ro_amp_prog(m, cfg, q, "X90", 1, 1, a << 16, 0)
    for want in (56, 24, 8):
        rq.setup(drv, m, {q: prog})
        rq.write_slot(drv, m, q, prog, "ro", 0, "dur", want)
        rq.check_magic(drv, m, q, prog)
        rq.write_var(drv, m, q, prog, "__rq_status", 0)
        rq.write_params(drv, m, q, prog, {"prep": 0})
        h = drv.sim.dac_capture_arm(m.ro_dac(q), 20000)
        rq.reset(drv, m, on=False)
        rq.poll_done(drv, m, q, prog, timeout=batch_timeout(period))
        rq.reset(drv, m, on=True)
        _, cap = drv.sim.dac_capture_get(h)
        active = cap.any(axis=1)
        runs = [i for i in range(len(active)) if active[i]]
        got = (runs[-1] - runs[0] + 1) if runs else 0
        print(f"\n[ro-dur] wrote {want} batches, DAC drive = {got}")
        assert got == want, f"wrote dur={want}, the readout DAC played {got} batches"


@pytest.mark.cosim
def test_multiqubit_both_cores_recover(cosim):
    """spec 13 Q5 — the load-bearing deliverable: a calibration takes `qubits=[0, 1]`, compiles one
    program per core and issues ONE run, and the host fits each core. Simultaneous readout is a
    DIFFERENT measurement from serial (spec §8): on this build cores 0 and 1 share the readout DAC/ADC
    (ro_dac=14 / adc_of=0), so the two readouts are frequency multiplexed — the models emit at demod
    codes 2048 / 1024 and their tones SUM on ADC 0 (MultiModel), each core's demod integrating out its
    own. We plant a DIFFERENT Rabi rate on each core and require BOTH to recover in the one run (the
    Rabi-rate cosine is robust to the residual crosstalk / halved per-tone amplitude of the summed
    readout, where an exp-decay T1 fit is not)."""
    drv, m = cosim
    ro = {0: _readout_freq(m), 1: units.demod_code_to_freq(1024, m.params)}   # codes 2048 / 1024
    code = {0: 2048, 1: 1024}
    sig_max = _sig_max(m, F_GE)
    planted = {0: float(4 * math.pi / sig_max), 1: float(6 * math.pi / sig_max)}   # ~2 / 3 periods
    cfg = Config()
    for q in (0, 1):
        cfg[f"qubit/{q}/freq"] = F_GE
        cfg[f"qubit/{q}/x90/amp"] = 0.5
        cfg[f"qubit/{q}/T1"] = _s(120, m)
        cfg[f"readout/{q}/freq"] = float(ro[q])
        cfg[f"readout/{q}/amp"] = 0.5
        cfg[f"readout/{q}/dur"] = _s(56, m)               # drive covers the demod window + SEP
        cfg[f"readout/{q}/demod/dur"] = _s(40, m)
    cfg["reset/relax"] = _s(RO_RELAX, m)                  # relax ≫ T1 resets both cores each grid slot

    def model(rabis, seeds):
        """A summed two-qubit model: core q's tone at demod code `code[q]`, readout_amp halved so the
        two frequency-multiplexed tones sum inside the converter range."""
        sub = [{**_model(rabis[q], t1=RO_T1, t2=3000, noise=300.0, seed=seeds[q], collapse=True,
                          readout_amp=14000.0), "core": q, "readout_code": code[q]} for q in (0, 1)]
        return {"kind": "multi", "models": sub}

    # 1) fix each core's demod discrimination phase on a PREP-CALIBRATED model (X90·X90 = π gives clean
    #    |1> clusters); the phase is rabi-independent (readout_phase pinned 0) so it serves step 2. ONE run
    drv.sim.set_model(model({0: _rabi_pi(m), 1: _rabi_pi(m)}, {0: 21, 1: 22}))
    rc = ReadoutCalibration(cfg, [0, 1], shots=24).run(drv)
    assert rc.ok, f"clusters did not separate: {[rc.data[q]['separation'] for q in (0, 1)]}"
    rc.apply()                                            # writes BOTH cores' demod phase + res-sign

    # 2) recover BOTH cores' (different) Rabi rates in ONE run
    drv.sim.set_model(model(planted, {0: 31, 1: 32}))
    r = Amplitude(cfg, [0, 1], n_gates=1, points=13, shots=48).run(drv)
    got = {q: r.proposal[f"qubit/{q}/rabi"] for q in (0, 1)}
    print(f"\n[multi-q] planted Rabi {planted} recovered {got} "
          f"ratios={ {q: got[q] / planted[q] for q in (0, 1)} }")
    assert r.ok
    assert set(r.data) == {0, 1}, "one run must return BOTH cores' data"
    for q in (0, 1):
        assert abs(got[q] / planted[q] - 1) < 0.08, \
            f"core {q}: recovered Rabi {got[q]:.4e} vs planted {planted[q]:.4e}"


# ── §8 heralding: a pre-sequence herald read post-selects |0> (spec 13 Q5 Part C) ──

@pytest.mark.cosim
def test_heralding_matches_unheralded_on_clean_qubit(cosim, demod_phase):
    """A `readout/herald` counts run inserts a readout BEFORE the sequence and only counts shots that
    found the qubit in |0> (`P = count/kept`, spec 13 §8). On a clean qubit (relax ≫ T1 resets to |0>)
    the herald passes every shot, so the heralded Rabi curve and its recovered rate MATCH the
    unheralded ones — proving the two-window grid timing, the runtime `if h==0` gate, and the
    (count, kept) denominator all agree with count/shots. (The herald=off path is byte-identical C —
    every other cosim test runs it — so this only has to check the herald=on path lands on the same
    answer.)"""
    drv, m = cosim
    rabi = float(3 * math.pi / _sig_max(m, F_GE))
    drv.sim.set_model(_model(rabi, t1=RO_T1, t2=3000, noise=300.0, seed=11, collapse=True))
    cfg = _cfg(m, F_GE, relax=RO_RELAX)
    cfg["readout/0/demod/phase"] = demod_phase

    cfg["readout/herald"] = False
    off = Amplitude(cfg, 0, n_gates=1, points=9, shots=64).run(drv)
    cfg["readout/herald"] = True
    on = Amplitude(cfg, 0, n_gates=1, points=9, shots=64).run(drv)

    assert off.ok and on.ok, "both the heralded and unheralded Rabi must FIT (a corrupt curve wouldn't)"
    print(f"\n[herald] off P={np.round(off.data[0]['y'], 3).tolist()}"
          f"\n  on  P={np.round(on.data[0]['y'], 3).tolist()}")
    # THE spec gate (§8): heralded and unheralded POPULATIONS agree — on a clean qubit the herald
    # passes every shot, so P = count/kept equals count/shots (which also proves kept == shots: a wrong
    # denominator would scale P away). off and on are independent noise realisations (heralding also
    # consumes RNG draws), so compare at ~3σ of the 64-shot noise — loose enough to never flake, tight
    # enough to catch the drive dropping (that bug, a too-short herald offset, read a flat P≈0, ~0.8 off).
    # We compare the curves, NOT the fitted Rabi rate: a 9-point cosine can mis-seed to a harmonic.
    assert np.allclose(off.data[0]["y"], on.data[0]["y"], atol=0.2), \
        "heralded and unheralded populations must agree (herald passes every clean-|0> shot)"


@pytest.mark.cosim
def test_heralded_readout_fidelity_matches_unheralded(cosim, demod_phase):
    """The same herald fold in k_ro_amp (spec 13 §8): qcal's transpiler post-selects EVERY circuit,
    the Fidelity/ReadoutFidelity confusion circuits included. On a clean |0> qubit (relax ≫ T1) the
    pre-prep herald passes every shot, so the heralded confusion diagonal must MATCH the unheralded
    one — which also proves the (count, kept) denominator (kept == shots: a wrong denominator scales
    the populations away) and the two-window grid timing (a dropped prep drive would read P(1|1)≈0,
    diagonal ≈ 0.5)."""
    drv, m = cosim
    drv.sim.set_model(_model(_rabi_pi(m), t1=RO_T1, t2=3000, noise=300.0, seed=18, collapse=True))
    cfg = _cfg(m, F_GE, x90_amp=0.495, relax=RO_RELAX)
    cfg["readout/0/demod/phase"] = demod_phase
    cfg["readout/herald"] = False
    off = ReadoutFidelity(cfg, 0, shots=32).run(drv)
    cfg["readout/herald"] = True
    on = ReadoutFidelity(cfg, 0, shots=32).run(drv)
    d_off, d_on = np.diag(off.data[0]["confusion"]), np.diag(on.data[0]["confusion"])
    print(f"\n[herald-rof] off diag={np.round(d_off, 3).tolist()} "
          f"on diag={np.round(d_on, 3).tolist()}")
    assert on.data[0]["fidelity"] > 0.75 and off.data[0]["fidelity"] > 0.75
    assert np.allclose(d_on, d_off, atol=0.15), \
        "heralded and unheralded confusion diagonals must agree on a clean |0> qubit"


@pytest.mark.cosim
def test_heralded_phase_matches_unheralded(cosim, demod_phase):
    """The same herald fold in k_phase (spec 13 §8): qcal post-selects both Phase sequences. On a
    clean |0> qubit the heralded and unheralded line crossings agree within the fit error (no Stark
    planted, so both recover ≈ 0). A broken herald shape (the drive-drop trap, a wrong (count, kept)
    decode) flattens or scrambles the two lines — the crossing leaves the swept range (ok=False) or
    lands far from the unheralded one."""
    drv, m = cosim
    rabi = float((math.pi / 2) / gate_sigma(m, Pulse(GATE_ENV, freq_hz=F_GE, amp=0.5),
                                            F_GE, units._amp_code(0.5)))
    drv.sim.set_model(_model(rabi, t1=RO_T1, t2=3000, noise=300.0, seed=19, collapse=True))
    cfg = _cfg(m, F_GE, x90_amp=0.5, relax=RO_RELAX)
    cfg["readout/0/demod/phase"] = demod_phase
    cfg["readout/herald"] = False
    off = Phase(cfg, 0, points=7, span=0.3, shots=32).run(drv)
    cfg["readout/herald"] = True
    on = Phase(cfg, 0, points=7, span=0.3, shots=32).run(drv)
    assert off.ok and on.ok
    v_off, v_on = off.proposal["qubit/0/x90/vz"][0], on.proposal["qubit/0/x90/vz"][0]
    print(f"\n[herald-phase] off={v_off:+.4f} on={v_on:+.4f}"
          f"\n  off P(Y180_X90)={np.round(off.data[0]['p0'], 3).tolist()}"
          f"\n  on  P(Y180_X90)={np.round(on.data[0]['p0'], 3).tolist()}")
    # 3σ of the crossing difference under binomial noise at 7×32 shots/sequence with ~±0.45 slopes
    # (σ ≈ 0.075/run, √2 for two independent runs) — loose enough not to flake, tight enough that a
    # dropped drive or a scaled denominator (both send the crossing wild or out of range) fails.
    assert abs(v_on - v_off) < 0.3, \
        "heralded and unheralded Phase crossings must agree on a clean |0> qubit"


# ── §7 cost accounting: a batched cal is O(1) client seam ops in one run ──

@pytest.mark.cosim
def test_cost_accounting_amplitude(cosim):
    """spec 08 §7: a batched Amplitude cal costs O(1) client seam ops in ONE run (a few block writes
    in, one poll, one block read out) — not the ~50k of the old one-run-per-point host loop. The op
    count is independent of npts×shots (block writes/reads move O(n) bytes in O(1) ops), so this tiny
    5×8 sweep counts the same as a full 21×160 one. CountingDriver wraps the 4 seam ops (spec 08 B3)."""
    from test_rerun import CountingDriver
    drv, m = cosim
    drv.sim.set_model(_model(_rabi_pi(m), t1=200, t2=2000, noise=300.0, seed=12, collapse=True))
    cfg = _cfg(m, F_GE)
    cfg["readout/0/demod/phase"] = math.pi / 2          # any phase; the op count is physics-independent
    cd = CountingDriver(drv)
    Amplitude(cfg, 0, n_gates=1, points=5, shots=8).run(cd)
    print(f"\n[cost] one batched Amplitude cal = {cd.ops} client seam ops in 1 run "
          f"(setup+rerun; the rerun batch alone is 10 seam ops — test_rerun.py, B3)")
    assert cd.ops < 60, f"batched Amplitude cost {cd.ops} ops (expected O(1), a few dozen), not ~50k"


# ── the Calibration_X6Y3 sequence improves a deliberately-detuned Config ──

@pytest.mark.cosim
def test_x6y3_improves_detuned_config(cosim):
    """spec 13 Q6 — the payoff: calibration_x6y3 IS the notebook, driven through the qcal adapter. The
    config is loaded from a co-sim-scaled qcal tree by Config.from_qcal (the parity path — not a
    hand-built _cfg), a copy is deliberately detuned, and the whole 8-step chain must move it toward
    the model's ground truth on EVERY corrected step, not just freq + amp."""
    drv, m = cosim
    rabi = float(3 * math.pi / _sig_max(m, F_GE))
    # projective model (spec 08 §2.4): the whole X6Y3 chain — readout-cluster cals AND the counts-mode
    # qubit cals — runs on ONE model, so that model must clear the READOUT budget (the qcal SNR formula,
    # spec 13 §2/Q2): relax=RO_RELAX ≫ T1=RO_T1 ≫ SEP (=LEAD=96) resets to |0> in the idle head yet
    # retains |1> across the SEP. (The pre-Q2 t1=300/relax=1600 gave sep≈0.91 < 1.0 — starved for the
    # ~4×-tighter qcal SNR, cascading ReadoutCalibration ok=False → an un-applied demod phase → Frequency
    # and Phase failing.) High-SNR clusters (noise=300 vs readout_amp=20000).
    drv.sim.set_model(_model(rabi, t1=RO_T1, t2=3000, noise=300.0, seed=11, collapse=True))

    # THE headline: the chain is driven from a Config.from_qcal(...) tree (the qcal adapter path), whose
    # co-sim-scaled numbers reproduce _cfg's world exactly (readout on the model's demod code 2048, the
    # square GATE_ENV gate, the RO_RELAX relax head) through the real qcal layout.
    ro_freq = _readout_freq(m)
    cfg = Config.from_qcal(QCAL_COSIM_YAML)
    assert cfg["qubit/0/freq"] == F_GE and cfg["readout/0/freq"] == ro_freq
    assert cfg["reset/relax"] == pytest.approx(_s(RO_RELAX, m))
    assert cfg["qubit/0/x90/vz"] == [0.0, 0.0]                # a co-sim config: the frame bracket is a no-op

    # deliberately-detune a copy: qubit freq off by d0, X90 amp wrong (0.5 vs the true π/2 amp)
    d0_code = 60
    drive = units.code_to_freq(units._freq_code(F_GE, m.params) + d0_code, m.params)
    cfg["qubit/0/freq"] = float(drive)
    cfg["qubit/0/x90/amp"] = 0.5
    true_x90_amp = _true_x90_amp(m, rabi)

    freq_before = abs(cfg["qubit/0/freq"] - F_GE)
    amp_before = abs(cfg["qubit/0/x90/amp"] - true_x90_amp)
    results = calibration_x6y3(cfg, 0, drv, verbose=True)
    assert len(results) == 8 and all(r.label for r in results)
    rc, sep, fid, rof, freq, amp_c, amp_f, phase = results

    # spec 13 Q6 — every CORRECTED step actually worked end-to-end, not just freq + amp:
    assert rc.ok, f"ReadoutCalibration could not separate clusters (sep={rc.data[0]['separation']:.2f})"
    assert sep.ok, "Separation did not return a readout frequency"
    assert abs(sep.proposal["readout/0/freq"] - ro_freq) <= 2.5e6, \
        f"Separation returned an out-of-band readout freq {sep.proposal['readout/0/freq']:.5g} (want ≈{ro_freq:.5g})"
    assert fid.ok, "Fidelity did not return a readout amplitude"
    assert 0.4 < fid.proposal["readout/0/amp"] < 0.6, \
        f"Fidelity picked a wild readout amp {fid.proposal['readout/0/amp']}"
    assert rof.ok and 0.6 < rof.data[0]["fidelity"] <= 1.0, \
        f"ReadoutFidelity reported an implausible fidelity {rof.data[0]['fidelity']:.3f}"
    assert freq.ok, "Frequency fit failed"
    assert amp_c.ok, "coarse Amplitude fit failed"
    assert amp_f.ok, "fine Amplitude fit failed"
    assert phase.ok and "qubit/0/x90/vz" in phase.proposal, "Phase did not recover a virtual-Z"

    freq_after = abs(cfg["qubit/0/freq"] - F_GE)
    amp_after = abs(cfg["qubit/0/x90/amp"] - true_x90_amp)
    print(f"\n[x6y3] freq err {freq_before:.3g}→{freq_after:.3g}  "
          f"x90amp {cfg['qubit/0/x90/amp']:.4f} (true≈{true_x90_amp:.4f}) err {amp_before:.4f}→{amp_after:.4f}\n"
          f"  readout freq={sep.proposal['readout/0/freq']:.5g} amp={fid.proposal['readout/0/amp']:.4f} "
          f"fidelity={rof.data[0]['fidelity']:.3f}  vz={phase.proposal['qubit/0/x90/vz'][0]:+.4f}")
    assert freq_after < 0.4 * freq_before, "X6Y3 did not improve the qubit frequency"
    assert amp_after < 0.5 * amp_before, "X6Y3 did not improve the X90 amplitude"

