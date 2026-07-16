"""Host-side unit tests for the QuantumModel ADC-seam models (riscq.sim.models). Deterministic,
no sim — the cosim datapath pins live in test_readout."""

import math
from pathlib import Path

import numpy as np

from riscq.map import SocMap, SocParams
from riscq.pulses import units
from riscq.sim import models

M = SocMap(SocParams.load(Path(__file__).resolve().parents[1] / "configs" / "sim-2q.json"))


def test_zero_model():
    z = models.ZeroModel()
    assert z.dac_ids() == []
    assert z.adc_batch(7, {}) == {}


def test_loopback_decimates_every_fourth():
    lb = models.LoopbackModel(gain=1.0, src=0, dst=3)
    assert lb.dac_ids() == [0]
    out = lb.adc_batch(0, {0: np.arange(16, dtype=np.int64)})   # DAC samples 0..15
    assert list(out[3]) == [0, 4, 8, 12]                        # ADC lane j = DAC sample 4j


def test_loopback_gain_and_clip():
    lb = models.LoopbackModel(gain=2.0, src=0, dst=0)
    dac = {0: np.array([100, 0, 0, 0, 20000, 0, 0, 0, -20000, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)}
    assert list(lb.adc_batch(0, dac)[0]) == [200, 32767, -32768, 0]   # gain then SInt16 clip


def test_loopback_delay():
    lb = models.LoopbackModel(gain=1.0, delay=2, src=0, dst=0)
    batch = lambda v: {0: np.full(16, v, dtype=np.int64)}
    assert list(lb.adc_batch(0, batch(10))[0]) == [0, 0, 0, 0]
    assert list(lb.adc_batch(1, batch(20))[0]) == [0, 0, 0, 0]
    assert list(lb.adc_batch(2, batch(30))[0]) == [10, 10, 10, 10]   # first batch emerges after 2


def test_multi_model_unions_dacs_and_sums_shared_adc():
    """MultiModel drives several sub-models together (spec 13 §8 simultaneous multi-qubit): dac_ids()
    unions the sub-models', and sub-models sharing an ADC (the frequency-multiplexed readout on this
    build, cores 0–6 → ADC 0) SUM on it, re-clipped to the converter range; distinct ADCs do not sum."""
    a = models.LoopbackModel(gain=1.0, src=0, dst=0)   # DAC 0 -> ADC 0
    b = models.LoopbackModel(gain=1.0, src=1, dst=0)   # DAC 1 -> ADC 0 (shared with a)
    mm = models.MultiModel([a, b])
    assert mm.dac_ids() == [0, 1]                       # union, order-preserving
    dac = {0: np.full(16, 10000, dtype=np.int64), 1: np.full(16, 25000, dtype=np.int64)}
    out = mm.adc_batch(0, dac)
    assert set(out) == {0} and list(out[0]) == [32767] * 4   # 10000 + 25000 = 35000 → SInt16 clip
    c = models.LoopbackModel(gain=1.0, src=1, dst=3)   # DAC 1 -> ADC 3 (distinct)
    out2 = models.MultiModel([a, c]).adc_batch(0, dac)
    assert set(out2) == {0, 3}
    assert list(out2[0]) == [10000] * 4 and list(out2[3]) == [25000] * 4


def test_tone_model_batch_time_locked():
    f = units.code_to_freq(F := 1024, M.params)
    tm = models.ToneModel(M, adc=0, freq_hz=f, amp=1000.0)
    assert tm.dac_ids() == []
    s0 = tm.adc_batch(0, {})[0]
    assert len(s0) == models.ADC_BATCH and s0[0] == 1000          # cos(0) = 1
    w = 2 * math.pi * f / (models.ADC_BATCH * M.params.dsp_freq_hz)
    assert abs(s0[1] - round(1000 * math.cos(w))) <= 1            # per-ADC-sample phase advance
    # phase continues across batches: sample index is ADC_BATCH*t + j
    s1 = tm.adc_batch(1, {})[0]
    assert abs(s1[0] - round(1000 * math.cos(w * models.ADC_BATCH))) <= 1


def test_twolevel_pi_pulse_flips_state_and_tone():
    c = 1000.0
    rabi = math.pi / (math.sqrt(2) * c)          # one all-`c` batch -> amp_est = sqrt(2)*c -> pi
    tl = models.TwoLevelModel(M, core=0, rabi_rad_per_amp=rabi, readout_code=2048,
                              readout_amp=5000.0, init_excited=True)
    assert tl.dac_ids() == [M.gate_dac(0)]
    assert tl.sigma_z() == -1.0                  # excited

    excited = tl.adc_batch(0, {M.gate_dac(0): np.zeros(16, dtype=np.int64)})   # DAC ~ 0: no rotation
    assert tl.sigma_z() == -1.0
    assert excited[M.adc_of(0)][0] == -5000      # |1> tone (batch 0, lane 0) is -amp*cos(0)

    tl.adc_batch(1, {M.gate_dac(0): np.full(16, c, dtype=np.int64)})        # a pi pulse
    assert tl.sigma_z() > 0.99                    # flipped to ground
    ground = tl.adc_batch(0, {M.gate_dac(0): np.zeros(16, dtype=np.int64)})  # same batch 0
    assert ground[M.adc_of(0)][0] > 4900         # |0> tone is +amp*cos(0) (pi flip vs |1>)


def test_twolevel_drive_axis_from_detuning():
    """A carrier off f_ge rotates the drive AXIS: two equal batches of a detuned drive rotate the
    Bloch vector about axes that differ by the detuning phase (the Ramsey mechanism), where a
    resonant drive keeps a fixed axis. Here we just check the axis is f_ge-dependent."""
    c = 5000.0
    dac = {M.gate_dac(0): np.full(16, c, dtype=np.int64)}
    # resonant (f_ge=0 vs a DC-ish drive): axis fixed -> repeated batches rotate the same way
    res = models.TwoLevelModel(M, rabi_rad_per_amp=0.05 / c, f_ge=0.0, init_excited=False)
    b0 = res._b.copy()
    res.adc_batch(0, dac)
    res.adc_batch(1, dac)
    # detuned: give f_ge a nonzero code so the demod axis ramps with batch time
    det = models.TwoLevelModel(M, rabi_rad_per_amp=0.05 / c,
                               f_ge=units.code_to_freq(200, M.params), init_excited=False)
    det.adc_batch(0, dac)
    det.adc_batch(1, dac)
    # the detuned run's Bloch vector differs from the resonant one (axis ramped between batches)
    assert not np.allclose(res._b, det._b), "detuning did not change the drive axis"
    assert not np.allclose(res._b, b0), "resonant drive did not rotate the state"


def test_twolevel_shot_noise_reproducible_and_off_by_default():
    base = dict(core=0, readout_code=2048, readout_amp=5000.0, init_excited=False)
    dac = {M.gate_dac(0): np.zeros(16, dtype=np.int64)}
    clean = models.TwoLevelModel(M, **base).adc_batch(0, dac)[M.adc_of(0)]
    # off by default: identical to a second no-noise instance
    assert np.array_equal(clean, models.TwoLevelModel(M, **base).adc_batch(0, dac)[M.adc_of(0)])
    # same seed -> identical noisy ADC (reproducible); and noise actually perturbs the samples
    n1 = models.TwoLevelModel(M, noise_scale=400.0, noise_seed=7, **base).adc_batch(0, dac)[M.adc_of(0)]
    n2 = models.TwoLevelModel(M, noise_scale=400.0, noise_seed=7, **base).adc_batch(0, dac)[M.adc_of(0)]
    assert np.array_equal(n1, n2)
    assert not np.array_equal(n1, clean)


def test_twolevel_state_is_qutip_qobj():
    import qutip
    tl = models.TwoLevelModel(M, init_excited=False)
    rho = tl.state()
    assert isinstance(rho, qutip.Qobj)
    assert qutip.expect(qutip.sigmaz(), rho) == 1.0     # ground


# ── projective readout (collapse mode, spec 08 §2.4 / B1) ──

def _read_window(tl, bz, i):
    """Plant Bloch-z = bz (a re-prepared superposition), then drive one readout window (readout-drive
    rising edge). Returns the sampled shot s ∈ {0,1} from the latched tone sign."""
    gate0 = np.zeros(16, dtype=np.int64)
    ro_off = np.zeros(16, dtype=np.int64)
    ro_on = np.full(16, 8000, dtype=np.int64)         # readout drive present (amp_est ≫ floor)
    tl._b[:] = (0.0, 0.0, bz)
    tl.adc_batch(2 * i, {M.gate_dac(0): gate0, M.ro_dac(0): ro_off})   # ro idle → arm the edge
    tl.adc_batch(2 * i + 1, {M.gate_dac(0): gate0, M.ro_dac(0): ro_on})  # rising edge → sample+collapse
    return 0 if tl._shot_amp > 0 else 1


def test_twolevel_projective_reads_readout_drive():
    soft = models.TwoLevelModel(M, collapse=False)
    proj = models.TwoLevelModel(M, collapse=True)
    assert soft.dac_ids() == [M.gate_dac(0)]                           # soft ignores ch1 (unchanged)
    assert set(proj.dac_ids()) == {M.gate_dac(0), M.ro_dac(0)}         # projective watches the ro drive


def test_twolevel_projective_collapses_to_a_pole():
    tl = models.TwoLevelModel(M, collapse=True, readout_amp=5000.0, noise_seed=1)
    for i, bz in enumerate((0.4, -0.7, 0.0)):
        s = _read_window(tl, bz, i)
        assert s in (0, 1)
        assert abs(abs(tl._b[2]) - 1.0) < 1e-12, "state did not collapse to a ±1 pole"
        assert (tl._b[2] > 0) == (s == 0)                             # |0>→s=0 (+amp), |1>→s=1 (−amp)


def test_twolevel_projective_binomial_statistics():
    """Repeated reads of a re-prepared superposition sample s ~ Bernoulli((1−bz)/2)."""
    tl = models.TwoLevelModel(M, collapse=True, readout_amp=5000.0, noise_seed=5)
    n, bz = 6000, 0.3
    ones = sum(_read_window(tl, bz, i) for i in range(n))
    p1 = (1 - bz) / 2                                                  # 0.35
    sigma = math.sqrt(p1 * (1 - p1) / n)
    assert abs(ones / n - p1) < 4 * sigma, f"p̂={ones / n:.4f} vs planted p1={p1}"


def test_twolevel_projective_extremes_deterministic():
    tl = models.TwoLevelModel(M, collapse=True, readout_amp=5000.0)
    assert all(_read_window(tl, 1.0, i) == 0 for i in range(20))       # |0> always reads 0
    assert all(_read_window(tl, -1.0, i) == 1 for i in range(20))      # |1> always reads 1


# ── dispersive readout (chi != 0, spec 13 Q2) ──

C_RO = 512                                                # a readout-drive DAC code (demod code 4x it)
F_RO = units.code_to_freq(C_RO, M.params)


def _drive(code, amp=9948, n=16):
    """One batch of DAC samples of a square-envelope tone at `code` (what the readout drive is)."""
    return _clip16(amp * np.cos(math.pi * code * np.arange(n) / (1 << 15)))


def _clip16(x):
    return np.clip(np.rint(x), -(1 << 15), (1 << 15) - 1).astype(np.int64)


def _dispersive(chi_code, **kw):
    return models.TwoLevelModel(M, readout_amp=8000.0, f_r=F_RO, chi=units.code_to_freq(chi_code, M.params),
                                kappa=units.code_to_freq(170, M.params), **kw)


def test_dispersive_is_off_by_default():
    """chi = 0 (the default) ⇒ bit-identical to the flat-tone model: same ADC samples, same DACs."""
    base = dict(core=0, readout_code=2048, readout_amp=5000.0, collapse=True, noise_seed=3)
    dac = {M.gate_dac(0): np.zeros(16, dtype=np.int64), M.ro_dac(0): _drive(C_RO)}
    old = models.TwoLevelModel(M, **base)
    new = models.TwoLevelModel(M, f_r=F_RO, kappa=1e6, chi=0.0, **base)     # f_r/kappa set, chi = 0
    assert old.dac_ids() == new.dac_ids()
    for t in range(4):
        assert np.array_equal(old.adc_batch(t, dac)[M.adc_of(0)], new.adc_batch(t, dac)[M.adc_of(0)])


def test_dispersive_recovers_the_drive_carrier_code():
    """The emitted tone TRACKS the drive: its ADC code is 4x the drive's DAC code, recovered exactly
    from the DAC samples (so the matched demod always sees the response at DC)."""
    for c in (256, 512, 700, 1024):
        assert models.TwoLevelModel._carrier_code(_drive(c).astype(float)) == c


def test_dispersive_lorentzian_shifts_with_the_state():
    """|0> pulls the resonator to f_r + chi and |1> to f_r − chi, so max |S21| at |0> is NOT max
    two-state separation: |S(|0>)| peaks at f_r + chi while |S(|0>) − S(|1>)| peaks at f_r. This is
    the ground truth Separation must find (and the |0>-magnitude cal must miss)."""
    chi_code = 60
    tl = _dispersive(chi_code)
    codes = [C_RO + k * chi_code for k in (-2, -1, 0, 1, 2)]
    z0 = [tl._lorentzian(units.code_to_freq(c, M.params), +1.0) for c in codes]
    z1 = [tl._lorentzian(units.code_to_freq(c, M.params), -1.0) for c in codes]
    mag0 = [abs(z) for z in z0]
    sep = [abs(a - b) for a, b in zip(z0, z1)]
    assert int(np.argmax(mag0)) == 3, "the |0> response does not peak at f_r + chi"
    assert int(np.argmax(sep)) == 2, "the two-state separation does not peak at f_r"


def _window(tl, bz, c, batches=40, theta=0.0):
    """Demodulate the model's ADC against the matched carrier (ADC code 4c) over a readout window,
    with the drive playing at DAC code `c` and carrier phase `theta` — i.e. what the decoder integrates."""
    tl._b[:] = (0.0, 0.0, bz)
    tl._ro_last = 0.0
    z = 0j
    for t in range(batches):
        lanes = tl.adc_batch(t, {M.gate_dac(0): np.zeros(16, dtype=np.int64),
                                 M.ro_dac(0): _tone(c, t, theta)})[M.adc_of(0)]
        j = np.arange(models.ADC_BATCH) + models.ADC_BATCH * t
        z += np.sum(lanes * np.exp(-1j * math.pi * (4 * c) * j / (1 << 15)))
    return z


def _tone(c, t, theta=0.0):
    """The readout drive's DAC samples for batch t: a square-envelope tone at code c, phase theta."""
    k = np.arange(16) + 16 * t
    return _clip16(units._amp_code(0.5) * np.cos(math.pi * c * k / (1 << 15) + theta))


def test_dispersive_response_survives_demodulation():
    """What the DECODER sees must BE the resonator: demodulated against the matched carrier, the
    integral has to trace S(f) — magnitude and phase — at every swept drive code. (The regression:
    a batch spans a fraction of a readout-carrier cycle, so any per-batch RMS amplitude estimate
    oscillates at 2ω and folds an image term back onto DC through the demod, bending
    |z(|0>) − z(|1>)| enough to move its argmax — i.e. faking exactly the answer Separation looks for.)"""
    chi = 60
    tl = _dispersive(chi)
    codes = [C_RO + k * chi for k in (-2, -1, 0, 1, 2)]
    z0 = [_window(tl, 1.0, c) for c in codes]
    z1 = [_window(tl, -1.0, c) for c in codes]
    s0 = [tl._lorentzian(units.code_to_freq(c, M.params), +1.0) for c in codes]
    s1 = [tl._lorentzian(units.code_to_freq(c, M.params), -1.0) for c in codes]
    gain = abs(z0[3]) / abs(s0[3])                       # one common scale (drive amp x window x ½)
    for k, c in enumerate(codes):
        assert abs(z0[k] / gain - s0[k]) < 0.06, f"demodulated |0> response off S(f) at code {c}"
        assert abs((z0[k] - z1[k]) / gain - (s0[k] - s1[k])) < 0.06
    assert int(np.argmax([abs(z) for z in z0])) == 3          # |0> magnitude peaks at f_r + chi
    assert int(np.argmax([abs(a - b) for a, b in zip(z0, z1)])) == 2   # separation peaks at f_r


def test_dispersive_response_is_locked_to_the_drive_phase():
    """The resonator answers ITS DRIVE: shift the drive's carrier phase by θ and the demodulated
    response rotates by exactly θ. This is what forbids re-synthesizing the tone from absolute time —
    the readout carrier is referenced to the PULSE START, so an absolute-time tone would land at a
    different phase on every shot of a swept code (the grid period is rounded for one code only), and
    the IQ cluster would smear into a ring instead of a blob."""
    tl = _dispersive(60)
    z = _window(tl, 1.0, C_RO)
    for theta in (0.7, -2.0, 3.0):
        rotated = _window(tl, 1.0, C_RO, theta=theta)
        assert abs(abs(rotated) - abs(z)) < 0.02 * abs(z)                     # same magnitude
        assert abs(math.remainder(np.angle(rotated) - np.angle(z) - theta, 2 * math.pi)) < 0.02


def test_dispersive_tone_scales_with_the_drive_amplitude():
    """The resonator's answer is proportional to the drive — the lever qcal's Fidelity sweeps (the
    flat tone's amplitude is a model constant, which is why the old Fidelity could not sweep it)."""
    tl = _dispersive(60, collapse=True)
    gate = np.zeros(16, dtype=np.int64)
    peak = lambda amp: max(abs(tl.adc_batch(0, {M.gate_dac(0): gate,
                                                M.ro_dac(0): _drive(C_RO, amp)})[M.adc_of(0)]))
    lo, hi = peak(2000), peak(9948)
    assert 0 < lo < hi
    assert abs(hi / lo - 9948 / 2000) < 0.1 * (9948 / 2000)     # linear in the drive amplitude
    assert peak(50) == 0                                        # under the drive floor ⇒ silent


# ── the ac-Stark drive phase (stark_rad_per_sigma != 0, spec 13 Q3) ──

def _stark(stark, **kw):
    """A driven qubit whose drive also Stark-shifts it. rabi is set so one full-amplitude batch is a
    π/2 rotation (amp_est = √2·c), i.e. σ (Σ amp_est) is one batch's worth."""
    c = 9948.0
    return models.TwoLevelModel(M, rabi_rad_per_amp=(math.pi / 2) / (math.sqrt(2) * c),
                                stark_rad_per_sigma=stark / (math.sqrt(2) * c), **kw), c


def test_stark_is_off_by_default():
    """stark_rad_per_sigma = 0 (the default) ⇒ bit-identical to the drive-only model: same Bloch
    vector after a driven batch, same ADC samples."""
    dac = lambda c: {M.gate_dac(0): np.full(16, int(c), dtype=np.int64)}
    old, c = _stark(0.0, readout_amp=5000.0, noise_seed=3)
    new = models.TwoLevelModel(M, rabi_rad_per_amp=old.rabi_rad_per_amp, readout_amp=5000.0,
                               noise_seed=3)                       # the pre-Q3 model, argument for argument
    for t in range(3):
        assert np.array_equal(old.adc_batch(t, dac(c))[M.adc_of(0)],
                              new.adc_batch(t, dac(c))[M.adc_of(0)])
        assert np.array_equal(old._b, new._b)


def test_stark_rotates_the_phase_with_the_drive():
    """The planted Z rotation is stark_rad_per_sigma · Σ amp_est over the pulse: drive N identical
    batches and the accrued phase is N × (the per-batch value). It rotates ONLY the phase (|b| and the
    z-projection of a pole are untouched), and an undriven batch accrues nothing."""
    eps = 0.4                                             # the Z rotation one 1-batch pulse must accrue
    tl, c = _stark(eps)
    tl.rabi_rad_per_amp = 0.0                             # Stark only: no xy rotation to confuse it
    tl._b[:] = (1.0, 0.0, 0.0)                            # on the equator: the phase is visible
    tl.adc_batch(0, {M.gate_dac(0): np.zeros(16, dtype=np.int64)})       # idle ⇒ no phase
    assert math.atan2(tl._b[1], tl._b[0]) == 0.0
    for n in (1, 2, 3):
        tl.adc_batch(n, {M.gate_dac(0): np.full(16, int(c), dtype=np.int64)})
        assert abs(math.atan2(tl._b[1], tl._b[0]) - n * eps) < 1e-9      # +z rotation, linear in σ
        assert abs(np.linalg.norm(tl._b) - 1.0) < 1e-12 and abs(tl._b[2]) < 1e-12
