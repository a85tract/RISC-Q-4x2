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
