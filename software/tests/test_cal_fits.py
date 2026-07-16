"""Host unit tests for riscq.cal.fits: recover known parameters from synthetic + noisy data
within tolerance (ok=True), and ill-conditioned input returns ok=False (never a silent fallback)."""

import numpy as np

from riscq.cal import fits

RNG = np.random.default_rng(1)


def test_fit_cosine_recovers_frequency():
    x = np.linspace(0, 10, 80)
    y = 2.3 * np.cos(2 * np.pi * 0.37 * x + 0.8) + 1.1 + RNG.normal(0, 0.03, x.size)
    f = fits.fit_cosine(x, y)
    assert f.ok
    assert abs(f.value - 0.37) < 0.01
    assert abs(f.params["amp"] - 2.3) < 0.1


def test_fit_damped_cosine_recovers_frequency_and_tau():
    x = np.linspace(0, 20, 120)
    y = 1.5 * np.exp(-x / 6.0) * np.cos(2 * np.pi * 0.22 * x + 0.3) + 0.2 + RNG.normal(0, 0.02, x.size)
    f = fits.fit_damped_cosine(x, y)
    assert f.ok
    assert abs(f.value - 0.22) < 0.01
    assert abs(f.params["tau"] - 6.0) < 1.0


def test_fit_exp_decay_recovers_tau_with_negative_amplitude():
    x = np.linspace(0, 20, 60)
    y = -3.0 * np.exp(-x / 4.0) + 2.0 + RNG.normal(0, 0.02, x.size)   # A < 0 (rising toward C)
    f = fits.fit_exp_decay(x, y)
    assert f.ok
    assert abs(f.value - 4.0) < 0.3
    assert f.params["amp"] < 0


def test_fit_parabola_vertex():
    x = np.linspace(-3, 5, 40)
    y = 0.7 * (x - 1.6) ** 2 + 0.4 + RNG.normal(0, 0.01, x.size)
    f = fits.fit_parabola(x, y)
    assert f.ok
    assert abs(f.value - 1.6) < 0.05
    assert f.params["a"] > 0


def test_fit_absolute_value_recovers_the_vertex():
    """qcal's V-fit (spec 13 Q4): the UNSIGNED fringe frequency |δ + applied| vs the applied detuning
    is a·|x − b| + c with b = −δ. Four detunings (qcal's default) is the whole data set, so the seed
    has to be good — this is the shape Frequency actually fits."""
    delta = -60.0                                       # the config carrier's error, in freq codes
    x = np.array([-400.0, -200.0, 200.0, 400.0])        # the applied detunings
    y = np.abs(delta + x) + RNG.normal(0, 3.0, x.size)  # the measured fringe frequencies (unsigned)
    f = fits.fit_absolute_value(x, y)
    assert f.ok
    assert f.params["a"] > 0                            # positive curvature (qcal's guard)
    assert abs(f.value - (-delta)) < 10                 # b = −δ = +60


def test_fit_absolute_value_flags_negative_curvature():
    """An inverted V (a < 0) is qcal's 'negative curvature' failure — the fit must REPORT it (a < 0),
    so Frequency can refuse rather than write a nonsense frequency."""
    x = np.linspace(-4, 4, 9)
    y = -2.0 * np.abs(x - 1.0) + 10.0 + RNG.normal(0, 0.05, x.size)
    f = fits.fit_absolute_value(x, y)
    assert f.ok and f.params["a"] < 0


def test_rabi_rate_route_agrees_with_qcal_period_arithmetic():
    """spec 13 Q4 — the cross-check. We recover the gate amplitude through a physical Rabi RATE (fit P
    against the pulse's drive integral σ, then a* = target_angle / (rabi · dσ/dcode)); qcal reads it
    straight off the period of the same cosine in the amplitude axis (amp = period_frac / f_fit). σ is
    linear in the amp code, so the two are the SAME number — assert it on one synthetic sweep, for both
    gates (X90: a quarter period; X: a half)."""
    codes = np.linspace(600, 19000, 21)
    sigma_per_code = 3.7                                # gate_sigma is linear in the amp code
    sig = sigma_per_code * codes
    rabi = 4 * np.pi / sig[-1]                          # ~2 Rabi periods across the sweep
    P = (1 - np.cos(rabi * sig)) / 2 + RNG.normal(0, 0.01, codes.size)

    for gate, frac in (("X90", 0.25), ("X", 0.5)):
        target = 2 * np.pi * frac
        ours = fits.fit_cosine(sig, P)                  # riscq: the rate route (cal.qubit.Amplitude)
        a_star = (target / (2 * np.pi * ours.value)) / sigma_per_code
        qcal = fits.fit_cosine(codes, P)                # qcal: amp = period_frac / f_fit
        a_qcal = frac / qcal.value
        assert ours.ok and qcal.ok
        assert abs(a_star / a_qcal - 1) < 1e-3, f"{gate}: {a_star:.1f} (rate) vs {a_qcal:.1f} (qcal)"


def test_fit_linear_slope_and_root():
    x = np.linspace(0, 10, 30)
    y = 2.5 * x - 4.0 + RNG.normal(0, 0.05, x.size)
    f = fits.fit_linear(x, y)
    assert f.ok
    assert abs(f.value - 2.5) < 0.05
    assert abs(f.params["root"] - 1.6) < 0.05          # y = 0 at x = 4.0/2.5 = 1.6


def test_ill_conditioned_fits_return_not_ok():
    """Fewer data points than free parameters — no fit can be constrained, so ok=False."""
    assert not fits.fit_cosine([0.0, 1.0, 2.0], [1.0, 2.0, 3.0]).ok
    assert not fits.fit_damped_cosine([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 1.0, 2.0]).ok
    assert not fits.fit_exp_decay([0.0, 1.0], [1.0, 2.0]).ok
    assert not fits.fit_parabola([0.0, 1.0], [1.0, 2.0]).ok
    assert not fits.fit_linear([0.0, 1.0], [1.0, 2.0]).ok
    assert not fits.fit_absolute_value([0.0, 1.0], [1.0, 2.0]).ok
    # a NaN in the data is likewise refused, not silently fitted
    x = np.linspace(0, 5, 20)
    assert not fits.fit_cosine(x, np.full(20, np.nan)).ok
