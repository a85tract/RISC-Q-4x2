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
    # a NaN in the data is likewise refused, not silently fitted
    x = np.linspace(0, 5, 20)
    assert not fits.fit_cosine(x, np.full(20, np.nan)).ok
