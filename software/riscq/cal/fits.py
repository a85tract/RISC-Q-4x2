"""The scipy fit helpers the calibrations use (spec 06 §3): the subset qcal actually fits.

Each returns a `Fit(ok, value, error, params)`: `value` is the quantity the calibration reads
(frequency, decay time, parabola vertex, line slope), `params` a dict of every fitted coefficient,
`error` the 1σ uncertainty on `value`. A failed or ill-conditioned fit (curve_fit does not
converge, or the covariance is non-finite) returns `ok=False` — never a silent fallback, so a
calibration can refuse to update the config.
"""

from __future__ import annotations

import math
import warnings
from collections import namedtuple

import numpy as np
from scipy.optimize import curve_fit

Fit = namedtuple("Fit", ["ok", "value", "error", "params"])

_FAIL = (RuntimeError, ValueError, TypeError)
_NONE = Fit(False, math.nan, math.nan, {})


def _fft_freq_seed(x: np.ndarray, y: np.ndarray) -> float:
    """Dominant non-DC FFT frequency of y on a ~uniform x grid (cycles per x-unit) — the seed
    that lets curve_fit lock onto the right sinusoid instead of a harmonic."""
    n = len(x)
    dx = (x[-1] - x[0]) / (n - 1) if n > 1 else 1.0
    spec = np.abs(np.fft.rfft(y - np.mean(y)))
    freqs = np.fft.rfftfreq(n, d=dx)
    if len(spec) <= 1:
        return 1.0 / (n * dx)
    return float(freqs[1 + int(np.argmax(spec[1:]))]) or 1.0 / (n * dx)


def fit_cosine(x, y) -> Fit:
    """y ≈ A·cos(2π·f·x + φ) + C, frequency seeded from the FFT peak. value = f (≥ 0)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    p0 = [max((y.max() - y.min()) / 2, 1e-9), _fft_freq_seed(x, y), 0.0, y.mean()]

    def model(t, A, f, phi, C):
        return A * np.cos(2 * np.pi * f * t + phi) + C

    try:
        popt, pcov = curve_fit(model, x, y, p0=p0, maxfev=20000)
    except _FAIL:
        return _NONE
    perr = np.sqrt(np.diag(pcov))
    A, f, phi, C = popt
    params = {"amp": abs(A), "freq": abs(f), "phase": phi, "offset": C}
    ok = bool(np.all(np.isfinite(popt)) and np.all(np.isfinite(perr)))
    return Fit(ok, abs(f), float(perr[1]), params)


def fit_damped_cosine(x, y) -> Fit:
    """y ≈ A·exp(−x/τ)·cos(2π·f·x + φ) + C. value = f (≥ 0); params carry the decay τ too."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    span = (x.max() - x.min()) or 1.0
    p0 = [max((y.max() - y.min()) / 2, 1e-9), span, _fft_freq_seed(x, y), 0.0, y.mean()]

    def model(t, A, tau, f, phi, C):
        return A * np.exp(-t / tau) * np.cos(2 * np.pi * f * t + phi) + C

    try:
        popt, pcov = curve_fit(model, x, y, p0=p0, maxfev=20000)
    except _FAIL:
        return _NONE
    perr = np.sqrt(np.diag(pcov))
    A, tau, f, phi, C = popt
    params = {"amp": A, "tau": tau, "freq": abs(f), "phase": phi, "offset": C}
    ok = bool(np.all(np.isfinite(popt)) and np.all(np.isfinite(perr)))
    return Fit(ok, abs(f), float(perr[2]), params)


def fit_exp_decay(x, y) -> Fit:
    """y ≈ A·exp(−x/τ) + C (A may be either sign). value = τ (> 0)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    span = (x.max() - x.min()) or 1.0
    p0 = [(y[0] - y[-1]) or 1.0, span / 2, y[-1]]

    def model(t, A, tau, C):
        return A * np.exp(-t / tau) + C

    try:
        popt, pcov = curve_fit(model, x, y, p0=p0, maxfev=20000)
    except _FAIL:
        return _NONE
    perr = np.sqrt(np.diag(pcov))
    A, tau, C = popt
    params = {"amp": A, "tau": tau, "offset": C}
    ok = bool(np.all(np.isfinite(popt)) and np.all(np.isfinite(perr)) and tau > 0)
    return Fit(ok, tau, float(perr[1]), params)


def fit_parabola(x, y) -> Fit:
    """y ≈ a·x² + b·x + c. value = the vertex x = −b/(2a) (the Amplitude n_gates>1 optimum)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # RankWarning on degenerate x -> caught below
            (a, b, c), cov = np.polyfit(x, y, 2, cov=True)
    except (np.linalg.LinAlgError, *_FAIL):
        return _NONE
    if a == 0 or not np.all(np.isfinite(cov)):
        return Fit(False, math.nan, math.nan, {"a": a, "b": b, "c": c})
    vertex = -b / (2 * a)
    da, db = b / (2 * a * a), -1 / (2 * a)     # ∂vertex/∂a, ∂vertex/∂b for error propagation
    var = da * da * cov[0, 0] + db * db * cov[1, 1] + 2 * da * db * cov[0, 1]
    params = {"a": a, "b": b, "c": c, "vertex": vertex}
    return Fit(True, vertex, math.sqrt(var) if var > 0 else math.nan, params)


def fit_linear(x, y) -> Fit:
    """y ≈ slope·x + intercept. value = slope; params carry the x-intercept (`root`, where y = 0)
    for the Phase/Frequency crossing fits."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # RankWarning on degenerate x -> caught below
            (slope, intercept), cov = np.polyfit(x, y, 1, cov=True)
    except (np.linalg.LinAlgError, *_FAIL):
        return _NONE
    ok = bool(np.all(np.isfinite([slope, intercept])) and np.all(np.isfinite(cov)) and slope != 0)
    params = {"slope": slope, "intercept": intercept}
    if slope != 0:
        params["root"] = -intercept / slope
    return Fit(ok, slope, math.sqrt(cov[0, 0]) if np.isfinite(cov[0, 0]) else math.nan, params)
