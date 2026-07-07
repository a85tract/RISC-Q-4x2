"""Envelope shape library: complex float64 arrays on the CHANNEL's stored-sample grid, peak
|value| = FULL = 32767/32768 (the largest SF(16) value — "1.0" nominal, clip-safe by
construction).

Grid conventions (this build: batch = 16 DAC samples, gateInterp 4 / readoutInterp 16):
  - gate channel:    4 stored samples per batch (grid rate 4 * f_dsp), each held x4 at the DAC;
  - readout channel: 1 stored sample per batch (grid rate f_dsp), held x16.
An envelope shorter than a whole number of lines is zero-padded at packing time (riscq.pulses.pack)."""

from __future__ import annotations

import numpy as np

FULL = 32767 / 32768   # largest SF(16) value; all library envelopes peak here


def gaussian(n_samples: int, sigmas: float) -> np.ndarray:
    """Gaussian truncated at +-sigmas, peak FULL."""
    t = np.linspace(-sigmas, sigmas, n_samples)
    e = np.exp(-0.5 * t * t)
    return (FULL * e / e.max()).astype(complex)


def drag(n_samples: int, sigmas: float, alpha: float) -> np.ndarray:
    """DRAG: gaussian + 1j * alpha * d(gaussian)/dt, derivative in per-stored-sample units;
    the whole envelope is renormalized so max|env| = FULL (clip-safe for any alpha)."""
    g = np.exp(-0.5 * np.linspace(-sigmas, sigmas, n_samples) ** 2)
    e = g + 1j * alpha * np.gradient(g)
    return e * (FULL / np.abs(e).max())


def cos_edge_square(n_samples: int, ramp_frac: float) -> np.ndarray:
    """Square with raised-cosine edges; each ramp is round(n_samples * ramp_frac) samples."""
    n_ramp = round(n_samples * ramp_frac)
    if 2 * n_ramp > n_samples:
        raise ValueError(f"ramp_frac {ramp_frac}: 2 ramps of {n_ramp} exceed {n_samples} samples")
    e = np.full(n_samples, FULL + 0j)
    if n_ramp:
        ramp = FULL * 0.5 * (1 - np.cos(np.pi * np.arange(n_ramp) / n_ramp))
        e[:n_ramp] = ramp
        e[-n_ramp:] = ramp[::-1]
    return e


def square(n_samples: int) -> np.ndarray:
    return np.full(n_samples, FULL + 0j)


def arb(samples) -> np.ndarray:
    """User-supplied envelope; validated (1-D, |value| <= FULL) and converted to complex128."""
    a = np.asarray(samples, dtype=complex)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"arb envelope must be a non-empty 1-D array, got shape {a.shape}")
    peak = float(np.max(np.maximum(np.abs(a.real), np.abs(a.imag))))
    if peak > FULL:
        raise ValueError(f"arb envelope |value| {peak} exceeds FULL = 32767/32768")
    return a.copy()
