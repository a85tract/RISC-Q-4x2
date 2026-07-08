"""Physical-unit <-> hardware-code conversions. THE definitions — nothing else in the stack
converts units. Pinned by measurement in the M1 cosim tests (VNA frequency test, DAC amp test),
not by RTL reading."""

from __future__ import annotations

import math

from riscq.map import ADC_BATCH, BATCH_SIZE, SocParams
from riscq.pulses.golden import AMP_SCALE  # noqa: F401  (re-exported: THE amplitude scale)

_PERIOD = 1 << 16    # SF(16) phase wraps mod 2^16 (one full turn per sample == the sample rate)


def _wrap_sf16(f_hz: float, code: int, rate: float) -> int:
    """Fold a raw per-sample phase-advance code into signed SF(16) [-2^15, 2^15). The hardware
    phase accumulator wraps mod 2^16, so a positive code above Nyquist (2^15) is bit-for-bit the
    same tone as code - 2^16 below zero; fold it there rather than rejecting it. Only a code a
    full turn or more from DC (a tone at or past the sample rate) is a genuine out-of-range error
    and still fails loud."""
    if not -_PERIOD < code < _PERIOD:
        raise ValueError(f"freq {f_hz} Hz -> code {code} exceeds one full turn "
                         f"[{-_PERIOD + 1}, {_PERIOD - 1}] (rate {rate:g} Hz)")
    return ((code + (1 << 15)) & 0xFFFF) - (1 << 15)


def sample_rate(params: SocParams) -> float:
    """DAC sample rate: 16 samples per batch, one batch per dsp cycle."""
    return BATCH_SIZE * params.dsp_freq_hz


def freq_to_code(f_hz: float, params: SocParams) -> int:
    """Carrier/demod frequency code = per-SAMPLE phase advance in pi units, SF(16):
    code = round(f_hz * 2^16 / (16 * dsp_freq_hz)), folded mod 2^16 into signed SF(16) so a tone
    above Nyquist aliases to its negative-frequency code (the phase accumulator wraps the same way).
    Fails loud only past one full turn."""
    fs = sample_rate(params)
    return _wrap_sf16(f_hz, round(f_hz * (1 << 16) / fs), fs)


def code_to_freq(code: int, params: SocParams) -> float:
    return code * sample_rate(params) / (1 << 16)


def demod_freq_to_code(f_hz: float, params: SocParams) -> int:
    """Demod-LO frequency code = per-ADC-SAMPLE phase advance in pi units, SF(16), DERIVED from
    the DAC code: BATCH_SIZE/ADC_BATCH (4x) freq_to_code, folded mod 2^16 — exactly what the
    on-core set_freq(demod, 4 * dac_code) truncates into the same 16-bit register. It must NOT be
    rounded independently from f_hz: the demod has to track the tone the DAC actually synthesizes
    (its rounded code), and an independent round(f * 2^16 / adc_rate) lands 1-2 LSB (30-61 kHz at
    2 GS/s) off 4x the DAC code at most frequencies — a drive-vs-demod offset that rotates the
    readout phase shot to shot (the hardware iq_scatter ring, 2026-07). Accepts any tone
    freq_to_code accepts (the full DAC band = BATCH_SIZE/ADC_BATCH demod turns); fails loud past
    it. The 4x relation itself is pinned by measurement in the M3 loopback test."""
    code = (BATCH_SIZE // ADC_BATCH) * freq_to_code(f_hz, params)
    return ((code + (1 << 15)) & 0xFFFF) - (1 << 15)


def demod_code_to_freq(code: int, params: SocParams) -> float:
    return code * (ADC_BATCH * params.dsp_freq_hz) / (1 << 16)


def phase_to_code(rad: float) -> int:
    """Phase code in pi units SF(16); phase wraps by design (two's-complement wrap = exact
    2*pi wrap), so any angle is reduced mod 2*pi first."""
    x = math.remainder(rad, 2 * math.pi)          # [-pi, pi]
    code = round(x / math.pi * (1 << 15))
    return ((code + (1 << 15)) & 0xFFFF) - (1 << 15)   # +pi and -pi are the same angle


def amp_to_code(a: float) -> int:
    """Amplitude code = round(a * AMP_SCALE), a in [-1, 1]. AMP_SCALE (= 19896) is the max safe
    code under prescaleAmp: software pre-multiplies by 1/K_cordic (K = prod sqrt(1+2^-2i),
    ~1.64676) with the same error headroom the hardware gives its own phasor constant, so the
    no-saturate datapath never wraps. Pinned empirically by the M1 DAC amp test."""
    if not -1.0 <= a <= 1.0:
        raise ValueError(f"amplitude {a} outside [-1, 1]")
    return round(a * AMP_SCALE)


def batches(t_ns: float, params: SocParams) -> int:
    """Duration in batches (dsp cycles) nearest to t_ns."""
    return round(t_ns * 1e-9 * params.dsp_freq_hz)


def ns(n_batches: int, params: SocParams) -> float:
    return n_batches / params.dsp_freq_hz * 1e9
