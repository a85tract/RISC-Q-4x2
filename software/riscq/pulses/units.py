"""Physical-unit <-> hardware-code conversions. THE definitions — nothing else in the stack
converts units. Pinned by measurement in the M1 cosim tests (VNA frequency test, DAC amp test),
not by RTL reading."""

from __future__ import annotations

import math

from riscq.map import ADC_BATCH, BATCH_SIZE, SocParams
from riscq.pulses.golden import AMP_SCALE  # noqa: F401  (re-exported: THE amplitude scale)

_PERIOD = 1 << 16    # SF(16) phase wraps mod 2^16 (one full turn per sample == the sample rate)


def _wrap_sf16(f_hz: float, code: int, rate: float, label: str, max_turns: int = 1) -> int:
    """Fold a raw per-sample phase-advance code into signed SF(16) [-2^15, 2^15). The hardware
    phase accumulator wraps mod 2^16, so a positive code above Nyquist (2^15) is bit-for-bit the
    same tone as code - 2^16 below zero; fold it there rather than rejecting it. `max_turns` bounds
    how many full 2^16 turns of aliasing are legitimate: the DAC allows one (a tone must be below the
    sample rate), the demod allows BATCH_SIZE/ADC_BATCH — its per-sample turn is that many times
    smaller, so the same physical band up to the sample rate spans that many turns. Only a code past
    `max_turns` full turns from DC is a genuine out-of-range error and still fails loud."""
    if not -max_turns * _PERIOD < code < max_turns * _PERIOD:
        raise ValueError(f"{label} {f_hz} Hz -> code {code} exceeds {max_turns} full turn(s) "
                         f"[{-max_turns * _PERIOD + 1}, {max_turns * _PERIOD - 1}] (rate {rate:g} Hz)")
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
    return _wrap_sf16(f_hz, round(f_hz * (1 << 16) / fs), fs, "freq")


def code_to_freq(code: int, params: SocParams) -> float:
    return code * sample_rate(params) / (1 << 16)


def demod_freq_to_code(f_hz: float, params: SocParams) -> int:
    """Demod-LO frequency code = per-ADC-SAMPLE phase advance in pi units, SF(16). The ADC has
    ADC_BATCH (4) samples/batch vs the DAC's BATCH_SIZE (16), so for the same physical frequency
    the demod code is BATCH_SIZE/ADC_BATCH = 4x the DAC's freq_to_code. Pinned by measurement in
    the M3 loopback test (a DAC tone at code F loops back onto the ADC at demod code 4F), NOT by
    re-deriving the RTL: code = round(f_hz * 2^16 / (ADC_BATCH * dsp_freq_hz)), folded mod 2^16 into
    signed SF(16) — a tone above the demod's own Nyquist aliases exactly like freq_to_code (on-core,
    set_freq(demod, 4 * dac_code) writes the product into the same 16-bit register). The demod's
    per-sample turn is BATCH_SIZE/ADC_BATCH x smaller than the DAC's, so the DAC's full band up to
    the sample rate spans that many demod turns; fails loud only past it."""
    rate = ADC_BATCH * params.dsp_freq_hz
    return _wrap_sf16(f_hz, round(f_hz * (1 << 16) / rate), rate, "demod freq",
                      max_turns=BATCH_SIZE // ADC_BATCH)


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
