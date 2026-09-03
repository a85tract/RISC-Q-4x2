"""Reusable waveform generation helpers for RFSoC-style pulse sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np  
import matplotlib.pyplot as plt


WaveformKind = Literal["real", "complex"]


@dataclass(frozen=True)
class PulseSpec:
    """One sinusoidal component of a rectangular pulse."""

    frequency_MHz: float
    amplitude: float = 1.0
    phase_deg: float = 0.0


def generate_two_pulse_waveform(
    v_MHz: float = 82.0,
    omega_MHz: float = 1.765,
    first_duration_us: float = 100.0,
    second_duration_us: float = 20.0,
    sample_rate_MHz: float = 1000.0,
    first_amplitude: float = 1.0,
    second_amplitude: float = 1.0,
    second_frequency_MHz: float | None = None,
    second_phase_deg: float = 90.0,
    inter_pulse_gap_us: float = 5.0,
    waveform_kind: WaveformKind = "real",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a two-pulse waveform with a microsecond time base.

    The first pulse is the sum of two equal-amplitude tones:
    ``v + omega`` with phase 0 degrees, and ``v - omega`` with phase -180
    degrees. The second pulse is a single tone with phase 90 degrees.

    Frequencies are in MHz and times are in microseconds, so the sinusoid
    phase uses ``2*pi*frequency_mhz*time_us`` directly.

    Args:
        v_MHz: Center frequency, in MHz.
        omega_MHz: Frequency offset, in MHz.
        first_duration_us: Duration of the two-tone pulse, in microseconds.
        second_duration_us: Duration of the phase-90 pulse, in microseconds.
        sample_rate_MHz: Sample rate in samples per microsecond, equivalent
            to MS/s.
        first_amplitude: Amplitude of each tone in the first pulse.
        second_amplitude: Amplitude of the second pulse.
        second_frequency_MHz: Frequency of the second pulse. Defaults to
            ``v_MHz`` when omitted.
        second_phase_deg: Phase of the second pulse, in degrees.
        inter_pulse_gap_us: Optional zero-valued gap between pulses.
        waveform_kind: ``"real"`` for cosine samples, or ``"complex"`` for
            complex exponential/IQ samples.

    Returns:
        A tuple ``(time_us, waveform)``. ``time_us`` is in microseconds.
    """

    if sample_rate_MHz <= 0:
        raise ValueError("sample_rate_MHz must be positive")
    if first_duration_us < 0 or second_duration_us < 0 or inter_pulse_gap_us < 0:
        raise ValueError("pulse durations and gap must be non-negative")
    if waveform_kind not in ("real", "complex"):
        raise ValueError('waveform_kind must be either "real" or "complex"')

    second_frequency_MHz = v_MHz if second_frequency_MHz is None else second_frequency_MHz

    first_specs = (
        PulseSpec(v_MHz + omega_MHz, first_amplitude, 0.0),
        PulseSpec(v_MHz - omega_MHz, first_amplitude, -180.0),
    )
    second_specs = (PulseSpec(second_frequency_MHz, second_amplitude, second_phase_deg),)

    first = _make_pulse(first_specs, first_duration_us, sample_rate_MHz, waveform_kind)
    gap = np.zeros(_sample_count(inter_pulse_gap_us, sample_rate_MHz), dtype=first.dtype)
    second = _make_pulse(second_specs, second_duration_us, sample_rate_MHz, waveform_kind)

    waveform = np.concatenate((first, gap, second))
    time_us = np.arange(waveform.size, dtype=float) / sample_rate_MHz

    np.savez("waveform.npz", time_us=time_us, waveform=waveform)
    plt.plot(time_us, waveform)
    plt.xlabel("Time (us)")
    plt.ylabel("Amplitude")
    plt.title("Two-pulse waveform")
    plt.grid(True)
    plt.savefig("waveform.png")
    plt.show()
    return time_us, waveform


def _make_pulse(
    specs: tuple[PulseSpec, ...],
    duration_us: float,
    sample_rate_MHz: float,
    waveform_kind: WaveformKind,
) -> np.ndarray:
    sample_count = _sample_count(duration_us, sample_rate_MHz)
    t_us = np.arange(sample_count, dtype=float) / sample_rate_MHz

    if waveform_kind == "complex":
        pulse = np.zeros(sample_count, dtype=np.complex128)
        for spec in specs:
            phase_rad = np.deg2rad(spec.phase_deg)
            pulse += spec.amplitude * np.exp(1j * (2.0 * np.pi * spec.frequency_MHz * t_us + phase_rad))
        return pulse

    pulse = np.zeros(sample_count, dtype=float)
    for spec in specs:
        phase_rad = np.deg2rad(spec.phase_deg)
        pulse += spec.amplitude * np.cos(2.0 * np.pi * spec.frequency_MHz * t_us + phase_rad)
    return pulse


def _sample_count(duration_us: float, sample_rate_MHz: float) -> int:
    return int(round(duration_us * sample_rate_MHz))


if __name__ == "__main__":
    t_us, waveform = generate_two_pulse_waveform()
    print(f"Generated {waveform.size} samples from {t_us[0]:.3f} us to {t_us[-1]:.3f} us")
