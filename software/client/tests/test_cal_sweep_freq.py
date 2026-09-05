"""`sweep_freq` (cal.base): the on-core frequency ramp at the build's own word width — the kernel
params it hands k_vna / k_cz_pop and the Hz axis that mirrors what the register keeps."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from riscq.cal.base import sweep_freq, sweep_q16
from riscq.map import SocMap, SocParams
from riscq.pulses import units

BITS = Path(__file__).resolve().parents[2] / "server" / "bits"
M32 = 1 << 32


def _map(fw):
    p = SocParams.load(BITS / "rfsoc4x2-1q-fine" / "params.json")
    return SocMap(dataclasses.replace(p, freq_width=fw))


def test_fw16_is_the_code_sweep():
    """At freq_width 16 the ramp starts on the seated code and the axis is the realized CODES: what
    `sweep_q16` used to hand out, code-exact in both directions."""
    m = _map(16)
    f0, f1, n = 80e6, 84e6, 9
    x0, dx, hz = sweep_freq(f0, f1, n, m)
    c0, c1 = units._freq_code(f0, m.params), units._freq_code(f1, m.params)
    assert x0 & (M32 - 1) == units.freq_word(f0, m.params) & (M32 - 1)     # seated code, bit-exact
    assert x0 == sweep_q16(c0, c1, n)[0]                     # the code sweep started on the same word
    codes = ((c0 << 16) + np.arange(n) * dx) >> 16           # what the 16-bit register keeps per point
    assert abs(codes[-1] - c1) <= 1                          # the span rounds in Hz, not in codes
    assert np.array_equal(hz, codes * units.sample_rate(m.params) / (1 << 16))
    assert all(units._freq_code(f, m.params) == c for f, c in zip(hz, codes))   # re-derivable


def test_fw32_keeps_the_whole_word():
    """At freq_width 32 every point is the full-precision word: the axis steps by exactly dx LSBs
    (1.83 Hz at the 4x2's rate) and re-derives the register word bit-for-bit."""
    m = _map(32)
    fs = units.sample_rate(m.params)
    f0, f1, n = 80e6, 80.001e6, 11
    x0, dx, hz = sweep_freq(f0, f1, n, m)
    assert x0 == units.freq_word(f0, m.params)
    assert np.allclose(np.diff(hz), dx * fs / M32, rtol=0, atol=1e-9)
    assert abs(hz[-1] - hz[0] - (f1 - f0)) < (n - 1) * fs / M32       # dx rounds to a 32-bit LSB once
    words = [units._wrap_signed(x0 + i * dx, 32) for i in range(n)]
    assert [units.freq_word(f, m.params) for f in hz] == words


@pytest.mark.parametrize("fw", [16, 32])
def test_axis_stays_in_the_callers_nyquist_zone(fw):
    """A sweep above the DAC's half rate reports the caller's band, not the folded register's alias
    (the readout of a 2nd-zone resonator must come back at its own frequency)."""
    m = _map(fw)
    fs = units.sample_rate(m.params)
    f0, f1, n = 0.7 * fs, 0.7 * fs + 5e6, 5
    x0, dx, hz = sweep_freq(f0, f1, n, m)
    assert abs(hz[0] - f0) <= fs / (1 << fw) and abs(hz[-1] - f1) <= fs / (1 << fw) * n
    assert np.all(np.diff(hz) > 0)
    assert units.word_to_freq(x0, m.params) < 0                       # the register itself is aliased


def test_wider_than_a_turn_or_past_the_sample_rate_is_refused():
    m = _map(32)
    fs = units.sample_rate(m.params)
    with pytest.raises(AssertionError):
        sweep_freq(0.0, 1.5 * fs, 3, m)                 # more than one turn
    with pytest.raises(AssertionError):
        sweep_freq(1.10 * fs, 1.11 * fs, 3, m)          # axis values units.freq_word cannot take back
    u = fs / M32
    with pytest.raises(AssertionError):
        sweep_freq(fs - 3 * u, fs - u, 4, m)            # legal endpoints, the last point rounds to fs
    with pytest.raises(AssertionError):
        sweep_freq(-fs / 2 + u, fs / 2, 11, m)          # asked < one turn, the rounded step realizes more
    m16 = _map(16)
    with pytest.raises(AssertionError):
        sweep_freq(-fs + fs / (1 << 17), -fs + 2 * fs / (1 << 17), 3, m16)   # ... and at -fs, 16-bit


def test_params_are_int32():
    """The kernel takes int32 params; a half-band sweep in two points has dxq = 2^31, which must come
    back wrapped (the on-core int32 accumulator wraps exactly like the hardware's phase accumulator)."""
    m = _map(32)
    fs = units.sample_rate(m.params)
    x0, dx, hz = sweep_freq(0.0, fs / 2, 2, m)
    assert -(1 << 31) <= dx < (1 << 31) and -(1 << 31) <= x0 < (1 << 31)
    assert (x0 + dx) & (M32 - 1) == units.freq_word(fs / 2, m.params) & (M32 - 1)
    assert hz[1] == pytest.approx(fs / 2)
