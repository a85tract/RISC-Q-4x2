"""M7b co-sim: the 32-bit frequency word, proven against the phase law rather than assumed.

The law (units.word_phase16 / golden.phase16): the 16-bit angle the CORDIC sees after `n` converter
samples is `((word * n) mod 2^fw) >> (fw - 16)`. Two claims follow, and both are tested HERE on the
real RTL rather than by reading generated Verilog:

  1. **legacy equivalence** — a seated 16-bit code written into a 32-bit build (`code << 16`) plays
     exactly what the 16-bit build plays, cycle for cycle (the hardware pre-advances its time input
     by the extra product stages so the phase reference does not move);
  2. **fractional resolution** — a word between two 16-bit codes plays a tone the 16-bit build
     cannot express, and it is the tone the law predicts.

Canaries per Codex M7b review: F = 1 with t = 4096 (N = 16) exercises the exact bit where the
sliced product first turns over, and the signed/wrap cases cover the accumulator's fold.
"""

import numpy as np
import pytest

from riscq import run as rq
from riscq.lang import ParamTable, compile_kernel, kernel
from riscq.map import ADC_BATCH, SocMap, SocParams
from riscq.pulses import Pulse, envelopes, golden, units

pytestmark = pytest.mark.cosim

DUR = 300


@pytest.fixture(scope="module")
def cosim_f32(request):
    if not request.config.getoption("--cosim"):
        pytest.skip("needs --cosim")
    from tests.conftest import CONFIGS, SW_ROOT
    from riscq.sim import server

    drv = server.start(CONFIGS / "rfsoc4x2-1q-f32.json", SW_ROOT / "build" / "rfsoc4x2-1q-f32")
    m = SocMap(SocParams.from_json(drv.sim.get_params()))
    yield drv, m
    server.stop(drv)


@kernel
def k_tone(ro: ParamTable, dur: int, cq: int):
    init_pulse_params(ro.pulses)  # noqa: F821
    set_freq(ro, cq)  # noqa: F821
    t1 = now() + 8192  # noqa: F821
    t1 = (t1 >> 12 << 12) + 4096
    play(ro, ro["v"], t1)  # noqa: F821
    wait_until(t1 + dur + 64)  # noqa: F821


def _capture(drv, m, prog, word):
    """Play a constant-envelope tone at the raw 32-bit register `word`; return the captured trace."""
    rq.rerun(drv, m, {0: prog}, params={0: {"cq": word}},
             timeout=(DUR + 20000) * 4 + 20_000_000)
    x = np.frombuffer(drv.read_block(m.robs(), 4 * ADC_BATCH * DUR), dtype="<i4")
    return x.astype(float)


@pytest.fixture(scope="module")
def tone_ready(cosim_f32):
    drv, m = cosim_f32
    assert m.params.freq_width == 32, m.params.freq_width
    drv.sim.set_model({"kind": "loopback", "src": 0, "dst": 0, "gain": 1.0, "delay": 0})
    spl = m.channel(1).samples_per_line
    ro = ParamTable(1, 0.0, {"v": Pulse(envelopes.square(spl), amp=1.0)})
    # `cq` is deliberately NOT bound here: a compile-time binding would bake it as a constant, and
    # every test below retunes the word at RUN time through rq.rerun(params=...).
    prog = compile_kernel(k_tone, m, tables=dict(ro=ro), dur=DUR)
    rq.setup(drv, m, {0: prog})
    rq.write_envelope(drv, m, 0, 1, 0,
                      np.tile(Pulse(0.9 * envelopes.square(spl), amp=1.0).packed_lines(m, 1),
                              (m.params.env_depth, 1)))
    rq.write_slot(drv, m, 0, prog, "ro", 0, "dur", DUR)
    rq.write_slot(drv, m, 0, prog, "ro", 0, "phase", 0)
    rq.write_slot(drv, m, 0, prog, "ro", 0, "amp", units._amp_code(0.9))
    return drv, m, prog


def _tone_freq_hz(x, fs):
    """Peak frequency of a captured trace (parabolic-interpolated FFT peak)."""
    X = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    f = np.fft.rfftfreq(x.size, 1 / fs)
    i = int(np.argmax(X[1:])) + 1
    if 0 < i < X.size - 1:                       # parabolic refinement
        a, b, c = X[i - 1], X[i], X[i + 1]
        i = i + 0.5 * (a - c) / (a - 2 * b + c + 1e-30)
    return float(i) * (f[1] - f[0])


def test_legacy_seated_word_is_unchanged(tone_ready):
    """A legacy code written as `code << 16` must play the SAME tone the 16-bit build plays."""
    drv, m, prog = tone_ready
    code = 683                                    # 81.96 MHz at the 4x2's rates
    x = _capture(drv, m, prog, code << 16)
    fs = ADC_BATCH * m.params.dsp_freq_hz
    got = _tone_freq_hz(x[40:-40], fs)
    want = code * 16 * m.params.dsp_freq_hz / 65536
    print(f"\n[m7b legacy] word {code << 16:#x} -> {got/1e6:.4f} MHz (16-bit build: {want/1e6:.4f})")
    assert abs(got - want) < 3 * fs / x.size


def test_fractional_word_lands_between_16bit_codes(tone_ready):
    """A word HALFWAY between two 16-bit codes plays the half-step tone — a frequency the 16-bit
    build cannot express at all."""
    drv, m, prog = tone_ready
    fs = ADC_BATCH * m.params.dsp_freq_hz
    step = 16 * m.params.dsp_freq_hz / 65536      # one 16-bit code = 120 kHz
    word = (683 << 16) + (1 << 15)               # +0.5 code
    x = _capture(drv, m, prog, word)
    got = _tone_freq_hz(x[40:-40], fs)
    want = 683 * step + step / 2
    print(f"[m7b fractional] +0.5 code -> {got/1e6:.4f} MHz (want {want/1e6:.4f}, "
          f"16-bit neighbours {683*step/1e6:.4f} / {684*step/1e6:.4f})")
    assert abs(got - want) < 3 * fs / x.size
    assert abs(got - 683 * step) > step / 4 and abs(got - 684 * step) > step / 4


def test_phase_law_matches_golden_bit_exactly(tone_ready):
    """The captured samples equal the GOLDEN model evaluated with freq_width = 32 — the same law
    software uses to compute words, applied to the real RTL's output. Canaries included:
    F = 1 (the smallest word, whose product turns over exactly at t = 4096 for N = 16) and a
    negative/aliased word."""
    drv, m, prog = tone_ready
    fs = ADC_BATCH * m.params.dsp_freq_hz
    for word in ((683 << 16) + 12345, -(101 << 16) - 7):
        x = _capture(drv, m, prog, word)
        # The model's per-sample phase on the ADC grid. The capture's absolute start time is not
        # known here, so compare with a QUADRATURE projection: a constant phase offset must not
        # count as a mismatch, while any error in the phase LAW (rate, slice, sign, wrap) does.
        n = np.arange(x.size) * (16 // ADC_BATCH)
        ph = np.array([golden.phase16(word, int(k), 32) for k in n]) * np.pi / (1 << 15)
        seg, phseg = x[60:-60], ph[60:-60]
        z = np.dot(seg, np.cos(phseg)) + 1j * np.dot(seg, np.sin(phseg))
        coh = abs(z) / (np.linalg.norm(seg) * np.sqrt(seg.size / 2) + 1e-30)
        print(f"[m7b law] word {word:#x}: coherence vs golden(fw=32) = {coh:.4f}")
        assert coh > 0.95, f"word {word:#x} does not follow the fw=32 phase law"

    # (the F = 1 canary lives in its own test below, where the law's prediction is exact)


# ── the decisive one (Codex M7b round-2 finding 1): SAMPLE-EXACT cross-build equality ──
#
# The quadrature canaries above are phase-blind, so they would pass even if the hardware's time
# pre-compensation were missing — the omission is exactly a CONSTANT −extraMulLatency·N·F rotation.
# This test runs the SAME program at the SAME absolute time on the freq_width-16 and freq_width-32
# builds, with the 32-bit one given the legacy seated word, and compares the captures SAMPLE BY
# SAMPLE. Nothing but true cycle-for-cycle equivalence passes it.

@pytest.fixture(scope="module")
def cosim_f16(request):
    if not request.config.getoption("--cosim"):
        pytest.skip("needs --cosim")
    from tests.conftest import CONFIGS, SW_ROOT
    from riscq.sim import server

    drv = server.start(CONFIGS / "rfsoc4x2-1q.json", SW_ROOT / "build" / "rfsoc4x2-1q")
    m = SocMap(SocParams.from_json(drv.sim.get_params()))
    yield drv, m
    server.stop(drv)


def _play_capture(drv, m, word, code_is_seated):
    """Set up the identical tone program on `drv` and return its capture."""
    spl = m.channel(1).samples_per_line
    ro = ParamTable(1, 0.0, {"v": Pulse(envelopes.square(spl), amp=1.0)})
    prog = compile_kernel(k_tone, m, tables=dict(ro=ro), dur=DUR)
    rq.setup(drv, m, {0: prog})
    rq.write_envelope(drv, m, 0, 1, 0,
                      np.tile(Pulse(0.9 * envelopes.square(spl), amp=1.0).packed_lines(m, 1),
                              (m.params.env_depth, 1)))
    rq.write_slot(drv, m, 0, prog, "ro", 0, "dur", DUR)
    rq.write_slot(drv, m, 0, prog, "ro", 0, "phase", 0)
    rq.write_slot(drv, m, 0, prog, "ro", 0, "amp", units._amp_code(0.9))
    rq.rerun(drv, m, {0: prog}, params={0: {"cq": word}},
             timeout=(DUR + 20000) * 4 + 20_000_000)
    return np.frombuffer(drv.read_block(m.robs(), 4 * ADC_BATCH * DUR), dtype="<i4").copy()


def test_legacy_word_is_sample_exact_across_widths(cosim_f16, cosim_f32):
    """freq_width 32 with a legacy seated word must emit the IDENTICAL samples as freq_width 16 —
    same absolute time, same program, no phase rotation. This is the test that would fail if the
    carrier's time input were not pre-advanced by `extraMulLatency`."""
    drv16, m16 = cosim_f16
    drv32, m32 = cosim_f32
    assert (m16.params.freq_width, m32.params.freq_width) == (16, 32)
    for model in ({"kind": "loopback", "src": 0, "dst": 0, "gain": 1.0, "delay": 0},):
        drv16.sim.set_model(model)
        drv32.sim.set_model(model)
    code = 683
    x16 = _play_capture(drv16, m16, code << 16, True)      # seated 16-bit word
    x32 = _play_capture(drv32, m32, code << 16, True)      # the SAME word, wide build
    n_bad = int(np.count_nonzero(x16 != x32))
    print(f"\n[m7b cross-width] {x16.size} samples, mismatching: {n_bad}; "
          f"first values {x16[60:64].tolist()} vs {x32[60:64].tolist()}")
    assert n_bad == 0, (f"{n_bad}/{x16.size} samples differ — the wide build is NOT cycle-for-cycle "
                        f"equivalent for a legacy word (a missing time pre-compensation looks "
                        f"exactly like this)")


def test_smallest_word_is_a_2_pow_minus_32_tone(tone_ready):
    """F = 1 must mean 2^-32 turns per sample (1.83 Hz), NOT one 16-bit code (120 kHz).

    Over this window a 1.83 Hz tone is indistinguishable from DC — the capture is a constant, just
    rotated by the phase the absolute batch time has already accumulated (the capture starts at an
    aligned t1 of several thousand batches, so the rotation is a couple of phase LSB, not zero).
    A hardware that mis-weighted the word by 2^16 would instead emit a 120 kHz sinusoid: ~26 deg of
    phase ramp across these 1200 samples, i.e. a visibly non-constant trace. So: still constant,
    and only slightly offset from word 0."""
    drv, m, prog = tone_ready
    x1 = _capture(drv, m, prog, 1)
    x0 = _capture(drv, m, prog, 0)
    body1, body0 = x1[60:-60], x0[60:-60]
    ripple = float(np.std(body1) / (abs(np.mean(body1)) + 1e-9))
    offset = float(abs(np.mean(body1) - np.mean(body0)) / (abs(np.mean(body0)) + 1e-9))
    print(f"[m7b F=1] ripple {ripple:.5f} (a 120 kHz mis-weighting would give ~0.1), "
          f"offset vs word 0 {offset:.5f}")
    # Bounds sized to the PREDICTION, not to comfort: a 2^16 mis-weighting gives ~10 % ripple and a
    # large offset, while the true 1.83 Hz tone measures ~0 ripple and a few 1e-5 of offset.
    assert ripple < 0.001, "word 1 is not a ~DC tone — the low bit is being weighted like a code"
    assert offset < 0.001, "word 1 is far from word 0 — its phase advance is far too fast"
