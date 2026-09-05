"""Host-pure regression for `riscq.artiqapi` — the ARTIQ-shaped layer.

These check the layer's CONTRACT, not its plumbing: the phase-mode arithmetic against the golden
model, the timeline semantics (set() is a non-advancing state event; parallel branches share a
start), the edge snapping and its honest error report, and that the ion-trap sequence written in
the API lands on exactly the hardware parameters the hand-written script reached.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from riscq import artiqapi as A
from riscq.map import SocMap, SocParams
from riscq.pulses import Pulse, envelopes, golden, units

CONFIGS = Path(__file__).resolve().parents[3] / "gateware" / "configs"


def _map(name="rfsoc4x2-1q-fine"):
    path = CONFIGS / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{name}.json not in this checkout")
    return SocMap(SocParams.from_json(path.read_text()))


def _phase_at(m, W, P, n):
    """What the hardware puts out at absolute DAC sample n (16-bit phase units)."""
    return (golden.phase16(W, n - A._TTP16, m.params.freq_width) + (P >> 16)) % 65536


def _err16(got, want):
    return ((got - want + 32768) % 65536) - 32768


# ── the phase modes ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("f_hz", [82.0e6, 83.765e6, 80.235e6])
@pytest.mark.parametrize("p_turns", [0.0, 0.25, 0.5, 0.123456])
def test_tracking_is_the_per_frequency_metronome(f_hz, p_turns):
    """TRACKING: phi(n) = p + W*(n - T). The phase at any instant depends only on that instant,
    so two pulses at the same frequency are phase-coherent however far apart they are."""
    m = _map()
    c = A.Core(m)
    ch = A.DDSChannel(c, 1)
    ch.set(f_hz, phase=p_turns)          # TRACKING is the default
    ch.sw.pulse(1 * A.us)
    A.delay(c, 3.7 * A.us)               # a deliberately non-round gap
    ch.set(f_hz, phase=p_turns)
    ch.sw.pulse(1 * A.us)
    sch = A.plan(c)

    t1 = 123456                           # any sequence origin
    p16 = round(p_turns * 65536) % 65536
    for e in sch.events:
        P = (e.phase_const - e.freq_word * (t1 * 16)) % A._M32
        n = (t1 + e.batch) * 16 + e.lead_zeros
        want = (p16 + golden.phase16(e.freq_word, n - t1 * 16, m.params.freq_width)) % 65536
        assert abs(_err16(_phase_at(m, e.freq_word, P, n), want)) <= 1


def test_absolute_anchors_at_the_set_call_not_the_pulse():
    """ARTIQ ABSOLUTE: phi = p at the instant of `set()`. A `delay()` between set() and pulse()
    must therefore show up as f*delay of carrier phase at the pulse start."""
    m = _map()
    c = A.Core(m)
    ch = A.DDSChannel(c, 1)
    f, gap = 82.0e6, 3.7 * A.us
    ch.set(f, phase=0.0, phase_mode=A.PHASE_MODE_ABSOLUTE)
    set_mu = c.now_mu()
    A.delay(c, gap)                       # the pulse starts later than the set()
    ch.sw.pulse(1 * A.us)
    e = A.plan(c).events[0]
    assert e.set_mu == set_mu and e.start_mu > e.set_mu

    t1 = 987654
    P = (e.phase_const - e.freq_word * (t1 * 16)) % A._M32
    n_set = (t1 * 16) + e.set_mu
    assert abs(_err16(_phase_at(m, e.freq_word, P, n_set), 0)) <= 1      # p = 0 AT THE set()
    n0 = (t1 + e.batch) * 16 + e.lead_zeros                              # at the pulse start
    want = golden.phase16(e.freq_word, n0 - n_set, m.params.freq_width)
    assert abs(_err16(_phase_at(m, e.freq_word, P, n0), want)) <= 1


def test_continuous_carries_the_accumulated_phase_across_a_hop():
    """After hopping away to f2 and back to f1, the two modes disagree in a PREDICTABLE way:

      TRACKING   the phase at the third pulse is f1's metronome at that instant, so between two
                 runs whose excursion differs by dn it advances by W1*dn — the excursion itself is
                 irrelevant (this is what "coherent" buys);
      CONTINUOUS the accumulator ran at f2 during the excursion, so the same comparison advances by
                 W2*dn instead — where you land depends on how long you stayed away.
    """
    m = _map()
    f1, f2 = 82.0e6, 80.235e6
    W1, W2 = units.freq_word(f1, m.params), units.freq_word(f2, m.params)
    t1 = 4242

    def third_pulse(stay_s, mode):
        c = A.Core(m)
        ch = A.DDSChannel(c, 1)
        ch.set(f1, phase=0.0)                       # the first set must be TRACKING/ABSOLUTE
        ch.sw.pulse(1 * A.us)
        ch.set(f2, phase=0.0, phase_mode=mode)
        ch.sw.pulse(stay_s)
        ch.set(f1, phase=0.0, phase_mode=mode)
        ch.sw.pulse(1 * A.us)
        e = A.plan(c).events[-1]
        P = (e.phase_const - e.freq_word * (t1 * 16)) % A._M32
        n = (t1 + e.batch) * 16 + e.lead_zeros
        return _phase_at(m, e.freq_word, P, n), n

    for mode, W in ((A.PHASE_MODE_TRACKING, W1), (A.PHASE_MODE_CONTINUOUS, W2)):
        (pa, na), (pb, nb) = third_pulse(2.0 * A.us, mode), third_pulse(2.5 * A.us, mode)
        want = golden.phase16(W, nb - na, m.params.freq_width)
        assert abs(_err16(pb - pa, want)) <= 1, (
            f"{A._MODE_NAME[mode]}: phase advanced by {(pb - pa) % 65536} over dn = {nb - na}, "
            f"expected {want} (i.e. the "
            f"{'f1 metronome' if mode == A.PHASE_MODE_TRACKING else 'f2 accumulation'})")

    # and the two modes really do land in different places
    pt, _ = third_pulse(2.5 * A.us, A.PHASE_MODE_TRACKING)
    pc, _ = third_pulse(2.5 * A.us, A.PHASE_MODE_CONTINUOUS)
    assert abs(_err16(pt, pc)) > 100, "TRACKING and CONTINUOUS should not coincide after a hop"


def test_continuous_needs_a_prior_tone():
    m = _map()
    c = A.Core(m)
    ch = A.DDSChannel(c, 1)
    ch.set(82.0e6, phase=0.0, phase_mode=A.PHASE_MODE_CONTINUOUS)
    ch.sw.pulse(1 * A.us)
    with pytest.raises(RuntimeError, match="does not exist yet"):
        A.plan(c)


# ── the timeline ─────────────────────────────────────────────────────────────────────────────────

def test_parallel_branches_share_a_start_and_the_cursor_takes_the_latest_end():
    m = _map()
    c = A.Core(m)
    ch0, ch1 = A.DDSChannel(c, 0), A.DDSChannel(c, 1)
    A.delay(c, 1 * A.us)
    with A.parallel(c):
        with A.branch(c):
            ch0.set(80e6); ch0.sw.pulse(3 * A.us)
        with A.branch(c):
            ch1.set(81e6); ch1.sw.pulse(5 * A.us)
    assert c.now_mu() == c.seconds_to_mu(6 * A.us)          # 1 us + the LONGER branch
    e0, e1 = A.plan(c).events
    assert e0.start_mu == e1.start_mu == c.seconds_to_mu(1 * A.us)


def test_set_does_not_advance_the_cursor():
    m = _map()
    c = A.Core(m)
    ch = A.DDSChannel(c, 1)
    before = c.now_mu()
    ch.set(82e6, phase=0.3)
    assert c.now_mu() == before


def test_mu_is_one_dac_sample():
    m = _map()
    c = A.Core(m)
    assert c.seconds_to_mu(1 / (16 * m.params.dsp_freq_hz)) == 1
    assert c.mu_to_seconds(1) == pytest.approx(127.157e-12, rel=1e-4)


# ── edge snapping, reported honestly ─────────────────────────────────────────────────────────────

def test_edges_snap_to_the_channel_grid_and_the_error_is_reported():
    m = _map()
    c = A.Core(m)
    ch = A.DDSChannel(c, 1)
    step = 16 // m.channel(1).samples_per_line
    c.at_s(105 * A.us)                     # 105 us is NOT on the batch grid (2457.6 batches)
    ch.set(82e6, phase=0.25)
    ch.sw.pulse(20 * A.us)
    e = A.plan(c).events[0]
    assert e.start_dac % step == 0 and e.end_dac % step == 0
    assert abs(e.start_error_ps(c)) <= step / (16 * m.params.dsp_freq_hz) * 1e12 / 2 + 1e-6
    assert e.start_error_ps(c) == pytest.approx(50.9, abs=1.0)      # the known optimum


def test_overlapping_pulses_on_one_channel_are_refused():
    m = _map()
    c = A.Core(m)
    ch = A.DDSChannel(c, 1)
    ch.set(82e6); ch.sw.pulse(2 * A.us)
    c.at_s(1 * A.us)                       # rewind into the previous pulse
    ch.set(82e6); ch.sw.pulse(2 * A.us)
    with pytest.raises(RuntimeError, match="overlap"):
        A.plan(c)


# ── the ion-trap sequence must reach the hand-written script's numbers ────────────────────────────

def test_ion_trap_sequence_matches_the_hand_written_parameters():
    """`Tests/experiment/ms_waveform_4x2.py --exact --frame absolute --edge` reached: pulse 2 opens
    at batch 51609 with 10 leading DAC samples zeroed, 9831 batches, and the three exact tones."""
    m = _map()
    c = A.Core(m)
    ch0, ch1 = A.DDSChannel(c, 0), A.DDSChannel(c, 1)
    with A.parallel(c):
        with A.branch(c):
            ch0.set(83.765e6, phase=0.0, amplitude=0.4); ch0.sw.pulse(100 * A.us)
        with A.branch(c):
            ch1.set(80.235e6, phase=0.5, amplitude=0.4); ch1.sw.pulse(100 * A.us)
    A.delay(c, 5 * A.us)
    ch1.set(82.0e6, phase=0.25, amplitude=0.4, phase_mode=A.PHASE_MODE_ABSOLUTE)
    ch1.sw.pulse(20 * A.us)
    sch = A.plan(c)

    a, b, v = sch.events
    assert (a.batch, a.dur_batches) == (0, 49152)
    assert (b.batch, b.dur_batches) == (0, 49152)
    assert (v.batch, v.lead_zeros) == (51609, 10)             # the +50.9 ps optimum
    assert v.batch + v.dur_batches == 61440                   # the fall is exactly ideal
    for e, f in zip((a, b, v), (83.765e6, 80.235e6, 82.0e6)):
        assert e.freq_word == units.freq_word(f, m.params)
        assert abs(units.word_to_freq(e.freq_word, m.params) - f) < 2.0


def test_generated_kernel_is_valid_source():
    m = _map()
    c = A.Core(m)
    ch = A.DDSChannel(c, 1)
    ch.set(82e6, phase=0.25, amplitude=0.4)
    ch.sw.pulse(20 * A.us)
    src = A.generate_kernel_source(A.plan(c), doc="unit test")
    compile(src, "<generated>", "exec")                       # it must at least be Python
    assert "set_phase" in src and "play(" in src and "wait_until" in src


def test_envelope_images_carry_the_partial_lines():
    m = _map()
    c = A.Core(m)
    ch = A.DDSChannel(c, 1)
    c.at_s(105 * A.us)
    ch.set(82e6, phase=0.25, amplitude=0.4)
    ch.sw.pulse(20 * A.us)
    sch = A.plan(c)
    img = A.envelope_images(sch)[1]
    e = sch.events[0]
    spl = m.channel(1).samples_per_line
    step = 16 // spl
    lead = e.lead_zeros // step
    assert np.all(img[e.env_line][:lead] == 0)                # the silent head
    assert np.all(np.abs(img[e.env_line][lead:]) > 0)         # the rest is the unit envelope
    assert np.all(np.abs(img[sch.first_line]) > 0)            # full lines are untouched
    assert np.array_equal(img[e.env_line + 1], img[sch.first_line]), (
        "the pair's copy line must hold the full square the lead line free-runs into")


# ── the one-shot capture needs a continuously firing readout channel ──────────────────────────────

def test_fill_gaps_makes_the_capture_channel_contiguous():
    """The trace records only while the readout channel fires and its write address RESETS between
    fires, so a silent gap comes back as two traces overlaid at address 0. `fill_gaps` plays the
    gap at amplitude 0 instead — the DAC output is zero either way.

    This is a REAL failure that co-sim caught: without the filler, pulse 2 overwrote the start of
    the trace (pulse-2 region all zeros, spikes in the pulse-1 region).
    """
    m = _map()
    c = A.Core(m)
    ro = A.DDSChannel(c, 1)
    ro.set(80.235e6, phase=0.5, amplitude=0.4); ro.sw.pulse(100 * A.us)
    A.delay(c, 5 * A.us)
    ro.set(82.0e6, phase=0.25, amplitude=0.4, phase_mode=A.PHASE_MODE_ABSOLUTE)
    ro.sw.pulse(20 * A.us)

    assert A.fill_gaps(c, 1) == 1
    sch = A.plan(c)
    evs = [e for e in sch.events if e.channel == 1]
    assert len(evs) == 3
    for prev, nxt in zip(evs, evs[1:]):                  # contiguous in BATCHES, no reset
        assert prev.batch + prev.dur_batches == nxt.batch
    filler = evs[1]
    assert filler.amp_code == 0, "the filler must be silent on the DAC"
    assert A.fill_gaps(c, 1) == 0, "already contiguous — nothing more to insert"


def test_fill_gaps_leaves_a_contiguous_sequence_alone():
    m = _map()
    c = A.Core(m)
    ch = A.DDSChannel(c, 1)
    ch.set(82e6); ch.sw.pulse(2 * A.us)
    ch.set(82e6); ch.sw.pulse(2 * A.us)                  # back-to-back, no gap
    assert A.fill_gaps(c, 1) == 0


def test_phase_is_anchored_at_the_unrounded_request_not_the_mu_grid():
    """The phase anchor must be the instant the user ASKED for, not the mu-rounded cursor.

    105 us is 825753.6 DAC samples: rounding the anchor to 825754 mu would put 0.4 samples of time
    into the carrier phase — 1.5 deg at 82 MHz. Co-sim caught exactly this as a 311-code residual
    against the hand-written capture, which anchors at the un-rounded instant.
    """
    m = _map()
    f = 82.0e6
    W = units.freq_word(f, m.params)
    c = A.Core(m)
    ch = A.DDSChannel(c, 1)
    c.at_s(105 * A.us)                       # NOT on the mu grid: 825753.6 DAC samples
    ch.set(f, phase=0.0, phase_mode=A.PHASE_MODE_ABSOLUTE)
    ch.sw.pulse(20 * A.us)
    e = A.plan(c).events[0]

    assert e.set_s == 105 * A.us and e.set_mu == 825754     # the cursor IS rounded ...
    t1 = 7777
    P = (e.phase_const - e.freq_word * (t1 * 16)) % A._M32
    # ... but the phase evaluated at the exact instant must be the requested 0, so the phase at the
    # rounded instant must be off by exactly W * 0.4 samples (1.5 deg), not 0.
    at_rounded = _phase_at(m, W, P, t1 * 16 + 825754)
    want_at_rounded = round(W * 0.4) >> 16
    assert abs(_err16(at_rounded, want_at_rounded)) <= 1, (
        f"phase at the rounded instant {at_rounded * 360 / 65536:.3f} deg, expected "
        f"{want_at_rounded * 360 / 65536:.3f} deg (= W * 0.4 samples)")
    assert abs(want_at_rounded * 360 / 65536 - 1.5) < 0.1   # sanity: it really is ~1.5 deg


def test_phase_and_frequency_words_equal_the_hand_written_script():
    """The decisive equivalence check, and it needs no simulator: the API's phase CONSTANTS and
    frequency words must equal, bit for bit, what `ms_waveform_4x2.py --exact --frame absolute
    --edge` puts in the same registers — that script's output is verified on hardware.

    The hand-written kernel writes `P = p2q - W*((t1 + DUR1 + gap - ttp)*16 + ez)`; stripping the
    runtime `t1` term leaves the constant compared here.
    """
    m = _map()
    TTP16, M = A._TTP16, A._M32
    DUR1, GAP, EZ = 49152, 2458, 10          # the hand-written script's own numbers
    gap_b = GAP - 1                           # --edge opens one batch early
    wa, wb, wv = (units.freq_word(f, m.params) for f in (83.765e6, 80.235e6, 82.0e6))
    gap_ideal_dac = 5e-6 * m.params.dsp_freq_hz * 16          # 39321.6, from pulse 1's start
    late = (gap_b * 16 + EZ) - gap_ideal_dac                  # +0.4 DAC samples
    start_dac_abs = (DUR1 + gap_b) * 16 + EZ                  # 825754, from the sequence origin
    hand = {
        "A": ((0 << 16) + wa * TTP16) % M,
        "B": ((32768 << 16) + wb * TTP16) % M,
        "V": (((16384 << 16) + round(wv * late)) + wv * TTP16 - wv * start_dac_abs) % M,
    }

    c = A.Core(m)
    ch0, ch1 = A.DDSChannel(c, 0), A.DDSChannel(c, 1)
    with A.parallel(c):
        with A.branch(c):
            ch0.set(83.765e6, phase=0.0, amplitude=0.4); ch0.sw.pulse(100 * A.us)
        with A.branch(c):
            ch1.set(80.235e6, phase=0.5, amplitude=0.4); ch1.sw.pulse(100 * A.us)
    A.delay(c, 5 * A.us)
    ch1.set(82.0e6, phase=0.25, amplitude=0.4, phase_mode=A.PHASE_MODE_ABSOLUTE)
    ch1.sw.pulse(20 * A.us)
    A.fill_gaps(c, 1)
    sch = A.plan(c)

    api = {"A": sch.events[0], "B": sch.events[1], "V": sch.events[3]}
    for k, W in (("A", wa), ("B", wb), ("V", wv)):
        assert api[k].freq_word == W, f"tone {k} frequency word"
        d = (api[k].phase_const - hand[k]) % M
        d = d - M if d > (1 << 31) else d
        assert d == 0, (f"tone {k} phase constant differs by {d} LSB32 "
                        f"({d / 65536:+.4f} LSB16): api {api[k].phase_const:#011x} "
                        f"vs hand-written {hand[k]:#011x}")
        # ... and the amplitude code, which the hand-written script takes from its table Pulse
        assert api[k].amp_code == Pulse(envelopes.square(8), amp=0.4).amp_code() & 0xFFFF,             f"tone {k} amplitude code"

    # ... and the envelope RAM images, line for line: the hand-written script tiles the unit
    # envelope everywhere and writes ONE partial line (10 leading DAC samples zeroed) at line 0.
    images = A.envelope_images(sch)
    for ch in (0, 1):
        spl = m.channel(ch).samples_per_line
        want = np.tile(Pulse(envelopes.square(spl), amp=1.0).packed_lines(m, ch),
                       (m.params.env_depth, 1))
        if ch == 1:
            env = envelopes.square(spl).copy(); env[: EZ // (16 // spl)] = 0
            want[0] = Pulse(env, amp=1.0).packed_lines(m, ch)[0]
        bad = np.flatnonzero(np.any(images[ch].astype(np.int64) != want.astype(np.int64), axis=1))
        assert bad.size == 0, f"ch{ch} envelope image differs on {bad.size} lines, first {bad[:3]}"


def test_a_channel_without_reserved_lines_is_not_chunked():
    """Chunking exists only to keep the free-running envelope reader off a reserved partial line.
    A channel that reserves none has a uniform envelope RAM, so the reader may wrap freely and one
    play covers any duration — fewer plays, less queue pressure, and it is what the hand-written
    script does for the gate channel."""
    m = _map()
    c = A.Core(m)
    ch0, ch1 = A.DDSChannel(c, 0), A.DDSChannel(c, 1)
    with A.parallel(c):
        with A.branch(c):
            ch0.set(83.765e6, phase=0.0, amplitude=0.4); ch0.sw.pulse(100 * A.us)   # no partial edge
        with A.branch(c):
            ch1.set(80.235e6, phase=0.5, amplitude=0.4); ch1.sw.pulse(100 * A.us)
    A.delay(c, 5 * A.us)
    ch1.set(82.0e6, phase=0.25, amplitude=0.4, phase_mode=A.PHASE_MODE_ABSOLUTE)
    ch1.sw.pulse(20 * A.us)                                    # ch1 DOES need a partial line
    A.fill_gaps(c, 1)
    sch = A.plan(c)

    assert not sch.env_lines[0], "ch0 reserves nothing in this sequence"
    assert sch.env_lines[1], "ch1 reserves the partial line"
    ch0_ev = next(i for i, e in enumerate(sch.events) if e.channel == 0)
    assert len(sch.chunks[ch0_ev]) == 1, "an unreserved channel must not be chunked"
    assert sch.chunks[ch0_ev][0][1] == 49152
    ch1_long = next(i for i, e in enumerate(sch.events) if e.channel == 1 and e.dur_batches == 49152)
    assert len(sch.chunks[ch1_long]) > 1, "the reserved channel must still be chunked"
    for line, n in sch.chunks[ch1_long]:                       # and no run may reach the reserved block
        assert line + n <= m.params.env_depth
        assert line >= sch.first_line


# ── the latent cases Codex found: none of the sequences above reach them ──────────────────────────

def test_a_set_without_a_pulse_still_advances_the_continuous_chain():
    """Codex F2: `set()` is a state event in its own right. `set(TRACKING); set(CONTINUOUS); pulse()`
    must see the first set as its predecessor even though it played nothing."""
    m = _map()
    c = A.Core(m)
    ch = A.DDSChannel(c, 1)
    ch.set(82e6, phase=0.0)                                  # TRACKING, no pulse follows
    A.delay(c, 1 * A.us)
    ch.set(80.235e6, phase=0.0, phase_mode=A.PHASE_MODE_CONTINUOUS)
    ch.sw.pulse(2 * A.us)
    sch = A.plan(c)                                          # must NOT raise
    assert len(c.sets) == 2 and len(sch.events) == 1
    assert sch.events[0].set_index == 1

    # The continuation is exact at the EXACT set() instant, which is not an integer sample here
    # (1 us is 7864.32 DAC samples), so evaluate both phases with the same set_mu + frac split the
    # planner uses. Evaluating at the ROUNDED sample would legitimately differ by (W1-W2)*frac.
    W1, W2 = units.freq_word(82e6, m.params), units.freq_word(80.235e6, m.params)
    t = c.sets[1]
    frac = t.set_s * c.f_dac - t.set_mu

    def at_set(w, phase_const):                       # phi(the exact set instant), 32-bit units
        c_const = (phase_const - w * A._TTP16) % A._M32
        return (w * t.set_mu + round(w * frac) + c_const) % A._M32

    prev, now = at_set(W1, c.sets[0].phase_const), at_set(W2, c.sets[1].phase_const)
    assert prev == now, ("CONTINUOUS must pick up exactly where the previous tone was: "
                         f"{prev:#011x} vs {now:#011x}")


def test_a_capture_filler_does_not_rewrite_the_phase_history():
    """Codex F3: `fill_gaps` inserts a zero-amplitude tone; it must not become the predecessor of a
    later CONTINUOUS pulse."""
    m = _map()

    def chain(with_filler):
        c = A.Core(m)
        ch = A.DDSChannel(c, 1)
        ch.set(82e6, phase=0.0); ch.sw.pulse(2 * A.us)
        A.delay(c, 3 * A.us)                                 # a gap the filler would cover
        ch.set(80.235e6, phase=0.0, phase_mode=A.PHASE_MODE_CONTINUOUS)
        ch.sw.pulse(2 * A.us)
        if with_filler:
            assert A.fill_gaps(c, 1) == 1
        return A.plan(c)

    a, b = chain(False), chain(True)
    pa = [e for e in a.events if e.amp_code][-1]
    pb = [e for e in b.events if e.amp_code][-1]
    assert pa.phase_const == pb.phase_const, \
        "the filler changed the CONTINUOUS phase — it must be transparent to the chain"


def test_two_reserved_patterns_do_not_leak_into_each_other():
    """Codex F1 + the RX wrap bug: with several reserved lead patterns, a pulse starting on a
    lower one would free-run through the others, so it plays a 3-batch triplet prefix (its lead
    line + its two full-copy lines) and continues from the full region 3 batches later — the
    closest spacing the II=3 parameter queue (SrlShadow) pops on time. The channel's HIGHEST
    lead line still free-runs the whole pulse as one play (the board-verified shape)."""
    m = _map()
    c = A.Core(m)
    ch = A.DDSChannel(c, 1)
    step = 16 // m.channel(1).samples_per_line
    # two different sub-batch offsets -> two reserved lines
    c.at_s(10 * A.us + 2 * step / (16 * m.params.dsp_freq_hz))
    ch.set(82e6, phase=0.0); ch.sw.pulse(5 * A.us)
    c.at_s(30 * A.us + 4 * step / (16 * m.params.dsp_freq_hz))
    ch.set(82e6, phase=0.0); ch.sw.pulse(5 * A.us)
    sch = A.plan(c)

    leads = {k for k in sch.env_lines[1] if k[0]}   # two distinct LEADING patterns
    assert len(leads) == 2, sch.env_lines
    for idx, e in enumerate(sch.events):
        runs = sch.chunks[idx]
        top = max(sch.env_lines[1].values())
        for line, n in runs:
            # below the full region only a lead line may start a run: the channel's highest one
            # free-runs (only full lines above it), any other covers at most its own pair
            assert line >= sch.first_line or line == top or (n <= 3 and line % 3 == 0), (
                f"run ({line}, {n}) would read through another pattern's reserved triplet")
        starts, b = [], sch.events[idx].batch
        for _, n in runs:
            starts.append(b); b += n
        assert all(y - x >= 3 for x, y in zip(starts, starts[1:])), (
            f"plays at {starts} start closer than 3 batches apart (II=3)")


def test_dds_channel_index_is_validated():
    """An out-of-range index fails at construction with the channel count; the demod carrier
    (the old local channel 2) is never a DDS — driving it would corrupt the readout decoder's bank."""
    m = _map()
    c = A.Core(m)
    with pytest.raises(ValueError, match="has 2 dds channels"):
        A.DDSChannel(c, 5)
    with pytest.raises(ValueError, match="DemodChannel"):
        A.DDSChannel(c, 2)
    with pytest.raises(ValueError, match="traces 0..0"):
        A.ADCChannel(c, 1)
    with pytest.raises(ValueError, match="readouts 0..0"):
        A.DemodChannel(c, 1)


def test_play_starts_one_batch_apart_raise():
    """TimedQueue II=3 (SrlShadow, the deployed impl): a queued play starting < 3 batches after
    the previous one pops a cycle late — the duration counter passes through zero and the channel
    (and trace recorder) glitches for a batch. The planner must refuse such schedules."""
    m = _map()
    c = A.Core(m)
    ch = A.DDSChannel(c, 1)
    ch.set(82e6, phase=0.0, amplitude=0.4)
    ch.sw.pulse_mu(16)                                         # a single-batch pulse ...
    ch.set(82e6, phase=0.0, amplitude=0.4)
    ch.sw.pulse_mu(16 * 8)                                     # ... chained straight into another
    with pytest.raises(RuntimeError, match="closer than 3 batches"):
        A.plan(c)


def test_queue_barriers_let_long_sequences_through():
    """More plays than the queue holds are fine when they are spread out: before a channel's
    (k+depth)-th play the kernel waits (a `wait_until` barrier right before the push) for the k-th
    to have popped, and the planner checks every later push still keeps LEAD + PUSH_MARGIN."""
    m = _map()
    depth = m.params.queue_depth
    c = A.Core(m)
    ch = A.DDSChannel(c, 1)
    n = depth + 4
    for _ in range(n):
        ch.set(82e6, phase=0.0, amplitude=0.4)
        ch.sw.pulse_mu(16 * 4)
        A.delay_mu(c, 16 * 600)                    # 604 batches apart: depth plays span >> LEAD + margin
    sch = A.plan(c)
    src = A.generate_kernel_source(sch)
    lines = src.splitlines()
    bars = [l for l in lines if "queue barrier" in l]
    assert len(bars) == n - depth
    due = [e.batch for e in sch.events]
    want = [due[k - depth] + A.POP_LATE for k in range(depth, n)]
    got = [int(l.split("t1 + ")[1].split(")")[0]) for l in bars]
    assert got == want                              # each waits for the play pushed depth plays earlier
    for l in bars:                                  # ... right before the event's FIRST queue write
        assert lines[lines.index(l) + 1].lstrip().startswith("set_freq(")


def test_queue_barrier_without_lead_raises():
    """The play push has NO backpressure (a Flow into the generator): an overfull TimedQueue
    silently drops entries. A (depth+1)-th play due before the first has popped plus LEAD +
    PUSH_MARGIN cannot be pushed in time, so the planner refuses it."""
    m = _map()
    c = A.Core(m)
    ch = A.DDSChannel(c, 1)
    for _ in range(m.params.queue_depth + 1):
        ch.set(82e6, phase=0.0, amplitude=0.4)
        ch.sw.pulse_mu(16 * 4)
        A.delay_mu(c, 16 * 4)
    with pytest.raises(RuntimeError, match="can only be pushed after the queue entry"):
        A.plan(c)


def test_long_unreserved_runs_split_below_the_dur_field():
    """The dur register is 16 bits: a 65536-batch play wraps to dur=0 and plays NOTHING. Runs on
    an unchunked (no-reserved-line) channel must split below that."""
    m = _map()
    c = A.Core(m)
    ch = A.DDSChannel(c, 0)
    ch.set(82e6, phase=0.0, amplitude=0.4)
    ch.sw.pulse_mu(70000 * 16)                                 # > 2^16 - 1 batches
    sch = A.plan(c)
    runs = sch.chunks[0]
    assert len(runs) == 2 and all(n <= 65535 for _, n in runs), runs
    assert sum(n for _, n in runs) == 70000


def test_adc_gate_rob_check_uses_the_snapped_window():
    """The gate snaps OUTWARD to whole batches; a request that fits rob_depth before snapping can
    exceed it after. The check must see the snapped size."""
    m = _map()
    c = A.Core(m)
    adc = A.ADCChannel(c)
    A.delay_mu(c, 8)                                           # mid-batch start: snaps outward
    adc.gate_mu(m.params.rob_depth * 16)
    with pytest.raises(RuntimeError, match="snapped outward"):
        A.plan(c)


def test_fetch_iq_restarts_on_a_new_run_result():
    """fetch_iq walks the queued results of THE LAST RUN: a new RunResult restarts the cursor
    (a reused DemodChannel could otherwise never fetch a second run's results)."""
    import types
    m = _map()
    c = A.Core(m)
    dm = A.DemodChannel(c)
    mk = lambda v: types.SimpleNamespace(cores={0: types.SimpleNamespace(
        res=[v], real=[v * 10], imag=[v * 100], iq=[complex(v)])})
    c.last_result = mk(1)
    assert dm.fetch_iq().res == 1
    c.last_result = mk(2)                                      # a NEW run's result object
    assert dm.fetch_iq().res == 2                              # cursor restarted, not exhausted


# ── the receive side ──────────────────────────────────────────────────────────────────────────────

def test_adc_gate_inserts_fillers_and_snaps_outward():
    """The trace records only while ch1 fires and its address resets between fires: the gate must
    make ch1 fire contiguously (lead-in, holes, tail) and snap outward to whole batches."""
    m = _map()
    c = A.Core(m)
    r, adc = A.DDSChannel(c, 1), A.ADCChannel(c)
    with A.parallel(c):
        with A.branch(c):
            A.delay(c, 2 * A.us)                       # a hole before the pulse
            r.set(82e6, amplitude=0.4); r.sw.pulse(3 * A.us)
            A.delay(c, 1 * A.us)                       # and one after
            r.set(82e6, amplitude=0.4); r.sw.pulse(2 * A.us)
        with A.branch(c):
            adc.gate(10 * A.us)
    sch = A.plan(c)
    tg = c.trace_gates[0]
    assert tg.batch0 == 0 and tg.batches == -(-c.seconds_to_mu(10 * A.us) // 16)
    ch1 = sorted((e for e in sch.events if e.channel == 1), key=lambda e: e.batch)
    fillers = [e for e in ch1 if c.sets[e.set_index].is_filler]
    assert len(fillers) == 3, "lead-in, hole, tail"
    assert all(f.amp_code == 0 for f in fillers)
    cur = tg.batch0                                     # contiguous coverage, batch by batch
    for e in ch1:
        assert e.batch == cur, f"hole at batch {cur}"
        cur = e.batch + e.dur_batches
    assert cur == tg.batch0 + tg.batches


def test_adc_gate_pure_listen_and_rules():
    m = _map()
    c = A.Core(m)
    adc = A.ADCChannel(c)
    adc.gate(5 * A.us)                                  # no ch1 pulses at all: pure listening
    sch = A.plan(c)
    fillers = [e for e in sch.events if e.channel == 1]
    assert len(fillers) == 1 and fillers[0].amp_code == 0

    c2 = A.Core(m)                                      # a second gate is refused
    adc2 = A.ADCChannel(c2)
    adc2.gate(1 * A.us); A.delay(c2, 1 * A.us); adc2.gate(1 * A.us)
    with pytest.raises(RuntimeError, match="one gate per trace per run"):
        A.plan(c2)

    c3 = A.Core(m)                                      # ch1 AFTER the gate is refused (it would
    r3, adc3 = A.DDSChannel(c3, 1), A.ADCChannel(c3)    # restart the trace and clobber it)
    adc3.gate(1 * A.us)
    r3.set(82e6); r3.sw.pulse(1 * A.us)
    with pytest.raises(RuntimeError, match="restart the recording"):
        A.plan(c3)

    c5 = A.Core(m)                                      # ch1 fully BEFORE the gate, with a break,
    r5, adc5 = A.DDSChannel(c5, 1), A.ADCChannel(c5)    # is harmless: the refire overwrites it
    r5.set(82e6, amplitude=0.4); r5.sw.pulse(1 * A.us)
    A.delay(c5, 1 * A.us)                               # >= one batch of silence
    adc5.gate(2 * A.us)
    sch5 = A.plan(c5)                                   # must NOT raise
    tg5 = c5.trace_gates[0]
    fill5 = [e for e in sch5.events if e.channel == 1 and c5.sets[e.set_index].is_filler]
    assert len(fill5) == 1 and fill5[0].batch == tg5.batch0 and fill5[0].dur_batches == tg5.batches

    c4 = A.Core(m)                                      # rob_depth is enforced
    A.ADCChannel(c4).gate(200 * A.us)                   # 98304 batches > 65536
    with pytest.raises(RuntimeError, match="rob_depth"):
        A.plan(c4)


def test_demod_word_is_the_matched_pair_and_batch_granular():
    """The user gives the RF frequency; the ADC-rate word must be units.demod_freq_word (the 4x
    law), and the demod grid is one whole batch (1 envelope sample per line)."""
    m = _map()
    c = A.Core(m)
    dm = A.DemodChannel(c)
    c.at_s(1 * A.us + 0.3e-9)                           # deliberately off-grid
    dm.set(82e6, phase=0.0)
    dm.gate(2 * A.us)
    e = A.plan(c).events[0]
    assert e.is_demod and e.freq_word == units.demod_freq_word(82e6, m.params) % A._M32
    assert e.freq_word == (4 * units.freq_word(82e6, m.params)) % A._M32
    assert e.start_dac % 16 == 0 and e.lead_zeros == 0 and e.trail_zeros == 0


def test_demod_phase_rotates_the_constant_exactly():
    """The RELATIVE phase contract: phase=phi must shift the register constant by exactly
    round(phi * 2^16) << 16, whatever the (unknown) absolute pipeline offset is."""
    m = _map()

    def const(phi):
        c = A.Core(m)
        dm = A.DemodChannel(c)
        dm.set(82e6, phase=phi)
        dm.gate(2 * A.us)
        return A.plan(c).events[0].phase_const

    base = const(0.0)
    for phi in (0.25, 0.5, 0.123456):
        d = (const(phi) - base) % A._M32
        assert d == (round(phi * 65536) % 65536) << 16, f"phase {phi}"


def test_demod_codegen_obeys_the_stall_contract():
    """riscq.h: wait past the window's opening (t + READOUT_LEAD) BEFORE read_res, and read res
    BEFORE real/imag; results land in `out`, 3 words per gate, in gate order."""
    m = _map()
    c = A.Core(m)
    dm = A.DemodChannel(c)
    dm.set(82e6); dm.gate(2 * A.us)
    A.delay(c, 1 * A.us)                                # > guard after the first window
    dm.set(80e6); dm.gate(2 * A.us)
    sch = A.plan(c)
    src = A.generate_kernel_source(sch, doc="t")
    assert "out: Array" in src
    for k, e in enumerate(ev for ev in sch.events if ev.is_demod):
        i_wait = src.index(f"wait_until(t1 + {e.batch} + {A.READOUT_LEAD})")
        i_res, i_re, i_im = (src.index(f"out[{3*k + j}] = read_{n}()")
                             for j, n in enumerate(("res", "real", "imag")))
        assert i_wait < i_res < i_re < i_im
    # the runtime phase term advances at the ADC rate (4 per batch), not the DAC rate
    assert "* (t1 * 4))" in src


def test_readout_guard_blocks_tight_followers():
    m = _map()
    c = A.Core(m)
    r, dm = A.DDSChannel(c, 1), A.DemodChannel(c)
    dm.set(82e6); dm.gate(2 * A.us)
    r.set(82e6); r.sw.pulse(1 * A.us)                   # starts right at the window end
    with pytest.raises(RuntimeError, match="read_res halts"):
        A.plan(c)


# ── two hardware cores on one timeline (rfsoc4x2-2q-fine) ────────────────────────────────────────

def _ion_trap(c, gate, ro, adc):
    """The verified ion-trap sequence on the given channels."""
    with A.parallel(c):
        with A.branch(c):
            with A.parallel(c):
                with A.branch(c):
                    gate.set(83.765e6, phase=0.0, amplitude=0.4); gate.sw.pulse(100 * A.us)
                with A.branch(c):
                    ro.set(80.235e6, phase=0.5, amplitude=0.4); ro.sw.pulse(100 * A.us)
            A.delay(c, 5 * A.us)
            ro.set(82.0e6, phase=0.25, amplitude=0.4, phase_mode=A.PHASE_MODE_ABSOLUTE)
            ro.sw.pulse(20 * A.us)
        with A.branch(c):
            adc.gate(125 * A.us)


def test_two_core_channel_ids_and_labels():
    """dds 2k/2k+1 are core k's gate/readout drives (flat ids 3k/3k+1), adc/demod k its trace and
    IQ readout; the report and the errors speak the user's names."""
    m = _map("rfsoc4x2-2q-fine")
    c = A.Core(m)
    chans = [A.DDSChannel(c, i) for i in range(4)]
    assert [ch.flat for ch in chans] == [0, 1, 3, 4]
    assert [ch.core_index for ch in chans] == [0, 0, 1, 1]
    assert (A.ADCChannel(c, 1).core_index, A.DemodChannel(c, 1).flat) == (1, 5)
    assert [A._label(f, m) for f in (0, 1, 2, 3, 4, 5)] == ["ch0", "ch1", "demod0", "ch2", "ch3", "demod1"]
    with pytest.raises(ValueError, match="has 4 dds channels"):
        A.DDSChannel(c, 4)
    chans[3].set(82e6, amplitude=0.4); chans[3].sw.pulse(2 * A.us)
    assert "ch3" in A.plan(c).report().splitlines()[1]


def test_two_core_core0_plan_equals_the_single_core_plan():
    """Core 0 of the 2-core build must land on exactly the words, lines, batches and envelope images
    the board-verified 1-core build lands on for the same timeline (the DAC combine depth is the
    same: two channels summed on one DAC in both)."""
    one, two = A.Core(_map("rfsoc4x2-1q-fine")), A.Core(_map("rfsoc4x2-2q-fine"))
    for c in (one, two):
        _ion_trap(c, A.DDSChannel(c, 0), A.DDSChannel(c, 1), A.ADCChannel(c, 0))
    s1, s2 = A.plan(one), A.plan(two)
    key = lambda e: (e.channel, e.batch, e.dur_batches, e.lead_zeros, e.freq_word, e.phase_const,
                     e.amp_code, e.env_line, e.is_demod)
    assert [key(e) for e in s1.events] == [key(e) for e in s2.events]
    assert s1.chunks == s2.chunks and s1.env_lines == s2.env_lines and s1.first_line == s2.first_line
    i1, i2 = A.envelope_images(s1), A.envelope_images(s2)
    assert i1.keys() == i2.keys() and all(np.array_equal(i1[k], i2[k]) for k in i1)
    assert two.trace_gates[0].core_index == 0
    assert two.m.dac_pipe(1) == one.m.dac_pipe(0) == 4


def test_two_core_kernels_split_per_core_with_shared_origin():
    """One kernel per hardware core: each carries only its own channels under their local names,
    takes the shared origin `t_origin` as a runtime parameter and reports its boot-time clock."""
    m = _map("rfsoc4x2-2q-fine")
    c = A.Core(m)
    ch = [A.DDSChannel(c, i) for i in range(4)]
    dm = [A.DemodChannel(c, k) for k in range(2)]
    adc = [A.ADCChannel(c, k) for k in range(2)]
    with A.parallel(c):
        for k in range(2):
            with A.branch(c):
                ch[2 * k].set(83.765e6, amplitude=0.4); ch[2 * k].sw.pulse(10 * A.us)
            with A.branch(c):
                ch[2 * k + 1].set(80.235e6, phase=0.5, amplitude=0.4); ch[2 * k + 1].sw.pulse(10 * A.us)
            with A.branch(c):
                adc[k].gate(12 * A.us)
        with A.branch(c):
            A.delay(c, 20 * A.us)
            dm[1].set(82e6); dm[1].gate(2 * A.us)
            A.delay(c, 1 * A.us)
            dm[1].set(82e6); dm[1].gate(2 * A.us)
    sch = A.plan(c)
    assert {e.channel for e in sch.events} == {0, 1, 3, 4, 5}
    src = [A.generate_kernel_source(sch, core_index=k, origin="run_origin()") for k in range(2)]
    for s in src:
        compile(s, "<generated>", "exec")
        assert "t1 = run_origin()" in s and ", tele: Array" in s
        i_entry, i_init, i_t1, i_armed, i_play = (s.index(x) for x in (
            "tele[1] = now()", "init_pulse_params(", "tele[0] = t1", "tele[2] = now()", "play("))
        assert i_entry < i_init < i_t1 < i_armed < i_play      # entry, init, origin, armed, first play
        assert s.count("tele[2] = now()") == 1
        assert "ch2" not in s and "ch3" not in s and "demod1" not in s   # local names only
    assert "out: Array" not in src[0] and "demod" not in src[0] and "tele[3]" not in src[0]
    assert "out: Array" in src[1] and "out[3] = read_res()" in src[1]   # core 1's own 2 results
    assert src[1].index("read_imag()") < src[1].index("tele[3] = now()") < src[1].index("tele[4] = now()")
    n_plays = lambda s: s.count("play(")
    assert n_plays(src[0]) + n_plays(src[1]) == sum(len(v) for v in sch.chunks.values())
    # the single-core form is unchanged: own clock read, no origin, no telemetry
    s0 = A.generate_kernel_source(sch, core_index=0)
    assert "t1 = now() + 8192" in s0 and "run_origin" not in s0 and "tele" not in s0


def test_telemetry_checks_catch_a_late_or_disagreeing_core():
    m = _map("rfsoc4x2-2q-fine")
    c = A.Core(m)
    ch1, ch3, dm1 = A.DDSChannel(c, 1), A.DDSChannel(c, 3), A.DemodChannel(c, 1)
    with A.parallel(c):
        with A.branch(c):
            A.delay(c, 2 * A.us); ch1.set(82e6, amplitude=0.4); ch1.sw.pulse(2 * A.us)
        with A.branch(c):
            A.delay(c, 2 * A.us); ch3.set(82e6, amplitude=0.4); ch3.sw.pulse(2 * A.us)
            dm1.set(82e6); dm1.gate(2 * A.us)
            A.delay(c, 1 * A.us)
            ch3.set(82e6, amplitude=0.4); ch3.sw.pulse(1 * A.us)
    sch = A.plan(c)
    t1, first = 5_000_000, min(e.batch for e in sch.events)
    good = [t1, t1 - 8192, t1 - 8192 + 600]                     # armed 7.6k batches before the first play
    origins = {}
    A._check_telemetry(sch, 0, good, origins)
    reads = [e for e in sch.events if e.is_demod and e.channel // 3 == 1]
    nxt = min(e.batch for e in sch.events if e.channel // 3 == 1 and e.batch > reads[0].batch)
    A._check_telemetry(sch, 1, good + [t1 + nxt - A.LEAD], origins)           # exactly the lead left
    with pytest.raises(RuntimeError, match="disagree on the run origin"):
        A._check_telemetry(sch, 1, [t1 + 1, t1 - 8192, t1 - 8000, t1], dict(origins))
    with pytest.raises(RuntimeError, match="pushed only"):
        A._check_telemetry(sch, 0, [t1, t1 - 100, t1 + first - 10], {})       # armed too late
    with pytest.raises(RuntimeError, match="lead left"):
        A._check_telemetry(sch, 1, good + [t1 + nxt - A.LEAD + 1], {})
    # the wrap: the same numbers just below 2^32 pass (signed differences, not magnitudes)
    w = (1 << 32) - 3000
    A._check_telemetry(sch, 0, [w, w - 8192, w - 8192 + 600], {})


def test_two_core_gates_and_readout_guard_are_per_core():
    m = _map("rfsoc4x2-2q-fine")
    c = A.Core(m)
    ro0, ro1 = A.DDSChannel(c, 1), A.DDSChannel(c, 3)
    adc0, adc1 = A.ADCChannel(c, 0), A.ADCChannel(c, 1)
    with A.parallel(c):
        with A.branch(c):
            adc0.gate(4 * A.us)
        with A.branch(c):
            A.delay(c, 1 * A.us); adc1.gate(2 * A.us)
        with A.branch(c):
            A.delay(c, 1.5 * A.us); ro1.set(82e6, amplitude=0.4); ro1.sw.pulse(1 * A.us)
    sch = A.plan(c)
    fill = {f: [e for e in sch.events if e.channel == f and e.amp_code == 0] for f in (1, 4)}
    assert len(fill[1]) == 1 and len(fill[4]) == 2              # pure listening / lead-in + tail
    assert all(e.channel in (1, 4) for e in sch.events)
    assert sorted(tg.core_index for tg in c.trace_gates) == [0, 1]

    c2 = A.Core(m)                                              # a second gate on the SAME trace
    a1 = A.ADCChannel(c2, 1)
    a1.gate(1 * A.us); A.delay(c2, 1 * A.us); a1.gate(1 * A.us)
    with pytest.raises(RuntimeError, match="2 gates on adc1"):
        A.plan(c2)

    c3 = A.Core(m)                                              # read_res halts only ITS core
    dm0, r0, r1 = A.DemodChannel(c3, 0), A.DDSChannel(c3, 1), A.DDSChannel(c3, 3)
    dm0.set(82e6); dm0.gate(2 * A.us)
    r1.set(82e6, amplitude=0.4); r1.sw.pulse(1 * A.us)          # core 1 right after core 0's window
    A.plan(c3)
    r0.set(82e6, amplitude=0.4)
    c3.at_mu(dm0.core.events[0].start_mu + dm0.core.events[0].dur_mu)
    r0.sw.pulse(1 * A.us)                                       # core 0 itself: refused
    with pytest.raises(RuntimeError, match="read_res halts"):
        A.plan(c3)


def test_shared_trace_multicore_build_refuses_a_per_channel_adc():
    m = _map("sim-2q")                                          # 2 cores, upstream shared trace
    c = A.Core(m)
    with pytest.raises(ValueError, match="ONE shared trace"):
        A.ADCChannel(c, 0)
    assert A.DDSChannel(c, 3).flat == 4                          # the drives are still addressable
