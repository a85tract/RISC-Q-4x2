"""M7a co-sim: env_depth 16384 with the envAddrWidth-follows-depth fix (M7_UPGRADES_PLAN.md 1a).

The canary (Codex M7 finding 1): before the fix `envAddrWidth` stayed 10 while the RAMs grew, so a
pulse whose `env` field points ABOVE line 0x3ff silently truncated (0x400 -> 0, 16000 -> 640) and
played the WRONG envelope. Here EVERY line the 10-bit alias can reach (0..0x3ff) holds a LOW
envelope and the target lines hold a HIGH one, so the captured amplitude says which line played.

Codex round-2 on this test found the first version non-falsifying: `Pulse.packed_lines()` packs
only `Pulse.env` and IGNORES `Pulse.amp`, so a `Pulse(square(n), amp=0.2)` vs `amp=0.9` pair
packs IDENTICAL full-scale lines. The amplitude difference must be in the ENVELOPE ARRAY, and the
test asserts the packed words really differ before trusting the result.
"""

import os

import numpy as np
import pytest

from riscq import run as rq
from riscq.lang import ParamTable, compile_kernel, kernel
from riscq.map import ADC_BATCH, SocMap, SocParams
from riscq.pulses import Pulse, envelopes, units

pytestmark = pytest.mark.cosim

CODE = 683
DUR = 200                     # batches (~0.4 us)
ALIAS_SPAN = 0x400            # every line a 10-bit address can reach
LINES_HI = DUR + 8            # enough HIGH lines for the free-running reader to walk
LOW, HIGH = 0.2, 0.9          # envelope scale factors (NOT Pulse.amp — see the docstring)


@pytest.fixture(scope="module")
def cosim_deep(request):
    if not request.config.getoption("--cosim"):
        pytest.skip("needs --cosim")
    from tests.conftest import CONFIGS, SW_ROOT
    from riscq_sim import cosim as server

    # RISCQ_DEEP_BUILD lets a falsification run point at an UNFIXED (envAddrWidth = 10) build of
    # the same config, proving this canary really does fail without the fix.
    build = os.environ.get("RISCQ_DEEP_BUILD", "rfsoc4x2-1q-deep")
    drv = server.start(CONFIGS / "rfsoc4x2-1q-deep.json", SW_ROOT.parents[1] / "sim" / "build" / build)
    m = SocMap(SocParams.from_json(drv.sim.get_params()))
    yield drv, m
    server.stop(drv)


@kernel
def k_canary(ro: ParamTable, dur: int, cq: int):
    init_pulse_params(ro.pulses)  # noqa: F821
    set_freq(ro, cq)  # noqa: F821
    t1 = now() + 4096  # noqa: F821
    play(ro, ro["v"], t1)  # noqa: F821
    wait_until(t1 + dur + 64)  # noqa: F821


def _lines(m, scale):
    """Packed constant lines at `scale` of full scale — the scale lives in the ENVELOPE array."""
    n = m.channel(1).samples_per_line
    return Pulse(scale * envelopes.square(n), amp=1.0).packed_lines(m, 1)


def _play_from(drv, m, prog, line):
    rq.write_slot(drv, m, 0, prog, "ro", 0, "dur", DUR)
    rq.write_slot(drv, m, 0, prog, "ro", 0, "phase", 0)
    rq.write_slot(drv, m, 0, prog, "ro", 0, "env", line)
    rq.rerun(drv, m, {0: prog}, timeout=(DUR + 12000) * 4 + 20_000_000)
    x = np.frombuffer(drv.read_block(m.robs(), 4 * ADC_BATCH * DUR), dtype="<i4").astype(float)
    return float(np.abs(x[40:-40]).max())


@pytest.fixture(scope="module")
def deep_ready(cosim_deep):
    """One setup + one RAM image for both canaries: LOW over every aliasable line, HIGH at the
    two target lines (0x400 and a near-top line whose alias also lands in the LOW region)."""
    drv, m = cosim_deep
    assert m.params.env_depth == 16384
    drv.sim.set_model({"kind": "loopback", "src": 0, "dst": 0, "gain": 1.0, "delay": 0})

    lo, hi = _lines(m, LOW), _lines(m, HIGH)
    assert not np.array_equal(lo, hi), \
        "LOW and HIGH pack to identical words — the canary cannot falsify anything"

    n = m.channel(1).samples_per_line
    ro = ParamTable(1, CODE * 16 * m.params.dsp_freq_hz / 65536,
                    {"v": Pulse(envelopes.square(n), amp=1.0)})
    prog = compile_kernel(k_canary, m, tables=dict(ro=ro), dur=DUR, cq=CODE << 16)
    rq.setup(drv, m, {0: prog})

    rq.write_envelope(drv, m, 0, 1, 0, np.tile(lo, (ALIAS_SPAN, 1)))          # 0..0x3ff LOW
    rq.write_envelope(drv, m, 0, 1, 0x400, np.tile(hi, (LINES_HI, 1)))        # canary A HIGH
    top = m.params.env_depth - LINES_HI                                       # 16192; alias 832 -> LOW
    assert (top & (ALIAS_SPAN - 1)) < ALIAS_SPAN
    rq.write_envelope(drv, m, 0, 1, top, np.tile(hi, (LINES_HI, 1)))          # canary B HIGH
    return drv, m, prog, top


def test_line_above_10bit_plays(deep_ready):
    """A pulse pointed at line 0x400 must play the HIGH envelope, not line 0's LOW one."""
    drv, m, prog, _ = deep_ready
    amp = _play_from(drv, m, prog, 0x400)
    print(f"\n[m7a canary 0x400] max|s| = {amp:.0f}  (HIGH ~ {HIGH:.1f} FS, LOW ~ {LOW:.1f} FS)")
    assert amp > 0.5 * 32767, "line 0x400 truncated to 0 — envAddrWidth still 10 bits"


def test_near_top_line_plays(deep_ready):
    """A near-top line (alias inside the LOW region) must play HIGH — proves the FULL address
    width, not just bit 10 (Codex round-2 finding 5: the write-only bank needs a PLAYBACK proof)."""
    drv, m, prog, top = deep_ready
    amp = _play_from(drv, m, prog, top)
    print(f"[m7a canary {top}] alias {top & 0x3ff} holds LOW; max|s| = {amp:.0f}")
    assert amp > 0.5 * 32767, f"line {top} aliased to {top & 0x3ff} — address width too narrow"


# ── banked trace RAM: all 16 banks, non-aliasing signature (Codex round-3 findings 1-2) ──

@kernel
def k_long_ramp(ro: ParamTable, dur: int):
    init_pulse_params(ro.pulses)  # noqa: F821
    set_freq(ro, 0)  # noqa: F821          carrier code 0 -> the DAC sample IS amp x envelope
    t1 = now() + 8192  # noqa: F821
    t1 = (t1 >> 12 << 12) + 4096
    play(ro, ro["v"], t1)  # noqa: F821
    wait_until(t1 + dur + 64)  # noqa: F821


def test_all_banks_beat_integrity(cosim_deep):
    """Every one of the 16 trace banks receives its own beats, in order, with no duplicate or
    missing beat — over the FULL 65535-beat window (the 61440-beat run only reached banks 0-14).

    Signature: carrier code 0 + a per-line ramp envelope through a unity loopback, so beat n
    carries envelope line (n + k0) mod env_depth. With env_depth 16384 the signature period spans
    FOUR banks, so a bank swapped with any of its 3 neighbours breaks the ramp — unlike a
    1024-line ramp, which repeats 4x inside every 4096-line bank and would hide a swap.
    (A swap by exactly 4 banks stays invisible to ANY periodic signature; that is excluded by
    construction instead: the bank select is a registered one-hot of the write address' high
    bits, PulseTableSoc.scala.)"""
    drv, m = cosim_deep
    assert m.params.rob_depth == 65536
    depth = m.params.env_depth
    banks = m.params.rob_depth // 4096
    dur = 65535                                     # durWidth 16 max: every bank, every boundary
    drv.sim.set_model({"kind": "loopback", "src": 0, "dst": 0, "gain": 1.0, "delay": 0})

    spl = m.channel(1).samples_per_line
    ramp = np.linspace(-0.9, 0.9, depth)
    lines = np.concatenate([Pulse(np.full(spl, v + 0j), amp=1.0).packed_lines(m, 1) for v in ramp])
    ro = ParamTable(1, 0.0, {"v": Pulse(np.full(spl, 0.9 + 0j), amp=1.0)})
    prog = compile_kernel(k_long_ramp, m, tables=dict(ro=ro), dur=dur)
    rq.setup(drv, m, {0: prog})
    rq.write_envelope(drv, m, 0, 1, 0, lines)
    rq.write_slot(drv, m, 0, prog, "ro", 0, "dur", dur)
    rq.write_slot(drv, m, 0, prog, "ro", 0, "amp", units._amp_code(0.9))
    rq.rerun(drv, m, {0: prog}, timeout=(dur + 20000) * 4 + 20_000_000)

    nbytes = 4 * ADC_BATCH * dur
    parts = [drv.read_block(m.robs() + off, min(1 << 17, nbytes - off))
             for off in range(0, nbytes, 1 << 17)]
    trace = np.frombuffer(b"".join(parts), dtype="<i4").reshape(dur, ADC_BATCH)
    assert (trace == trace[:, :1]).all(), "lanes disagree within a beat"
    beats = trace[:, 0].astype(np.int64)

    head, tail = 8, 32
    core = beats[head:dur - tail]
    steps = np.diff(core)
    wraps = np.flatnonzero(steps < 0)
    gaps = np.diff(wraps)
    up = steps[steps > 0]
    written = [np.abs(beats[b * 4096:(b + 1) * 4096]).max() for b in range(banks)]
    print(f"\n[all banks] {dur} beats over {banks} banks; wrap spacing "
          f"{set(gaps.tolist()) if gaps.size else 'n/a'} (expect {{{depth}}}); "
          f"step {np.median(up):.1f} +- {up.std():.2f}; every bank written: {min(written) > 0}")
    assert gaps.size and (gaps == depth).all(), "a beat was dropped, duplicated or misplaced"
    assert up.std() < 1.5, f"ramp step not uniform (std {up.std():.2f})"
    assert min(written) > 0, f"a bank received no data: {written}"
    for b in range(1, banks):
        s = int(beats[b * 4096]) - int(beats[b * 4096 - 1])
        assert abs(s - np.median(up)) <= 2 or s < 0, f"bank {b} boundary step {s}"
