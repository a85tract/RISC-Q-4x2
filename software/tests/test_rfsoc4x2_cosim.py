"""RFSoC 4x2 (rfsoc4x2-1q.json) co-sim: the 1-qubit / 2-DAC / 2-ADC geometry boots, runs a
kernel, and the robs readout trace — the M5 raw-ADC capture instrument on this board — records
through the same read_robs code path the board will use (PORT_PLAN_4X2.md M2.6 / M5.15).

The trace (PulseTableSoc robs(0)) records the mapped ADC's lanes (with adc_map [0] the "sum" IS
ADC 0's raw samples, sign-extended to i32/lane) at batch rate while a READOUT-channel pulse
fires; rbAddr resets between fires, so the trace holds the LAST fire from address 0 up."""

import math

import numpy as np
import pytest

from riscq import run as rq
from riscq.cal import kernels
from riscq.lang import Array, ParamTable, compile_kernel, kernel
from riscq.map import ADC_BATCH
from riscq.pulses import Pulse, envelopes, units

pytestmark = pytest.mark.cosim

F_GE = 50e6
RO_CODE = 8192            # quarter-turn per ADC sample (see test_batch.py RO_CODE rationale)
RO_DUR = 40               # demod window (batches)
PERIOD = 256


def _tables(m):
    ro_freq = units.demod_code_to_freq(RO_CODE, m.params)
    gate = ParamTable(0, F_GE, {"x90": Pulse(envelopes.square(16), freq_hz=F_GE, amp=0.5)})
    ro = ParamTable(1, ro_freq, {"meas": Pulse(envelopes.square(RO_DUR + 16), freq_hz=ro_freq, amp=0.5)})
    demod = ParamTable(2, 0.0, {"sq": Pulse(envelopes.square(RO_DUR), amp=1.0)})
    return gate, ro, demod


def test_geometry(cosim_4x2):
    """The generated SoC carries the 4x2 config: 1 qubit core, rob_depth 32768."""
    drv, m = cosim_4x2
    assert m.params.qubit_num == 1
    assert m.params.dac_num == 2 and m.params.adc_num == 2
    assert m.params.rob_depth >= 61440, "trace must hold the whole 125 us ion-trap sequence"
    assert m.params.dsp_freq_hz == 491.52e6


def test_vna_and_robs_trace(cosim_4x2):
    """A 1-point matched-pair VNA runs on the 1-core map (demod integral lands, IQSUM), and the
    robs trace holds the readout tone's raw ADC samples from the last fire."""
    drv, m = cosim_4x2
    gate, ro, demod = _tables(m)
    F, npts, shots = 256, 1, 1
    drv.sim.set_model({"kind": "tone", "adc": m.adc_of(0),
                       "freq_hz": units.code_to_freq(1024, m.params), "amp": 20000.0})
    prog = compile_kernel(kernels.k_vna, m, fw32=int(m.params.freq_width == 32), tables=dict(ro=ro, demod=demod),
                          out=Array(2 * npts), npts=npts, shots=shots, period=PERIOD, sh=0,
                          ddly=0, mode=kernels.IQSUM, c0q=4 * F << 16, dcq=0)
    out = rq.run(drv, m, {0: prog}, timeout=PERIOD * 4 + 20_000_000)[0]["out"]
    mag = math.hypot(int(out[0]), int(out[1]))
    print(f"\n[4x2 vna] iq=({int(out[0])}, {int(out[1])}) |.|={mag:.0f}")
    assert mag > 0, "demod integral is zero — readout path dead on the 4x2 map"

    # the trace: at least the readout window's batches of the model tone, from address 0.
    lanes = ADC_BATCH                                # 4 lanes x i32 per batch line
    trace = rq.read_robs(drv, m, nbytes=4 * lanes * 4 * RO_DUR)
    samples = trace.astype(np.int64)
    amp = float(np.abs(samples).max())
    print(f"[4x2 robs] first {RO_DUR} batches: max|s|={amp:.0f} nonzero={np.count_nonzero(samples)}"
          f"/{samples.size}")
    assert amp > 1000, "robs trace empty — capture path (readoutPulse-gated) did not record"
    assert np.count_nonzero(samples) > samples.size // 4, "robs trace mostly zeros"


# ── banked trace RAM: per-beat integrity across every bank boundary (Codex acceptance) ──

@kernel
def k_long_ramp(ro: ParamTable, dur: int):
    init_pulse_params(ro.pulses)  # noqa: F821
    set_freq(ro, 0)  # noqa: F821          carrier code 0 -> the DAC sample IS amp * envelope
    t1 = now() + 8192  # noqa: F821
    t1 = (t1 >> 12 << 12) + 4096
    play(ro, ro["v"], t1)  # noqa: F821
    wait_until(t1 + dur + 64)  # noqa: F821


def test_banked_trace_has_no_duplicate_or_missing_beat(cosim_4x2):
    """The 65536-line trace RAM is split into 4096-line banks (PulseTableSoc `robBanks`) because a
    single deep memory infers a 64-deep BRAM cascade that Vivado refuses to route (DRC CASC-31).
    Banking must be invisible: every captured beat must land at its own address, in order, across
    all 15 bank boundaries.

    Signature: carrier code 0 (so the DAC sample is amp x envelope) + a RAMP envelope, one distinct
    value per line, played through a unity loopback. Beat n then carries envelope line
    (n + k) mod env_depth on all four lanes — a duplicated or dropped beat breaks the step by
    construction."""
    drv, m = cosim_4x2
    dur = 61440                                    # the ion-trap sequence length; 15 bank crossings
    assert m.params.rob_depth >= dur
    depth = m.params.env_depth
    drv.sim.set_model({"kind": "loopback", "src": 0, "dst": 0, "gain": 1.0, "delay": 0})

    spl = m.channel(1).samples_per_line
    assert spl == 1, "readout interp is expected to give 1 envelope sample per line"
    ramp = np.linspace(-0.9, 0.9, depth)           # one distinct value per line
    lines = np.concatenate([Pulse(np.full(spl, v + 0j), amp=1.0).packed_lines(m, 1)
                            for v in ramp])
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

    # all four lanes of a beat share the beat's envelope value (interp 16 => one value per batch)
    assert (trace == trace[:, :1]).all(), "lanes disagree within a beat — capture is not coherent"
    beats = trace[:, 0].astype(np.int64)
    # Skip the head/tail edges: the first beats carry the generator's pipeline fill and the last
    # ones the duration gate closing (the pulse output is forced to zero there), neither of which
    # is a capture property. Everything between must be an unbroken ramp.
    head, tail = 8, 32
    core = beats[head:dur - tail]
    steps = np.diff(core)
    # the ramp rises monotonically and wraps once per envelope pass: ONE large negative step, and
    # the wraps must be EXACTLY `depth` beats apart — a dropped or duplicated beat shifts a wrap.
    wraps = np.flatnonzero(steps < 0)
    gaps = np.diff(wraps)
    up = steps[steps > 0]
    print(f"\n[banked trace] {dur} beats, {len(wraps)} envelope wraps, wrap spacing "
          f"{set(gaps.tolist()) if gaps.size else 'n/a'} (expect {{{depth}}}); "
          f"ramp step {np.median(up):.0f} +- {up.std():.2f}")
    assert gaps.size and (gaps == depth).all(), \
        f"envelope wraps are not {depth} beats apart — a beat was dropped or duplicated"
    assert len(wraps) in (core.size // depth, core.size // depth + 1), \
        f"{len(wraps)} wraps over {core.size} beats (depth {depth})"
    assert up.std() < 1.5, f"ramp step is not uniform (std {up.std():.2f}) — beat integrity broken"
    # explicit bank-boundary check: the step ACROSS every 4096 boundary is an ordinary ramp step
    med = np.median(up)
    for b in range(4096, dur - tail, 4096):
        s = beats[b] - beats[b - 1]
        assert abs(s - med) <= 2 or s < 0, f"bank boundary at beat {b}: step {s} != {med}"
