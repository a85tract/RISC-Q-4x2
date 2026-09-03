"""F1 acceptance (spec 14 F1): the n-gate trains hold at the REFERENCE depths.

`Calibration_X6Y3_Two_Qubits` runs Rabi ladders to n_gates = 96 (GE) and 24 (EF), but every
PulseGenerator parameter sits behind a depth-4 TimedQueue on the posted RF link and a push into a full
queue is silently dropped — an UNPACED train plays its first four gates and nothing after (measured
here). The kernels now walk a paced `step` grid (base.train_step / TRAIN_AHEAD); these tests capture
the gate DAC across one shot and check the train the hardware actually played: the exact gate count,
each gate full length, on the expected grid, a full LEAD after the retuned prep.

The gate envelope is sized to the X6Y3 X90 (35 ns = 18 batches), so the trains here run at the same
per-gate rate the real config asks for. One --cosim run for the whole file.
"""

import numpy as np
import pytest

from riscq import run as rq
from riscq.cal import kernels
from riscq.cal.base import SEP, TRAIN_STEP, X90, grid_period, train_step
from riscq.lang import Array, ParamTable, compile_kernel, kernel
from riscq.map import LEAD, READOUT_LEAD, pack16
from riscq.pulses import Pulse, envelopes, units

pytestmark = pytest.mark.cosim

F_GE = 50e6                 # planted qubit frequency
F_EF = 40e6                 # the EF carrier the train retunes to
RO_CODE = 2048
RO_DUR = 40
X90_SAMPLES = 72            # 18 batches at gateInterp 4 — the X6Y3 X90 (35 ns) to scale
RELAX = 200                 # a short co-sim relax head

# Captures are SIZED, not generous (01 §3.3): `dac_capture_get` BLOCKS until the armed window is
# full, so every armed batch past the last gate is simulated for nothing. A capture armed before the
# reset release pays the core's boot + preamble first (measured ~800 batches on this build), and the
# one shot these kernels play lives entirely inside the ONE grid period that follows — so
# BOOT_NCAP + period covers it with the whole idle relax head to spare. Each test prints its
# windows and asserts an exact gate count, so an undersized capture fails loudly, never silently.
BOOT_NCAP = 1400


@kernel
def _k_unpaced(gate: ParamTable, ro: ParamTable, demod: ParamTable, out: Array, period: int,
               ngates: int, code: int):
    """k_rabi's train as it was BEFORE F1: `ngates` pushes issued back-to-back, no pacing."""
    init_pulse_params(demod.pulses)  # noqa: F821
    set_freq(demod, code)  # noqa: F821
    init_pulse_params(ro.pulses)  # noqa: F821
    set_freq(ro, ro.freq)  # noqa: F821
    init_pulse_params(gate.pulses)  # noqa: F821
    set_freq(gate, gate.freq)  # noqa: F821
    d = gate["x90"].dur  # noqa: F821
    t_ro = now() + period  # noqa: F821
    play(gate, gate["x90"], t_ro - SEP - ngates * d)  # noqa: F821
    for g in range(ngates - 1):
        fire(gate, gate["x90"])  # noqa: F821
    play(ro, ro["meas"], t_ro)  # noqa: F821
    play(demod, demod["sq"], t_ro)  # noqa: F821
    wait_until(t_ro + READOUT_LEAD)  # noqa: F821
    out[0] = read_res()  # noqa: F821


def _tables(m, samples=X90_SAMPLES):
    ro_freq = units.demod_code_to_freq(RO_CODE, m.params)
    env = envelopes.square(samples)
    gate = ParamTable(0, F_GE, {"x90": Pulse(env, freq_hz=F_GE, amp=0.5),
                                "ef": Pulse(env, amp=0.5)})       # baseband: the EF segment retunes
    ro = ParamTable(1, ro_freq, {"meas": Pulse(envelopes.square(RO_DUR + 16), freq_hz=ro_freq,
                                               amp=0.5)})
    demod = ParamTable(2, 0.0, {"sq": Pulse(envelopes.square(RO_DUR), amp=1.0)})
    return gate, ro, demod


def _capture(drv, m, prog, period):
    """Run one program with the gate DAC captured from before the reset release, over a window
    sized to boot + the shot's own grid period (BOOT_NCAP)."""
    ncap = BOOT_NCAP + period
    rq.setup(drv, m, {0: prog})
    rq.check_magic(drv, m, 0, prog)
    rq.write_var(drv, m, 0, prog, "__rq_status", 0)
    rq.write_params(drv, m, 0, prog)
    handle = drv.sim.dac_capture_arm(m.gate_dac(0), ncap)
    rq.reset(drv, m, on=False)
    rq.poll_done(drv, m, 0, prog, timeout=period * 8 + 20_000_000)
    rq.reset(drv, m, on=True)
    return drv.sim.dac_capture_get(handle)


def _windows(cap):
    """The (start, length) of every contiguous run of active DAC batches."""
    active = cap.any(axis=1)
    out, i = [], 0
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def _check_train(wins, n, d, step, where):
    """`n` gates, each `d` batches long, on a `step` grid — one window per gate when the step leaves a
    gap, one merged window of n·d when the pulse fills its step (step == d, back-to-back)."""
    if step == d:
        assert len(wins) == 1 and wins[0][1] == n * d, \
            f"{where}: want one back-to-back window of {n * d}, got {[w[1] for w in wins]}"
        return wins[0][0]
    assert len(wins) == n, f"{where}: want {n} gates, got {len(wins)} ({[w[1] for w in wins][:8]})"
    assert all(w[1] == d for w in wins), f"{where}: short gates {sorted({w[1] for w in wins})}"
    starts = [w[0] for w in wins]
    assert np.array_equal(np.diff(starts), np.full(n - 1, step)), \
        f"{where}: train off its {step}-batch grid (diffs {sorted(set(np.diff(starts)))})"
    return starts[0]


@pytest.mark.parametrize("ngates", [4, 96])
def test_x90_train_holds_at_the_reference_depth(cosim, ngates):
    """The GE Rabi ladder's deepest reference rung (n_gates = 96, qcal's X90 ladder) plays all 96
    gates, each full length, on the paced train grid. n=4 is the shallow rung the old unpaced train
    could still do."""
    drv, m = cosim
    drv.sim.set_model({"kind": "zero"})              # the gate DAC then carries only the core's gates
    gate, ro, demod = _tables(m)
    d = gate.pulses["x90"].dur_batches(m, 0)
    step = train_step(d)
    assert (d, step) == (18, TRAIN_STEP), f"probe sizing changed: d={d} step={step}"
    seq = (ngates - 1) * step + d
    period = grid_period(RELAX, seq, RO_DUR, 0)
    prog = compile_kernel(kernels.k_rabi, m, tables=dict(gate=gate, ro=ro, demod=demod),
                          out=Array(1), npts=1, shots=1, period=period, ngates=ngates, step=step,
                          code=pack16(RO_CODE), mode=kernels.COUNTS, ddly=0, prep_gate=X90,
                          vz0=0, vzsum=0, a0q=(16000 << 16), daq=0, prep=1, herald=0, hoff=0)
    t0, cap = _capture(drv, m, prog, period)
    wins = _windows(cap)
    first = _check_train(wins, ngates, d, step, f"k_rabi n={ngates}")
    print(f"\n[x90-train] n={ngates} d={d} step={step} gates={len(wins)} span={wins[-1][0] - first}")
    assert wins[-1][0] + wins[-1][1] <= first + seq, "the train overran its own grid"


def test_unpaced_train_would_drop_gates(cosim):
    """The finding this milestone is built on (spec 14 §3.1): with the pacing removed — the pushes
    issued back-to-back, as every train did before — the depth-4 queues swallow everything past the
    fourth gate. Guards the pacing against being 'simplified' away."""
    drv, m = cosim
    drv.sim.set_model({"kind": "zero"})
    gate, ro, demod = _tables(m)
    d = gate.pulses["x90"].dur_batches(m, 0)
    ngates = 96
    period = grid_period(RELAX, ngates * d, RO_DUR, 0)
    prog = compile_kernel(_k_unpaced, m, tables=dict(gate=gate, ro=ro, demod=demod), out=Array(1),
                          period=period, ngates=ngates, code=pack16(RO_CODE))
    t0, cap = _capture(drv, m, prog, period)
    played = sum(w[1] for w in _windows(cap)) / d
    print(f"\n[unpaced] n={ngates} gates played ~ {played:.0f}")
    assert played < ngates / 2, f"the unpaced train played {played:.0f}/{ngates} — is the queue deeper?"


def test_cz_train_holds_at_the_reference_depth(cosim):
    """The CZ ladder's reference rung (n_gates = 15, qcal's 1→3→…→15 pulse-shape ladder). A CZ pulse
    is far longer than the pacing floor, so its train stays BACK-TO-BACK — only the queue pacing was
    missing, which capped the ladder at (1, 3). Driven through k_cz_cond's COUPLER role, which fires
    the train and never reads out, so one core is the whole measurement."""
    drv, m = cosim
    drv.sim.set_model({"kind": "zero"})
    _, ro, demod = _tables(m)
    czd, ngates = 50, 15                                  # 50 batches ≫ the pacing floor
    cz = ParamTable(0, F_GE, {"cz": Pulse(envelopes.square(czd * 4), freq_hz=F_GE, amp=0.5)})
    assert cz.pulses["cz"].dur_batches(m, 0) == czd and train_step(czd) == czd
    xd = 18                                               # the qubit cores' X90 (unused by the coupler)
    period = grid_period(RELAX, 3 * xd + ngates * czd, RO_DUR, 0)
    prog = compile_kernel(kernels.k_cz_cond, m, tables=dict(gate=cz, ro=ro, demod=demod),
                          out=Array(1), npts=1, shots=1, period=period, code=pack16(RO_CODE),
                          ddly=0, role=kernels.COUPLER, knob=kernels.AMP, form=kernels.COUPLER_FORM,
                          ngates=ngates, vz0=0, vzsum=0, hpi=0, zi=0, iz=0, xd=xd, czd=czd,
                          fcz=0, fef=0, sw=0, tail=0, x0=(16000 << 16), dx=0, prep=0, quad=0)
    t0, cap = _capture(drv, m, prog, period)
    wins = _windows(cap)
    _check_train(wins, ngates, czd, czd, "k_cz_cond n=15")
    print(f"\n[cz-train] n={ngates} czd={czd} one window of {wins[0][1]} batches")


def test_ef_train_holds_at_the_reference_depth(cosim):
    """The EF Rabi ladder's reference rung (n_gates = 24): the paced EF train plays every gate after
    the GE prep, a full LEAD later (the retune's phasor-regen gap)."""
    drv, m = cosim
    drv.sim.set_model({"kind": "zero"})
    gate, ro, demod = _tables(m)
    ge = gate.pulses["x90"].dur_batches(m, 0)
    ef = gate.pulses["ef"].dur_batches(m, 0)
    step, ngates = train_step(ef), 24
    seq = SEP + (ngates - 1) * step + ef + LEAD + 2 * ge
    period = grid_period(RELAX, seq, RO_DUR, 0)
    prog = compile_kernel(kernels.k_ef_rabi, m, tables=dict(gate=gate, ro=ro, demod=demod),
                          out=Array(2), npts=1, shots=1, period=period, ngates=ngates, step=step,
                          code=pack16(RO_CODE), ddly=0,
                          ge_freq=units.freq_to_code(F_GE, m.params),
                          ef_freq=units.freq_to_code(F_EF, m.params), vz0=0, vzsum=0,
                          evz0=0, evzsum=0, a0q=(16000 << 16), daq=0)
    t0, cap = _capture(drv, m, prog, period)
    wins = _windows(cap)
    prep, train = wins[0], wins[1:]                  # the two-X90 GE prep is one merged window
    assert prep[1] == 2 * ge, f"the GE prep is not 2 X90s: {prep[1]} batches"
    start = _check_train(train, ngates, ef, step, "k_ef_rabi n=24")
    print(f"\n[ef-train] n={ngates} ef={ef} step={step} gates={len(train)} "
          f"prep_gap={start - (prep[0] + prep[1])}")
    assert start - (prep[0] + prep[1]) >= LEAD, "the EF train must start a full LEAD after the prep"
