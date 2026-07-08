"""M4a model-physics sanity (cosim): prove the extended TwoLevelModel BEFORE the M4b
calibrations build on it.

- Ramsey (X90, wait, X90, measure) on a DETUNED model: the integrated readout oscillates vs the
  wait at the planted detuning. The extended model takes the drive AXIS from the gate DAC
  demodulated against f_ge, so a carrier off f_ge ramps the axis at (f_drive − f_ge) and the fringe
  falls out with no explicit free-precession term — <σz> = −exp(−t/T2)·cos(2π·f·τ) with
  f = δ_code/4096 cycles per wait-batch (δ_code = f_drive − f_ge in SF16). Recovered with
  fits.fit_damped_cosine.
- T1 (prepare |1> with a π pulse, wait, measure): <σz> recovers toward ground as 1 − 2·exp(−Δt/T1),
  recovered with fits.fit_exp_decay. To keep the demod phase and the decay-clock reference constant
  across the sweep, the readout time is FIXED and the π pulse is placed `delay` batches before it
  (a longer delay = an earlier pulse), so the elapsed decay is exactly `delay`.

Each sweep point re-prepares the qubit from |0> (set_model rebuilds the model per point — the model
is one evolving Bloch vector, so it MUST be reset between points). The demod pipeline phase is
calibrated once (a ReadoutCalibration in miniature, like M3) so the integrated real carries the
signed <σz>. Every model is selected over Pyro5; the fixture is session-scoped.
"""

import math

import numpy as np
import pytest

from riscq import run as rq
from riscq.cal import fits
from riscq.cal.base import SEP, demod_table
from riscq.lang import Array, ParamTable, compile_kernel, kernel
from riscq.map import LEAD, READOUT_LEAD, pack16
from riscq.pulses import Pulse, envelopes, golden, units

pytestmark = pytest.mark.cosim

RO_CODE = 2048
RO_DUR = 40
RO_AMP = 20000.0
F_DRIVE_HZ = 50e6           # freq_to_code = 2048 at 100 MHz dsp
DETUNE_CODE = 64           # planted Ramsey detuning: f_ge sits DETUNE_CODE below the drive


# ── kernels ──

@kernel
def k_read(demod: ParamTable, out: Array, code: int):
    """One readout window against demod `code` — the demod-phase calibration probe."""
    init_pulse_params(demod.pulses)  # noqa: F821
    set_freq(demod, code)  # noqa: F821
    t = now() + LEAD  # noqa: F821
    play(demod, demod["sq"], t)  # noqa: F821  firing the demod IS the readout
    wait_until(t + READOUT_LEAD)  # noqa: F821  past the window's opening
    out[0] = read_res()  # noqa: F821
    out[1] = read_real()  # noqa: F821
    out[2] = read_imag()  # noqa: F821


@kernel
def k_ramsey(gate: ParamTable, demod: ParamTable, out: Array, wait: int, maxw: int, code: int):
    """X90 — wait — X90 — measure. The two X90s are `wait` batches apart; on a detuned model their
    axes differ by the accumulated detuning phase, so <σz> fringes vs `wait`. The readout is at a
    FIXED t_ro (maxw covers every wait) so the demod carrier phase is the same across the sweep — a
    varying t_ro would ramp the demod phase with wait and beat against the fringe."""
    init_pulse_params(demod.pulses)  # noqa: F821
    set_freq(demod, code)  # noqa: F821
    init_pulse_params(gate.pulses)  # noqa: F821
    set_freq(gate, gate.freq)  # noqa: F821
    d = gate["x90"].dur  # noqa: F821
    t_ro = now() + LEAD + maxw + 2 * d + SEP  # noqa: F821  fixed readout time
    play(demod, demod["sq"], t_ro)  # noqa: F821  (demod carrier at the fixed window; firing it IS the readout)
    ta = t_ro - SEP - d  # noqa: F821  2nd X90 start
    tb = ta - wait - d  # noqa: F821  1st X90 start (`wait` batches earlier)
    play(gate, gate["x90"], tb)  # noqa: F821
    play(gate, gate["x90"], ta)  # noqa: F821
    wait_until(t_ro + READOUT_LEAD)  # noqa: F821  past the window's opening
    read_res()  # noqa: F821  (HALTS until the window settles)
    out[0] = read_real()  # noqa: F821
    out[1] = read_imag()  # noqa: F821


@kernel
def k_t1(gate: ParamTable, demod: ParamTable, out: Array, delay: int, maxd: int, code: int):
    """Prepare |1> with a resonant π pulse, then read at a FIXED time (t_ro): the π pulse ends
    `delay` batches before it, so the elapsed decay is exactly `delay` and the readout LO phase is
    the same for every point (t_ro constant)."""
    init_pulse_params(demod.pulses)  # noqa: F821
    set_freq(demod, code)  # noqa: F821
    init_pulse_params(gate.pulses)  # noqa: F821
    set_freq(gate, gate.freq)  # noqa: F821
    t_ro = now() + LEAD + maxd + gate["x180"].dur  # noqa: F821
    play(demod, demod["sq"], t_ro)  # noqa: F821  (demod carrier at the fixed readout time; firing it IS the readout)
    tp = t_ro - delay - gate["x180"].dur  # noqa: F821
    play(gate, gate["x180"], tp)  # noqa: F821
    wait_until(t_ro + READOUT_LEAD)  # noqa: F821  past the window's opening
    read_res()  # noqa: F821
    out[0] = read_real()  # noqa: F821
    out[1] = read_imag()  # noqa: F821


# ── helpers ──

def _rabi_for(m, pulse: Pulse, carrier_hz: float, target_rad: float) -> float:
    """The model Rabi rate that makes `pulse` rotate by `target_rad`: rate = target / Σ amp_est,
    where amp_est is the model's per-batch phase-blind RMS over the bit-exact DAC golden (the same
    samples the model sees) — exactly M3's Rabi calibration, generalized to an arbitrary angle."""
    lines = pulse.packed_lines(m, 0)
    w = golden.pulse_window(lines, pulse.amp_code(), units._freq_code(carrier_hz, m.params),
                            0, 0, len(lines))
    sigma = sum(math.sqrt(2 * np.mean(row.astype(float) ** 2)) for row in w)
    return target_rad / sigma


def _sweep(cosim, prog, spec, name, points, extra=None):
    """Run `prog` once per sweep value, re-preparing the model (fresh Bloch state) each point;
    return the complex integrated readout z = real + i·imag."""
    drv, m = cosim
    z = []
    for v in points:
        drv.sim.set_model(spec)
        params = {name: int(v), "code": pack16(RO_CODE), **(extra or {})}
        out = rq.run(drv, m, {0: prog}, params={0: params}, timeout=3_000_000)[0]["out"]
        z.append(complex(float(out[0]), float(out[1])))
    return np.array(z)


@pytest.fixture(scope="module")
def demod_cal(cosim):
    """The demod-LO phase (rad) that lands a |0> readout tone on +real — measured once (the M3
    two-probe method); the readout pipeline phase depends only on RO_CODE/RO_DUR, so both tests
    reuse it."""
    drv, m = cosim
    prog = compile_kernel(k_read, m, tables=dict(demod=demod_table(RO_DUR)),
                          out=Array(3), code=pack16(RO_CODE))

    def read(phase):
        drv.sim.set_model(dict(kind="twolevel", rabi_rad_per_amp=0.0, readout_code=RO_CODE,
                               readout_amp=RO_AMP, readout_phase=phase, init_excited=False))
        out = rq.run(drv, m, {0: prog}, timeout=1_000_000)[0]["out"]
        return int(out[1]), int(out[2])

    r0, i0 = read(0.0)
    r1, i1 = read(0.4)
    a0, a1 = math.atan2(i0, r0), math.atan2(i1, r1)
    s = 1.0 if math.remainder(a1 - a0, 2 * math.pi) >= 0 else -1.0
    return -s * a0


@pytest.fixture(autouse=True)
def _zero_model_after(cosim):
    yield
    cosim[0].sim.set_model({"kind": "zero"})


# ── tests ──

def test_ramsey_oscillates_at_planted_detuning(cosim, demod_cal):
    _, m = cosim
    f_ge_hz = units.code_to_freq(units._freq_code(F_DRIVE_HZ, m.params) - DETUNE_CODE, m.params)
    x90 = Pulse(envelopes.square(16), freq_hz=F_DRIVE_HZ, amp=0.5)     # short => little intra-pulse smear
    gate = ParamTable(0, F_DRIVE_HZ, {"x90": x90})
    rabi = _rabi_for(m, x90, F_DRIVE_HZ, math.pi / 2)

    prog = compile_kernel(k_ramsey, m, tables=dict(gate=gate, demod=demod_table(RO_DUR)),
                          out=Array(2))
    spec = dict(kind="twolevel", core=0, rabi_rad_per_amp=rabi, readout_code=RO_CODE,
                readout_amp=RO_AMP, readout_phase=demod_cal, f_ge=f_ge_hz, t2=300)
    waits = list(range(0, 195, 13))                                   # 15 points, ~3 fringe periods
    # fixed readout time => the demod phase is constant across the sweep; demod_cal lands the fringe on real
    reals = _sweep(cosim, prog, spec, "wait", waits, extra={"maxw": max(waits)}).real

    fit = fits.fit_damped_cosine(waits, reals)
    f_expect = DETUNE_CODE / (1 << 12)                                # δ_code/4096 cycles per wait-batch
    print(f"\n[ramsey] fit ok={fit.ok} f={fit.value:.5f} expect={f_expect:.5f} "
          f"tau={fit.params.get('tau'):.1f}")
    assert fit.ok, "damped-cosine fit failed on the Ramsey fringe"
    assert abs(fit.value - f_expect) < 0.1 * f_expect, \
        f"recovered Ramsey freq {fit.value:.5f} != planted {f_expect:.5f}"


def test_t1_exponential_decay(cosim):
    _, m = cosim
    t1 = 120                                                          # planted T1, in batches
    x180 = Pulse(envelopes.square(64), freq_hz=F_DRIVE_HZ, amp=0.9)
    gate = ParamTable(0, F_DRIVE_HZ, {"x180": x180})
    rabi = _rabi_for(m, x180, F_DRIVE_HZ, math.pi)

    prog = compile_kernel(k_t1, m, tables=dict(gate=gate, demod=demod_table(RO_DUR)), out=Array(2))
    spec = dict(kind="twolevel", core=0, rabi_rad_per_amp=rabi, readout_code=RO_CODE,
                readout_amp=RO_AMP, readout_phase=0.0, f_ge=F_DRIVE_HZ, t1=t1)  # resonant π; phase via projection
    # delays skip 0 (a pulse abutting the window smears the flip) and stay small enough that the
    # earliest π pulse (furthest in the future) is inside the fire-scheduling horizon.
    delays = list(range(45, 450, 45))                                # 9 points, ~3·T1
    extra = {"maxd": delays[-1]}
    z = _sweep(cosim, prog, spec, "delay", delays, extra=extra)
    # the fixed readout time fixes the demod-LO phase across the sweep, but at an arbitrary absolute
    # angle; project onto a ground-state reference at the SAME readout structure (rabi=0 => the π
    # pulse is a no-op, qubit stays |0>) so the signal carries the signed <σz> cleanly.
    zg = _sweep(cosim, prog, {**spec, "rabi_rad_per_amp": 0.0}, "delay", [delays[0]], extra=extra)[0]
    reals = (z * zg.conjugate()).real / abs(zg)

    fit = fits.fit_exp_decay(delays, reals)
    print(f"\n[t1] fit ok={fit.ok} tau={fit.value:.1f} expect={t1} amp={fit.params.get('amp'):.0f}")
    assert fit.ok, "exp-decay fit failed on the T1 recovery"
    assert fit.params["amp"] < 0, "|1> recovers toward ground (+real), so the exp amplitude is negative"
    assert 0.85 * t1 < fit.value < 1.15 * t1, \
        f"recovered T1 {fit.value:.1f} batches != planted {t1}"
