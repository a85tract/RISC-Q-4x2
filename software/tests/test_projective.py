"""B1 acceptance (spec 08 §2.4 / §9): the projective TwoLevelModel (collapse=True) drives the REAL
readout datapath (readout drive → demod carrier → carrier-triggered decoder → res/real/imag) to
bimodal IQ clusters with binomial counts — the substrate counts mode needs. An X90 prepares a bz=0
superposition every shot (T1-reset in the idle grid head), so each shot samples s ~ Bernoulli(0.5):
the shots split into two antipodal IQ clusters and res is ~50/50 within the binomial CI.

The soft model (default) is exercised unchanged by the whole existing suite (test_readout/test_cal).
"""

import math

import numpy as np
import pytest

from riscq import run as rq
from riscq.cal.base import SEP, gate_sigma
from riscq.lang import Array, ParamTable, compile_kernel, kernel
from riscq.map import LEAD, READOUT_LEAD, pack16
from riscq.pulses import Pulse, envelopes, units

pytestmark = pytest.mark.cosim

F_GE = 50e6
RO_CODE = 2048
RO_DUR = 40
GATE_ENV = envelopes.square(16)   # 4-batch square X90


def _tables(m):
    ro_freq = units.demod_code_to_freq(RO_CODE, m.params)   # ~physical readout freq (model ignores ch1 freq)
    gate = ParamTable(0, F_GE, {"x90": Pulse(GATE_ENV, freq_hz=F_GE, amp=0.5)})
    ro = ParamTable(1, ro_freq, {"meas": Pulse(envelopes.square(RO_DUR + 16), freq_hz=ro_freq, amp=0.5)})
    demod = ParamTable(2, 0.0, {"sq": Pulse(envelopes.square(RO_DUR), amp=1.0)})
    return gate, ro, demod


@kernel
def k_read(demod: ParamTable, out: Array, code: int):
    """A demod-only |0> read (for the demod-phase calibration; the model synthesizes the tone)."""
    init_pulse_params(demod.pulses)  # noqa: F821
    set_freq(demod, code)  # noqa: F821
    t = now() + LEAD  # noqa: F821
    play(demod, demod["sq"], t)  # noqa: F821
    wait_until(t + READOUT_LEAD)  # noqa: F821
    read_res()  # noqa: F821
    out[0] = read_real()  # noqa: F821
    out[1] = read_imag()  # noqa: F821


@kernel
def k_shots(gate: ParamTable, ro: ParamTable, demod: ParamTable, res_out: Array, iq: Array,
            code: int, period: int, nshots: int):
    """N shots of [T1-reset idle → X90 prep → readout drive + demod → read]. The readout drive (ch1)
    is the projective model's window trigger; firing the demod carrier IS the readout."""
    init_pulse_params(demod.pulses)  # noqa: F821
    set_freq(demod, code)  # noqa: F821
    init_pulse_params(ro.pulses)  # noqa: F821
    set_freq(ro, ro.freq)  # noqa: F821
    init_pulse_params(gate.pulses)  # noqa: F821
    set_freq(gate, gate.freq)  # noqa: F821
    t_ro = now() + period                          # first grid slot (period idle head resets to |0>)
    k = 0
    for s in range(nshots):
        tpi = t_ro - SEP - gate["x90"].dur  # noqa: F821
        play(gate, gate["x90"], tpi)  # noqa: F821  X90 → bz=0 superposition
        play(ro, ro["meas"], t_ro)  # noqa: F821     readout drive (projective window trigger; hardware)
        play(demod, demod["sq"], t_ro)  # noqa: F821  demod carrier IS the readout
        wait_until(t_ro + READOUT_LEAD)  # noqa: F821
        res_out[s] = read_res()  # noqa: F821
        iq[k] = read_real()  # noqa: F821
        iq[k + 1] = read_imag()  # noqa: F821
        k = k + 2
        t_ro = t_ro + period


def _phase_cal(drv, m, demod):
    """The demod-LO phase (rad) that lands a soft |0> read on +real (two probe reads pin the fixed
    ADC→integrator pipeline phase and its slope). Measured under the SOFT model; applied to projective."""
    base = dict(kind="twolevel", rabi_rad_per_amp=0.0, readout_code=RO_CODE, readout_amp=20000.0)

    def rd(ph):
        drv.sim.set_model({**base, "readout_phase": ph})
        prog = compile_kernel(k_read, m, tables=dict(demod=demod), out=Array(2), code=pack16(RO_CODE))
        out = rq.run(drv, m, {0: prog}, timeout=1_000_000)[0]["out"]
        return float(out[0]), float(out[1])

    r0, i0 = rd(0.0)
    r1, i1 = rd(0.4)
    a0, a1 = math.atan2(i0, r0), math.atan2(i1, r1)
    s = 1.0 if math.remainder(a1 - a0, 2 * math.pi) >= 0 else -1.0
    return -s * a0


@pytest.fixture(autouse=True)
def _zero_after(cosim):
    yield
    cosim[0].sim.set_model({"kind": "zero"})


def test_projective_bimodal_binomial(cosim):
    drv, m = cosim
    gate, ro, demod = _tables(m)

    # calibrate the model so the X90 rotates by exactly π/2 (→ bz=0), and the demod phase lands |0> on +real
    x90 = gate.pulses["x90"]
    rabi = (math.pi / 2) / gate_sigma(m, x90, F_GE, x90.amp_code())
    phase = _phase_cal(drv, m, demod)

    # t1 ≫ SEP so the equator survives the idle before the readout (a short t1 would relax bz→+1 and
    # bias counts toward 0). No between-shot reset is needed: the collapse leaves a PURE pole
    # (bx=by=0) and the X90 maps any (0,0,z) to bz=0, so every shot re-prepares p=0.5 on its own.
    drv.sim.set_model(dict(kind="twolevel", core=0, collapse=True, rabi_rad_per_amp=rabi,
                           readout_code=RO_CODE, readout_amp=20000.0, readout_phase=phase, f_ge=F_GE,
                           t1=4000, t2=6000, noise_scale=300.0, noise_seed=2))
    nshots = 150
    prog = compile_kernel(k_shots, m, tables=dict(gate=gate, ro=ro, demod=demod),
                          res_out=Array(nshots), iq=Array(2 * nshots),
                          code=pack16(RO_CODE), period=256, nshots=nshots)
    out = rq.run(drv, m, {0: prog}, timeout=20_000_000)[0]

    res = out["res_out"].astype(int)
    iq = out["iq"].astype(np.int64)
    z = np.array([complex(int(iq[2 * k]), int(iq[2 * k + 1])) for k in range(nshots)])
    real = z.real

    print(f"\n[projective] n={len(res)} counts p̂={res.mean():.3f} |z|̄={np.abs(z).mean():.3e} "
          f"real+̄={real[res == 0].mean():.3e} real−̄={real[res == 1].mean():.3e}")

    # 1. definite-state signal: |z| is full-scale every shot (collapse ⇒ ±amp, not the ⟨σz⟩=0 soft ~0)
    assert np.abs(z).mean() > 1_000_000, "projective |z| too small — not collapsing to a definite state"

    # 2. binomial counts: X90 → p=0.5; both outcomes occur, fraction within the binomial CI
    assert 0 in res and 1 in res, f"one outcome never sampled: res={res.tolist()}"
    p = res.mean()
    sigma = math.sqrt(0.25 / len(res))
    assert abs(p - 0.5) < 4 * sigma, f"counts fraction {p:.3f} not ~0.5 (±{4 * sigma:.3f})"

    # 3. bimodal + phase-calibrated res: |0>-shots (res=0) on +real, |1>-shots on −real, well separated
    assert (real[res == 0] > 0).all() and (real[res == 1] < 0).all(), "res sign disagrees with real sign"
    sep = abs(real[res == 0].mean() - real[res == 1].mean())
    spread = max(real[res == 0].std(), real[res == 1].std())
    assert sep > 5 * spread, f"IQ clusters not well separated (bimodal): sep={sep:.2e} spread={spread:.2e}"
