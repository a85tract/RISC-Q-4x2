"""B2 acceptance (spec 08 §9): the batched kernels — one run returns a whole sweep. Each scenario
drives a batched kernel DIRECTLY (the cal-class rewiring is B5), so this file is where the four
kernels' own circuits and the batched run layer are pinned, split by tier
(specs/software-test-refactor/01, migration 02 §3.3):

- **L1** (01 §3) — what the SoC EMITS, model off or on a deterministic `ToneModel`: the matched-pair
  IQSUM VNA and its grid invariance, the computed amp sweep bit-exact on the gate DAC, the demod
  table's phase and window knobs through the real `write_slot` + rerun path, and the RAW per-shot
  cursor (row 0 is not an outlier; the two preps land antipodal).
- **L2** (01 §4) — what those circuits do to the QUBIT, read off `drv.sim.model_state()` /the real
  soft readout through `tests.probe.Probe`: one shot per point, an ANALYTIC target, no fit and no
  shot statistics. These four replace `test_counts_{rabi,frequency,t1,t2}`, which re-measured
  `test_cal.py`'s physics at the kernel level through 21×160 / 15×48 / 9×120 projective sweeps.

The fits those counts sweeps ran, and the statistics that made them meaningful, are host-pure:
`riscq.cal.fits` in `test_cal_fits.py`, the cluster/binomial properties in `test_models.py`, and the
whole fit→propose→apply loop in `test_cal_host.py` against the analytic responder.
"""

import math

import numpy as np
import pytest

from riscq import build, run as rq
from riscq.cal.base import SEP, X, X90, gate_sigma, sweep_q16
from riscq.cal import kernels
from riscq.lang import Array, ParamTable, compile_kernel
from riscq.map import READOUT_MAX_WIN_LOG2, pack16
from riscq.pulses import Pulse, envelopes, golden, units
from tests.probe import Probe, rabi_for

pytestmark = pytest.mark.cosim

F_GE = 50e6                       # planted qubit frequency (freq_code 2048)
# The demod carrier code, = the model's readout tone code. 8192 is a QUARTER TURN per ADC sample, so
# the demod product's counter-rotating 2ω term closes exactly every batch and the integral is a clean
# multiple of the overlapping batches. At the model's other usual code, 2048, it does not: the
# leftover ripple is a state-independent ±3.8 % of a 40-batch integral whose PHASE turns a quarter
# per batch of window start — and `now()` (and so the grid) lands wherever the host released the
# reset, so it differs run to run. Every claim below that compares two RERUNS' integrals (the
# antipodal preps, the demod-phase rotation, the T1 ratios) would carry that ripple as an
# irreducible few-percent error. test_cal.py::test_readout_timing_knob_moves_the_readout picks 8192
# for exactly this reason.
RO_CODE = 8192
RO_DUR = 40                       # demod window (batches)
GATE_ENV = envelopes.square(16)   # 4-batch square gate
COUNTS, RAW = kernels.COUNTS, kernels.RAW

# The grid period every test here runs on. The longest prelude any of these kernels schedules is
# LEAD (the pulse's scheduling lead) + the sequence + SEP (pulse end → readout), i.e. ~200 batches,
# so 256 is the shortest legal 8-multiple that clears it. The relax head is GONE: nothing below
# needs the qubit reset by an idle, because `set_model` rebuilds it in zero simulated cycles
# (01 §4.2) and the L1 tests carry no quantum state at all. That is a 32× cut on the old 8192.
PERIOD = 256

# The soft L2 model: no noise, no collapse, no decay unless the test is about decay (01 §4.6).
L2_MODEL = dict(kind="twolevel", core=0, f_ge=F_GE, readout_code=RO_CODE, readout_amp=20000.0,
                readout_phase=0.0, noise_scale=0.0, collapse=False)


def _tables(m, gate_amp=0.5):
    ro_freq = units.demod_code_to_freq(RO_CODE, m.params)   # physical readout freq (model ignores ch1 freq)
    gate = ParamTable(0, F_GE, {"x90": Pulse(GATE_ENV, freq_hz=F_GE, amp=gate_amp)})
    gate180 = ParamTable(0, F_GE, {"x": Pulse(GATE_ENV, freq_hz=F_GE, amp=0.99)})   # the "X" prep
    ro = ParamTable(1, ro_freq, {"meas": Pulse(envelopes.square(RO_DUR + 16), freq_hz=ro_freq, amp=0.5)})
    demod = ParamTable(2, 0.0, {"sq": Pulse(envelopes.square(RO_DUR), amp=1.0)})
    return gate, gate180, ro, demod


def _rabi_pi(m):
    return float(math.pi / gate_sigma(m, Pulse(GATE_ENV, freq_hz=F_GE, amp=0.99), F_GE,
                                      units._amp_code(0.99)))


def _timeout(nbatches):
    return int(nbatches) * 4 + 20_000_000


@pytest.fixture(scope="session")
def sub(cosim):
    """(drv, m, (gate, gate180, ro, demod)) — the shared table substrate for the L1 tests."""
    drv, m = cosim
    return drv, m, _tables(m)


@pytest.fixture(autouse=True)
def _zero_after(cosim):
    yield
    cosim[0].sim.set_model({"kind": "zero"})


def _mags(out, npts):
    o = np.asarray(out, dtype=np.int64)
    return np.array([math.hypot(int(o[2 * i]), int(o[2 * i + 1])) for i in range(npts)])


# ── L1 iqsum: the matched-pair VNA batches on-core; argmax matches the Separation result ──

def test_vna_argmax(sub):
    drv, m, (gate, gate180, ro, demod) = sub
    F, points, shots = 256, 7, 2
    c0q, dcq, xs = sweep_q16(F, points * F, points)      # xs = F..7F — matched (4x) at real index 3 (1024)
    npts = points
    sh = max(0, (shots - 1).bit_length())
    drv.sim.set_model({"kind": "tone", "adc": m.adc_of(0),
                       "freq_hz": units.code_to_freq(1024, m.params), "amp": 20000.0})
    prog = compile_kernel(kernels.k_vna, m, fw32=int(m.params.freq_width == 32), tables=dict(ro=ro, demod=demod),
                          out=Array(2 * npts), npts=npts, shots=shots, period=PERIOD, sh=sh, ddly=0,
                          mode=kernels.IQSUM, c0q=int(c0q), dcq=int(dcq))
    out = rq.run(drv, m, {0: prog},
                 timeout=_timeout(npts * shots * PERIOD))[0]["out"]
    mag = _mags(out, npts)
    print(f"\n[vna] codes/F={[int(x) // F for x in xs]} mag={[int(x) for x in mag]} argmax={int(np.argmax(mag))}")
    assert int(np.argmax(mag)) == 3, "matched-pair VNA peak not at 4x the DAC code"
    second = np.sort(mag)[-2]
    assert mag[3] > 3 * second, f"VNA peak not dominant: peak={mag[3]:.0f} second={second:.0f}"


# ── L1 grid-invariance: repeated identical rows replay identically across the batch ──

def test_grid_invariance(sub):
    drv, m, (gate, gate180, ro, demod) = sub
    npts, shots = 8, 2
    c0q, dcq = 1024 << 16, 0                              # dcq=0 → every row realizes code 1024
    sh = max(0, (shots - 1).bit_length())
    drv.sim.set_model({"kind": "tone", "adc": m.adc_of(0),
                       "freq_hz": units.code_to_freq(1024, m.params), "amp": 20000.0})
    prog = compile_kernel(kernels.k_vna, m, fw32=int(m.params.freq_width == 32), tables=dict(ro=ro, demod=demod),
                          out=Array(2 * npts), npts=npts, shots=shots, period=PERIOD, sh=sh, ddly=0,
                          mode=kernels.IQSUM, c0q=c0q, dcq=dcq)
    out = rq.run(drv, m, {0: prog},
                 timeout=_timeout(npts * shots * PERIOD))[0]["out"]
    mag = _mags(out, npts)
    print(f"\n[grid] mag={[int(x) for x in mag]} std/mean={mag.std() / mag.mean():.5f}")
    assert mag.std() / mag.mean() < 0.01, "identical point-table rows disagree across the batch"


# ── L1 raw: the two preps land ANTIPODAL through the real per-shot IQ cursor ──

def test_raw_clusters(sub):
    """L1 (spec 08 §9) — k_t1 in RAW mode writes each shot's IQ, and the |0>/|1> preps land as the
    SAME phasor negated: the two states' readout tones are π out of phase.

    Noiseless and one shot per prep. The old version ran 15 shots per prep on an 8192-batch grid
    (a 245 k-batch test) because a projective |1> prep needs the qubit RESET before every shot —
    the collapse leaves a pole and the next π flips it back — and the only reset available on-core
    is a relax head ≫ T1. `set_model` rebuilds the model in |0> for free (01 §4.2), so one shot per
    prep is a complete measurement: with noise off, a collapsed shot is a DEFINITE state and the
    two phasors are exactly antipodal. That tightens the claim from "the cluster means have opposite
    real parts" to "z(|1>) = −z(|0>) to 1 %", and drops the calibrated demod phase the sign test
    needed (the orientation claim lives in test_applied_demod_phase, on the real write_slot path).

    The cluster STATISTICS the old assertion really rested on — the qcal SNR of two noisy blobs —
    are host-pure in test_models.py::test_twolevel_projective_clusters_separate."""
    drv, m, (gate, gate180, ro, demod) = sub
    spec = dict(kind="twolevel", core=0, collapse=True, rabi_rad_per_amp=_rabi_pi(m),
                readout_code=RO_CODE, readout_amp=20000.0, readout_phase=0.0, f_ge=F_GE,
                noise_scale=0.0)                          # no decay: the π prep is exact
    npts, shots = 1, 1
    prog = compile_kernel(kernels.k_t1, m, tables=dict(gate=gate180, ro=ro, demod=demod),
                          out=Array(2 * npts * shots), npts=npts, shots=shots, period=PERIOD, ddly=0,
                          code=pack16(RO_CODE), mode=RAW, d0=SEP, dd=0, prep_gate=X)   # prep runtime
    probe = Probe((drv, m), {0: prog})
    z0 = probe.iq(spec, {0: {"prep": 0}})[0][0]           # |0>
    z1 = probe.iq(spec, {0: {"prep": 1}})[0][0]           # |1>: one X (π)
    print(f"\n[raw] z(|0>)={z0:.4e}  z(|1>)={z1:.4e}  |z1+z0|/|z0|={abs(z1 + z0) / abs(z0):.4f}")
    assert abs(z0) > 1_000_000, "the |0> shot is not a full-scale definite state"
    assert abs(z1 + z0) < 0.01 * abs(z0), \
        f"the two preps are not antipodal: z(|0>)={z0:.3e} z(|1>)={z1:.3e}"


# ── L1 W0 (spec 11): row 0 needs no warm-up — the run's FIRST window matches the ensemble ──

def test_first_row_clean(sub):
    """Spec 11 W0: identical-knob rows through the undiluted per-shot RAW path — every shot of
    every row, INCLUDING shot (0,0), the first window the program ever integrates, matches the
    ensemble. Noise-free model, prep=0 (no drive, deterministic collapse to |0>), dd=0: the
    matched-pair integral is window-start-invariant on the fixed grid, so a cold-first-read
    artifact would make shot 0 an outlier. Guards the warm-up-row removal (spec 11).

    The grid period is the schedule's floor rather than 8192: a |0> row plays no gate, so there is
    nothing for a relax head to reset and the 32× shorter grid tests exactly the same windows."""
    drv, m, (gate, gate180, ro, demod) = sub
    drv.sim.set_model(dict(kind="twolevel", core=0, collapse=True, rabi_rad_per_amp=_rabi_pi(m),
                           readout_code=RO_CODE, readout_amp=20000.0, readout_phase=0.0, f_ge=F_GE,
                           noise_scale=0.0))
    npts, shots = 4, 4
    prog = compile_kernel(kernels.k_t1, m, tables=dict(gate=gate180, ro=ro, demod=demod),
                          out=Array(2 * npts * shots), npts=npts, shots=shots, period=PERIOD, ddly=0,
                          code=pack16(RO_CODE), mode=RAW, d0=SEP, dd=0, prep_gate=X)
    out = rq.run(drv, m, {0: prog}, params={0: {"prep": 0}},
                 timeout=_timeout(npts * shots * PERIOD))[0]["out"]
    iq = out.reshape(npts * shots, 2).astype(float)
    z = iq[:, 0] + 1j * iq[:, 1]
    zm = z.mean()
    dev = np.abs(z - zm) / abs(zm)
    print(f"\n[first-row] |z|={abs(zm):.3e} max dev={dev.max():.2e} shot0 dev={dev[0]:.2e}")
    assert dev.max() < 0.005, f"a shot deviates {dev.max():.3%} from the ensemble (first-window artifact?)"


# ── L1 computed sweep: the on-core Q16 amp sweep realizes EXACTLY sweep_q16's xs (spec 09 §4.2) ──

def test_computed_amp_matches_xs(sub):
    """spec 09 §4.2 acceptance: the on-core computed amp sweep reproduces sweep_q16's xs. Capture the
    gate DAC across a computed-amp k_rabi run and match each per-point pulse window bit-exactly vs the
    golden at the host-mirrored code — `set_amp(aq)` raw == sweep_q16 point-for-point (a bit-exact
    readback, not a fit)."""
    drv, m, (gate, gate180, ro, demod) = sub
    drv.sim.set_model({"kind": "zero"})                  # the gate DAC carries only the core's gate pulses
    points, shots = 5, 1
    # This is the one test here that retunes a slot BETWEEN points mid-run, so it is the one that
    # needs slack in the grid: at PERIOD the point's `set_amp` push and the next point's pulse race,
    # and the capture shows the previous point's amplitude. 400 is the original grid — the subject
    # here is which codes the sweep realizes, not how short the period can be.
    period = 400
    a0q, daq, xs = sweep_q16(600, units.AMP_SCALE - 600, points)
    npts = points
    expected = [int(x) for x in xs]                      # realized code per point
    x90 = Pulse(GATE_ENV, freq_hz=F_GE, amp=0.5)
    lines = x90.packed_lines(m, 0)
    f_code, ph_code, dur = x90.freq_code(m), x90.phase_code(), len(lines)
    prog = compile_kernel(kernels.k_rabi, m, tables=dict(gate=gate, ro=ro, demod=demod),
                          out=Array(npts), npts=npts, shots=shots, period=period, ddly=0,
                          ngates=1, step=4, code=pack16(RO_CODE), mode=COUNTS, prep_gate=X90,
                          vz0=0, vzsum=0,
                          a0q=int(a0q), daq=int(daq), prep=1)
    # sized, not generous (01 §7): the capture is armed before the reset release, so it pays for the
    # core's boot (~1 500 batches) and then the npts x period grid. `dac_capture_get` BLOCKS until the
    # armed window is full, so every batch over that is simulated for nothing (the old 8000 wasted 4 000).
    ncap = 4000
    rq.setup(drv, m, {0: prog})
    rq.check_magic(drv, m, 0, prog)
    rq.write_var(drv, m, 0, prog, "__rq_status", 0)
    rq.write_params(drv, m, 0, prog)
    handle = drv.sim.dac_capture_arm(m.gate_dac(0), ncap)
    rq.reset(drv, m, on=False)
    rq.poll_done(drv, m, 0, prog, timeout=_timeout(npts * shots * period))
    rq.reset(drv, m, on=True)
    t0, cap = drv.sim.dac_capture_get(handle)

    active = cap.any(axis=1)                             # per-batch: any DAC lane nonzero?
    starts = [i for i in range(len(active)) if active[i] and (i == 0 or not active[i - 1])]
    print(f"\n[amp-xs] xs={[int(x) for x in xs]} windows={len(starts)} "
          f"first_offset={starts[0] if starts else None} t0={int(t0)}")
    assert len(starts) == npts, f"expected {npts} gate windows, found {len(starts)} (capture too short?)"
    for k, s in enumerate(starts):
        gold = golden.pulse_window(lines, expected[k], f_code, ph_code, int(t0 + s), dur)
        assert np.array_equal(cap[s:s + dur], gold), \
            f"point {k} DAC window != golden at code {expected[k]} (set_amp diverged from sweep_q16)"


# ── L1 B3 (run layer): the demod TABLE is the real discriminator/window knob — write_slot + rerun ──
#
# These retune a demod-table slot field through the actual hardware path (demod carrier → carrier-
# triggered decoder), NOT the soft-model shortcut: the model's `readout_phase` is pinned at 0 so the
# ONLY phase knob in play is the demod slot. The demod carrier is a real PulseGenerator whose slot
# phase rotates the decoder's integrated z, so the discriminator's phase and its integration window
# are both pure table writes + rerun (no recompile).

def test_applied_demod_phase(sub):
    """B3 (spec 08 §2.1): the discriminator's one programmable knob is the DEMOD-TABLE carrier phase,
    applied through the REAL hardware path (demod carrier → decoder), not the model's readout_phase
    (pinned at 0). `write_slot(..., "phase", code)` + rerun rotates the measured IQ by exactly that
    phase; calibrated so |0> lands on +real / |1> on −real, sign(sumR) then discriminates. No recompile.

    The |0> and |1> reads are separate `set_model` + rerun points rather than shots on a long grid:
    the soft model has no auto-reset, so without the rebuild the |1> prep would find the qubit
    wherever the previous point left it — which is the only thing the retired 8192-batch relax head
    was doing."""
    drv, m, (gate, gate180, ro, demod) = sub
    spec = dict(kind="twolevel", core=0, rabi_rad_per_amp=_rabi_pi(m),   # soft |0>/|1> tones
                readout_code=RO_CODE, readout_amp=20000.0, readout_phase=0.0, f_ge=F_GE)
    npts, shots = 1, 1
    prog = compile_kernel(kernels.k_t1, m, tables=dict(gate=gate180, ro=ro, demod=demod),
                          out=Array(2 * npts * shots), npts=npts, shots=shots, period=PERIOD, ddly=0,
                          code=pack16(RO_CODE), mode=RAW, d0=SEP, dd=0, prep_gate=X)
    runs = build.CC_RUNS
    probe = Probe((drv, m), {0: prog})

    z0 = probe.iq(spec, {0: {"prep": 0}})[0][0]                     # demod phase 0 → fixed pipeline angle
    a0 = math.atan2(z0.imag, z0.real)
    rq.write_slot(drv, m, 0, prog, "demod", 0, "phase", units._phase_code(-a0))   # rotate |0> → +real
    z0p = probe.iq(spec, {0: {"prep": 0}})[0][0]                    # rerun, no recompile
    z1p = probe.iq(spec, {0: {"prep": 1}})[0][0]
    rot = math.remainder(math.atan2(z0p.imag, z0p.real) - a0, 2 * math.pi)          # measured rotation
    print(f"\n[demod-phase] a0={math.degrees(a0):.1f}° applied={math.degrees(-a0):.1f}° "
          f"rot={math.degrees(rot):.1f}° |z0|={abs(z0):.0f}->{abs(z0p):.0f} |1>re={z1p.real:.0f}")
    assert abs(math.remainder(rot - (-a0), 2 * math.pi)) < math.radians(3), \
        "demod slot phase did not rotate the IQ by the applied angle"
    assert abs(abs(z0p) - abs(z0)) < 0.03 * abs(z0), "demod-phase rotation changed |z| (not a pure rotation)"
    assert z0p.real > 0 and z1p.real < 0, "calibrated demod phase does not separate |0>/|1> by sign(real)"
    assert build.CC_RUNS == runs, "applying the demod phase via write_slot must not recompile"


def test_fidelity_dur_retune(sub):
    """B3 (spec 08 §4): the demod integration WINDOW is a table-slot field, so `write_slot(demod, 0,
    "dur", d)` + rerun retunes it with two writes and NO recompile (the Fidelity window sweep). A
    longer window coherently integrates more of the steady readout tone, so |z| grows with `dur`;
    both lengths run from the SAME compiled image (compile + setup once, then write_slot + rerun/dur)."""
    drv, m, (gate, gate180, ro, demod) = sub
    drv.sim.set_model(dict(kind="twolevel", core=0, rabi_rad_per_amp=0.0,           # a fixed |0> tone
                           readout_code=RO_CODE, readout_amp=20000.0, readout_phase=0.0, f_ge=F_GE))
    npts, shots = 1, 1                                              # one |0> row (no drive)
    prog = compile_kernel(kernels.k_t1, m, tables=dict(gate=gate180, ro=ro, demod=demod),
                          out=Array(2 * npts * shots), npts=npts, shots=shots, period=PERIOD, ddly=0,
                          code=pack16(RO_CODE), mode=RAW, d0=SEP, dd=0, prep=0, prep_gate=X)
    runs = build.CC_RUNS
    rq.setup(drv, m, {0: prog})

    def mag(d):
        assert d <= (1 << READOUT_MAX_WIN_LOG2), f"demod window {d} over the decoder cap"   # host-side (§4)
        rq.write_slot(drv, m, 0, prog, "demod", 0, "dur", d)
        out = rq.rerun(drv, m, {0: prog}, timeout=_timeout(npts * shots * PERIOD))[0]["out"]
        z = out.reshape(shots, 2).astype(float).mean(0)                 # the |0> row
        return math.hypot(*z)

    d1, d2 = 16, 40                                                 # both ≤ RO_DUR (loaded demod env) & cap
    m1, m2 = mag(d1), mag(d2)
    print(f"\n[dur] |z|(d={d1})={m1:.0f} |z|(d={d2})={m2:.0f} ratio={m2 / m1:.2f} (window ratio {d2 / d1:.2f})")
    assert m1 > 0 and m2 > 1.4 * m1, \
        f"longer window did not integrate more: |z|({d1})={m1:.0f} |z|({d2})={m2:.0f}"
    assert build.CC_RUNS == runs, "retuning the demod window via write_slot must not recompile"


# ── L2 state probes: what each kernel's CIRCUIT does to the qubit (01 §4) ──
#
# One `rq.setup` per test, one shot per point, `set_model` re-preparing |0> for free between points,
# and an ANALYTIC target every time — never another run of the same model (01 §4.4). The rate that
# makes a pulse an exact rotation is PLANTED with `probe.rabi_for`, so every gate is correct by
# construction. These replace the four counts sweeps, which measured the same physics through
# thousands of projective shots and a fit.


def test_counts_rabi(cosim):
    """L2 — k_rabi's circuit: the on-core Q16 amplitude sweep is LINEAR in the rotation angle.

    The gate's drive integral σ is linear in the amplitude code, so a swept code k·A rotates by
    k·θ(A). Plant the rate that makes code A an exact X90 and the three rungs are the textbook
    quarter, half and three-quarter turns from |0>:

        k=1 → (0, −1, 0)   the equator, on −y      k=2 → (0, 0, −1)   |1>
        k=3 → (0, +1, 0)   the equator, on +y

    which is the Rabi cosine the counts version fitted — read directly, at three points instead of
    21 × 160 projective shots, and to 0.02 instead of the fit's 1 %. That the realized codes really
    are sweep_q16's is bit-exact on the DAC in test_computed_amp_matches_xs; this is what they DO.

    The base amplitude is 0.3 rather than 0.5 so the third rung (3 × 5969 = 17907) still fits under
    `units.AMP_SCALE` = 19896 — a hand-written "half scale" code would silently mis-scale every
    angle here (01 §4.6)."""
    _, m = cosim
    gate, _, ro, demod = _tables(m, gate_amp=0.3)
    x90 = gate.pulses["x90"]
    a0 = x90.amp_code()                                  # 0.3 * AMP_SCALE — never hand-written
    prog = compile_kernel(kernels.k_rabi, m, tables=dict(gate=gate, ro=ro, demod=demod),
                          out=Array(1), npts=1, shots=1, period=PERIOD, ddly=0, ngates=1, step=4,
                          code=pack16(RO_CODE), mode=COUNTS, prep_gate=X90, vz0=0, vzsum=0,
                          daq=0, prep=1)
    spec = {**L2_MODEL, "rabi_rad_per_amp": rabi_for(m, x90, F_GE, math.pi / 2)}   # code a0 = an X90
    p = Probe(cosim, {0: prog})

    for k, want in ((1, [0.0, -1.0, 0.0]), (2, [0.0, 0.0, -1.0]), (3, [0.0, 1.0, 0.0])):
        assert k * a0 <= units.AMP_SCALE, "the swept code must stay on scale"
        b = p.state(spec, {0: {"a0q": (k * a0) << 16}})["bloch"]
        print(f"\n[rabi k={k}] code={k * a0} bloch={np.round(b, 4).tolist()} want={want}")
        assert b == pytest.approx(want, abs=0.02), \
            f"the swept code {k * a0} turned the qubit to {b}, not the {k}·π/2 rotation {want}"


def test_counts_frequency(cosim):
    """L2 — k_ramsey's circuit: the detuning `Frequency` applies as a per-wait virtual-Z really
    separates the two X90s' AXES by the angle the phase word says.

    `Frequency` sweeps the wait w and seats the phase pair so point i carries `16·d_code·w_i` of
    phase code — one turn is 2^16, so the second X90's axis sits at

        φ = 2π · 16 · d_code · w / 2^16 = 2π · d_code · w / 4096   rad

    from the first's. Both axes lie in the xy-plane (the carrier is resonant and each pulse is a
    fixed-axis rotation), so Rodrigues through the pair from |0> gives, exactly,

        b = (−sin φ · cos φ,  −sin² φ,  −cos φ)

    — the fringe the counts version recovered by fitting a cosine through 15 × 48 shots and reading
    `4096 · f_fit` back as a code. Asserting the whole vector also pins the SIGN of the frame
    convention, which ⟨σz⟩ = −cos φ alone cannot see (bx is odd in φ, bz is even).

    Three waits at d_code = 128 put φ at π/4, 3π/4 and π: two off-axis points that pin the sign and
    one full turn back to |0>."""
    _, m = cosim
    gate, _, ro, demod = _tables(m)
    x90 = gate.pulses["x90"]
    d_code, waits = 128, (4, 12, 16)
    prog = compile_kernel(kernels.k_ramsey, m, tables=dict(gate=gate, ro=ro, demod=demod),
                          out=Array(1), npts=1, shots=1, period=PERIOD, ddly=0,
                          code=pack16(RO_CODE), mode=COUNTS, vz0=0, vzsum=0, dw=0, dp=0)
    spec = {**L2_MODEL, "rabi_rad_per_amp": rabi_for(m, x90, F_GE, math.pi / 2)}   # exact X90s
    p = Probe(cosim, {0: prog})

    for w in waits:
        b = p.state(spec, {0: {"w0": w, "p0": pack16(16 * d_code * w)}})["bloch"]
        phi = 2 * math.pi * d_code * w / (1 << 12)
        want = [-math.sin(phi) * math.cos(phi), -math.sin(phi) ** 2, -math.cos(phi)]
        print(f"\n[ramsey w={w}] φ={phi:+.4f} bloch={np.round(b, 4).tolist()} "
              f"want={np.round(want, 4).tolist()}")
        assert b == pytest.approx(want, abs=0.02), \
            f"wait {w} (φ={phi:+.4f}): the X90 pair landed at {b}, not {want}"


def test_counts_t1(cosim):
    """L2 — k_t1's circuit: the computed delay pair (d0, dd) really places the |1> prep `dly`
    batches before the readout, measured as the decay it lets happen.

    The prep drives to |1> and the qubit then idles for exactly `dly` batches before the window, so
    the model's amplitude damping leaves ⟨σz⟩(dly) = 1 − B·e^(−dly/T1), and the soft readout is
    linear in ⟨σz⟩ — the integrated phasor is z(dly) = A·⟨σz⟩(dly) for a complex A fixed by the
    readout chain. Neither A nor B is known here (A carries the demod pipeline's gain and angle, B
    the prep's own imperfection and the window's averaging), so the statement is made on SUCCESSIVE
    DIFFERENCES at EQUALLY SPACED delays, which cancel both exactly:

        z(d+2Δ) − z(d+Δ)        e^(−(d+2Δ)/T1) − e^(−(d+Δ)/T1)
        ────────────────  =  ────────────────────────────────  =  e^(−Δ/T1)
        z(d+Δ) − z(d)           e^(−(d+Δ)/T1) − e^(−d/T1)

    — the textbook T1 law with every prep- and readout-side constant divided out, and a direct
    measurement of the delay knob: a `d0` short by a factor would move the ratio. It is a COMPLEX
    ratio that has to come back REAL, which additionally pins that the three points share one demod
    frame (a delay that moved the readout in time, rather than the pulse, would rotate them).

    Read through the real demod/decoder rather than off `model_state()`, because a planted T1 keeps
    decaying during the host's poll after the run — the readout latches ⟨σz⟩ at the window, the
    state does not stand still.

    Replaces a 9 × 120-shot projective sweep + `fit_exp_decay` at ±20 %; the decay physics itself is
    L2 in test_cal_model::test_t1_exponential_decay and the fit is host-pure."""
    _, m = cosim
    gate, gate180, ro, demod = _tables(m)
    t1, step, delays = 600, 300, None
    delays = tuple(SEP + i * step for i in range(3))
    prog = compile_kernel(kernels.k_t1, m, tables=dict(gate=gate180, ro=ro, demod=demod),
                          out=Array(2), npts=1, shots=1, period=PERIOD + max(delays), ddly=0,
                          code=pack16(RO_CODE), mode=RAW, dd=0, prep=1, prep_gate=X)
    spec = {**L2_MODEL, "rabi_rad_per_amp": _rabi_pi(m), "t1": t1}
    p = Probe(cosim, {0: prog})

    z = np.array([p.iq(spec, {0: {"d0": d}})[0][0] for d in delays])
    ratio = (z[2] - z[1]) / (z[1] - z[0])
    want = math.exp(-step / t1)
    print(f"\n[t1] delays={list(delays)} z={np.round(z / 1e6, 4).tolist()}e6\n"
          f"  successive-difference ratio={ratio:.4f}  want e^(−{step}/{t1})={want:.4f}")
    assert abs(z[1] - z[0]) > 0.1 * abs(z[0]), \
        "the delay knob moved nothing — the |1> prep or the decay is missing"
    assert ratio.real == pytest.approx(want, abs=0.03), \
        f"the successive differences shrink by {ratio.real:.4f}, the T1 law wants {want:.4f}"
    assert abs(ratio.imag) < 0.03, \
        f"the three readouts are not collinear ({ratio.imag:+.4f}i) — the demod frame moved with the delay"


def test_counts_t2(cosim):
    """L2 — k_ramsey's other knob: the computed wait (w0, dw) is a REAL idle, and the coherence
    decays across exactly its length.

    With the swept virtual-Z at zero the two X90s share an axis, so the pair is a π and ⟨σz⟩ would
    be −1 — except that the equatorial vector shrinks by e^(−w/T2) during the wait. The pulses are
    identical at every point, so their own (interleaved) decay is a constant C and the analytic
    statement is again the ratio against the shortest wait:

        ⟨σz⟩(w) / ⟨σz⟩(w₀) = e^(−(w − w₀)/T2)

    T1 is left unplanted so ⟨σz⟩ is FROZEN once the second X90 has run (the model damps bz only
    through t1), which is what lets this read the state directly instead of the readout.

    Replaces a 15 × 48-shot projective sweep + `fit_damped_cosine` at ±20/25 %."""
    _, m = cosim
    gate, _, ro, demod = _tables(m)
    x90 = gate.pulses["x90"]
    t2, waits = 200, (8, 80, 152)
    prog = compile_kernel(kernels.k_ramsey, m, tables=dict(gate=gate, ro=ro, demod=demod),
                          out=Array(1), npts=1, shots=1, period=PERIOD + max(waits), ddly=0,
                          code=pack16(RO_CODE), mode=COUNTS, vz0=0, vzsum=0, dw=0, dp=0, p0=0)
    spec = {**L2_MODEL, "rabi_rad_per_amp": rabi_for(m, x90, F_GE, math.pi / 2), "t2": t2}
    p = Probe(cosim, {0: prog})

    bs = [p.state(spec, {0: {"w0": w}})["bloch"] for w in waits]
    print(f"\n[t2] waits={list(waits)} bloch={[np.round(b, 4).tolist() for b in bs]}")
    assert bs[0][2] < -0.85, f"the aligned X90 pair is not a π: bz({waits[0]}) = {bs[0][2]:.3f}"
    for w, b in zip(waits, bs):
        assert math.hypot(b[0], b[1]) < 0.02, \
            f"wait {w}: an aligned X90 pair must leave nothing in the equator, got {b}"
    for w, b in zip(waits[1:], bs[1:]):
        got, want = b[2] / bs[0][2], math.exp(-(w - waits[0]) / t2)
        print(f"  w={w}: bz ratio={got:.4f} want e^(−{w - waits[0]}/{t2})={want:.4f}")
        assert got == pytest.approx(want, abs=0.03), \
            f"wait {w} kept {got:.4f} of the coherence, the T2 law wants {want:.4f}"
