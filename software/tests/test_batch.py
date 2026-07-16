"""B2 acceptance (spec 08 §9): the batched kernels — one run returns a whole sweep. Each scenario
drives a batched kernel directly (the cal-class rewiring is B5) and checks it against the SAME
physics `tests/test_cal.py` recovers with the host-loop-per-point kernels:

  - counts mode (k_rabi / k_ramsey / k_t1) against the PROJECTIVE TwoLevelModel recovers the planted
    Rabi rate (<1%), detuning, and T1/T2 within test_cal's tolerances — self-normalised populations
    P = out/shots, no |0> projection;
  - raw mode (k_t1 with a prep column) reproduces acquire_shots cluster statistics (bimodal, well
    separated);
  - iqsum mode (k_vna, matched readout+demod retune) reproduces the Separation VNA argmax;
  - a grid-invariance check: repeated identical point-table rows agree across the batch.

counts mode needs a calibrated demod phase (sign(real) discriminator) — `_phase_cal` pins it once,
exactly as tests/test_projective.py does. The soft model is untouched (every other test still uses it).
These are long single-boot batches (vs test_cal's many boots), so the whole file is one --cosim run.
"""

import math

import numpy as np
import pytest

from riscq import build, run as rq
from riscq.cal.base import SEP, X, X90, gate_sigma, sweep_q16
from riscq.cal import fits, kernels
from riscq.cal.readout import Classifier
from riscq.lang import Array, ParamTable, compile_kernel, kernel
from riscq.map import LEAD, READOUT_LEAD, READOUT_MAX_WIN_LOG2, pack16
from riscq.pulses import Pulse, envelopes, golden, units

pytestmark = pytest.mark.cosim

F_GE = 50e6                       # planted qubit frequency (freq_code 2048)
RO_CODE = 2048                    # demod carrier code = the model's readout tone code
RO_DUR = 40                       # demod window (batches)
GATE_ENV = envelopes.square(16)   # 4-batch square gate
COUNTS, RAW = kernels.COUNTS, kernels.RAW


def _tables(m):
    ro_freq = units.demod_code_to_freq(RO_CODE, m.params)   # physical readout freq (model ignores ch1 freq)
    gate = ParamTable(0, F_GE, {"x90": Pulse(GATE_ENV, freq_hz=F_GE, amp=0.5)})
    gate180 = ParamTable(0, F_GE, {"x": Pulse(GATE_ENV, freq_hz=F_GE, amp=0.99)})   # the "X" prep
    ro = ParamTable(1, ro_freq, {"meas": Pulse(envelopes.square(RO_DUR + 16), freq_hz=ro_freq, amp=0.5)})
    demod = ParamTable(2, 0.0, {"sq": Pulse(envelopes.square(RO_DUR), amp=1.0)})
    return gate, gate180, ro, demod


@kernel
def _k_read(demod: ParamTable, out: Array, code: int):
    """A demod-only |0> read, for the demod-phase calibration (the model synthesizes the tone)."""
    init_pulse_params(demod.pulses)  # noqa: F821
    set_freq(demod, code)  # noqa: F821
    t = now() + LEAD  # noqa: F821
    play(demod, demod["sq"], t)  # noqa: F821
    wait_until(t + READOUT_LEAD)  # noqa: F821
    read_res()  # noqa: F821
    out[0] = read_real()  # noqa: F821
    out[1] = read_imag()  # noqa: F821


def _phase_cal(drv, m, demod):
    """The demod-LO phase that lands a soft |0> read on +real (so read_res = sign(real) means 0 for
    |0>, 1 for |1>). Two probe reads pin the fixed ADC→integrator phase + slope (tests/test_projective)."""
    base = dict(kind="twolevel", rabi_rad_per_amp=0.0, readout_code=RO_CODE, readout_amp=20000.0)

    def rd(ph):
        drv.sim.set_model({**base, "readout_phase": ph})
        prog = compile_kernel(_k_read, m, tables=dict(demod=demod), out=Array(2), code=pack16(RO_CODE))
        out = rq.run(drv, m, {0: prog}, timeout=1_000_000)[0]["out"]
        return float(out[0]), float(out[1])

    r0, i0 = rd(0.0)
    r1, i1 = rd(0.4)
    a0, a1 = math.atan2(i0, r0), math.atan2(i1, r1)
    s = 1.0 if math.remainder(a1 - a0, 2 * math.pi) >= 0 else -1.0
    return -s * a0


def _sig(m, amp_code):
    return gate_sigma(m, Pulse(GATE_ENV, freq_hz=F_GE, amp=0.5), F_GE, int(amp_code))


def _rabi_pi(m):
    return float(math.pi / gate_sigma(m, Pulse(GATE_ENV, freq_hz=F_GE, amp=0.99), F_GE,
                                      units._amp_code(0.99)))


def _rabi_pi2(m):
    return float((math.pi / 2) / gate_sigma(m, Pulse(GATE_ENV, freq_hz=F_GE, amp=0.5), F_GE,
                                            units._amp_code(0.5)))


def _timeout(nbatches):
    return int(nbatches) * 4 + 20_000_000


@pytest.fixture(scope="session")
def sub(cosim):
    """(drv, m, (gate, gate180, ro, demod), phase) — the shared substrate, phase calibrated once."""
    drv, m = cosim
    tabs = _tables(m)
    phase = _phase_cal(drv, m, tabs[3])
    return drv, m, tabs, phase


@pytest.fixture(autouse=True)
def _zero_after(cosim):
    yield
    cosim[0].sim.set_model({"kind": "zero"})


def _twolevel(drv, phase, t1, t2, rabi, seed, noise=300.0):
    drv.sim.set_model(dict(kind="twolevel", core=0, collapse=True, rabi_rad_per_amp=rabi,
                           readout_code=RO_CODE, readout_amp=20000.0, readout_phase=phase,
                           f_ge=F_GE, t1=t1, t2=t2, noise_scale=noise, noise_seed=seed))


def _mags(out, npts):
    o = np.asarray(out, dtype=np.int64)
    return np.array([math.hypot(int(o[2 * i]), int(o[2 * i + 1])) for i in range(npts)])


# ── iqsum: the matched-pair VNA batches on-core; argmax matches the Separation result ──

def test_vna_argmax(sub):
    drv, m, (gate, gate180, ro, demod), phase = sub
    F, points, shots, period = 256, 7, 8, 400
    c0q, dcq, xs = sweep_q16(F, points * F, points)      # xs = F..7F — matched (4x) at real index 3 (1024)
    npts = points
    sh = max(0, (shots - 1).bit_length())
    drv.sim.set_model({"kind": "tone", "adc": m.adc_of(0),
                       "freq_hz": units.code_to_freq(1024, m.params), "amp": 20000.0})
    prog = compile_kernel(kernels.k_vna, m, tables=dict(ro=ro, demod=demod),
                          out=Array(2 * npts), npts=npts, shots=shots, period=period, sh=sh, ddly=0,
                          mode=kernels.IQSUM, c0q=int(c0q), dcq=int(dcq))
    out = rq.run(drv, m, {0: prog},
                 timeout=_timeout(npts * shots * period))[0]["out"]
    mag = _mags(out, npts)
    print(f"\n[vna] codes/F={[int(x) // F for x in xs]} mag={[int(x) for x in mag]} argmax={int(np.argmax(mag))}")
    assert int(np.argmax(mag)) == 3, "matched-pair VNA peak not at 4x the DAC code"
    second = np.sort(mag)[-2]
    assert mag[3] > 3 * second, f"VNA peak not dominant: peak={mag[3]:.0f} second={second:.0f}"


# ── grid-invariance: repeated identical rows replay identically across the batch ──

def test_grid_invariance(sub):
    drv, m, (gate, gate180, ro, demod), phase = sub
    npts, shots, period = 8, 8, 400
    c0q, dcq = 1024 << 16, 0                              # dcq=0 → every row realizes code 1024
    sh = max(0, (shots - 1).bit_length())
    drv.sim.set_model({"kind": "tone", "adc": m.adc_of(0),
                       "freq_hz": units.code_to_freq(1024, m.params), "amp": 20000.0})
    prog = compile_kernel(kernels.k_vna, m, tables=dict(ro=ro, demod=demod),
                          out=Array(2 * npts), npts=npts, shots=shots, period=period, sh=sh, ddly=0,
                          mode=kernels.IQSUM, c0q=c0q, dcq=dcq)
    out = rq.run(drv, m, {0: prog},
                 timeout=_timeout(npts * shots * period))[0]["out"]
    mag = _mags(out, npts)
    print(f"\n[grid] mag={[int(x) for x in mag]} std/mean={mag.std() / mag.mean():.5f}")
    assert mag.std() / mag.mean() < 0.01, "identical point-table rows disagree across the batch"


# ── raw: k_t1 in RAW mode reproduces acquire_shots' |0>/|1> clusters (bimodal, separated) ──

def test_raw_clusters(sub):
    drv, m, (gate, gate180, ro, demod), phase = sub
    _twolevel(drv, phase, t1=2000, t2=3000, rabi=_rabi_pi(m), seed=2)   # t1 ≫ SEP: |1> survives to readout
    npts, shots, period = 1, 15, 8192
    prog = compile_kernel(kernels.k_t1, m, tables=dict(gate=gate180, ro=ro, demod=demod),
                          out=Array(2 * npts * shots), npts=npts, shots=shots, period=period, ddly=0,
                          code=pack16(RO_CODE), mode=RAW, d0=SEP, dd=0, prep_gate=X)          # prep runtime (per rerun)
    rq.setup(drv, m, {0: prog})

    def cluster(prep):                                                  # prep=0 → |0>, prep=1 → |1>
        out = rq.rerun(drv, m, {0: prog}, params={0: {"prep": prep}},
                       timeout=_timeout(npts * shots * period))[0]["out"]
        return out.reshape(shots, 2).astype(float)

    iq0, iq1 = cluster(0), cluster(1)
    clf = Classifier(iq0, iq1)
    print(f"\n[raw] real0={iq0[:,0].mean():.3e} real1={iq1[:,0].mean():.3e} "
          f"spread0={iq0[:,0].std():.2e} sep={clf.separation:.2f}")
    assert iq0[:, 0].mean() > 0 and iq1[:, 0].mean() < 0, "clusters not antipodal (|0>=+real, |1>=−real)"
    # Classifier.separation is qcal's cluster SNR now (‖Δmeans‖ / (2σ₀ + 2σ₁), spec 13 §2) — a quarter
    # of the old distance/σ number, and the |1> cluster here also carries the un-reset shots of a
    # relax head only ~4·T1 long, which that denominator charges for. > 2 is still means ≥ 8σ apart.
    assert clf.separation > 2, f"raw clusters not well separated: sep={clf.separation:.2f}"


# ── W0 (spec 11): row 0 needs no warm-up — the run's FIRST window matches the ensemble ──

def test_first_row_clean(sub):
    """Spec 11 W0: identical-knob rows through the undiluted per-shot RAW path — every shot of
    every row, INCLUDING shot (0,0), the first window the program ever integrates, matches the
    ensemble. Noise-free model, prep=0 (no drive, deterministic collapse to |0>), dd=0: the
    matched-pair integral is window-start-invariant on the fixed grid, so a cold-first-read
    artifact would make shot 0 an outlier. Guards the warm-up-row removal (spec 11)."""
    drv, m, (gate, gate180, ro, demod), phase = sub
    _twolevel(drv, phase, t1=2000, t2=3000, rabi=_rabi_pi(m), seed=3, noise=0.0)
    npts, shots, period = 4, 4, 8192
    prog = compile_kernel(kernels.k_t1, m, tables=dict(gate=gate180, ro=ro, demod=demod),
                          out=Array(2 * npts * shots), npts=npts, shots=shots, period=period, ddly=0,
                          code=pack16(RO_CODE), mode=RAW, d0=SEP, dd=0, prep_gate=X)
    out = rq.run(drv, m, {0: prog}, params={0: {"prep": 0}},
                 timeout=_timeout(npts * shots * period))[0]["out"]
    iq = out.reshape(npts * shots, 2).astype(float)
    z = iq[:, 0] + 1j * iq[:, 1]
    zm = z.mean()
    dev = np.abs(z - zm) / abs(zm)
    print(f"\n[first-row] |z|={abs(zm):.3e} max dev={dev.max():.2e} shot0 dev={dev[0]:.2e}")
    assert dev.max() < 0.005, f"a shot deviates {dev.max():.3%} from the ensemble (first-window artifact?)"


# ── counts: recover the planted physics within test_cal's tolerances ──

def test_counts_rabi(sub):
    drv, m, (gate, gate180, ro, demod), phase = sub
    nper, points, shots, period = 4, 21, 160, 1280
    rabi = nper * math.pi / _sig(m, units.AMP_SCALE - 600)
    _twolevel(drv, phase, t1=200, t2=2000, rabi=rabi, seed=1)
    a0q, daq, xs = sweep_q16(600, units.AMP_SCALE - 600, points)       # realized amp codes on-core
    npts = points
    prog = compile_kernel(kernels.k_rabi, m, tables=dict(gate=gate, ro=ro, demod=demod),
                          out=Array(npts), npts=npts, shots=shots, period=period, ddly=0,
                          ngates=1, code=pack16(RO_CODE), mode=COUNTS, prep_gate=X90, vz0=0, vzsum=0,
                          a0q=int(a0q), daq=int(daq), prep=1)
    out = rq.run(drv, m, {0: prog},
                 timeout=_timeout(npts * shots * period))[0]["out"]
    P = out.astype(float) / shots
    fit = fits.fit_cosine(np.array([_sig(m, a) for a in xs]), P)
    rec = 2 * math.pi * fit.value
    print(f"\n[rabi] rec={rec:.6e} planted={rabi:.6e} ratio={rec / rabi:.4f}")
    assert fit.ok
    assert abs(rec / rabi - 1) < 0.01, f"counts Rabi rate off: {rec / rabi:.4f}"


def test_counts_frequency(sub):
    drv, m, (gate, gate180, ro, demod), phase = sub
    points, shots, d_code, t0, dt, period = 15, 48, 128, 8, 8, 1280
    _twolevel(drv, phase, t1=200, t2=3000, rabi=_rabi_pi2(m), seed=3)
    waits = [t0 + i * dt for i in range(points)]                       # fit x-axis (host mirror of the on-core sweep)
    npts = points
    prog = compile_kernel(kernels.k_ramsey, m, tables=dict(gate=gate, ro=ro, demod=demod),
                          out=Array(npts), npts=npts, shots=shots, period=period, ddly=0,
                          code=pack16(RO_CODE), mode=COUNTS, w0=t0, dw=dt, vz0=0, vzsum=0,
                          p0=pack16(16 * d_code * t0), dp=pack16(16 * d_code * dt))     # detuning applied via virtual-Z
    out = rq.run(drv, m, {0: prog},
                 timeout=_timeout(npts * shots * period))[0]["out"]
    P = out.astype(float) / shots
    fit = fits.fit_cosine(np.array(waits, float), P)
    rec_code = 4096 * fit.value                                        # cycles/batch → detuning code
    print(f"\n[freq] rec_code={rec_code:.1f} planted={d_code}")
    assert fit.ok
    assert abs(rec_code - d_code) < 20, f"counts Ramsey detuning off: {rec_code:.1f} vs {d_code}"


def test_counts_t1(sub):
    drv, m, (gate, gate180, ro, demod), phase = sub
    t1, points, shots, dt, period = 300, 9, 120, 100, 1600
    _twolevel(drv, phase, t1=t1, t2=3000, rabi=_rabi_pi(m), seed=4)
    delays = [SEP + i * dt for i in range(points)]                     # fit x-axis (host mirror)
    npts = points
    prog = compile_kernel(kernels.k_t1, m, tables=dict(gate=gate180, ro=ro, demod=demod),
                          out=Array(npts), npts=npts, shots=shots, period=period, ddly=0,
                          code=pack16(RO_CODE), mode=COUNTS, d0=SEP, dd=dt, prep=1, prep_gate=X)
    out = rq.run(drv, m, {0: prog},
                 timeout=_timeout(npts * shots * period))[0]["out"]
    P = out.astype(float) / shots                                 # |1> population, decaying
    fit = fits.fit_exp_decay(np.array(delays, float), P)
    print(f"\n[t1] rec={fit.value:.1f} planted={t1} ratio={fit.value / t1:.3f} P={np.round(P, 3).tolist()}")
    assert fit.ok
    assert 0.8 * t1 < fit.value < 1.2 * t1, f"counts T1 off: {fit.value:.1f} vs {t1}"


def test_counts_t2(sub):
    drv, m, (gate, gate180, ro, demod), phase = sub
    t2, points, shots, d_code, t0, dt, period = 200, 15, 48, 70, 8, 16, 1600
    _twolevel(drv, phase, t1=400, t2=t2, rabi=_rabi_pi2(m), seed=5)
    waits = [t0 + i * dt for i in range(points)]                       # fit x-axis (host mirror)
    npts = points
    prog = compile_kernel(kernels.k_ramsey, m, tables=dict(gate=gate, ro=ro, demod=demod),
                          out=Array(npts), npts=npts, shots=shots, period=period, ddly=0,
                          code=pack16(RO_CODE), mode=COUNTS, w0=t0, dw=dt, vz0=0, vzsum=0,
                          p0=pack16(16 * d_code * t0), dp=pack16(16 * d_code * dt))
    out = rq.run(drv, m, {0: prog},
                 timeout=_timeout(npts * shots * period))[0]["out"]
    P = out.astype(float) / shots
    fit = fits.fit_damped_cosine(np.array(waits, float), P)
    tau = fit.params["tau"]
    print(f"\n[t2] rec_tau={tau:.1f} planted={t2} freq_code={4096 * fit.value:.1f}")
    assert fit.ok
    assert 0.8 * t2 < tau < 1.25 * t2, f"counts T2 off: {tau:.1f} vs {t2}"


# ── computed sweep: the on-core Q16 amp sweep realizes EXACTLY sweep_q16's xs (spec 09 §4.2) ──

def test_computed_amp_matches_xs(sub):
    """spec 09 §4.2 acceptance: the on-core computed amp sweep reproduces sweep_q16's xs. Capture the
    gate DAC across a computed-amp k_rabi run and match each per-point pulse window bit-exactly vs the
    golden at the host-mirrored code — `set_amp(aq)` raw == sweep_q16 point-for-point (a bit-exact
    readback, not a fit)."""
    drv, m, (gate, gate180, ro, demod), phase = sub
    drv.sim.set_model({"kind": "zero"})                  # the gate DAC carries only the core's gate pulses
    points, shots, period = 5, 1, 400                    # period ≥ LEAD+SEP+dur so the early gate has lead
    a0q, daq, xs = sweep_q16(600, units.AMP_SCALE - 600, points)
    npts = points
    expected = [int(x) for x in xs]                      # realized code per point
    x90 = Pulse(GATE_ENV, freq_hz=F_GE, amp=0.5)
    lines = x90.packed_lines(m, 0)
    f_code, ph_code, dur = x90.freq_code(m), x90.phase_code(), len(lines)
    prog = compile_kernel(kernels.k_rabi, m, tables=dict(gate=gate, ro=ro, demod=demod),
                          out=Array(npts), npts=npts, shots=shots, period=period, ddly=0,
                          ngates=1, code=pack16(RO_CODE), mode=COUNTS, prep_gate=X90, vz0=0, vzsum=0,
                          a0q=int(a0q), daq=int(daq), prep=1)
    ncap = 8000                                          # armed before reset release: covers boot + grid
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


# ── B3 (run layer): the demod TABLE is the real discriminator/window knob — write_slot + rerun ──
#
# These retune a demod-table slot field through the actual hardware path (demod carrier → carrier-
# triggered decoder), NOT the soft-model shortcut the counts tests use (_phase_cal injects the phase
# as the model's readout_phase). Confirmed in co-sim: the demod carrier is a real PulseGenerator whose
# slot phase rotates the decoder's integrated z (a +90° slot phase rotates the measured IQ by +90°,
# magnitude preserved), so the discriminator's phase and its integration window are both pure table
# writes + rerun (no recompile). The soft model's readout_phase is pinned at 0 here so the ONLY phase
# knob in play is the demod slot — the point of B3.

def test_applied_demod_phase(sub):
    """B3 (spec 08 §2.1): the discriminator's one programmable knob is the DEMOD-TABLE carrier phase,
    applied through the REAL hardware path (demod carrier → decoder), not the model's readout_phase
    (pinned at 0). `write_slot(..., "phase", code)` + rerun rotates the measured IQ by exactly that
    phase; calibrated so |0> lands on +real / |1> on −real, sign(sumR) then discriminates. No recompile."""
    drv, m, (gate, gate180, ro, demod), phase = sub
    drv.sim.set_model(dict(kind="twolevel", core=0, rabi_rad_per_amp=_rabi_pi(m),   # soft |0>/|1> tones
                           readout_code=RO_CODE, readout_amp=20000.0, readout_phase=0.0,
                           f_ge=F_GE, t1=2000, t2=3000))
    npts, shots, period = 1, 1, 8192
    prog = compile_kernel(kernels.k_t1, m, tables=dict(gate=gate180, ro=ro, demod=demod),
                          out=Array(2 * npts * shots), npts=npts, shots=shots, period=period, ddly=0,
                          code=pack16(RO_CODE), mode=RAW, d0=SEP, dd=0, prep_gate=X)      # prep runtime (per-prep rerun)
    runs = build.CC_RUNS
    rq.setup(drv, m, {0: prog})

    def cluster(prep):
        out = rq.rerun(drv, m, {0: prog}, params={0: {"prep": prep}},
                       timeout=_timeout(npts * shots * period))[0]["out"]
        return complex(*out.reshape(shots, 2).astype(float).mean(0))

    def read():
        return cluster(0), cluster(1)                               # |0>, |1>

    z0, z1 = read()                                                 # demod phase 0 → fixed pipeline angle
    a0 = math.atan2(z0.imag, z0.real)
    rq.write_slot(drv, m, 0, prog, "demod", 0, "phase", units._phase_code(-a0))   # rotate |0> → +real
    z0p, z1p = read()                                               # rerun, no recompile
    rot = math.remainder(math.atan2(z0p.imag, z0p.real) - a0, 2 * math.pi)          # measured rotation
    print(f"\n[demod-phase] a0={math.degrees(a0):.1f}° applied={math.degrees(-a0):.1f}° "
          f"rot={math.degrees(rot):.1f}° |z0|={abs(z0):.0f}->{abs(z0p):.0f} |1>re {z1.real:.0f}->{z1p.real:.0f}")
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
    drv, m, (gate, gate180, ro, demod), phase = sub
    drv.sim.set_model(dict(kind="twolevel", core=0, rabi_rad_per_amp=0.0,           # a fixed |0> tone
                           readout_code=RO_CODE, readout_amp=20000.0, readout_phase=0.0,
                           f_ge=F_GE, t1=4000, t2=6000))
    npts, shots, period = 1, 1, 8192                                # one |0> row (no drive)
    prog = compile_kernel(kernels.k_t1, m, tables=dict(gate=gate180, ro=ro, demod=demod),
                          out=Array(2 * npts * shots), npts=npts, shots=shots, period=period, ddly=0,
                          code=pack16(RO_CODE), mode=RAW, d0=SEP, dd=0, prep=0, prep_gate=X)
    runs = build.CC_RUNS
    rq.setup(drv, m, {0: prog})

    def mag(d):
        assert d <= (1 << READOUT_MAX_WIN_LOG2), f"demod window {d} over the decoder cap"   # host-side (§4)
        rq.write_slot(drv, m, 0, prog, "demod", 0, "dur", d)
        out = rq.rerun(drv, m, {0: prog}, timeout=_timeout(npts * shots * period))[0]["out"]
        z = out.reshape(shots, 2).astype(float).mean(0)                 # the |0> row
        return math.hypot(*z)

    d1, d2 = 16, 40                                                 # both ≤ RO_DUR (loaded demod env) & cap
    m1, m2 = mag(d1), mag(d2)
    print(f"\n[dur] |z|(d={d1})={m1:.0f} |z|(d={d2})={m2:.0f} ratio={m2 / m1:.2f} (window ratio {d2 / d1:.2f})")
    assert m1 > 0 and m2 > 1.4 * m1, \
        f"longer window did not integrate more: |z|({d1})={m1:.0f} |z|({d2})={m2:.0f}"
    assert build.CC_RUNS == runs, "retuning the demod window via write_slot must not recompile"
