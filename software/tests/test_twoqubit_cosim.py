"""Two-qubit Q0 co-sim gate (specs/two-qubit/01 §7): the 3-core sim-2q1c build boots, its host map
decodes every core, and per-shot readout on two cores lines up by shot index — the cross-core shot
alignment the joint two-qubit populations rely on. The zip logic itself is unit-tested host-pure in
test_twoqubit; here we prove the real capture path on the 3-core build."""

import math

import numpy as np
import pytest

from riscq import run as rq
from riscq.cal import (JAZZ, ClassifierN, Config, CZAmpFreqSweep, EFAmplitude, EFFrequency, EFPhase,
                       ReadoutCalibration, cz_drive_table, cz_table, joint_populations, kernels)
from riscq.cal.base import (GATE_CH, GATE_ENV, SEP, acquire_shots, batch_timeout, ef_pulse, ef_table,
                            gate_pulse, gate_sigma, grid_period, qubit_freq, readout_tables,
                            relax_batches, x90_vz)
from riscq.cal.readout import _rawiq_prog
from riscq.cal.twoqubit import _cz_dur_batches, _cz_pulse, _sandwich_binds
from riscq.lang import Array, ParamTable, compile_kernel
from riscq.map import LEAD, pack16
from riscq.pulses import Pulse, units

pytestmark = pytest.mark.cosim

F_GE = 50e6                       # planted qubit frequency (DAC freq code 2048)
# EF cal: GE and EF carriers a full demod-null (4096 codes) apart so the ThreeLevelModel picks the
# right transition cleanly (GE code 6144 = 150 MHz, EF code 2048 = 50 MHz).
EF_F_GE = 150e6
EF_F_EF = 50e6


def _s(n_batches, m):             # batches → seconds (the Config is physical, spec 13 §2)
    return n_batches / m.params.dsp_freq_hz


def _rabi_pi(m):
    """The Rabi rate that makes the default X90·X90 |1> prep a full π (mirrors test_cal._rabi_pi)."""
    return float(math.pi / gate_sigma(m, Pulse(GATE_ENV, freq_hz=F_GE, amp=0.99), F_GE,
                                      units._amp_code(0.99)))


def test_3core_boots_and_host_map_decodes(cosim_2q1c):
    """The build boots with 3 cores; SocMap read the config's dac_map/adc_map (not the ZCU216 default);
    and every core's RAM window decodes over real AXI."""
    drv, m = cosim_2q1c
    assert m.params.qubit_num == 3
    assert [m.gate_dac(c) for c in range(3)] == [0, 1, 3]     # core 2 = coupler on DAC 3
    assert [m.ro_dac(c) for c in range(3)] == [2, 2, 2]       # readout drives summed on DAC 2
    assert [m.adc_of(c) for c in range(3)] == [0, 0, 0]
    rq.reset(drv, m, on=True)                                  # cores held; RAM host port is independent
    for c in range(3):
        val = (0xA5A50000 + c) & 0xFFFFFFFF
        drv.write_block(m.imem(c), val.to_bytes(4, "little"))
        assert int.from_bytes(drv.read_block(m.imem(c), 4), "little") == val


def test_cross_core_shot_alignment(cosim_2q1c):
    """Two qubit cores captured per-shot in ONE run, zipped by shot index. Core 0 is prepped |1>
    (rabi = π) while core 1 stays |0> (rabi = 0) under the SAME prep=1 — so distinct marginals prove
    the two per-shot streams are genuinely separate (each core's own demod/ADC lane, freq-multiplexed
    on the shared readout DAC/ADC), not one core read twice, and equal shot counts + the dominant
    joint P(10) prove shot k on both cores is the same repetition."""
    drv, m = cosim_2q1c
    code = {0: 2048, 1: 1024}                                  # distinct demod codes → freq-multiplexed
    cfg = Config()
    for q in (0, 1):
        cfg[f"qubit/{q}/freq"] = F_GE
        cfg[f"qubit/{q}/x90/amp"] = 0.5
        cfg[f"qubit/{q}/T1"] = _s(120, m)
        cfg[f"readout/{q}/freq"] = float(units.demod_code_to_freq(code[q], m.params))
        cfg[f"readout/{q}/amp"] = 0.5
        cfg[f"readout/{q}/dur"] = _s(56, m)                    # drive covers the demod window + SEP
        cfg[f"readout/{q}/demod/dur"] = _s(40, m)
    cfg["reset/relax"] = _s(3200, m)                          # relax ≫ T1 resets both cores each slot

    # the two tones SUM on the shared ADC (readout_amp halved so the sum stays in converter range),
    # each core's demod integrating out its own — the multi-qubit summed-readout model (spec 13 §8).
    sub = [{"kind": "twolevel", "core": q, "rabi_rad_per_amp": (_rabi_pi(m) if q == 0 else 0.0),
            "readout_code": code[q], "readout_amp": 14000.0, "f_ge": F_GE, "t1": 600, "t2": 3000}
           for q in (0, 1)]
    drv.sim.set_model({"kind": "multi", "models": sub})

    shots = 32
    progs, timeout = {}, 0
    for q in (0, 1):
        prog, period = _rawiq_prog(m, cfg, q, "X90", shots)
        progs[q] = prog
        timeout = max(timeout, batch_timeout(shots * period))
    rq.setup(drv, m, progs)
    iq = acquire_shots(drv, m, progs, 1, shots, timeout)      # ONE run, both cores prep=1

    assert iq[0].shape == (shots, 2) == iq[1].shape, "per-core shot streams must be the same length"
    bits = {q: (iq[q][:, 0] < 0).astype(int) for q in (0, 1)}  # sign(real): |0> on +real, |1> on −real
    assert bits[0].mean() > 0.9, f"core 0 prepped |1> but reads P1={bits[0].mean()} (iq={iq[0][:, 0]})"
    assert bits[1].mean() < 0.1, f"core 1 stayed |0> but reads P1={bits[1].mean()} (iq={iq[1][:, 0]})"
    p = joint_populations(bits, order=(0, 1))                 # zip by shot index
    assert p[2] > 0.9, f"joint P(10) should dominate the aligned zip, got {p}"


def test_three_level_clusters_separate(cosim_2q1c):
    """Q1 readout gate (spec 01 §5): |0>/|1>/|2> read out as three separated IQ clusters on the
    ThreeLevelModel, so the host ClassifierN tells all three apart. Each level is captured as its own
    RAW-shot run (the model sits in that level and emits the readout tone at the level's phase); the
    EF PREP that actually reaches |2> is exercised by the EF calibration."""
    drv, m = cosim_2q1c
    q = 0
    cfg = Config()
    cfg[f"qubit/{q}/freq"] = F_GE
    cfg[f"qubit/{q}/x90/amp"] = 0.5
    cfg[f"qubit/{q}/T1"] = _s(120, m)
    cfg[f"readout/{q}/freq"] = float(units.demod_code_to_freq(2048, m.params))
    cfg[f"readout/{q}/amp"] = 0.5
    cfg[f"readout/{q}/dur"] = _s(56, m)
    cfg[f"readout/{q}/demod/dur"] = _s(40, m)
    cfg["reset/relax"] = _s(3200, m)

    shots = 48
    prog, period = _rawiq_prog(m, cfg, q, "X90", shots)
    progs = {q: prog}
    timeout = batch_timeout(shots * period)
    clouds = []
    for level in range(3):
        drv.sim.set_model({"kind": "threelevel", "core": q, "readout_code": 2048,
                           "readout_amp": 18000.0, "init_level": level, "collapse": True,
                           "noise_scale": 400.0, "noise_seed": 7 + level})
        rq.setup(drv, m, progs)
        clouds.append(acquire_shots(drv, m, progs, 0, shots, timeout)[q])

    clf = ClassifierN(clouds)
    conf = clf.confusion()
    print(f"\n[3-level] separation={clf.separation:.2f} confusion diag={np.diag(conf)}")
    assert clf.separation > 1.0, f"3-level clusters not separated (min pairwise SNR={clf.separation:.2f})"
    assert np.all(np.diag(conf) > 0.9), f"3-level confusion diagonal weak: {np.diag(conf)}"


# ── EF calibration (spec two-qubit/01 §4.1 / Q1): recover a planted EF Rabi rate / EF detuning ──

def _carrier_code(win):
    """The (unsigned) carrier code of a clean tone from its DAC window (TwoLevelModel._carrier_code):
    amplitude- and phase-blind, exact for a square-envelope tone."""
    x = np.asarray(win, float).reshape(-1)
    num = float(np.sum(x[1:-1] * (x[2:] + x[:-2])))
    den = 2.0 * float(np.sum(x[1:-1] ** 2))
    w = math.acos(min(1.0, max(-1.0, num / den))) if den else 0.0
    return round(w / math.pi * (1 << 15))


def test_ef_drive_carriers_in_rtl(cosim_2q1c):
    """The mid-shot freq switch (spec 01 §4.1): capture the gate DAC across one k_ef_rabi shot and prove
    the RTL retunes the ONE gate NCO between segments — the GE pi prep comes out at f_GE (2 X90s) and the
    EF drive at f_EF, each at the programmed amplitude. This is the bit-exact readback under the novel
    per-shot double set_freq (the drive whose end-to-end recovery the EF cals verify)."""
    drv, m = cosim_2q1c
    q = 0
    cfg = _ef_cfg(m, q)
    cfg["reset/relax"] = _s(400, m)                          # short relax → short capture
    drv.sim.set_model({"kind": "zero"})
    table, ge_freq, ef_freq = ef_table(cfg, q, m)
    ro, demod, code, dur, ddly = readout_tables(cfg, q, m)
    ge = table.pulses["x90"].dur_batches(m, 0)
    ef = table.pulses["ef"].dur_batches(m, 0)
    period = grid_period(relax_batches(cfg, m), SEP + ef + LEAD + 2 * ge, dur, ddly)
    prog = compile_kernel(kernels.k_ef_rabi, m, tables=dict(gate=table, ro=ro, demod=demod),
                          out=Array(2), npts=1, shots=1, period=period, ngates=1, step=ef,
                          code=code, ddly=ddly,
                          ge_freq=ge_freq, ef_freq=ef_freq, **x90_vz(cfg, q))
    rq.setup(drv, m, {0: prog})
    rq.check_magic(drv, m, 0, prog)
    rq.write_var(drv, m, 0, prog, "__rq_status", 0)
    rq.write_params(drv, m, 0, prog, {"a0q": int(units._amp_code(0.5)) << 16, "daq": 0})
    handle = drv.sim.dac_capture_arm(m.gate_dac(0), 8000)    # armed before release: covers boot + shot
    rq.reset(drv, m, on=False)
    rq.poll_done(drv, m, 0, prog, timeout=period * 8 + 20_000_000)
    rq.reset(drv, m, on=True)
    t0, cap = drv.sim.dac_capture_get(handle)

    active = cap.any(axis=1)
    starts = [i for i in range(len(active)) if active[i] and (i == 0 or not active[i - 1])]
    ends = [i for i in range(len(active)) if active[i] and (i == len(active) - 1 or not active[i + 1])]
    wins = [(s, e) for s, e in zip(starts, ends)]
    ge_code = units._freq_code(EF_F_GE, m.params)
    ef_code = units._freq_code(EF_F_EF, m.params)
    print(f"\n[ef-carriers] windows={[(s, e - s + 1, _carrier_code(cap[s:e + 1])) for s, e in wins]}")
    assert len(wins) == 2, f"expected GE-prep + EF windows, got {len(wins)}"
    (gs, gee), (es, ee) = wins
    assert ee - es + 1 == ef, "EF drive should be one EF X90 window"
    assert gee - gs + 1 == 2 * ge, "GE prep should be two back-to-back X90s"
    assert abs(_carrier_code(cap[gs:gee + 1]) - ge_code) < 40, "GE prep not at f_GE"
    assert abs(_carrier_code(cap[es:ee + 1]) - ef_code) < 40, "EF drive not at f_EF"


def _ef_cfg(m, q, ge_amp=0.5, ef_amp=0.4):
    """The EF co-sim Config: GE at EF_F_GE, EF at EF_F_EF, a 3-level readout tone at demod code 2048."""
    cfg = Config()
    cfg[f"qubit/{q}/freq"] = EF_F_GE
    cfg[f"qubit/{q}/x90/amp"] = ge_amp
    cfg[f"qubit/{q}/T1"] = _s(120, m)
    cfg[f"qubit/{q}/EF/freq"] = EF_F_EF
    cfg[f"qubit/{q}/EF/x90/amp"] = ef_amp
    cfg[f"readout/{q}/freq"] = float(units.demod_code_to_freq(2048, m.params))
    cfg[f"readout/{q}/amp"] = 0.5
    cfg[f"readout/{q}/dur"] = _s(56, m)
    cfg[f"readout/{q}/demod/dur"] = _s(40, m)
    cfg["reset/relax"] = _s(3200, m)
    return cfg


def _train_3level(drv, m, cfg, q, shots=48):
    """Train the 3-level ClassifierN: capture |0>/|1>/|2> reference clouds by PLANTING each level on the
    model (init_level) and reading it out (no drive), exactly the readout half's gate — the EF cal reads
    P(|2>) against these centroids."""
    prog, period = _rawiq_prog(m, cfg, q, "X90", shots)
    progs, timeout = {q: prog}, batch_timeout(shots * period)
    clouds = []
    for level in range(3):
        drv.sim.set_model({"kind": "threelevel", "core": q, "readout_code": 2048,
                           "readout_amp": 18000.0, "init_level": level, "collapse": True,
                           "noise_scale": 400.0, "noise_seed": 7 + level})
        rq.setup(drv, m, progs)
        clouds.append(acquire_shots(drv, m, progs, 0, shots, timeout)[q])
    return ClassifierN(clouds)


def _ef_model(rabi_ge, rabi_ef, f_ef=EF_F_EF, seed=1):
    """The planted qutrit: a GE Rabi rate for the |1> prep, an EF Rabi rate for the swept EF drive, the
    true EF frequency `f_ef` (which the EF Frequency test detunes the config away from), and a T1 so the
    grid's idle head RESETS |1>/|2> to |0> between shots — without it the model (no auto-reset) carries a
    shot's collapsed level into the next prep and the sweep scrambles. T1 (600) ≪ relax (3200) resets
    each shot but ≫ the LEAD prep→drive gap, so the prep survives to the EF drive."""
    return {"kind": "threelevel", "core": 0, "f_ge": EF_F_GE, "f_ef": f_ef,
            "rabi_ge_rad_per_amp": float(rabi_ge), "rabi_ef_rad_per_amp": float(rabi_ef),
            "readout_code": 2048, "readout_amp": 18000.0, "init_level": 0, "collapse": True,
            "t1": 600, "noise_scale": 400.0, "noise_seed": int(seed)}


def _rabi_ge_pi(m, ge_amp):
    """The GE Rabi rate that makes k_ef_rabi's two-X90 prep a full π (|0> → |1>)."""
    sig = gate_sigma(m, Pulse(GATE_ENV, freq_hz=EF_F_GE, amp=ge_amp), EF_F_GE, units._amp_code(ge_amp))
    return math.pi / (2 * sig)


def test_ef_amplitude_recovers_the_ef_rabi(cosim_2q1c):
    """Q1 EF gate (spec 01 §4.1): the EF X90 amplitude cal recovers a planted EF Rabi rate and writes
    `qubit/0/EF/x90/amp`. GE-π prep reaches |1>, the swept EF drive rotates |1>->|2>, and the host reads
    P(|2>) off the 3-level clusters — the EF keys land in co-sim."""
    drv, m = cosim_2q1c
    q = 0
    cfg = _ef_cfg(m, q)
    clf = _train_3level(drv, m, cfg, q, shots=48)
    assert clf.separation > 1.0, f"3-level clusters not separated ({clf.separation:.2f})"

    efp = ef_pulse(cfg, q, m)
    # ONE Rabi period over the amp sweep: a single-hump cosine the fit locks onto cleanly (2 periods
    # over noisy 3-level P(2) let fit_cosine catch a harmonic).
    rabi_ef = 2 * math.pi / gate_sigma(m, efp, EF_F_EF, units._amp_code(0.97))
    drv.sim.set_model(_ef_model(_rabi_ge_pi(m, float(cfg[f"qubit/{q}/x90/amp"])), rabi_ef, seed=1))
    cal = EFAmplitude(cfg, q, clf, n_gates=1, points=15, shots=48)
    r = cal.run(drv)
    y = r.data[q]["y"]
    print(f"\n[ef-amp] P(2)={np.round(y, 2).tolist()}")
    assert r.ok, "EF amplitude fit failed"
    recovered = cal.recovered_rabi[q]
    print(f"[ef-amp] recovered rabi_ef={recovered:.4e} planted={rabi_ef:.4e} "
          f"ratio={recovered / rabi_ef:.4f}  EF/x90/amp={r.proposal[f'qubit/{q}/EF/x90/amp']:.4f}")
    # the EF drive genuinely rotates |1>->|2> (P(2) rises into the transition; the peak sits below 1
    # because the prep partly relaxes over the LEAD gap) and the recovered rate lands near the plant
    # (3-level readout is noisier than the GE counts, so a looser bound than the GE cal)
    assert y.min() < 0.2 and y.max() > 0.6, f"P(2) did not span the |1>->|2> transition: {y}"
    assert abs(recovered / rabi_ef - 1) < 0.15, f"recovered {recovered} vs planted {rabi_ef}"
    assert 0.0 < r.proposal[f"qubit/{q}/EF/x90/amp"] < 1.0


@pytest.mark.parametrize("d0_code", [60, -60])
def test_ef_frequency_recovers_the_ef_detuning(cosim_2q1c, d0_code):
    """Q1 EF gate (spec 01 §4.1): the EF Ramsey MEASURES a planted EF detuning of EITHER sign on the
    {|1>, |2>} subspace (P(|2>)). The robust gate is the sign — the fringe MINIMISES where the applied
    detuning cancels the config error (at applied = −δ), so the slowest fringe sits on the opposite side
    of 0 from the plant — the same sign lock-step as the GE Frequency. When the V-fit converges on the
    (noisier, 3-level) fringes it writes `qubit/0/EF/freq` toward the true f_ef; the V-fit MATH itself is
    the GE-Frequency code, verified there in counts."""
    drv, m = cosim_2q1c
    q = 0
    detuned = float(units.code_to_freq(units._freq_code(EF_F_EF, m.params) + d0_code, m.params))
    cfg = _ef_cfg(m, q)
    cfg[f"qubit/{q}/EF/freq"] = detuned                       # config EF carrier off the true f_ef by δ0
    clf = _train_3level(drv, m, cfg, q, shots=48)
    assert clf.separation > 1.0, f"3-level clusters not separated ({clf.separation:.2f})"

    efp = ef_pulse(cfg, q, m)
    rabi_ef = (math.pi / 2) / gate_sigma(m, efp, EF_F_EF, units._amp_code(float(cfg[f"qubit/{q}/EF/x90/amp"])))
    drv.sim.set_model(_ef_model(_rabi_ge_pi(m, float(cfg[f"qubit/{q}/x90/amp"])), rabi_ef,
                                f_ef=EF_F_EF, seed=3))
    # detunings placed so every fringe is fittable (≥1 cycle over the wait grid, below Nyquist): the
    # slowest, |δ + applied|, still sits ~100 codes off zero at the nearest point.
    cal = EFFrequency(cfg, q, clf, detune=units.code_to_freq(160, m.params), n_detune=4,
                      t0=_s(8, m), dt=_s(4, m), points=11, shots=36)
    r = cal.run(drv)
    applied, obs = r.data[q]["applied"], r.data[q]["obs"]
    rec = cal.recovered_detuning_code.get(q, float("nan"))
    print(f"\n[ef-freq δ={d0_code:+d}] applied={applied} |fringe|={np.round(obs)}\n"
          f"  V-fit ok={r.ok} b={r.fit[q].value:+.1f} → recovered δ={rec:+.1f} (planted {d0_code:+d})")
    # robust sign gate: the fringe minimum is on the side that cancels the config error (opposite δ)
    assert len(obs) >= 3, "too few fringes fit for a V"
    vertex_side = float(applied[int(np.argmin(obs))])
    assert np.sign(vertex_side) == -np.sign(d0_code), \
        f"EF Ramsey measured the wrong detuning side: min |fringe| at applied={vertex_side:+.0f}"
    # when the V-fit converges it must move EF/freq TOWARD the true f_ef (never away)
    if r.ok:
        assert r.fit[q].params["a"] > 0 and np.sign(rec) == np.sign(d0_code), "recovered wrong sign"
        r.apply()
        assert abs(cfg[f"qubit/{q}/EF/freq"] - EF_F_EF) < abs(detuned - EF_F_EF), \
            "config EF freq moved away from the true f_ef"


def test_ef_phase_writes_the_vz_pair(cosim_2q1c):
    """X4 co-sim gate (spec 04 §5): the EF keys land — one EFPhase run at minimal knobs on the
    ThreeLevelModel drives the NEW k_ef_phase program end-to-end (GE π prep → retune f_EF → the two
    Rz(±π/2)-decorated 3-EF-X90 sequences → RAW P(|2>) through the 3-level classifier) and writes
    `qubit/0/EF/x90/vz`. The model plants no EF Stark shift and the retune frame slip conjugates
    out of z-basis populations (the kernel docstring), so the crossing sits near 0 — the gate is
    the code path + the key, not the physics (the crossing math is host-verified in
    test_twoqubit.test_ef_phase_recovers_planted_vz)."""
    drv, m = cosim_2q1c
    q = 0
    cfg = _ef_cfg(m, q)
    clf = _train_3level(drv, m, cfg, q, shots=48)
    assert clf.separation > 1.0, f"3-level clusters not separated ({clf.separation:.2f})"

    efp = ef_pulse(cfg, q, m)
    rabi_ef = (math.pi / 2) / gate_sigma(m, efp, EF_F_EF,
                                         units._amp_code(float(cfg[f"qubit/{q}/EF/x90/amp"])))
    drv.sim.set_model(_ef_model(_rabi_ge_pi(m, float(cfg[f"qubit/{q}/x90/amp"])), rabi_ef, seed=9))
    cal = EFPhase(cfg, q, clf, points=7, span=0.4, shots=24)
    r = cal.run(drv)
    p_y, p_x = r.data[q]["p0"], r.data[q]["p1"]
    print(f"\n[ef-phase] P_y={np.round(p_y, 2).tolist()}\n           P_x={np.round(p_x, 2).tolist()}"
          f"\n           ok={r.ok} recovered={cal.recovered_vz[q]:+.3f}")
    assert r.ok, "EF phase line crossing failed"
    v = r.proposal[f"qubit/{q}/EF/x90/vz"]
    assert v[0] == v[1] == cal.recovered_vz[q]                   # ONE crossing, BOTH slots
    assert abs(v[0]) < 0.15, f"crossing should sit near 0 on the shift-free model, got {v[0]:+.3f}"


@pytest.mark.parametrize("planted", [0.0, 0.6])
def test_ef_phase_x_gate_recovers_the_planted_axis(cosim_2q1c, planted):
    """spec 14 F1 — `EFPhase(gate='X')`, qcal's second Phase mode in the EF subspace. After the GE π
    prep the circuit is EF-X90 · EF-X · EF-X90, a 2π rotation inside {|1>, |2>} that returns to |1>
    only when the EF X's own AXIS matches the EF X90s'. Plant that axis by giving the EF X90s a
    phase of `planted`: the EF X's calibrated axis must follow them, and the proposal lands on
    `qubit/0/EF/x/phase` (an axis phase — NOT the virtual-Z pair the X90 mode writes). The EF X is
    the X6Y3 shape (double LENGTH, same amplitude → π when the X90 is π/2, spec 13 §4).

    An exact two-level Bloch model of the circuit gives P(|2>) = sin²(phi − axis) — MINIMAL on
    alignment, period π — so the recovered axis is only defined mod π (the two solutions are the
    same gate: R_{φ+π}(π) = −R_φ(π)), which is what is checked. Like every level-mode cal here the
    discrimination is pinned by training the 3-level classifier on this same Config's readout
    (`_train_3level` — the EF counterpart of the counts cals' measured `demod_phase`); without it
    the cluster assignment is whatever the grid leaves the demod LO at and the recovered axis lands
    half a fringe out."""
    drv, m = cosim_2q1c
    q = 0
    cfg = _ef_cfg(m, q)
    cfg[f"qubit/{q}/EF/x90/phase"] = planted       # the reference axis both EF X90s sit on
    cfg[f"qubit/{q}/EF/x/env"] = "square"          # the EF X: double LENGTH, same amp → π
    cfg[f"qubit/{q}/EF/x/dur"] = _s(8, m)
    cfg[f"qubit/{q}/EF/x/amp"] = float(cfg[f"qubit/{q}/EF/x90/amp"])
    cfg[f"qubit/{q}/EF/x/phase"] = 0.0             # deliberately off the X90s when planted != 0
    clf = _train_3level(drv, m, cfg, q, shots=48)
    assert clf.separation > 1.0, f"3-level clusters not separated ({clf.separation:.2f})"

    efp = ef_pulse(cfg, q, m)
    rabi_ef = (math.pi / 2) / gate_sigma(m, efp, EF_F_EF,
                                         units._amp_code(float(cfg[f"qubit/{q}/EF/x90/amp"])))
    drv.sim.set_model(_ef_model(_rabi_ge_pi(m, float(cfg[f"qubit/{q}/x90/amp"])), rabi_ef, seed=11))
    cal = EFPhase(cfg, q, clf, gate="X", points=13, shots=32)
    r = cal.run(drv)
    got = r.proposal.get(f"qubit/{q}/EF/x/phase")
    print(f"\n[ef-phase-X] planted axis={planted:+.3f} recovered={got} "
          f"fallback={cal.fallback.get(q)}\n  P(2)={np.round(r.data[q]['y'], 3).tolist()}")
    assert r.ok and f"qubit/{q}/EF/x90/vz" not in r.proposal   # the X mode writes the axis, not the pair
    assert abs(math.remainder(got - planted, math.pi)) < 0.3, \
        f"recovered {got:+.4f} rad, the EF X90s sit at {planted:+.4f} (mod π)"


# ── JAZZ: residual-ZZ Ramsey (spec two-qubit/01 §4.3 / Q3) ──

def test_jazz_runs_end_to_end_no_spurious_zz(cosim_2q1c):
    """Q3 light gate (spec 01 §4.3): the `JAZZ` cal runs end-to-end on the real 3-core build — `k_jazz`
    compiled for BOTH roles (control / target) walking the shared grid, the four sequences rerun over
    one resident image, both fringes fit — and with NO planted ZZ (ζ=0) the two control states give the
    SAME fringe (the applied detuning), i.e. no spurious control-conditional split. This pins the
    kernel + cal + two-qubit-model integration and the absence of a readout/timing artifact.

    That the ZZ split TRACKS a planted ζ (sign and non-vanishing) is the model physics, verified
    deterministically host-side in test_models.test_twoqubit_jazz_zz_physics. Recovering the ζ MAGNITUDE
    to fit tolerance end-to-end in co-sim is relaxation-SNR-limited (the model's only reset is
    relaxation, which also decoheres the multi-pulse Ramsey) and is deferred (spec 01 §4.3 note)."""
    drv, m = cosim_2q1c
    ctrl, tgt = 0, 1
    code = {0: 2048, 1: 1024}
    detune = 1.5e6
    cfg = Config()
    for q in (ctrl, tgt):
        cfg[f"qubit/{q}/freq"] = F_GE
        cfg[f"qubit/{q}/x90/amp"] = 0.5
        cfg[f"readout/{q}/freq"] = float(units.demod_code_to_freq(code[q], m.params))
        cfg[f"readout/{q}/amp"] = 0.5
        cfg[f"readout/{q}/dur"] = _s(56, m)
        cfg[f"readout/{q}/demod/dur"] = _s(40, m)
    cfg["reset/relax"] = _s(800, m)
    cfg["two_qubit/(0, 1)/core"] = 2

    rabi = _rabi_pi(m)                                               # 2 X90 = π, one X90 ≈ π/2
    drv.sim.set_model({"kind": "twoqubit", "control": ctrl, "target": tgt, "coupler": 2,
                       "f_ge": [F_GE, F_GE], "f_ef": [F_GE / 2, F_GE / 2],
                       "rabi_ge": [rabi, rabi], "rabi_ef": [0.0, 0.0],
                       "zz_rad_per_batch": 0.0,                      # NO planted ZZ ⇒ no split expected
                       "readout_code": [code[ctrl], code[tgt]], "readout_amp": [0.0, 16000.0],
                       "level_phases": [0.0, math.pi, math.pi / 2],  # 2-level target: |0>/|1> π apart
                       "collapse": True, "t1": 250, "noise_scale": 250.0, "noise_seed": 4})

    cal = JAZZ(cfg, (ctrl, tgt), detune=detune, points=10, t0=40e-9, dt=80e-9, shots=40)
    r = cal.run(drv)
    d = r.data[(ctrl, tgt)]
    zz = r.proposal.get("two_qubit/(0, 1)/ZZ11", float("nan"))
    print(f"\n[jazz ζ=0] f0={d['f0'] / 1e6:+.3f}MHz f1={d['f1'] / 1e6:+.3f}MHz ZZ={zz / 1e6:+.3f}MHz ok={r.ok}")
    assert r.ok, "JAZZ fringe fits failed"
    # both control states see only the applied detuning (no ZZ) ⇒ same fringe, no spurious split
    assert abs(abs(d["f0"]) - detune) < 0.4e6, f"control-|0> fringe not at the detuning: {d['f0'] / 1e6:.3f} MHz"
    assert abs(abs(d["f1"]) - detune) < 0.4e6, f"control-|1> fringe not at the detuning: {d['f1'] / 1e6:.3f} MHz"
    assert abs(zz) < 0.4e6, f"spurious ZZ split with no planted ζ: {zz / 1e6:.3f} MHz"


# ── CZ resonance & conditionality (spec two-qubit/01 §4.4-4.5 / Q4) ──

F_CZ = 25e6                       # the model's |11>-|02> resonance = |f_ef[1] − f_ge[0]| (code 1024)
CZ_BATCHES = 30                   # the coupler drive length in batches


def _cz_cfg(m, ctrl=0, tgt=1, cz_amp=0.35):
    """The two-qubit co-sim Config: two qubits + a coupler-drive CZ entry at F_CZ (spec 01 §2), the
    coupler on core 2 of sim-2q1c. Two demod codes so the qubits' readouts are frequency-multiplexed."""
    code = {ctrl: 2048, tgt: 1024}
    cfg = Config()
    for q in (ctrl, tgt):
        cfg[f"qubit/{q}/freq"] = F_GE
        cfg[f"qubit/{q}/x90/amp"] = 0.5
        cfg[f"readout/{q}/freq"] = float(units.demod_code_to_freq(code[q], m.params))
        cfg[f"readout/{q}/amp"] = 0.5
        cfg[f"readout/{q}/dur"] = _s(56, m)
        cfg[f"readout/{q}/demod/dur"] = _s(40, m)
    cfg["reset/relax"] = _s(800, m)
    cfg["two_qubit/(0, 1)/core"] = 2
    cfg["two_qubit/(0, 1)/CZ/freq"] = F_CZ
    cfg["two_qubit/(0, 1)/CZ/pulse"] = [
        {"channel": "C0_1", "time": _s(CZ_BATCHES, m), "kwargs": {"amp": cz_amp, "phase": 0.0},
         "env": "square"},
        {"channel": "Q0", "env": "virtualz", "kwargs": {"phase": 0.0}},
        {"channel": "Q1", "env": "virtualz", "kwargs": {"phase": 0.0}},
    ]
    return cfg, code


def _abs_window(drv, dac, handle):
    """(absolute batch start, length, carrier code) of a captured DAC's single active window."""
    t0, cap = drv.sim.dac_capture_get(handle)
    act = np.abs(cap).sum(axis=1) > 0
    if not act.any():
        return None
    s = int(np.argmax(act))
    e = len(act) - 1 - int(np.argmax(act[::-1]))
    return t0 + s, e - s + 1, _carrier_code(cap[s:e + 1])


def test_cz_3core_drives_coupler_at_fcz_aligned(cosim_2q1c):
    """Q4 integration gate (spec 01 §4.4 / §7): the 3-core `k_cz_pop` build drives the COUPLER at f_CZ,
    for the CZ length, in a window that ABUTS the |11> prep and ends exactly SEP before the readout —
    the novel 3-core-timed coupler-in-lockstep path (JAZZ only used two cores). Captured off the real
    DACs with the model OFF, so it is deterministic and NOT relaxation-SNR-limited.

    This pins the kernel + fabric integration (the hard, novel part). The full P(11)-dip / conditionality
    R RECOVERY through the two-qubit COLLAPSE readout is relaxation-SNR-sensitive on this model (its only
    reset is relaxation, which also decoheres the multi-pulse sequence — the JAZZ ζ-magnitude precedent,
    spec 01 §4.3), and the recovery PHYSICS is proven deterministically host-side (test_twoqubit's
    resonance-dip fit + test_models.test_twoqubit_cz_conditionality_R_peaks_at_the_cz); a tight cosim
    recovery gate wants a long non-decohering grid and is deferred (spec 01 §4.4 note)."""
    drv, m = cosim_2q1c
    cfg, _ = _cz_cfg(m)
    drv.sim.set_model({"kind": "zero"})
    czd = _cz_dur_batches(cfg, (0, 1), m)
    ro0, demod0, code0, dur0, ddly0 = readout_tables(cfg, 0, m)
    xd = gate_pulse(cfg, 0, m).dur_batches(m, GATE_CH)
    period = grid_period(relax_batches(cfg, m), 2 * xd + czd, dur0, ddly0)
    progs = {}
    for q in (0, 1):                                          # both qubit cores prep |11>
        gate = ParamTable(GATE_CH, qubit_freq(cfg, q), {"x90": gate_pulse(cfg, q, m)})
        ro, demod, cd, dur, ddly = readout_tables(cfg, q, m)
        progs[q] = compile_kernel(kernels.k_cz_pop, m, tables=dict(gate=gate, ro=ro, demod=demod),
                                  out=Array(1), npts=1, shots=1, period=period, code=cd, ddly=ddly,
                                  role=kernels.CONTROL, knob=kernels.FREQ,
                                  form=kernels.COUPLER_FORM, xd=xd, czmax=czd, fcz=0,
                                  fef=0, sw=0, tail=0, x0=0, dx=0, **x90_vz(cfg, q))
    czt = cz_table(cfg, (0, 1), m, czd)                       # coupler: CZ drive at f_CZ
    progs[2] = compile_kernel(kernels.k_cz_pop, m, tables=dict(gate=czt, ro=ro0, demod=demod0),
                              out=Array(1), npts=1, shots=1, period=period, code=code0, ddly=ddly0,
                              role=kernels.COUPLER, knob=kernels.FREQ,
                              form=kernels.COUPLER_FORM, xd=xd, czmax=czd, fcz=0,
                              fef=0, sw=0, tail=0,
                              x0=int(units._freq_code(F_CZ, m.params)) << 16, dx=0, **x90_vz(cfg, 0))
    rq.setup(drv, m, progs)
    for c in (0, 1, 2):
        rq.check_magic(drv, m, c, progs[c]); rq.write_var(drv, m, c, progs[c], "__rq_status", 0)
    caps = {d: drv.sim.dac_capture_arm(d, 6000) for d in (0, 3, 2)}   # control gate, coupler, readout
    rq.reset(drv, m, on=False)
    for c in (0, 1, 2):
        rq.poll_done(drv, m, c, progs[c], timeout=period * 8 + 20_000_000)
    rq.reset(drv, m, on=True)
    prep = _abs_window(drv, 0, caps[0])
    cz = _abs_window(drv, 3, caps[3])
    ro = _abs_window(drv, 2, caps[2])
    f_cz_code = units._freq_code(F_CZ, m.params)
    print(f"\n[cz-3core] period={period} czd={czd} xd={xd} f_cz_code={f_cz_code}\n"
          f"  prep(DAC0)={prep} coupler(DAC3)={cz} readout(DAC2)={ro}")
    assert prep and cz and ro, "a prep / coupler / readout window was silent"
    assert cz[1] == czd, f"coupler window {cz[1]} != CZ length {czd}"
    assert abs(cz[2] - f_cz_code) < 40, f"coupler not at f_CZ: code {cz[2]} vs {f_cz_code}"
    assert prep[1] == 2 * xd, f"prep {prep[1]} != two X90s ({2 * xd})"
    assert 0 <= cz[0] - (prep[0] + prep[1]) <= 2, \
        f"coupler must abut the |11> prep end: prep_end={prep[0] + prep[1]} cz_start={cz[0]}"
    assert abs((ro[0] - (cz[0] + cz[1])) - SEP) <= 1, \
        f"coupler must end SEP before readout: gap={ro[0] - (cz[0] + cz[1])} vs SEP={SEP}"


# ── Two-qubit-drive CZ (spec 04 §4.1 / X2): the 2-core drive-form fire off the real DACs ──

DRIVE_F_GE = {0: 50e6, 1: 75e6}   # DISTINCT GE carriers (codes 2048/3072) — the retune is per core
DRIVE_F_CZ = 25e6                 # the shared in-band CZ tone (code 1024)
DRIVE_REL_PHASE = 0.267           # the target line's calibrated relative phase (rad)


def _cz_drive_cfg(m):
    """The two-qubit-drive co-sim Config (spec 04 §1): NO coupler core — both CZ lines on the pair's
    own gate channels at `CZ/freq`, equal amps, the TARGET line carrying the relative phase."""
    code = {0: 2048, 1: 1024}
    cfg = Config()
    for q in (0, 1):
        cfg[f"qubit/{q}/freq"] = DRIVE_F_GE[q]
        cfg[f"qubit/{q}/x90/amp"] = 0.5
        cfg[f"readout/{q}/freq"] = float(units.demod_code_to_freq(code[q], m.params))
        cfg[f"readout/{q}/amp"] = 0.5
        cfg[f"readout/{q}/dur"] = _s(56, m)
        cfg[f"readout/{q}/demod/dur"] = _s(40, m)
    cfg["reset/relax"] = _s(400, m)                            # short relax → short capture
    cfg["two_qubit/(0, 1)/CZ/freq"] = DRIVE_F_CZ
    cfg["two_qubit/(0, 1)/CZ/pulse"] = [
        {"channel": "Q0", "time": _s(CZ_BATCHES, m), "env": "square",
         "kwargs": {"amp": 0.35, "phase": 0.0}},
        {"channel": "Q1", "time": _s(CZ_BATCHES, m), "env": "square",
         "kwargs": {"amp": 0.35, "phase": DRIVE_REL_PHASE}},
        {"channel": "Q0", "env": "virtualz", "kwargs": {"phase": 0.0}},
        {"channel": "Q1", "env": "virtualz", "kwargs": {"phase": 0.0}},
    ]
    return cfg


def _abs_windows(drv, handle):
    """[(absolute batch start, length, carrier code, samples)] of EVERY active window of a captured
    DAC — the multi-window version of `_abs_window` (a drive-form gate DAC fires prep AND cz)."""
    t0, cap = drv.sim.dac_capture_get(handle)
    act = np.abs(cap).sum(axis=1) > 0
    wins, s = [], None
    for i, a in enumerate(act):
        if a and s is None:
            s = i
        if not a and s is not None:
            wins.append((t0 + s, i - s, _carrier_code(cap[s:i]), cap[s:i]))
            s = None
    if s is not None:
        wins.append((t0 + s, len(act) - s, _carrier_code(cap[s:]), cap[s:]))
    return wins


def test_cz_drive_form_two_tone_fire_aligned(cosim):
    """X2 integration gate (spec 04 §4.1 / §5, the drive-form mirror of
    test_cz_3core_drives_coupler_at_fcz_aligned): ONE drive-form `k_cz_pop` point on the 2-core
    sim-2q build, captured off the real DACs with the model OFF (deterministic). BOTH gate DACs fire
    the CZ tone at the f_CZ code for the CZ length, in LOCK-STEP (same absolute start); each core's
    GE prep before it is at its OWN GE code (the f_GE → f_CZ retune round-trip is real, not a
    carrier that was parked at f_CZ); the prep ends exactly LEAD before the tone (the retune's
    phasor-regen gap, the k_ef precedent — the drive form's 'abutting', spec 04 §1) and the tone
    ends SEP before the shared readout. The raw samples also expose the two lines' RELATIVE phase:
    the target tone leads the control's by the configured `kwargs/phase` (both NCOs retuned to the
    same f_CZ at the same pinned batch, so the slot phases are the only difference)."""
    drv, m = cosim
    cfg = _cz_drive_cfg(m)
    drv.sim.set_model({"kind": "zero"})
    czd = _cz_dur_batches(cfg, (0, 1), m)
    ro0, demod0, code0, dur0, ddly0 = readout_tables(cfg, 0, m)
    xd = gate_pulse(cfg, 0, m).dur_batches(m, GATE_CH)
    period = grid_period(relax_batches(cfg, m), 2 * xd + LEAD + czd, dur0, ddly0)
    fcz = units.freq_to_code(DRIVE_F_CZ, m.params)
    progs = {}
    for q, drive in ((0, 0), (1, 1)):
        gate = cz_drive_table(cfg, (0, 1), q, drive, m, czd)
        ro, demod, cd, dur, ddly = readout_tables(cfg, q, m)
        progs[q] = compile_kernel(kernels.k_cz_pop, m, tables=dict(gate=gate, ro=ro, demod=demod),
                                  out=Array(1), npts=1, shots=1, period=period, code=cd, ddly=ddly,
                                  role=kernels.CONTROL, knob=kernels.FREQ, form=kernels.DRIVE_FORM,
                                  xd=xd, czmax=czd, fcz=fcz, fef=0, sw=0, tail=0,
                                  x0=int(fcz), dx=0, **x90_vz(cfg, q))
    rq.setup(drv, m, progs)
    for c in (0, 1):
        rq.check_magic(drv, m, c, progs[c]); rq.write_var(drv, m, c, progs[c], "__rq_status", 0)
    # sim-2q has no dac_map → SocMap defaults: gate DACs 0/1, both readout drives summed on DAC 14
    rd = m.ro_dac(0)
    caps = {d: drv.sim.dac_capture_arm(d, 8000) for d in (m.gate_dac(0), m.gate_dac(1), rd)}
    rq.reset(drv, m, on=False)
    for c in (0, 1):
        rq.poll_done(drv, m, c, progs[c], timeout=period * 8 + 20_000_000)
    rq.reset(drv, m, on=True)
    wins = {d: _abs_windows(drv, caps[d]) for d in caps}

    ge_code = {q: units._freq_code(DRIVE_F_GE[q], m.params) for q in (0, 1)}
    fcz_code = units._freq_code(DRIVE_F_CZ, m.params)
    print(f"\n[cz-drive] period={period} czd={czd} xd={xd} f_cz_code={fcz_code}\n"
          f"  DAC0={[(s, n, c) for s, n, c, _ in wins[0]]}\n"
          f"  DAC1={[(s, n, c) for s, n, c, _ in wins[1]]}\n"
          f"  RO(DAC{rd})={[(s, n, c) for s, n, c, _ in wins[rd]]}")
    for q in (0, 1):
        assert len(wins[q]) == 2, f"core {q}: expected GE-prep + CZ windows, got {len(wins[q])}"
    (ps0, pl0, pc0, _), (cs0, cl0, cc0, w0) = wins[0]
    (ps1, pl1, pc1, _), (cs1, cl1, cc1, w1) = wins[1]
    # each core's GE prep at its OWN carrier — proves the retune starts from f_GE every shot
    assert pl0 == 2 * xd == pl1, f"preps {pl0}/{pl1} != two X90s ({2 * xd})"
    assert abs(pc0 - ge_code[0]) < 40, f"core 0 prep not at its f_GE: {pc0} vs {ge_code[0]}"
    assert abs(pc1 - ge_code[1]) < 40, f"core 1 prep not at its f_GE: {pc1} vs {ge_code[1]}"
    # both CZ tones: the f_CZ code, the CZ length, the same absolute start (lock-step)
    assert cl0 == czd == cl1, f"CZ windows {cl0}/{cl1} != CZ length {czd}"
    assert abs(cc0 - fcz_code) < 40, f"control line not at f_CZ: {cc0} vs {fcz_code}"
    assert abs(cc1 - fcz_code) < 40, f"target line not at f_CZ: {cc1} vs {fcz_code}"
    # each core's grid is t_ro = now() + period, so cross-core starts carry the boot skew of the
    # independent now() reads — the SAME ≤2-batch slack the coupler-form gate tolerates; the tones
    # stay phase-coherent regardless (the carrier phase is time-referenced, checked below)
    assert abs(cs0 - cs1) <= 2, f"the two CZ lines must fire in lock-step: {cs0} vs {cs1}"
    # the prep ends exactly LEAD before the tone; the tone ends SEP before the shared readout
    assert cs0 - (ps0 + pl0) == LEAD == cs1 - (ps1 + pl1), \
        f"prep→tone gap must be the LEAD phasor-regen gap: {cs0 - (ps0 + pl0)} vs {LEAD}"
    assert len(wins[rd]) == 1, "expected ONE summed readout window"
    ro_start = wins[rd][0][0]
    assert abs((ro_start - (cs0 + cl0)) - SEP) <= 1, \
        f"CZ must end SEP before readout: gap={ro_start - (cs0 + cl0)} vs SEP={SEP}"
    # the relative phase off the raw samples: project each tone onto ONE ABSOLUTE-TIME carrier frame
    # (the NCO phase is time-referenced — a start skew windows the same free-running carrier), so
    # the projected difference is exactly the slot-phase difference = the target line's phase
    w = math.pi * fcz_code / (1 << 15)                          # rad per DAC sample

    def tone_phase(samples, start_batch):
        x = np.asarray(samples, float).reshape(-1)
        n = start_batch * 16 + np.arange(len(x))                # absolute DAC sample index
        z = complex(np.sum(x * np.exp(-1j * w * n)))
        return math.atan2(z.imag, z.real)

    dphi = (tone_phase(w1, cs1) - tone_phase(w0, cs0) + math.pi) % (2 * math.pi) - math.pi
    print(f"  [cz-drive] relative phase Δφ={dphi:+.3f} rad (configured {DRIVE_REL_PHASE})")
    assert dphi == pytest.approx(DRIVE_REL_PHASE, abs=0.05), \
        f"target−control tone phase {dphi:+.3f} != configured {DRIVE_REL_PHASE}"


# ── SpectatorPhase geometry (spec 04 §4.5 / X3): the 3-core drive-form bracket off the real DACs ──

SPECT_F_GE = 100e6                # the spectator's own GE carrier (code 4096, distinct from the pair)


def test_spectator_ramsey_brackets_the_cz_fire(cosim_2q1c):
    """X3 optional gate (spec 04 §5): SpectatorPhase's 3-core geometry off the real DACs, model OFF
    (deterministic) — ONE point/shot of the class's exact compile recipe on sim-2q1c (cores 0/1 =
    the drive-form pair on DACs 0/1, core 2 = the spectator on DAC 3). The pair fires its two CZ
    lines at the f_CZ code in lock-step with the conditional |1> prep (sp=1 bound) ending LEAD
    before the tone; the SPECTATOR — the COUPLER_FORM ACTIVE Ramsey, window czd + 2·LEAD — plays
    its two Y90s at its OWN GE code so they BRACKET the tone with ~LEAD margin each side. The
    margins are the design (SpectatorPhase docstring): the spectator's 1-slot gate table makes its
    init preamble shorter than the pair's 2-slot ones, so its `now() + period` grid runs tens of
    batches EARLY (~58 here) — the first unequal-table multi-core kernel, beyond the ≤2-batch
    equal-table boot skew — and the symmetric bracket absorbs any sub-LEAD offset. Only the
    spectator fires the readout, SEP after its close."""
    drv, m = cosim_2q1c
    cfg = _cz_drive_cfg(m)
    cfg["qubit/2/freq"] = SPECT_F_GE
    cfg["qubit/2/x90/amp"] = 0.5
    cfg["readout/2/freq"] = float(units.demod_code_to_freq(3072, m.params))
    cfg["readout/2/amp"] = 0.5
    cfg["readout/2/dur"] = _s(56, m)
    cfg["readout/2/demod/dur"] = _s(40, m)
    drv.sim.set_model({"kind": "zero"})
    czd = _cz_dur_batches(cfg, (0, 1), m)
    fcz = units.freq_to_code(DRIVE_F_CZ, m.params)
    hpi = pack16(units._phase_code(math.pi / 2))
    ro_s, demod_s, code_s, dur_s, ddly_s = readout_tables(cfg, 2, m)
    xd = gate_pulse(cfg, 2, m).dur_batches(m, GATE_CH)
    period = grid_period(relax_batches(cfg, m), 3 * xd + czd + 2 * LEAD, dur_s, ddly_s)
    common = dict(npts=1, shots=1, period=period, ddly=ddly_s, hpi=hpi, xd=xd, p0=0, dp=0)
    progs = {2: compile_kernel(                       # the spectator: the bystander Ramsey (X3)
        kernels.k_cz_local, m,
        tables=dict(gate=ParamTable(GATE_CH, qubit_freq(cfg, 2),
                                    {"x90": gate_pulse(cfg, 2, m)}), ro=ro_s, demod=demod_s),
        out=Array(1), code=code_s, role=kernels.ACTIVE, form=kernels.COUPLER_FORM,
        czd=czd + 2 * LEAD, fcz=0, fef=0, sw=0, tail=0, sp=0, **common, **x90_vz(cfg, 2))}
    for q, drive, sp in ((0, 0, 1), (1, 1, 0)):       # the pair; the conditional (0) preps |1>
        gate = cz_drive_table(cfg, (0, 1), q, drive, m, czd)
        ro, demod, cd, dur, ddly = readout_tables(cfg, q, m)
        progs[q] = compile_kernel(kernels.k_cz_local, m, tables=dict(gate=gate, ro=ro, demod=demod),
                                  out=Array(1), code=cd, role=kernels.SPECTATOR,
                                  form=kernels.DRIVE_FORM, czd=czd, fcz=fcz, fef=0, sw=0, tail=0,
                                  sp=sp, **common, **x90_vz(cfg, q))
    rq.setup(drv, m, progs)
    for c in (0, 1, 2):
        rq.check_magic(drv, m, c, progs[c]); rq.write_var(drv, m, c, progs[c], "__rq_status", 0)
    caps = {d: drv.sim.dac_capture_arm(d, 8000) for d in (0, 1, 3, 2)}
    rq.reset(drv, m, on=False)
    for c in (0, 1, 2):
        rq.poll_done(drv, m, c, progs[c], timeout=period * 8 + 20_000_000)
    rq.reset(drv, m, on=True)
    wins = {d: _abs_windows(drv, caps[d]) for d in caps}

    fcz_code = units._freq_code(DRIVE_F_CZ, m.params)
    sge_code = units._freq_code(SPECT_F_GE, m.params)
    print(f"\n[spectator] period={period} czd={czd} xd={xd} f_cz_code={fcz_code}\n"
          f"  cond(DAC0)={[(s, n, c) for s, n, c, _ in wins[0]]}\n"
          f"  pair(DAC1)={[(s, n, c) for s, n, c, _ in wins[1]]}\n"
          f"  spect(DAC3)={[(s, n, c) for s, n, c, _ in wins[3]]}\n"
          f"  RO(DAC2)={[(s, n, c) for s, n, c, _ in wins[2]]}")
    # the pair: conditional = |1> prep (2 X90s at its GE code) + its CZ line; the other = CZ only
    assert len(wins[0]) == 2 and len(wins[1]) == 1
    (pps, ppl, ppc, _), (cs0, cl0, cc0, _) = wins[0]
    (cs1, cl1, cc1, _) = wins[1][0]
    assert ppl == 2 * xd and abs(ppc - units._freq_code(DRIVE_F_GE[0], m.params)) < 40
    assert cl0 == czd == cl1
    assert abs(cc0 - fcz_code) < 40 and abs(cc1 - fcz_code) < 40
    assert abs(cs0 - cs1) <= 2, f"CZ lines not in lock-step: {cs0} vs {cs1}"
    assert cs0 - (pps + ppl) == LEAD, f"cond prep→tone gap {cs0 - (pps + ppl)} != LEAD {LEAD}"
    # the spectator: two GE-code Y90s that BRACKET the pair's tone (sub-2·LEAD margins each side —
    # nominal LEAD ± the deterministic preamble-offset the docstring pins)
    assert len(wins[3]) == 2, f"spectator: expected prep + close Y90s, got {len(wins[3])}"
    (sps, spl, spc, _), (scs, scl, scc, _) = wins[3]
    assert spl == xd == scl
    assert abs(spc - sge_code) < 40 and abs(scc - sge_code) < 40, "spectator not at its OWN GE code"
    head, tail = cs0 - (sps + spl), scs - (cs0 + czd)
    print(f"  [spectator] bracket margins: head={head} tail={tail} (window czd + 2*LEAD = {czd + 2 * LEAD})")
    assert 0 < head < 2 * LEAD, \
        f"spectator prep must end BEFORE the tone starts (margin < 2*LEAD): head={head}"
    assert 0 < tail < 2 * LEAD, \
        f"spectator close must fire AFTER the tone ends (margin < 2*LEAD): tail={tail}"
    assert head + tail == pytest.approx(2 * LEAD, abs=4), \
        f"bracket margins must sum to the 2*LEAD slack: {head} + {tail}"
    # only the spectator reads out, SEP after its close
    assert len(wins[2]) == 1, "expected ONE readout window (the spectator's)"
    assert abs((wins[2][0][0] - (scs + scl)) - SEP) <= 1, \
        f"readout must open SEP after the close: gap={wins[2][0][0] - (scs + scl)} vs SEP={SEP}"


# ── EF-sandwich CZ playback (spec 04 §1 / X4): the shelved train off the real DACs ──

SAND_F_EF = 125e6                 # the shelf's EF carrier (code 5120, distinct from the GE/CZ codes)


def _cz_sandwich_cfg(m):
    """`_cz_drive_cfg` rebuilt as an EF-sandwich pair (the X6Y3 (5,6)/(6,7) layout on the sim-2q
    labels): qubit 1 is the SHELF — its EF keys land in the config and the pulse list brackets the
    two drives with the identical `single_qubit/1/EF/X/pulse` string references."""
    cfg = _cz_drive_cfg(m)
    cfg["qubit/1/EF/freq"] = SAND_F_EF
    cfg["qubit/1/EF/x/amp"] = 0.6
    pl = cfg["two_qubit/(0, 1)/CZ/pulse"]
    cfg["two_qubit/(0, 1)/CZ/pulse"] = (["single_qubit/1/EF/X/pulse"] + pl[:2]
                                        + ["single_qubit/1/EF/X/pulse"] + pl[2:])
    return cfg


def test_cz_sandwich_dac_train_aligned(cosim):
    """X4 co-sim gate (spec 04 §5): the sandwich DAC capture — ONE drive-form `k_cz_pop` point on
    sim-2q with a synthetic sandwich config, model OFF (deterministic, seconds). The SHELF core's
    gate DAC shows the full window train prep(f_GE) → EF-X(f_EF) → cz(f_CZ) → EF-X(f_EF), each
    segment at its carrier code with exactly the LEAD retune gap between segments and SEP before
    the readout; the PARTNER core plays prep(f_GE) → cz(f_CZ) with its cz tone in LOCK-STEP with
    the shelf's — the padded 3-slot gate table keeps the two boot preambles equal (the X3
    unequal-table grid finding), so the ≤2-batch equal-table skew still holds."""
    drv, m = cosim
    cfg = _cz_sandwich_cfg(m)
    drv.sim.set_model({"kind": "zero"})
    czd = _cz_dur_batches(cfg, (0, 1), m)
    shelf, sw_binds, tail = _sandwich_binds(cfg, (0, 1), m)
    efd = tail - LEAD
    assert shelf == 1
    xd = gate_pulse(cfg, 0, m).dur_batches(m, GATE_CH)
    ro0, demod0, code0, dur0, ddly0 = readout_tables(cfg, 0, m)
    period = grid_period(relax_batches(cfg, m), 2 * xd + LEAD + czd + 2 * tail, dur0, ddly0)
    fcz = units.freq_to_code(DRIVE_F_CZ, m.params)
    progs = {}
    for q, drive in ((0, 0), (1, 1)):
        gate = cz_drive_table(cfg, (0, 1), q, drive, m, czd)
        assert list(gate.pulses) == ["x90", "cz", "ef"]          # the ef slot pads BOTH cores
        ro, demod, cd, dur, ddly = readout_tables(cfg, q, m)
        progs[q] = compile_kernel(kernels.k_cz_pop, m, tables=dict(gate=gate, ro=ro, demod=demod),
                                  out=Array(1), npts=1, shots=1, period=period, code=cd, ddly=ddly,
                                  role=kernels.CONTROL, knob=kernels.FREQ, form=kernels.DRIVE_FORM,
                                  xd=xd, czmax=czd, fcz=fcz, tail=tail, x0=int(fcz), dx=0,
                                  **sw_binds[q], **x90_vz(cfg, q))
    rq.setup(drv, m, progs)
    for c in (0, 1):
        rq.check_magic(drv, m, c, progs[c]); rq.write_var(drv, m, c, progs[c], "__rq_status", 0)
    rd = m.ro_dac(0)
    caps = {d: drv.sim.dac_capture_arm(d, 8000) for d in (m.gate_dac(0), m.gate_dac(1), rd)}
    rq.reset(drv, m, on=False)
    for c in (0, 1):
        rq.poll_done(drv, m, c, progs[c], timeout=period * 8 + 20_000_000)
    rq.reset(drv, m, on=True)
    wins = {d: _abs_windows(drv, caps[d]) for d in caps}

    ge_code = {q: units._freq_code(DRIVE_F_GE[q], m.params) for q in (0, 1)}
    ef_code = units._freq_code(SAND_F_EF, m.params)
    fcz_code = units._freq_code(DRIVE_F_CZ, m.params)
    print(f"\n[cz-sandwich] period={period} czd={czd} xd={xd} efd={efd} tail={tail}\n"
          f"  partner(DAC0)={[(s, n, c) for s, n, c, _ in wins[0]]}\n"
          f"  shelf(DAC1)={[(s, n, c) for s, n, c, _ in wins[1]]}\n"
          f"  RO(DAC{rd})={[(s, n, c) for s, n, c, _ in wins[rd]]}")
    # the shelf: prep(f_GE) → EF-X(f_EF) → cz(f_CZ) → EF-X(f_EF), LEAD gaps between segments
    assert len(wins[1]) == 4, f"shelf: expected prep+ef+cz+ef windows, got {len(wins[1])}"
    (ps, pl, pc, _), (e1s, e1l, e1c, _), (czs, czl, czc, _), (e2s, e2l, e2c, _) = wins[1]
    assert pl == 2 * xd and abs(pc - ge_code[1]) < 40, "shelf prep not two X90s at its OWN f_GE"
    assert e1l == efd == e2l, f"EF-X windows {e1l}/{e2l} != EF X length {efd}"
    assert abs(e1c - ef_code) < 40 and abs(e2c - ef_code) < 40, "EF-X not at f_EF"
    assert czl == czd and abs(czc - fcz_code) < 40, "shelf cz tone not at f_CZ for the CZ length"
    assert e1s - (ps + pl) == LEAD, f"prep→EF-X gap {e1s - (ps + pl)} != LEAD {LEAD}"
    assert czs - (e1s + e1l) == LEAD, f"EF-X→cz gap {czs - (e1s + e1l)} != LEAD {LEAD}"
    assert e2s - (czs + czl) == LEAD, f"cz→EF-X gap {e2s - (czs + czl)} != LEAD {LEAD}"
    # the partner: its own prep + cz line, the tone in LOCK-STEP with the shelf's
    assert len(wins[0]) == 2, f"partner: expected prep + cz windows, got {len(wins[0])}"
    (pps, ppl, ppc, _), (cs0, cl0, cc0, _) = wins[0]
    assert ppl == 2 * xd and abs(ppc - ge_code[0]) < 40, "partner prep not at its OWN f_GE"
    assert cl0 == czd and abs(cc0 - fcz_code) < 40, "partner cz tone not at f_CZ for the CZ length"
    assert abs(cs0 - czs) <= 2, f"cz tones must fire in lock-step: partner {cs0} vs shelf {czs}"
    # the summed readout opens SEP after the shelf's un-shelving EF X ends (± the boot skew)
    assert len(wins[rd]) == 1, "expected ONE summed readout window"
    gap = wins[rd][0][0] - (e2s + e2l)
    assert abs(gap - SEP) <= 3, f"readout must open SEP after the post EF-X: gap={gap} vs SEP={SEP}"


# ── CZ 2D amp x freq seed landscape (spec 14 F4): the drive-form co-sim gate ──

AF_F_GE = {0: 50e6, 1: 150e6}     # planted GE carriers (codes 2048 / 6144)
AF_F_EF = {0: 450e6, 1: 650e6}    # planted EF carriers (codes 18432 / 26624)
AF_F_CZ = (AF_F_GE[0] + 2 * AF_F_GE[1] + AF_F_EF[1]) / 4   # the model's OWN (f11+f02)/4 = 250 MHz
AF_AMP = 0.35                     # the per-line amp planted for a FULL 2*pi round trip (the CZ)
AF_RELAX, AF_T1 = 3200, 1600      # relax = 2*t1 head resets the pair; t1 keeps the sequence coherent


def _amp_freq_cfg(m):
    """The drive-form co-sim Config for the 2D seed sweep: `_cz_drive_cfg`'s layout retuned to the
    MODEL's own in-band resonance, with both CZ lines ALIGNED (relative phase 0 → |E| = 2A, the
    coherent-sum optimum) so the amp axis alone sets the round-trip angle. Only the TARGET's demod
    matters (`_cond_R` reads the target), so it gets the whole readout tone."""
    code = {0: 2048, 1: 1024}
    cfg = Config()
    for q in (0, 1):
        cfg[f"qubit/{q}/freq"] = AF_F_GE[q]
        cfg[f"qubit/{q}/x90/amp"] = 0.5
        cfg[f"readout/{q}/freq"] = float(units.demod_code_to_freq(code[q], m.params))
        cfg[f"readout/{q}/amp"] = 0.5
        cfg[f"readout/{q}/dur"] = _s(56, m)
        cfg[f"readout/{q}/demod/dur"] = _s(40, m)
    cfg["reset/relax"] = _s(AF_RELAX, m)
    cfg["two_qubit/(0, 1)/CZ/freq"] = AF_F_CZ
    cfg["two_qubit/(0, 1)/CZ/pulse"] = [
        {"channel": "Q0", "time": _s(CZ_BATCHES, m), "env": "square",
         "kwargs": {"amp": AF_AMP, "phase": 0.0}},
        {"channel": "Q1", "time": _s(CZ_BATCHES, m), "env": "square",
         "kwargs": {"amp": AF_AMP, "phase": 0.0}},
        {"channel": "Q0", "env": "virtualz", "kwargs": {"phase": 0.0}},
        {"channel": "Q1", "env": "virtualz", "kwargs": {"phase": 0.0}},
    ]
    return cfg, code


def test_cz_amp_freq_sweep_peaks_at_the_planted_cz(cosim):
    """F4 co-sim gate (spec 14 §5): `CZAmpFreqSweep` on the 2-core drive-form build against a
    planted `TwoQubitModel` whose CZ is a full 2*pi round trip (the conditional pi) at
    `AF_AMP` on its own in-band resonance `AF_F_CZ`. The 3x3 landscape — the HOST lockstep-amp
    loop (`write_slot('amp')` on BOTH cores' cz slot of ONE resident image) x the ON-CORE freq
    sweep — is extremal at that planted cell, and the seed is written to `CZ/freq` + both lines'
    `CZ/pulse` amps.

    Physics of record (verified against the model itself, not just this curve): the CZ rotates the
    {|11>, |02>} pseudo-qubit by theta = rabi_cz*|E|*czd, and under our ONE-BIT discriminator (the
    target's |2> reads as a coin flip, spec 01 §4.5) the on-resonance conditionality is
    R = (1 - cos(theta/2))/2 — 0.5 at the half amp, 1 at `AF_AMP`, 0.5 at 1.5x — while a detuning
    starves the round trip. Hence the 0.5x/1x/1.5x amp axis and the +-6 MHz freq axis: the exact
    model puts the landscape at

        [[0.11 0.50 0.11]      (0.5x amp)
         [0.38 1.00 0.38]      (1.0x amp — the planted CZ)
         [0.65 0.50 0.65]]     (1.5x amp; the off-resonance arms are the chevron of 2*pi contours)

    so the apex beats the runner-up by 0.35 — the margin this gate spends on relaxation contrast
    loss and 48-shot noise. The discriminator is pinned first by a real `ReadoutCalibration` on the
    target (its demod phase + res_sign baked into the config): R is invariant under a polarity flip
    but NOT under a discriminator with no contrast."""
    drv, m = cosim
    cfg, code = _amp_freq_cfg(m)
    czd = _cz_dur_batches(cfg, (0, 1), m)
    sig_cz = gate_sigma(m, _cz_pulse(cfg, (0, 1), m, czd, 0), AF_F_CZ, units._amp_code(AF_AMP))
    rabi_cz = math.pi / sig_cz                    # two aligned lines: |E| = 2A → 2*pi over czd

    def rabi_ge(q):                               # the X90/X90 prep is a full pi
        gp = gate_pulse(cfg, q, m)
        return (math.pi / 2) / gate_sigma(m, gp, AF_F_GE[q],
                                          units._amp_code(float(cfg[f"qubit/{q}/x90/amp"])))

    def model(seed):
        return {"kind": "twoqubit", "control": 0, "target": 1,   # no "coupler" key ⇒ the drive form
                "f_ge": [AF_F_GE[0], AF_F_GE[1]], "f_ef": [AF_F_EF[0], AF_F_EF[1]],
                "rabi_ge": [rabi_ge(0), rabi_ge(1)], "rabi_ef": [0.0, 0.0],
                "rabi_cz_rad_per_amp": rabi_cz, "zz_rad_per_batch": 0.0,
                "readout_code": [code[0], code[1]], "readout_amp": [0.0, 20000.0],
                "level_phases": [0.0, math.pi, math.pi / 2],     # |2> lands on the discriminator null
                "collapse": True, "t1": AF_T1, "noise_scale": 150.0, "noise_seed": seed}

    drv.sim.set_model(model(seed=5))
    ro = ReadoutCalibration(cfg, [1], shots=32).run(drv)         # pin the target's discriminator
    sep = ro.data[1]["separation"]
    cfg["readout/1/demod/phase"] = float(ro.proposal["readout/1/demod/phase"])
    cfg["readout/1/res_sign"] = int(ro.proposal["readout/1/res_sign"])
    assert ro.ok, f"target readout clusters did not separate (separation={sep:.2f})"

    drv.sim.set_model(model(seed=17))
    amps = np.array([0.5, 1.0, 1.5]) * AF_AMP
    r = CZAmpFreqSweep(cfg, (0, 1), amps=amps, span=6e6, points=3, ngates=1, shots=48).run(drv)
    d = r.data[(0, 1)]
    print(f"\n[cz-ampfreq] separation={sep:.2f} czd={czd} rabi_cz={rabi_cz:.3e}\n"
          f"  amps={np.round(d['amps'], 4)}  freqs[MHz]={np.round(d['freqs'] / 1e6, 3)}\n"
          f"  R=\n{np.round(d['R'], 3)}\n  fit={r.fit[(0, 1)]}")
    ka, kf = np.unravel_index(int(np.argmax(d["R"])), d["R"].shape)
    assert d["R"].shape == (3, 3)
    assert d["freqs"][1] == pytest.approx(AF_F_CZ)               # the axis hits the resonance exactly
    assert (ka, kf) == (1, 1), \
        f"landscape argmax at amp {d['amps'][ka]:.4f} / {d['freqs'][kf] / 1e6:.3f} MHz, " \
        f"planted {AF_AMP} / {AF_F_CZ / 1e6:.3f} MHz"
    assert r.ok and float(d["R"][1, 1]) > 0.5
    assert r.proposal["two_qubit/(0, 1)/CZ/freq"] == pytest.approx(AF_F_CZ)
    written = r.proposal["two_qubit/(0, 1)/CZ/pulse"]
    assert [p["kwargs"]["amp"] for p in written[:2]] == pytest.approx([AF_AMP, AF_AMP])
