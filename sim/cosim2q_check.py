"""Co-simulation acceptance suite of the 2-core `rfsoc4x2-2q-fine` build under `riscq.artiqapi`.

Model: two ideal loopbacks (DAC1 -> ADC1 = core 0's pair, DAC0 -> ADC0 = core 1's pair), gain 0.9,
5 batches of delay. Every check prints PASS/FAIL; the last line is the tally.
  A     the ion-trap reference on ch0/ch1 -> core 0's trace (size, level)
  ISO   ch1 alone lights only trace 0 and ch3 alone only trace 1 — INCLUDING the last batch of the
        window, which the previous run had left non-zero (regression of the 2026-09-04 recorder fix)
  ISO-3 different tones / amplitudes / delays on the two cores land in their own traces
  EXT   a full-scale tone keeps both signs of the 16-bit lanes
  DEP   a gate of the whole trace depth (65536 batches): one sinusoid to the very last sample
  B     the SAME tone on ch1 and ch3 at the same instant with deliberately ASYMMETRIC kernels
        (core 1 also plays a gate pulse and takes a demod readout) -> identical traces (one origin)
  B'    control: half a turn on ch3 inverts its trace
  PM    ABSOLUTE with a frequency hop, then a TRACKING -> CONTINUOUS chain, on both cores -> identical
        (batch-aligned starts: a sub-batch start costs two queue entries, 4 hops + 2 fillers > depth 8)
  WRAP  the 32-bit batch clock wraps INSIDE a run: the setup distance origin - batch_time() is measured
        on an unshifted run, then the time offset puts the next origin 8000 batches below 2^32
  1Q    the 2-core core-0 ion-trap trace equals the 1-core rfsoc4x2-1q-fine build's trace within one
        16-bit phase LSB (the start-time phase compensation is truncated to the phase register: +-3 codes)
Run inside the co-sim container:
  cd /work/RISC-Q && PYTHONPATH=software/client:sim python sim/cosim2q_check.py [out_dir]
Takes several hours (each run is 10-30 min of Verilator time).
"""
import sys
import time
from pathlib import Path

import numpy as np

from riscq import artiqapi as A
from riscq import run as rq
from riscq.map import SocMap, SocParams
from riscq_sim import cosim

REPO = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "sim" / "build" / "cosim2q_check"
LOOP = {"kind": "loopback", "gain": 0.9, "delay": 5}
MULTI = {"kind": "multi", "models": [dict(LOOP, src=1, dst=1), dict(LOOP, src=0, dst=0)]}
M32 = 1 << 32
results = []


def ion_trap(c, gate, ro, adc):
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


def start(name, model):
    t0 = time.time()
    drv = cosim.start(REPO / f"gateware/configs/{name}.json", REPO / f"sim/build/{name}")
    drv.sim.set_model(model)
    m = SocMap(SocParams.from_json(drv.sim.get_params()))
    print(f"[{name}] co-sim up in {time.time() - t0:.0f} s; rob_per_core={m.params.rob_per_core} "
          f"run_origin={m.params.run_origin}", flush=True)
    return drv, m


def run(drv, core, tag):
    t0 = time.time()
    r = A.run(drv, core, OUT / f"gen_{tag}", doc=tag)
    tele = {k: (list(map(int, cr.tele)) if cr.tele is not None else None) for k, cr in r.cores.items()}
    print(f"[{tag}] run {time.time() - t0:.0f} s; origin={r.origin}; tele={tele}", flush=True)
    return r


def verdict(ok, msg):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    return ok


def peak_hz(x, fs):
    x = x.astype(float) - x.astype(float).mean()
    sp = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    return np.fft.rfftfreq(x.size, 1 / fs)[np.argmax(sp)]


def sin_fit(x, f, fs, skip=400):
    """Least-squares sinusoid at f over x[skip:] (the DAC -> ADC pipeline head is ~10 zero batches):
    (max residual, amplitude)."""
    n = np.arange(x.size)
    basis = np.stack([np.cos(2 * np.pi * f * n / fs), np.sin(2 * np.pi * f * n / fs), np.ones(n.size)], 1)
    coef, *_ = np.linalg.lstsq(basis[skip:], x[skip:].astype(float), rcond=None)
    return np.abs(x[skip:] - basis[skip:] @ coef).max(), float(np.hypot(coef[0], coef[1]))


def two_tone_traces(m, extra_core1=False, phase3=0.25, gate_us=30):
    """ch1 and ch3 play the same tone at the same instant; core 1 optionally does extra, different work
    (a gate pulse before, a demod readout after) so its kernel is asymmetric to core 0's."""
    c = A.Core(m)
    ch = [A.DDSChannel(c, i) for i in range(4)]
    adc = [A.ADCChannel(c, k) for k in range(2)]
    with A.parallel(c):
        with A.branch(c):
            A.delay(c, 5 * A.us); ch[1].set(82e6, phase=0.25, amplitude=0.4); ch[1].sw.pulse(20 * A.us)
        with A.branch(c):
            A.delay(c, 5 * A.us); ch[3].set(82e6, phase=phase3, amplitude=0.4); ch[3].sw.pulse(20 * A.us)
        if extra_core1:
            with A.branch(c):
                ch[2].set(83.765e6, amplitude=0.3); ch[2].sw.pulse(3 * A.us)      # core 1 only: gate pulse
            with A.branch(c):
                A.delay(c, 27 * A.us)
                dm = A.DemodChannel(c, 1); dm.set(82e6); dm.gate(1 * A.us)        # core 1 only: a readout
        with A.branch(c):
            adc[0].gate(gate_us * A.us)
        with A.branch(c):
            adc[1].gate(gate_us * A.us)
    return c


def tone_window(m, r, lo_us=6, hi_us=24):
    fs = 4 * m.params.dsp_freq_hz
    sl = slice(int(lo_us * 1e-6 * fs), int(hi_us * 1e-6 * fs))
    return r.cores[0].trace.astype(int)[sl], r.cores[1].trace.astype(int)[sl]


def horizon(r):
    return max(e.batch + e.dur_batches for e in r.schedule.events)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    drv, m = start("rfsoc4x2-2q-fine", MULTI)
    fs = 4 * m.params.dsp_freq_hz
    try:
        # ---- A: the ion trap on core 0 ----
        c = A.Core(m)
        ion_trap(c, A.DDSChannel(c, 0), A.DDSChannel(c, 1), A.ADCChannel(c, 0))
        rA = run(drv, c, "2q_ionA")
        trA = rA.cores[0].trace
        np.savez(OUT / "cosim2q_ionA.npz", trace=trA, t=rA.cores[0].t)
        verdict(trA is not None and np.abs(trA).max() > 1000 and trA.size == 61440 * 4,
                f"A: core-0 ion-trap trace: {trA.size} samples, max |s| = {np.abs(trA).max()}")

        # ---- ISO: one channel at a time (right after A, whose 125 us trace fills the RAM the ISO gate
        #      covers: the silent core's window, last batch included, must read back all-zero) ----
        for k, name in ((0, "ch1"), (1, "ch3")):
            c = A.Core(m)
            ro = A.DDSChannel(c, 1 if k == 0 else 3)
            adc = [A.ADCChannel(c, j) for j in range(2)]
            with A.parallel(c):
                with A.branch(c):
                    A.delay(c, 5 * A.us); ro.set(82e6, amplitude=0.4); ro.sw.pulse(10 * A.us)
                with A.branch(c):
                    adc[0].gate(20 * A.us)
                with A.branch(c):
                    adc[1].gate(20 * A.us)
            r = run(drv, c, f"2q_iso_{name}")
            own, other = r.cores[k].trace, r.cores[1 - k].trace
            verdict(np.abs(own).max() > 1000 and np.abs(own).max() < 12000 and np.abs(other).max() == 0,
                    f"ISO: {name} alone -> trace{k} max |s| {np.abs(own).max()} (one tone, no stale batch), "
                    f"trace{1 - k} max |s| {np.abs(other).max()} (silent DAC, must be 0 incl. the last batch)")

        # ---- ISO-3: different tones, amplitudes and delays on the two cores ----
        c = A.Core(m)
        ch1, ch3 = A.DDSChannel(c, 1), A.DDSChannel(c, 3)
        adc = [A.ADCChannel(c, j) for j in range(2)]
        with A.parallel(c):
            with A.branch(c):
                A.delay(c, 5 * A.us); ch1.set(82e6, amplitude=0.4); ch1.sw.pulse(10 * A.us)
            with A.branch(c):
                A.delay(c, 7 * A.us); ch3.set(80.235e6, amplitude=0.2); ch3.sw.pulse(10 * A.us)
            with A.branch(c):
                adc[0].gate(20 * A.us)
            with A.branch(c):
                adc[1].gate(20 * A.us)
        r = run(drv, c, "2q_iso_tones")
        t0, t1 = r.cores[0].trace, r.cores[1].trace
        w0 = t0[int(5.5e-6 * fs):int(14.5e-6 * fs)]; w1 = t1[int(7.5e-6 * fs):int(16.5e-6 * fs)]
        f0, f1 = peak_hz(w0, fs), peak_hz(w1, fs)
        a0, a1 = np.abs(w0).max(), np.abs(w1).max()
        on1 = np.argmax(np.abs(t1) > 500) / fs * 1e6
        verdict(abs(f0 - 82e6) < 0.5e6 and abs(f1 - 80.235e6) < 0.5e6 and 0.45 < a1 / a0 < 0.55
                and 6.9 < on1 < 7.6,
                f"ISO-3: distinct tones per trace: trace0 peak {f0/1e6:.3f} MHz amp {a0}, trace1 peak "
                f"{f1/1e6:.3f} MHz amp {a1} (ratio {a1/a0:.3f}, want 0.5), trace1 onset {on1:.2f} us (want 7)")

        # ---- EXT: full-scale tone keeps both signs ----
        c = A.Core(m)
        ch1, adc0 = A.DDSChannel(c, 1), A.ADCChannel(c, 0)
        with A.parallel(c):
            with A.branch(c):
                A.delay(c, 2 * A.us); ch1.set(82e6, amplitude=1.0); ch1.sw.pulse(10 * A.us)
            with A.branch(c):
                adc0.gate(14 * A.us)
        r = run(drv, c, "2q_ext")
        t0 = r.cores[0].trace
        verdict(t0.min() < -25000 and t0.max() > 25000,
                f"EXT: full-scale tone spans [{t0.min()}, {t0.max()}] codes (int16 lanes, both signs)")

        # ---- DEP: the whole trace depth, one sinusoid to the last sample ----
        c = A.Core(m)
        ch1, adc0 = A.DDSChannel(c, 1), A.ADCChannel(c, 0)
        with A.parallel(c):
            with A.branch(c):
                ch1.set(82e6, amplitude=0.4); ch1.sw.pulse_mu(m.params.rob_depth * 16)
            with A.branch(c):
                adc0.gate_mu(m.params.rob_depth * 16)
        r = run(drv, c, "2q_depth")
        t0 = r.cores[0].trace
        resid, amp = sin_fit(t0, 82e6, fs)
        verdict(t0.size == m.params.rob_depth * 4 and np.abs(t0[-4:]).max() > 0 and resid < 0.02 * amp,
                f"DEP: {t0.size} samples (= {m.params.rob_depth} batches x 4), sinusoid fit incl. the last "
                f"batch: max residual {resid:.0f} of amplitude {amp:.0f}, last 4 samples {t0[-4:].tolist()}")

        # ---- B: one origin, asymmetric kernels ----
        rB = run(drv, two_tone_traces(m, extra_core1=True), "2q_syncB")
        w0, w1 = tone_window(m, rB)
        d = np.abs(w0 - w1).max()
        np.savez(OUT / "cosim2q_syncB.npz", trace0=rB.cores[0].trace, trace1=rB.cores[1].trace)
        verdict(d == 0 and np.abs(w0).max() > 1000,
                f"B: same tone on ch1/ch3 (asymmetric kernels): tone windows identical, max diff {d} codes, "
                f"max |s| {np.abs(w0).max()}; origin {rB.origin}")
        rB2 = run(drv, two_tone_traces(m, extra_core1=True, phase3=0.75), "2q_syncB2")
        w0, w1 = tone_window(m, rB2)
        d2 = np.abs(w0 + w1).max()
        verdict(d2 <= 2, f"B': half a turn on ch3 inverts its trace: max |t0 + t1| = {d2} codes")

        # ---- PM: phase modes with a hop, both cores, batch-aligned back-to-back segments ----
        c = A.Core(m)
        ch = [A.DDSChannel(c, i) for i in range(4)]
        adc = [A.ADCChannel(c, k) for k in range(2)]
        with A.parallel(c):
            for k in (1, 3):
                with A.branch(c):
                    c.delay_mu(23600)                                                                         # 3.0 us
                    ch[k].set(82e6, phase=0.1, amplitude=0.4, phase_mode=A.PHASE_MODE_ABSOLUTE); ch[k].sw.pulse_mu(39328)
                    ch[k].set(80.235e6, phase=0.3, amplitude=0.4, phase_mode=A.PHASE_MODE_ABSOLUTE); ch[k].sw.pulse_mu(39328)
                    ch[k].set(82e6, phase=0.0, amplitude=0.4, phase_mode=A.PHASE_MODE_TRACKING); ch[k].sw.pulse_mu(31456)
                    ch[k].set(80.235e6, phase=0.2, amplitude=0.4, phase_mode=A.PHASE_MODE_CONTINUOUS); ch[k].sw.pulse_mu(31456)
            with A.branch(c):
                adc[0].gate(25 * A.us)
            with A.branch(c):
                adc[1].gate(25 * A.us)
        rPM = run(drv, c, "2q_modes")
        t0, t1 = rPM.cores[0].trace.astype(int), rPM.cores[1].trace.astype(int)
        d = np.abs(t0 - t1).max()
        segs = [(3.5, 7.5, 82e6), (8.5, 12.5, 80.235e6), (13.5, 16.5, 82e6), (17.5, 20.5, 80.235e6)]
        peaks = [peak_hz(t0[int(lo * 1e-6 * fs):int(hi * 1e-6 * fs)], fs) for lo, hi, _ in segs]
        hop_ok = all(abs(p - f) < 0.5e6 for p, (_, _, f) in zip(peaks, segs))
        verdict(d == 0 and np.abs(t0).max() > 1000 and hop_ok,
                f"PM: ABSOLUTE hop + TRACKING->CONTINUOUS chain on both cores: identical, max diff {d}; "
                f"segment peaks {' / '.join(f'{p/1e6:.3f}' for p in peaks)} MHz (want 82 / 80.235 / 82 / 80.235)")

        # ---- WRAP: measure the setup distance, then land the origin just below 2^32 ----
        tb = int(drv.sim.batch_time())
        r0 = run(drv, two_tone_traces(m, extra_core1=True), "2q_wrap0")
        D = (int(r0.origin) - tb) % M32
        tb = int(drv.sim.batch_time())
        rq.set_time_offset(drv, m, (M32 - 8000 - (tb + D)) % M32)
        rW = run(drv, two_tone_traces(m, extra_core1=True), "2q_wrap")
        rq.set_time_offset(drv, m, 0)
        w0, w1 = tone_window(m, rW)
        d = np.abs(w0 - w1).max()
        O, h = int(rW.origin), horizon(rW)
        crossed = (O + h + 64 >= M32) or (O < 8192)
        verdict(d == 0 and np.abs(w0).max() > 1000 and crossed,
                f"WRAP: origin {O} (target {M32 - 8000}), horizon {h} -> wrap "
                f"{'CROSSED inside the run' if crossed else 'NOT reached'}; traces identical, max diff {d}; "
                f"core-1 telemetry {list(map(int, rW.cores[1].tele))}")
    finally:
        cosim.stop(drv)

    # ---- 1Q: the same ion trap on the 1-core build ----
    drv, m1 = start("rfsoc4x2-1q-fine", dict(LOOP, src=0, dst=0))
    try:
        c = A.Core(m1)
        ion_trap(c, A.DDSChannel(c, 0), A.DDSChannel(c, 1), A.ADCChannel(c))
        r1 = run(drv, c, "1q_ionA")
        tr1 = r1.trace
        np.savez(OUT / "cosim1q_ionA.npz", trace=tr1)
        n = min(len(tr1), len(trA))
        dd = np.abs(tr1[:n].astype(int) - trA[:n].astype(int))
        verdict(len(tr1) == len(trA) and dd.max() <= 3,
                f"1Q: 2q core-0 trace == 1q-fine trace within one 16-bit phase LSB: lengths {len(trA)}/{len(tr1)}, "
                f"max diff {dd.max()} codes (<= 3), differing samples {(dd > 0).sum()}")
    finally:
        cosim.stop(drv)

    print("COSIM2Q: " + ("PASS" if all(results) else "FAIL") + f" ({sum(results)}/{len(results)})", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
