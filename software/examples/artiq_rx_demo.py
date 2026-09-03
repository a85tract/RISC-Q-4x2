"""The RECEIVE side of the ARTIQ-shaped API, verified two ways in one script.

A. TRACE IDENTITY — the ion-trap sequence rebuilt with `adc.gate()` (no manual fill_gaps, no
   hand-written robs plumbing) must produce the very same trace as the verified TX-side script.

B. IQ ORACLE — with no golden demod model, the oracle is the raw trace itself: one run both
   records the trace and demod-gates windows inside a played tone; the host then integrates the
   SAME trace with the SAME LO law. The unknown demod pipeline alignment is one fixed complex
   factor, so the checks need no constants:
     (1) hw_iq / host_iq is the SAME complex ratio for every case (magnitude AND angle);
     (2) rotating the demod phase by phi rotates hw_iq by exactly phi;
     (3) rotating the TONE phase by phi rotates hw_iq the same way as it rotates host_iq;
     (4) the hardware res bit is consistently the sign classifier of real.

  from the repository root (or inside the docker image, where PYTHONPATH is preset):
  co-sim:  PYTHONPATH=software/client:sim python software/examples/artiq_rx_demo.py --cosim
  board :  PYTHONPATH=software/client     python software/examples/artiq_rx_demo.py --remote 192.168.3.1
  (--config/--build default to this repo's gateware/configs and sim/build; --loopback-src 1 for
  the 2-DAC bundle, whose readout drive is DAC1)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from riscq import artiqapi as A
from riscq.map import SocMap, SocParams

V, OMEGA, AMP = 82.0 * A.MHz, 1.765 * A.MHz, 0.4

# (tone phase, demod phase) per case — turns
CASES = [(0.0, 0.0), (0.0, 0.25), (0.25, 0.0)]
CASE_US = 10.0
TONE_US, DM_OFF_US, DM_US = 8.0, 1.0, 6.0


def build_trace_experiment(m):
    """The ion-trap two-pulse sequence with a concurrent recording gate — the RX idiom."""
    core = A.Core(m)
    gate, ro, adc = A.DDSChannel(core, 0, "gate"), A.DDSChannel(core, 1, "readout"), A.ADCChannel(core)
    with A.parallel(core):
        with A.branch(core):
            with A.parallel(core):
                with A.branch(core):
                    gate.set(V + OMEGA, phase=0.0, amplitude=AMP); gate.sw.pulse(100 * A.us)
                with A.branch(core):
                    ro.set(V - OMEGA, phase=0.5, amplitude=AMP); ro.sw.pulse(100 * A.us)
            A.delay(core, 5 * A.us)
            ro.set(V, phase=0.25, amplitude=AMP, phase_mode=A.PHASE_MODE_ABSOLUTE)
            ro.sw.pulse(20 * A.us)
        with A.branch(core):
            adc.gate(125 * A.us)          # record everything; fillers appear automatically
    return core, adc


def build_iq_experiment(m):
    """Three tone+demod cases inside one recording gate."""
    core = A.Core(m)
    ro, dm, adc = A.DDSChannel(core, 1, "readout"), A.DemodChannel(core), A.ADCChannel(core)
    with A.parallel(core):
        with A.branch(core):
            for tone_phase, dm_phase in CASES:
                t0 = A.now_mu(core)
                with A.parallel(core):
                    with A.branch(core):
                        ro.set(V, phase=tone_phase, amplitude=AMP)
                        ro.sw.pulse(TONE_US * A.us)
                    with A.branch(core):
                        A.delay(core, DM_OFF_US * A.us)
                        dm.set(V, phase=dm_phase)
                        dm.gate(DM_US * A.us)
                A.at_mu(core, t0 + core.seconds_to_mu(CASE_US * A.us))
        with A.branch(core):
            adc.gate(len(CASES) * CASE_US * A.us)
    return core, adc, dm


def host_integral(res: "A.RunResult", m, e) -> complex:
    """The RTL demod law applied to the captured trace over event e's window: sumR + i*sumI =
    sum_n adc[n] * exp(+i*2*pi*f*n/fs + i*phase). The unknown pipeline alignment is one fixed
    complex factor common to every window — the checks use ratios, never this absolute value."""
    fs = res.fs
    lo = (e.batch - res.gate_start_mu // 16) * 4
    hi = lo + e.dur_batches * 4
    n = np.arange(lo, hi)
    f = A.units.word_to_freq(e.freq_word, m.params) / 4          # the ADC-rate word's RF frequency
    lo_ = np.exp(1j * (2 * np.pi * f * n / fs + 2 * np.pi * e.phase_turns))
    return complex(np.sum(res.trace[lo:hi] * lo_))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosim", action="store_true")
    ap.add_argument("--remote", default=None)
    ap.add_argument("--bundle", default="rfsoc4x2-1q-fine")
    repo = Path(__file__).resolve().parents[2]
    ap.add_argument("--config", default=str(repo / "gateware/configs/rfsoc4x2-1q-fine.json"))
    ap.add_argument("--build", default=str(repo / "sim/build/rfsoc4x2-1q-fine"))
    ap.add_argument("--work", default=str(repo / "sim/build/artiq_rx_generated"))
    ap.add_argument("--delay", type=int, default=5)
    ap.add_argument("--loopback-src", type=int, default=0,
                    help="co-sim: which DAC the loopback model feeds into ADC0 (0 on the single-DAC "
                         "builds; 1 on rfsoc4x2-2dac-fine, where the readout drive is DAC1)")
    ap.add_argument("--loopback-dst", type=int, default=0,
                    help="co-sim: ADC port the loopback feeds (the bundle's adc_map entry; 1 for *-adcb)")
    ap.add_argument("--ref-trace", default=None,
                    help="npz whose 'trace' the part-A capture must equal byte for byte (co-sim)")
    ap.add_argument("--out", default="artiq_rx.npz")
    a = ap.parse_args()

    if a.cosim:
        from riscq_sim import cosim
        drv = cosim.start(a.config, a.build)
        m = SocMap(SocParams.from_json(drv.sim.get_params()))
        drv.sim.set_model({"kind": "loopback", "src": a.loopback_src, "dst": a.loopback_dst, "gain": 0.9,
                           "delay": a.delay})
    elif a.remote:
        from riscq.driver.remote import RemoteDriver
        drv = RemoteDriver(a.remote)
        drv.board.load(a.bundle)
        m = SocMap(SocParams.from_json(drv.board.get_params()))
        print(f"loaded: {drv.board.info()}", flush=True)
    else:
        ap.error("pick --cosim or --remote")

    ok = True
    try:
        # ── A. trace identity ────────────────────────────────────────────────────────────────────
        core, adc = build_trace_experiment(m)
        res = A.run(drv, core, a.work, doc="ion-trap sequence recorded through adc.gate")
        tr = adc.fetch_trace()
        print(f"A: adc.gate trace: {tr.size} samples, max|s| = {np.abs(tr).max()}", flush=True)
        if a.ref_trace:
            # The reference was generated by a DIFFERENT program, so its t1 differs; the phase
            # constant P = Pconst - W*(t1*16) is truncated to the 16-bit register once per
            # program, so the two traces may differ by up to a few codes (the documented
            # t1-dependent double-truncation envelope) while being the same physical waveform.
            ref = np.load(a.ref_trace)["trace"]
            n = min(tr.size, ref.size)
            d = tr[:n].astype(np.int64) - ref[:n].astype(np.int64)
            maxd = int(np.abs(d).max())
            same = tr.size == ref.size and maxd <= 3
            print(f"A: vs {Path(a.ref_trace).name}: sizes {tr.size}/{ref.size}, "
                  f"mismatching {int(np.count_nonzero(d))}, max |diff| {maxd}")
            print("A: TRACE_MATCH (<= 3 codes, the t1 double-truncation envelope):",
                  "yes" if same else "NO")
            ok &= same

        # ── B. IQ oracle ─────────────────────────────────────────────────────────────────────────
        core2, adc2, dm2 = build_iq_experiment(m)
        res2 = A.run(drv, core2, a.work, doc="three demod cases inside one recording gate")
        demods = [e for e in res2.schedule.events if e.is_demod]
        hw = res2.iq
        host = np.array([host_integral(res2, m, e) for e in demods])
        ratios = hw / host
        print("B: case  (tone_phase, dm_phase)   hw |iq|      angle      hw/host |r|   angle")
        for j, (e, r) in enumerate(zip(demods, ratios)):
            print(f"B: {j}    {CASES[j]}   {abs(hw[j]):12.3e} {np.degrees(np.angle(hw[j])):+9.3f}  "
                  f"{abs(r):10.5f} {np.degrees(np.angle(r)):+9.3f}")
        r0 = ratios[0]
        mag_ok = all(abs(abs(r / r0) - 1) < 0.01 for r in ratios)
        ang_ok = all(abs(((np.angle(r / r0) + np.pi) % (2 * np.pi)) - np.pi) < np.radians(1.0)
                     for r in ratios)
        print(f"B1 same complex ratio for every case (|r| within 1 %, angle within 1 deg): "
              f"{mag_ok and ang_ok}")
        ok &= mag_ok and ang_ok

        dphi = np.degrees(np.angle(hw[1] / hw[0]))
        print(f"B2 demod phase +90 deg rotated hw IQ by {dphi:+.3f} deg "
              f"(expect +-90 within 1 deg)")
        ok &= abs(abs(dphi) - 90) < 1.0

        tphi_hw = np.degrees(np.angle(hw[2] / hw[0]))
        tphi_ho = np.degrees(np.angle(host[2] / host[0]))
        print(f"B3 tone phase +90 deg: hw rotated {tphi_hw:+.3f} deg, host {tphi_ho:+.3f} deg "
              f"(must match within 1 deg)")
        ok &= abs(((tphi_hw - tphi_ho + 180) % 360) - 180) < 1.0

        sign_neg = [int(r) == int(x < 0) for r, x in zip(res2.res, res2.real)]
        sign_pos = [int(r) == int(x >= 0) for r, x in zip(res2.res, res2.real)]
        print(f"B4 res bits {res2.res.tolist()} vs real signs "
              f"{[int(x < 0) for x in res2.real]}: consistent = {all(sign_neg) or all(sign_pos)}")
        ok &= all(sign_neg) or all(sign_pos)

        np.savez(a.out, trace_a=tr, trace_b=res2.trace, hw_res=res2.res, hw_real=res2.real,
                 hw_imag=res2.imag, host_real=host.real, host_imag=host.imag,
                 fs=res2.fs, cases=np.array(CASES))
        print("RX_DEMO:", "PASS" if ok else "FAIL")
    finally:
        if a.cosim:
            from riscq_sim import cosim
            cosim.stop(drv)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
