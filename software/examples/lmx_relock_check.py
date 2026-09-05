"""Does the DAC -> ADC timing move when the board's clocks re-lock, and does the LMX2594's phase-SYNC
mode change that? (RFSoC 4x2, `rfsoc4x2-2q-fine`.)

Per trial the clocks are reprogrammed — the whole chain (LMK + both LMXs), or ONE LMX alone with the LMK
untouched — the bundle is reloaded (tiles restart, MTS re-runs and its pinned latencies are checked), and
the same ABSOLUTE-phase 82 MHz tone plays on dds 1 (DAC_A, LMX "lmxdac") and dds 3 (DAC_B) into ADC_A /
ADC_B (LMX "lmxadc"). Reported per lock: each path's carrier phase against the generator (a step of one
LMX VCO period, 127 ps, would be 3.75 deg at 82 MHz; run-to-run noise is ~0.05 deg), the envelope lag,
the levels, the strongest non-tone bin of trace A (a spur floor), the MTS latencies, the measured dsp clock.

    mode: default  — xrfclk's own LMX list at every reprogramming
          sync     — the bundle's lmx2594-491.52-sync.txt: the same list with VCO_PHASE_SYNC = 1 and
                     PLL_N 320 -> 80 (IncludedDivide 4), datasheet SNAS696C 7.3.10 category 1
                     (software/examples/clocks/make_lmx_sync_list.py derives it)

A bundle load restores the clock state its board.json asks for (per LMX), so EVERY arm can only be
measured with the list the bundle itself carries: run "sync" against rfsoc4x2-2q-fine (it ships the SYNC
list) and "default" against a copy of it whose board.json has no "lmx_regs". The script reads the
programmed state back after every load and stops if it is not this mode's.

Run from the co-sim container or any host with the client installed:
    PYTHONPATH=software/client python software/examples/lmx_relock_check.py <default|sync> <trials> [host] [arms] [bundle]
    arms: comma list of clocks,lmxdac,lmxadc (default: all three); bundle: the bundle to load
    (default rfsoc4x2-2q-fine; for "default" mode name your copy of it without "lmx_regs")
Result of 2026-09-04 (4 trials each): see docs/hardware-contract.md, "Clocks and re-locks".
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

from riscq import artiqapi as A
from riscq.driver import remote
from riscq.map import SocMap, SocParams

MODE, TRIALS = sys.argv[1], int(sys.argv[2])
HOST = sys.argv[3] if len(sys.argv) > 3 else "192.168.3.1"
ARMS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["clocks", "lmxdac", "lmxadc"]
BUNDLE = sys.argv[5] if len(sys.argv) > 5 else "rfsoc4x2-2q-fine"
F = 82e6
HERE = Path(__file__).resolve().parent
SYNC_LIST = HERE.parent / "server" / "bits" / "rfsoc4x2-2q-fine" / "lmx2594-491.52-sync.txt"
regs = None if MODE == "default" else [int(l.split()[1], 16) for l in SYNC_LIST.read_text().splitlines()
                                       if l.strip()]
# the driver's record of a programmed list (pynq_driver._regs_sha): None = xrfclk's own file
WANT = None if regs is None else hashlib.sha256(",".join(str(int(v)) for v in regs).encode()).hexdigest()


def check_state(info):
    """Stop unless both LMXs run this mode's list — a load restores the bundle's board.json list."""
    state = info["refclks"][2] if info["refclks"] else None
    if state is None or any(v != WANT for v in state.values()):
        raise SystemExit(f"programmed clock state {state} is not mode {MODE!r}'s ({WANT}): run this mode "
                         f"against a bundle whose board.json asks for the same list (see the docstring)")


def measure(drv, m, tag):
    fs = 4 * m.params.dsp_freq_hz
    c = A.Core(m)
    ch1, ch3 = A.DDSChannel(c, 1), A.DDSChannel(c, 3)
    adc = [A.ADCChannel(c, k) for k in range(2)]
    with A.parallel(c):
        with A.branch(c):
            A.delay(c, 5 * A.us); ch1.set(F, phase=0.0, amplitude=0.4, phase_mode=A.PHASE_MODE_ABSOLUTE); ch1.sw.pulse(10 * A.us)
        with A.branch(c):
            A.delay(c, 5 * A.us); ch3.set(F, phase=0.0, amplitude=0.4, phase_mode=A.PHASE_MODE_ABSOLUTE); ch3.sw.pulse(10 * A.us)
        with A.branch(c):
            adc[0].gate(20 * A.us)
        with A.branch(c):
            adc[1].gate(20 * A.us)
    r = A.run(drv, c, HERE / "artiq_compat_work" / f"lmx_relock_{MODE}", doc="lmx relock check")
    a, b = r.cores[0].trace.astype(float), r.cores[1].trace.astype(float)
    mid = slice(int(7e-6 * fs), int(13e-6 * fs))
    n = np.arange(mid.stop - mid.start)
    w = 2 * np.pi * F / fs
    basis = np.stack([np.cos(w * n), np.sin(w * n), np.ones(n.size)], 1)
    ca, *_ = np.linalg.lstsq(basis, a[mid], rcond=None)
    cb, *_ = np.linalg.lstsq(basis, b[mid], rcond=None)
    pa, pb = np.degrees(np.arctan2(-ca[1], ca[0])), np.degrees(np.arctan2(-cb[1], cb[0]))
    k = max(2, int(round(fs / F)))
    ea = np.convolve(np.abs(a), np.ones(k) / k, "same"); eb = np.convolve(np.abs(b), np.ones(k) / k, "same")
    ea -= ea.mean(); eb -= eb.mean()
    lag = int(np.argmax(np.correlate(ea, eb, "full"))) - (len(eb) - 1)
    seg = a[mid] * np.hanning(n.size)
    spec = np.abs(np.fft.rfft(seg)); freqs = np.fft.rfftfreq(n.size, 1 / fs)
    tone = spec[np.argmin(np.abs(freqs - F))]
    far = (np.abs(freqs - F) > 2e6) & (freqs > 5e6)          # not the tone, not the pulse's DC
    spur_db = 20 * np.log10(spec[far].max() / tone); spur_at = freqs[far][np.argmax(spec[far])]
    info = drv.board.info()
    check_state(info)
    print(f"{tag:>22}  phase A {pa:+8.2f}  B {pb:+8.2f}  B-A {(pb - pa + 180) % 360 - 180:+7.2f} deg  "
          f"lag {lag:+d}  amp A {np.hypot(ca[0], ca[1]):.0f} B {np.hypot(cb[0], cb[1]):.0f}  "
          f"spur {spur_db:+.1f} dBc @ {spur_at/1e6:.2f} MHz  mts {info['mts_latencies']}  "
          f"dsp {info['dsp_mhz']:.4f} MHz  refclks {info.get('refclks')}", flush=True)
    return pa, pb


def reload(drv, tag):
    """Reload the bundle (tiles restart, MTS re-runs against the pinned latencies); an MTS miss is a
    data point here, not a stop — report it and retry once."""
    for attempt in range(2):
        try:
            drv.board.load(BUNDLE)
            return True
        except Exception as e:                            # Pyro carries the server's RuntimeError
            print(f"{tag:>22}  load FAILED (attempt {attempt}): {str(e).splitlines()[0][:200]}", flush=True)
    return False


drv = remote.RemoteDriver(HOST)
try:
    if drv.board.info()["bundle"] != BUNDLE:
        drv.board.load(BUNDLE)
    check_state(drv.board.info())
    m = SocMap(SocParams.from_json(drv.board.get_params()))
    rows = {arm: [] for arm in ARMS}
    for t in range(TRIALS if "clocks" in ARMS else 0):
        drv.board.refclks(245.76, 491.52, regs)           # LMK + both LMXs (xrfclk's files, then the list)
        if reload(drv, f"{MODE} clocks #{t}"):
            rows["clocks"].append(measure(drv, m, f"{MODE} clocks #{t}"))
    for which in (arm for arm in ("lmxdac", "lmxadc") if arm in ARMS):
        for t in range(TRIALS):
            drv.board.lmx_program(which, regs)             # ONE LMX re-locks, the LMK is not touched
            if reload(drv, f"{MODE} {which} #{t}"):
                rows[which].append(measure(drv, m, f"{MODE} {which} #{t}"))
    for k, v in rows.items():
        v = np.array(v)
        if len(v):
            span = lambda x: (np.max(x) - np.min(x))
            print(f"{MODE} {k:>7}: phase A spread {span(v[:, 0]):.2f} deg, B spread {span(v[:, 1]):.2f} deg, "
                  f"B-A spread {span((v[:, 1] - v[:, 0] + 180) % 360 - 180):.2f} deg over {len(v)} locks")
finally:
    drv.close()
