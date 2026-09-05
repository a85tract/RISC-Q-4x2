"""Co-sim proof of the kernel's queue barriers (artiqapi._queue_barriers): more plays on one channel
than the TimedQueue holds, at the MINIMUM spacing the planner allows.

12 pulses of 32 batches every `period` batches on ch1, gaps filled (fill_gaps) so the trace records
continuously: 12 pulses + 11 gap fillers + 2 trace-window fillers = 25 plays on a queue of depth 8, i.e.
17 barriers, each before the event's first queue write (set_freq). A push that comes late is dropped
or misplaced by the queue (no backpressure): a missing pulse, or a missing filler that resets the trace
address and garbles the recording. Checks: the generated kernel has 17 barriers; the trace shows exactly
12 bursts whose onsets are 4 * period ADC samples apart.
Measured 2026-09-04 (RTL): period 73 (36.5 batches per play) FAILS — 8 bursts then garbage: the kernel
falls behind the plays and pushes arrive late; period 100 (50 per play), 121 and 160 PASS. The
planner's clock model (PUSH_MARGIN 300 for a wait, PUSH_COST 60 per further event, LEAD 96) refuses
this pattern up to period 115 and admits 121 — the shipped default, ~15 % above what the RTL sustained.
Run inside the co-sim container (a few minutes of Verilator time):
  cd /work/RISC-Q && PYTHONPATH=software/client:sim python sim/cosim2q_queue.py [out_dir] [period_batches]
"""
import sys

import numpy as np

from riscq import artiqapi as A

import cosim2q_check as S

N_PULSES, ON = 12, 32
PERIOD = int(sys.argv[2]) if len(sys.argv) > 2 else 121  # batches; 8 plays (pulses + fillers) span 4 periods


def main():
    S.OUT.mkdir(parents=True, exist_ok=True)
    drv, m = S.start("rfsoc4x2-2q-fine", S.MULTI)
    try:
        depth = m.params.queue_depth
        c = A.Core(m)
        ch1, adc0 = A.DDSChannel(c, 1), A.ADCChannel(c, 0)
        with A.parallel(c):
            with A.branch(c):
                A.delay(c, 2 * A.us)
                for _ in range(N_PULSES):
                    ch1.set(82e6, amplitude=0.4)
                    ch1.sw.pulse_mu(16 * ON)
                    A.delay_mu(c, 16 * (PERIOD - ON))
            with A.branch(c):
                adc0.gate(6 * A.us)
        fillers = A.fill_gaps(c, 1)
        r = S.run(drv, c, "2q_queue")
        src = (S.OUT / "gen_2q_queue" / "generated_sequence.py").read_text()
        n_bars = src.count("queue barrier")
        plays = sum(len(v) for v in r.schedule.chunks.values())   # pulses + gap fillers + the 2 window fillers
        S.verdict(fillers == N_PULSES - 1 and plays == 2 * N_PULSES + 1 and n_bars == plays - depth,
                  f"Q: period {PERIOD}: {N_PULSES} pulses + {fillers} gap fillers + 2 window fillers = {plays} "
                  f"plays on ch1 (queue depth {depth}); kernel has {n_bars} queue barriers (want {plays - depth})")
        t0 = r.cores[0].trace.astype(float)
        env = np.convolve(np.abs(t0), np.ones(24) / 24, "same")     # one 82 MHz period of smoothing
        on = env > 0.5 * env.max()
        onsets = np.flatnonzero(on[1:] & ~on[:-1]) + 1
        gaps = np.diff(onsets)
        want = PERIOD * 4                                  # ADC samples per period
        S.verdict(onsets.size == N_PULSES and np.all(np.abs(gaps - want) <= 8),
                  f"Q: period {PERIOD}: trace has {onsets.size} bursts (want {N_PULSES}), onset spacing "
                  f"{sorted(set(gaps.tolist()))} samples (want {want} +-8), max |s| {np.abs(t0).max():.0f}")
        np.savez(S.OUT / "cosim2q_queue.npz", trace=t0, onsets=onsets)
    finally:
        drv.close()
    ok = sum(S.results)
    print(f"{ok}/{len(S.results)} PASS", flush=True)
    sys.exit(0 if ok == len(S.results) else 1)


if __name__ == "__main__":
    main()
