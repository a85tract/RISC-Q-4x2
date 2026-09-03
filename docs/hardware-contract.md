# The hardware contract — what is exact, what is snapped, what is refused

Numbers below are for the deployed build `rfsoc4x2-1q-fine` (DAC 7.86432 GS/s, ADC
1.96608 GS/s, dsp clock 491.52 MHz, 16 DAC samples per batch).

## Time grids

| quantity | resolution | note |
|---|---|---|
| timeline arithmetic (`mu`) | 1 DAC sample = **127.157 ps** | finer than anything physical below — `seconds_to_mu` hides no hardware grid |
| pulse **leading** edge | envelope grid = **0.254 ns** (2 DAC samples) | realized start reported per event (`report()`, `err` column) |
| pulse **trailing** edge | whole-batch grid = **2.035 ns** (absolute positions) | realized duration = snapped end − snapped start, so it need not be a whole-batch multiple; `end err` reported |
| carrier phase | 16-bit register = **0.0055°** (0.19 ps at 82 MHz) | a phase shift of a tone IS an exact time shift — use phase when you need sub-grid timing |
| demod window edges | whole batches | |

A pulse asked at, e.g., 105.000000 µs lands at +50.9 ps (the nearest envelope-grid point);
the report prints exactly this.

## Phase modes (ARTIQ `ad9910` semantics)

* `PHASE_MODE_TRACKING` (default) — φ(t) = p + t·f against a **global** epoch: each frequency
  is its own free-running metronome, so hopping away and back is phase-reproducible.
* `PHASE_MODE_ABSOLUTE` — φ(t) = p + (t − t_set)·f, t_set = the instant of the `set()` call:
  `p` is the phase at the start, history irrelevant.
* `PHASE_MODE_CONTINUOUS` — the accumulator is not reset across a frequency change; the chain
  walks the `set()` calls (a `set` with no pulse still matters).

Verified on hardware: cross-pulse carrier coherence < 0.5°; the live notebook's capture agrees
with the ideal generator to ≤ 0.11° per tone.

## The trace recorder (why fillers exist)

The raw-trace BRAM records **only while the readout-drive channel (ch 1) plays**, and its write
address **resets to 0 whenever ch 1 stops** — a gap does not pause the recording, it makes the
next pulse overwrite it from the start. Consequences, all handled by the scheduler:

* `adc.gate` auto-inserts amplitude-0 filler pulses over every silent stretch of the window
  (lead-in, holes between your pulses, tail). DAC output is genuinely zero either way.
* **One gate per run**; the outward-snapped window must fit the trace memory:
  ≤ 65 536 batches ≈ **133 µs** here.
* A ch-1 pulse that overlaps the gate boundary, or sub-batch gaps the filler cannot express,
  are refused with named errors rather than recorded wrongly.

## Scheduling limits (the play parameter queue)

Each play (a pulse, a filler, an envelope chunk of a long pulse) is one entry in a hardware
timed queue. Two limits, both enforced by `plan()` before anything runs:

* **Spacing ≥ 3 batches** between play starts on one channel. The deployed queue pops at most
  one entry per 3 cycles (SrlShadow II=3); a closer play would pop a batch late, glitch the
  channel low for one batch, *and* reset the trace recorder. Practical corollaries: no 1– or
  2-batch gaps between pulses on a recorded channel; pulses starting mid-batch reserve an
  envelope-line triplet internally (invisible to you, but it is why the rule is 3).
* **≤ 8 queued plays per channel** (`queue_depth` of this build). The push has no backpressure
  — an overfull queue silently drops entries — so the planner refuses schedules that could
  exceed it. Long pulses split internally, each chunk one play: ≈ 16 k-batch chunks on a
  channel that also has mid-batch pulses; up to 65 535 batches (the duration field) per chunk
  otherwise.
* Two pulses on one channel may not overlap in time (a channel plays one pulse at a time).

## IQ readout rules

* The demod gate's window is the integration window; **≤ 16 384 batches ≈ 33 µs** (the RTL
  accumulator's no-overflow bound).
* Reading a result **halts the core** until the window's integral settles, so an event may
  start at or before the demod window's START, or at least **144 batches ≈ 293 ns** after its
  END (`READOUT_LEAD 48` + push margin `LEAD 96`); anything starting in between is refused.
* Results return per gate, in gate order: the sign bit `res`, and the 32-bit `real`/`imag`
  integrals. `phase` on the demod `set()` rotates the returned IQ by that many turns, exact
  to the 16-bit phase register (0.0055°).
* There is no golden model of the demod pipeline; it is verified by the ratio oracle in
  `software/examples/artiq_rx_demo.py` (hardware/host integrals constant to 0.01 % / 0.004°
  across phase cases, on co-sim and on the board).

## The errors, and what to do (the common ones — every check raises a specific message)

| error (abbreviated) | meaning | fix |
|---|---|---|
| `pulses overlap — one ends at ..., the next starts at ...` | two pulses on one channel overlap (remember trailing edges round to whole batches) | separate them |
| `queued plays at batches A and B start closer than 3 batches` | the II=3 spacing rule | ≥ 3 batches (6.1 ns) between play starts; avoid 1–2-batch pulses/gaps |
| `N queued plays exceed the hardware queue depth 8` | too many pulses+fillers+chunks on one channel in one run | split into several runs, or merge/space events |
| `adc gate of N batches (snapped outward) exceeds rob_depth 65536` | window > trace memory | shorten the gate |
| `... adc gates in one run` | more than one gate | one gate per run |
| `event at batch B starts within 144 batches of the readout window ending at E` | too close after a demod window (the core is halted there) | space the sequence out |
| `integration window of N batches exceeds the accumulator's no-overflow contract` | window too long | ≤ 33 µs per window |
| `channel 2 is the demod carrier, not a drive channel` | `DDSChannel`/`dds` on index 2 | use the demod device |
| `unknown channel index N (have 3 channels)` | bad channel number | 0 = gate, 1 = readout |
| `... .pulse() before ... .set()` | pulse with no tone set on the channel | `set()` first |
| `amplitude ... outside [0, 1]`, `unknown phase mode ...` | bad `set()` arguments | fix the value |
| `PHASE_MODE_CONTINUOUS continues a phase that does not exist yet` | first `set()` on a channel is CONTINUOUS | start with TRACKING or ABSOLUTE |
| `ch1 pulse at batches ... overlaps the adc gate ... boundary or follows it` | a readout-drive pulse crossing or after the gate would merge with or restart the recording | keep ch1 pulses inside the gate, or ended ≥ 1 batch before it |
| `pulse at batch A is squeezed to nothing by the next pulse at batch B` | sub-batch pulses packed tighter than the envelope grid | space them |
| `envelope RAM exhausted`, `no full envelope lines left to chunk into` | too many distinct mid-batch edge patterns / not enough envelope memory | reuse edge offsets or shorten |
| `no trace: run() the sequence first`, `no readout results ...`, `all N queued results already fetched` | fetch before/beyond what the last run produced | run first / fetch once per gate |

Every limit in this file is enforced at planning time — before the kernel executes on the
hardware. (`run_experiment` does open the board/co-sim connection first, to read the build's
parameters; but nothing plays unless every check passed.)
