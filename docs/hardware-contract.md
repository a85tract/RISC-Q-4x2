# The hardware contract — what is exact, what is snapped, what is refused

Numbers below are for the deployed "fine" builds — `rfsoc4x2-1q-fine` (one core) and
`rfsoc4x2-2q-fine` (two cores, both DACs, both ADCs): DAC 7.86432 GS/s, ADC 1.96608 GS/s, dsp
clock 491.52 MHz, 16 DAC samples per batch.

## Channels, cores and the one timeline

A build has `qubit_num` identical hardware cores; each core owns a gate drive, a readout drive
(summed on the same DAC in the fine builds), one ADC and an IQ demodulator. You address them
across cores with flat indices:

| device / class | index | meaning (`rfsoc4x2-2q-fine`) |
|---|---|---|
| `dds` / `DDSChannel` | 2k, 2k+1 | core k's gate drive, core k's readout drive — core 0 → **DAC_A**, core 1 → **DAC_B** |
| `adc` / `ADCChannel` | k | core k's raw trace: the ADC in `adc_map[k]`, recorded while dds 2k+1 fires — 0 → **ADC_A**, 1 → **ADC_B** |
| `demod` / `DemodChannel` | k | core k's IQ readout (same ADC) |

On a one-core build these are simply dds 0/1, adc 0, demod 0. There is still ONE timeline: `run()`
compiles one kernel per hardware core from the same recorded schedule. The kernels anchor on the
SoC's **shared run origin** — the batch time latched when the cores are released from reset, plus
a fixed lead (8192 batches) — so events scheduled at the same instant start on the same batch on
every DAC and the phase-mode law holds *between* DACs (TRACKING phases are relative to that
common origin). The cores' own clock reads after the release differ by their boot skew (≤ ~2
batches = 60° at 82 MHz), which is why the origin is a hardware latch and not a `now()` read; each
kernel reports the origin it used and the clock just before its first play, and `run()` refuses
the result if the cores disagree or a play was pushed with less than `LEAD` (96 batches) to spare.

From the digital origin to the **connectors**: DAC_A and DAC_B sit on two RF tiles (230 and 228).
On `rfsoc4x2-2q-fine` the two tiles are **multi-tile synchronized** (as upstream RISC-Q and QubiC run
their ZCU216): tile 230 owns the sample-clock PLL and distributes its clock to tile 228, the ADC
tile 226 is synchronized to the same SYSREF (the other three RF-ADC tiles are enabled idle, which is
what makes the converter IP offer ADC MTS and keeps the on-chip SYSREF distribution chain intact), the
fabric clock is tile 230's own output clock (so the SoC runs on the very clock MTS aligns), and at
every bundle load the board runs the xrfdc
MTS procedure against SYSREF and pins the tile latencies to the values recorded in the bundle's
`board.json` (`"mts": {..., "required": true}` — a miss refuses to load). What remains between the
two connectors is the fixed board/cable path difference; the notebook's second demo measures it
(the phase offset between the two loopback traces) and it is the number to fold into the DAC_B
channels' `phase` if the connectors themselves must be aligned. The one-core bundles
(`rfsoc4x2-1q-*`, `rfsoc4x2-2dac-*`) predate MTS and keep their unsynchronized tile clocks.

**Clocks and re-locks** (measured 2026-09-04, `software/examples/lmx_relock_check.py`, 4 locks per
arm, 82 MHz loopback phase of each path against the generator): the two LMX2594 synthesizers that
make the 491.52 MHz tile reference clocks are programmed from PYNQ's xrfclk register files. With
those files, re-locking either LMX alone (the LMK untouched) leaves both paths within 0.06°, but
reprogramming the whole chain (LMK + both LMXs — what the first load in a fresh server process or a
power cycle does) lands the paths in one of two states about 0.8° (DAC_A) / 1.8° (DAC_B) apart, i.e.
25–60 ps, while MTS still reaches its pinned latencies. `rfsoc4x2-2q-fine` therefore carries its own
LMX list (`board.json` `"lmx_regs"`: `lmx2594-491.52-sync.txt`, the xrfclk list with the LMX's
phase-SYNC mode enabled — VCO_PHASE_SYNC = 1 and PLL_N 320 → 80 for the included ÷4, TI category 1
for this frequency plan, no SYNC pulse needed), which the server programs into both LMXs after the
defaults at every clock programming: with it the whole-chain reprogramming repeats within 0.06° /
0.12° (≤ 3 ps) and the single-LMX re-locks within 0.1°. The absolute path phase moves by a fixed
~3° between the two lists (the notebook's numbers are taken with the bundle's list). Loads take
4–22 s in either state. Not covered from here: real power cycles, scope probing of the clock edges.

## Time grids

| quantity | resolution | note |
|---|---|---|
| timeline arithmetic (`mu`) | 1 DAC sample = **127.157 ps** | finer than anything physical below — `seconds_to_mu` hides no hardware grid |
| pulse **leading** edge | envelope grid = **0.254 ns** (2 DAC samples) | realized start reported per event (`report()`, `err` column) |
| pulse **trailing** edge | whole-batch grid = **2.035 ns** (absolute positions) | realized duration = snapped end − snapped start, so it need not be a whole-batch multiple; `end err` reported |
| carrier phase | 16-bit register = **0.0055°** (0.19 ps at 82 MHz); in co-simulation two runs of the same timeline reproduce their carrier phases to within that one LSB (the start-time compensation is truncated to the register), so replayed traces agree within ±3 codes at full scale, not bit-exactly | a phase shift of a tone IS an exact time shift — use phase when you need sub-grid timing |
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

Verified on hardware: cross-pulse carrier coherence < 0.5°; the live notebook's capture agrees with
the ideal generator to ≤ 0.3° per tone (2026-09-04, 0.6 % rms residual; ≤ 0.11° on the 2026-08-31
bench). A lossy cable on one loop showed up as 0.5° and 11 % — the cable, not the design
(`software/server/bits/rfsoc4x2-2q-fine/PROVENANCE.md`).

## The trace recorder (why fillers exist)

Core k's raw-trace BRAM records **only while that core's readout-drive channel (dds 2k+1)
plays**, and its write address **resets to 0 whenever that channel stops** — a gap does not
pause the recording, it makes the next pulse overwrite it from the start. Consequences, all
handled by the scheduler:

* `adc.gate` auto-inserts amplitude-0 filler pulses on that channel over every silent stretch
  of the window (lead-in, holes between your pulses, tail). DAC output is genuinely zero either
  way.
* **One gate per trace per run** (adc 0 and adc 1 may each have one); the outward-snapped
  window must fit the trace memory: ≤ 65 536 batches ≈ **133 µs** here.
* A readout-drive pulse that overlaps the gate boundary, or sub-batch gaps the filler cannot
  express, are refused with named errors rather than recorded wrongly.
* Storage: on `rfsoc4x2-2q-fine` each core has its own trace (`rob_per_core`: 4 lanes × 16-bit
  = the ADC's samples as they are); the one-core builds keep the upstream single trace (32-bit
  per-lane sum over the mapped ADCs — one ADC there). Both come back as int32 samples.

## Scheduling limits (the play parameter queue)

Each play (a pulse, a filler, an envelope chunk of a long pulse) is one entry in a hardware
timed queue. Two limits, both enforced by `plan()` before anything runs:

* **Spacing ≥ 3 batches** between play starts on one channel. The deployed queue pops at most
  one entry per 3 cycles (SrlShadow II=3); a closer play would pop a batch late, glitch the
  channel low for one batch, *and* reset the trace recorder. Practical corollaries: no 1– or
  2-batch gaps between pulses on a recorded channel; pulses starting mid-batch reserve an
  envelope-line triplet internally (invisible to you, but it is why the rule is 3).
* **The parameter queues hold 8 entries per channel** (`queue_depth` of this build; `set_freq` and
  every fire push one entry each). The push has no backpressure — an overfull queue silently drops
  entries — so before a channel's 9th, 10th, … event the generated kernel waits for the play pushed
  8 plays earlier to pop (its due batch + 3) and only then writes the event (`set_freq` first). Such
  a wait holds the whole core, so from the first one on the planner budgets the kernel's clock —
  PUSH_MARGIN = 300 batches (0.61 µs) for a wait that waits and its event, PUSH_COST = 60 batches
  per further event — and refuses the schedule when a play would be pushed with less than LEAD
  (96 batches) to its due. In short: **no 9 plays of one channel within 399 batches (0.81 µs), and
  a long burst of tightly spaced plays behind such a wait may be refused too**; beyond that a
  channel may play any number of pulses in one run. The budget comes from co-sim
  (`sim/cosim2q_queue.py`): the kernel sustained 25 plays at 50 batches per play and failed at 36.5;
  the model keeps ~15 % over that. Long pulses split internally, each chunk one play: ≈ 16 k-batch
  chunks on a channel that also has mid-batch pulses; up to 65 535 batches (the duration field) per
  chunk otherwise. Capture fillers (`fill_gaps`) are plays too.
* Two pulses on one channel may not overlap in time (a channel plays one pulse at a time).

## IQ readout rules

* The demod gate's window is the integration window; **≤ 16 384 batches ≈ 33 µs** (the RTL
  accumulator's no-overflow bound).
* Reading a result **halts that core** until the window's integral settles, so an event on
  the same core may start at or before the demod window's START, or at least **144 batches ≈
  293 ns** after its END (`READOUT_LEAD 48` + push margin `LEAD 96`); anything starting in
  between is refused. Other cores run their own kernels and are not affected.
* Results return per gate, in gate order: the sign bit `res`, and the 32-bit `real`/`imag`
  integrals. `phase` on the demod `set()` rotates the returned IQ by that many turns, exact
  to the 16-bit phase register (0.0055°).
* There is no golden model of the demod pipeline; it was verified with a ratio oracle
  (hardware/host integrals of the same raw trace constant to 0.01 % / 0.004° across phase
  cases, in co-sim and on the board, 2026-08-30 / 09-03 bench records).

## The errors, and what to do (the common ones — every check raises a specific message)

| error (abbreviated) | meaning | fix |
|---|---|---|
| `pulses overlap — one ends at ..., the next starts at ...` | two pulses on one channel overlap (remember trailing edges round to whole batches) | separate them |
| `queued plays at batches A and B start closer than 3 batches` | the II=3 spacing rule | ≥ 3 batches (6.1 ns) between play starts; avoid 1–2-batch pulses/gaps |
| `the play at batch N (event k) can only be pushed after the queue entry from batch M pops` | 9 plays (pulses + fillers + chunks) of one channel within ~400 batches, or a dense burst of plays behind such a wait | space or merge events, or split the run |
| `adc gate of N batches (snapped outward) exceeds rob_depth 65536` | window > trace memory | shorten the gate |
| `N gates on adcK in one run` | more than one gate on the same trace | one gate per trace per run |
| `this timeline spans hardware cores [...] but the build has no shared run origin` | a multi-core timeline on a build without `run_origin` (the one-core bundles) | use a `run_origin` build (`rfsoc4x2-2q-fine`) |
| `cores disagree on the run origin`, `... pushed only N batches before it was due` | the post-run telemetry check failed: the run is invalid (hardware fault or an overlong kernel setup) | report it; do not trust that run |
| `schedule horizon of N batches exceeds the 32-bit batch clock's safe range` | a timeline longer than 2.18 s | shorten the run |
| `event at batch B starts within 144 batches of the readout window ending at E` | too close after a demod window (the core is halted there) | space the sequence out |
| `integration window of N batches exceeds the accumulator's no-overflow contract` | window too long | ≤ 33 µs per window |
| `dds channel N does not exist: a K-core build has M dds channels` | bad dds index (2k = core k's gate, 2k+1 its readout; IQ readout is `demod`, never a dds) | pick 0..M-1 |
| `adc K does not exist`, `demod K does not exist` | bad core index | 0..qubit_num-1 |
| `this multi-core build has ONE shared trace (rob_per_core off)` | `adc` on a multi-core build without per-core traces (e.g. `sim-2q`) | use a `rob_per_core` build |
| `... .pulse() before ... .set()` | pulse with no tone set on the channel | `set()` first |
| `amplitude ... outside [0, 1]`, `unknown phase mode ...` | bad `set()` arguments | fix the value |
| `PHASE_MODE_CONTINUOUS continues a phase that does not exist yet` | first `set()` on a channel is CONTINUOUS | start with TRACKING or ABSOLUTE |
| `chN pulse at batches ... overlaps the adc gate ... boundary or follows it` | a readout-drive pulse crossing or after its gate would merge with or restart the recording | keep the readout-drive pulses inside the gate, or ended ≥ 1 batch before it |
| `pulse at batch A is squeezed to nothing by the next pulse at batch B` | sub-batch pulses packed tighter than the envelope grid | space them |
| `envelope RAM exhausted`, `no full envelope lines left to chunk into` | too many distinct mid-batch edge patterns / not enough envelope memory | reuse edge offsets or shorten |
| `no trace: run() the sequence first`, `no readout results ...`, `all N queued results already fetched` | fetch before/beyond what the last run produced | run first / fetch once per gate |

Every limit in this file is enforced at planning time — before the kernel executes on the
hardware. (`run_experiment` does open the board/co-sim connection first, to read the build's
parameters; but nothing plays unless every check passed.)
