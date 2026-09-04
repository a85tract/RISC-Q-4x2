# The ARTIQ-shaped interface — `riscq.artiq_compat`

`from riscq.artiq_compat import *` gives you the names an ARTIQ user already knows:

`EnvExperiment`, `kernel`, `run_experiment`, `parallel`, `sequential`,
`delay`, `delay_mu`, `now_mu`, `at_mu`,
`s ms us ns Hz kHz MHz GHz`,
`PHASE_MODE_CONTINUOUS`, `PHASE_MODE_ABSOLUTE`, `PHASE_MODE_TRACKING`.

**The honest contract** (also in the module docstring): this is an *ARTIQ-syntax restricted
subset*, not ARTIQ's compiler. Your `run()` is executed once as ordinary Python to *record* the
pulse schedule; the RISC-V kernel is generated from that record and then runs on the hardware.
Two visible consequences:

1. The whole `run()` is **one kernel**, executed after `run()` returns. You cannot interleave
   host I/O with hardware execution inside `run()` (that is why the `@kernel` mark on `run` is
   *required* — it states this). Read results in `analyze()`.
2. `with parallel:` cannot see bare statements the way ARTIQ's compiler does, so **every
   parallel arm must be wrapped in `with sequential:`** (ARTIQ's own recommended style). A
   timeline operation placed directly under `parallel` raises
   `RuntimeError: ... wrap each parallel arm in 'with sequential:'` instead of silently
   serializing.

---

## The experiment class

```python
class MyExp(EnvExperiment):
    def build(self): ...      # declare devices:  self.setattr_device("name")
    def prepare(self): ...    # optional host-side preparation
    @kernel
    def run(self): ...        # the pulse program (recorded, then executed as one kernel)
    def analyze(self): ...    # read results; runs ONLY after a successful hardware run
```

* `setattr_device(name)` — binds `self.name` to the device declared under `name` in the
  device db. Unknown names raise `KeyError` at build time.
* `@kernel` — a marker (no compilation happens at decoration time). `run_experiment` refuses
  an experiment whose `run` is not marked.

## `run_experiment(exp_cls, device_db, workdir="artiq_compat_work", doc="") -> exp`

The `artiq_run` role: validate the db, open the driver, instantiate, `build`, `prepare`,
`run` (records), compile + execute on the hardware, then `analyze`. Returns the experiment
instance; the raw `riscq.artiqapi.RunResult` is at `exp.last_result`, and the board's report on
the loaded bundle (`mts_result`, `mts_latencies`, `dsp_mhz`, `xsa_sha`; `None` in co-sim) at
`exp.board_info`. The driver is closed on every path (success or failure); `analyze` is skipped if
the hardware run failed.

## The device db

```python
device_db = {
    "core":     {"type": "board", "host": "192.168.3.1", "bundle": "rfsoc4x2-2q-fine"},
    # or:       {"type": "cosim", "config": "gateware/configs/rfsoc4x2-2q-fine.json",
    #            "build": "sim/build/rfsoc4x2-2q-fine",
    #            "model": {"kind": "multi", "models": [           # the two loopback "cables"
    #                {"kind": "loopback", "src": 1, "dst": 1, "gain": 0.9, "delay": 5},
    #                {"kind": "loopback", "src": 0, "dst": 0, "gain": 0.9, "delay": 5}]}},
    "gate_dds": {"type": "dds", "channel": 0},     # core 0's gate drive     -> DAC_A
    "ro_dds":   {"type": "dds", "channel": 1},     # core 0's readout drive  -> DAC_A (the recorded channel)
    "dds_b0":   {"type": "dds", "channel": 2},     # core 1's gate drive     -> DAC_B
    "dds_b1":   {"type": "dds", "channel": 3},     # core 1's readout drive  -> DAC_B
    "adc":      {"type": "adc"},                   # raw trace of ADC_A (= "channel": 0)
    "adc_b":    {"type": "adc", "channel": 1},     # raw trace of ADC_B
    "dm":       {"type": "demod"},                 # IQ readout of core 0 (= "channel": 0)
    "readout":  "ro_dds",                          # a string is an ALIAS (same device object)
}
```

`dds` needs `channel`; `adc` and `demod` take an optional `channel` = the hardware core (default
0). On the one-core bundles (`rfsoc4x2-1q-fine`, …) only dds 0/1, adc 0 and demod 0 exist.

Validated eagerly, before anything runs: a `core` entry is required and must itself be the
`board`/`cosim` dict (not an alias); `board`/`cosim` may appear only as (the target of)
`core`; each type's required keys are checked; alias chains are followed and cycles refused.
An alias and its target resolve to the **same** device object (ARTIQ semantics), tone state
included.

This is a RISC-Q device db — smaller than ARTIQ's `type=local/module/class` schema on purpose.

## Devices

### `dds` — one drive channel (`artiq.coredevice.ad9910` shape)

```python
self.ro_dds.set(frequency, phase=0.0, amplitude=1.0, phase_mode=None)
self.ro_dds.set_phase_mode(phase_mode)    # sets the default mode for later set() calls
self.ro_dds.sw.pulse(duration)            # play for `duration` at the cursor, advance cursor
self.ro_dds.sw.pulse_mu(duration_mu)
self.ro_dds.pulse(duration)               # alias of sw.pulse
```

* `frequency` in Hz (`82.0*MHz`); `phase` in **turns** (0.25 = 90°), exact to the 16-bit phase
  register (0.0055°); `amplitude` in [0, 1].
* `phase_mode` — ARTIQ's AD9910 semantics (default `TRACKING`); see
  [hardware-contract.md](hardware-contract.md#phase-modes).
* `sw.on()/off()` are **not** implemented (a free-running switch has no end time to schedule);
  use `pulse`.
* channels: 2k = hardware core k's gate drive, 2k+1 its readout drive (`rfsoc4x2-2q-fine`: 0/1
  on DAC_A, 2/3 on DAC_B; a one-core build has 0/1). The demod carrier is never a `dds` (use
  `demod`); out-of-range indices are refused with the count.
* several cores, one timeline: what you write under one `with parallel:` starts together on
  every DAC — the kernels share one hardware time origin (see
  [hardware-contract.md](hardware-contract.md#channels-cores-and-the-one-timeline), including
  the note on the analog offset between the DAC connectors).

### `adc` — raw trace capture (the `gate_rising → count` idiom)

```python
self.adc.gate(duration)        # record the raw ADC stream over [now, now+duration); advances
self.adc.gate_mu(duration_mu)  #   the cursor; returns the cursor after the gate
self.adc.fetch_trace()         # after the run: numpy int array (fs = exp.last_result.fs,
                               #   1.96608 GS/s on this build)
```

* `"channel": k` selects hardware core k's trace: ADC_A (recorded while dds 1 fires) for
  k = 0, ADC_B (while dds 3 fires) for k = 1 on `rfsoc4x2-2q-fine`.
* **One gate per trace per run.** The gate must fit the trace memory (65 536 batches ≈ 133 µs
  on this build; the *outward-snapped* window is what is checked).
* A trace records **only while its core's readout-drive channel is playing**; the scheduler
  automatically inserts amplitude-0 filler pulses on that channel over every silent stretch of
  the gate (lead-in, holes, tail), so you do not have to keep it busy yourself.
* Typical shape: the gate sits in its own `with sequential:` arm of an outer `with parallel:`
  so it spans the pulses it should record.

### `demod` — hardware IQ readout (the `EdgeCounter.gate → fetch_count` idiom)

```python
self.dm.set(frequency, phase=0.0)   # the demod LO: RF frequency in Hz, phase in turns
self.dm.set_phase_mode(phase_mode)  # default mode for later set() calls
self.dm.gate(duration)              # integrate over [now, now+duration): this IS the readout
self.dm.gate_mu(duration_mu)        #   (both return the cursor after the window, like adc.gate)
self.dm.fetch_iq()                  # after the run, one record per gate, in gate order:
                                    #   .res (hardware sign bit), .real, .imag (int integrals),
                                    #   .iq (complex)
```

* The integration window is the gate duration; windows are batch-granular and must be
  ≤ 16 384 batches (≈ 33 µs) — the RTL's no-overflow bound.
* `phase` rotates the returned IQ by the requested turns, exact to the 16-bit phase
  register (0.0055°).
* `"channel": k` selects hardware core k's readout (its ADC); default 0.
* Scheduling rule: reading a result stalls that core until the integral settles, so an event
  on the same core may start at or before a demod window's START, or ≥ 144 batches (≈ 293 ns)
  after its END — anything starting in between is refused by the planner.
* `fetch_iq()` walks the queued results of the **latest** run and restarts automatically on a
  new run. Calling it more times than there were gates raises.
* The 4× ADC-rate LO word, the phase-register law and the result registers are all internal —
  you never see them.

### `core`

```python
self.core.reset()                 # drop the recorded timeline (call at the TOP of run();
                                  #   refused inside parallel/sequential)
self.core.seconds_to_mu(t); self.core.mu_to_seconds(mu)
```

`mu` is one DAC sample = 127.157 ps on this build. Unlike ARTIQ's `core.reset()` this touches
no hardware — there is no RTIO to reset; it only clears the recording.

## Timeline verbs

```python
delay(5*us); delay_mu(64)      # advance the cursor
at_mu(t); now_mu()             # jump / read the cursor (absolute mu)
```

Bare (no core argument) — they act on the experiment being recorded, and raise
`RuntimeError: no experiment is running` outside one.

## Common errors

All but the fetch errors are raised while recording/planning — before the kernel executes
(`run_experiment` does connect to the board/co-sim first, to read the build's parameters);
the fetch errors occur after the run, when reading results.

| message (abbreviated) | cause |
|---|---|
| `wrap each parallel arm in 'with sequential:'` | timeline op directly under `parallel` |
| `run must be decorated with @kernel` | unmarked `run()` |
| `no experiment is running` | bare `delay`/`parallel`… outside `run_experiment` |
| `core.reset() inside 'with parallel:'/'with sequential:'` | reset not at top level |
| `device_db ...` (several) | schema violation, unknown type/name, alias cycle |
| `... .pulse() before ... .set()` | a pulse with no tone set on that channel |
| `amplitude ... outside [0, 1]`, `unknown phase mode` | bad `set()` arguments |
| `PHASE_MODE_CONTINUOUS continues a phase that does not exist yet` | the FIRST `set()` on a channel must be TRACKING or ABSOLUTE |
| `no trace: run() the sequence first`, `no readout results`, `all N queued results already fetched` | fetch before/beyond what the last run produced |
| `chN pulse ... overlaps the adc gate ... boundary or follows it`, `squeezed to nothing`, `envelope RAM exhausted` | recording/envelope limits — details in [hardware-contract.md](hardware-contract.md) |
| `dds channel N does not exist ...`, `adc K does not exist`, `demod K does not exist` | bad channel / core index for this build |
| `... no shared run origin`, `cores disagree on the run origin`, `pushed only N batches before it was due` | multi-core runs: wrong build, or the post-run telemetry check failed (the run is invalid) |
| scheduling limits (`pulses overlap`, `closer than 3 batches`, queue depth, gate size, readout guard, …) | the fuller table is in [hardware-contract.md](hardware-contract.md) |
