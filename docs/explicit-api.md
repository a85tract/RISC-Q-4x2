# The explicit layer — `riscq.artiqapi`

Everything `artiq_compat` does is sugar over this module. Use it directly when you want the
timeline as a first-class object (scripts, tests, tools) instead of an experiment class. Same
semantics, one difference in shape: the timeline lives in an explicit `Core` object that every
call takes.

```python
from riscq import artiqapi as A
from riscq.map import SocMap, SocParams

m = SocMap(SocParams.from_json(open("gateware/configs/rfsoc4x2-1q-fine.json").read()))
core = A.Core(m)
ro, adc = A.DDSChannel(core, 1, "readout"), A.ADCChannel(core)

with A.parallel(core):
    with A.branch(core):                       # `branch` = one parallel arm
        ro.set(82.0*MHz, phase=0.25, amplitude=0.4)
        ro.sw.pulse(20*A.us)
    with A.branch(core):
        adc.gate(30*A.us)

res = A.run(drv, core, "workdir")              # drv: board or co-sim driver
trace = adc.fetch_trace()
```

## `Core(soc_map)`

The timeline cursor and machine unit (`mu` = one DAC sample = 127.157 ps here).

* `now_mu() / at_mu(t) / delay_mu(dt) / delay(dt) / at_s(t)` — cursor ops (also available as
  module-level verbs taking the core: `A.delay(core, 5*A.us)` …).
* `seconds_to_mu(t) / mu_to_seconds(mu)`.
* `clear()` — drop the recorded sequence AND `last_result` (fetches then raise until the
  next run; this is what `artiq_compat`'s `core.reset()` calls).
* `last_result` — the `RunResult` of the latest `A.run` (what `fetch_trace`/`fetch_iq` read).

## Timing blocks

* `with A.parallel(core):` — every `with A.branch(core):` inside starts at the same instant;
  afterwards the cursor is the **latest** branch end.
* `with A.branch(core):` — one arm; outside a `parallel` it is a no-op passthrough.
* `with A.sequential(core):` — explicit "statements advance the cursor" (the default).

Note the naming: this layer says `branch` where `artiq_compat` (and ARTIQ) say `sequential`
inside `parallel`. Unlike the compat layer there is **no bare-statement guard** here — an event
issued directly under `parallel` simply runs at the block-start cursor like any other statement,
so wrap arms deliberately.

## Channels

* `A.DDSChannel(core, index, name=None)` — drive channel. `set(frequency, phase=0.0,
  amplitude=1.0, phase_mode=None)`, `set_phase_mode(phase_mode)`,
  `sw.pulse(duration)` / `sw.pulse_mu(duration_mu)`.
  Index is validated (0 = gate, 1 = readout on this build); index 2 (the demod carrier) is
  refused — use `DemodChannel`.
* `A.ADCChannel(core, name="adc")` — `gate(duration)` / `gate_mu(duration_mu)` (records the
  window, advances the cursor, returns the cursor), `fetch_trace()`.
* `A.DemodChannel(core, name="demod")` — `set(frequency, phase=0.0, amplitude=1.0,
  phase_mode=None)`, `set_phase_mode(phase_mode)`, `gate(duration)` / `gate_mu(duration_mu)`
  (the integration window IS the readout; both return the cursor), `fetch_iq()`.

All limits and semantics are the same as through `artiq_compat` — see
[hardware-contract.md](hardware-contract.md).

## Planning and inspection

```python
sch = A.plan(core)          # snap, allocate, chunk, and CHECK the whole timeline
print(sch.report())         # the honest table: asked vs got, per event
```

Full signature: `A.plan(core, reserved_base=0, max_run=None)`. It is where every scheduling
rule is enforced (overlaps, play spacing, queue depth, gate size, readout guard — the full
list with numbers is in the hardware contract). It FINALIZES the recorded timeline: events are
snapped in place and the gate's amplitude-0 fillers are appended to `core.events` (repeating
`plan`, or `plan` then `A.run`, re-snaps the same finalized state). `report()` RETURNS the
table as a string — one row per event: `start asked / start got / err / dur asked / dur got /
end err / freq / phase / mode` — plus the recording-gate line.

Useful `Schedule` fields: `events` (the snapped `PulseEvent`s, including auto-inserted
amplitude-0 fillers), `chunks` (event index → the `(envelope line, batches)` runs its plays
use), `env_lines` / `first_line` (envelope RAM layout).

## Execution

```python
res = A.run(drv, core, work_dir, doc="", max_run=None)   # -> RunResult
```

Plans, generates + compiles the kernel (into `work_dir`), writes the envelope images, executes,
and reads everything back. `drv` is a co-sim driver (`riscq_sim.cosim.start(config, build)`)
or a board driver (`riscq.driver.remote.RemoteDriver(host)` after `drv.board.load(bundle)`).
`max_run` forces a smaller envelope-chunk size (diagnostics only; it applies to channels
with mid-batch pulses — a channel without mid-batch pulses uses free-running chunks of up to
65 535 batches regardless).

```python
@dataclass
class RunResult:
    schedule: Schedule
    fs: float                        # trace sample rate (1.96608 GS/s here)
    trace: np.ndarray | None = None  # int32 ADC samples over the gate window (None: no gate)
    t: np.ndarray | None = None      # seconds, relative to the gate start
    gate_start_mu: int = 0
    res: np.ndarray = <empty>        # per demod gate, in order: hardware sign bits
    real: np.ndarray = <empty>       # 32-bit integrals
    imag: np.ndarray = <empty>
    # .iq is a @property: real + 1j*imag
```

## Legacy helper

`A.fill_gaps(core, channel, amplitude=0.0) -> int` — manually make a channel fire
continuously by filling the gaps BETWEEN its existing pulses with amplitude-0 pulses; returns
how many it inserted. Superseded for capture purposes: `adc.gate()` inserts its fillers
(including lead-in and tail) automatically; you only need `fill_gaps` if you manage recording
without a gate.
