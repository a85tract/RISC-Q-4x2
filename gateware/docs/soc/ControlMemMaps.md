# ControlMemMaps — the per-core Time / Host control block

**Source:** `src/riscq/soc/rf/ControlMemMaps.scala` · **Package:** `riscq.soc.rf` · **Type:** `Area`s
contributing to a `MemMapFiber` (`TimeMemMap`, `HostMemMap`)

The small CPU-mapped control surface every qubit core sees: the wall-clock `time`, the wait-compare the
scheduling software spins on, and the host→CPU mailbox. These are the RF reads that are kept **core-local**
(no link crossing), the counterpart to the readout result that does cross
([ReadoutResultLink](ReadoutResultLink.md)). Ported from the RISC-Q reference (`riscq.soc.Misc`).

## Role in the system

Both are `Area`s whose `mapping(factory)` is contributed to a core-local [MemMapFiber](MemMapFiber.md) off
the CPU `dBus` decode (see [RiscqRfWithPulseTableFiber](RiscqRfWithPulseTableFiber.md) / `RiscvSoc`). The
batch `time` arrives on the down-link's broadcast (a pipelined copy of the DSP-side `refTime + timeOffset`
counter, [PulseTableSoc](PulseTableSoc.md)); everything the CPU reads here is then a **short local arc**.

## `TimeMemMap` — batch time and the scheduling spin-wait

- `time`@0xbff8 — the current batch time, a registered copy of the external `time` input.
- `timeCmp`@0x4000 — a software-written compare value (read/write).
- `waitTimeCmp`@0x4008 — a read that **halts the bus** until `time + delay ≥ timeCmp` (`delay = 3`). This
  is the spin-wait the control software blocks on: the CPU stalls until the wall clock catches up to a
  scheduled instant, then proceeds to fire pulses.

Why a local halt: the halt arc must be short, so the `time` net is pipelined down to the core region and the
compare is done core-locally — the CPU never blocks on a long bus round-trip. The few cycles of `delay`
slack are harmless because real-time precision lives in the DSP `TimedQueue` (see
[PulseGenerator](../dsp/PulseGenerator.md)), not in the CPU wait. The whole posted-link design relies on
`time` being a pipelined local copy that reads `dspTime − D`; the lead-time contract (program `startTime`
far enough ahead) absorbs the constant `D` — see [PulseParamBuffer](PulseParamBuffer.md) and
[ARCH](ARCH.md) §5.3.

## `HostMemMap` — the host mailbox

- `fromHost`@0x2000 — a read returns the host-written `fromHost` word (registered for the host→dsp domain
  crossing upstream). The control software polls it for commands from the host.

## What is *not* here

- **`startTime`** is no longer a control-block register: under the posted link it is **per-buffer**, written
  down the `RfCmd` stream into each [PulseParamBuffer](PulseParamBuffer.md) (`@0x4100` within the RF
  window). The control-map sim adds a local `startTime` reg only to exercise a write; production
  `startTime` rides the link.
- **`res`/`real`/`imag`** (readout result) are served by the core-local
  [ReadoutResultSink](ReadoutResultLink.md), fed by the up-`Flow` — not by this block.

## Verification

`riscq.soc.sim.ControlMapFiberSim` drives a [MemMapFiber](MemMapFiber.md) carrying `TimeMemMap` +
`HostMemMap` (+ a `startTime` write) over TileLink with a `MasterAgent`, and asserts: `time` reads the
registered external time; `timeCmp` read/writes; `startTime` write-only; `fromHost` reads the mailbox; and
`waitTimeCmp`@0x4008 **halts** until `time + 3 ≥ timeCmp` (released only once the externally-ramped time
catches up).

```bash
mill runMain riscq.soc.sim.ControlMapFiberSim
```

## Related

- [MemMapFiber](MemMapFiber.md) — the TileLink slave the mappings are contributed to.
- [ReadoutResultLink](ReadoutResultLink.md) — the other (crossing) RF read path.
- [PulseParamBuffer](PulseParamBuffer.md) — where `startTime` now lives.
- [RiscqRfWithPulseTableFiber](RiscqRfWithPulseTableFiber.md) / [PulseTableSoc](PulseTableSoc.md) — the core and the `time` source.
