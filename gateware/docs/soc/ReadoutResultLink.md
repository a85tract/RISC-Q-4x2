# ReadoutResultLink — the readout result up-path

**Source:** `src/riscq/soc/link/ReadoutResultLink.scala` · **Package:** `riscq.soc.link` ·
**Type:** Bundle + source helper (`object ReadoutResultLink`) + `ReadoutResultSink` (`Area`)

The link's **only** return path: the readout decoder's settled result travels **upstream** (DSP → core)
on a narrow posted `Flow` into a near-core register the CPU polls **locally**. This makes the halting `res`
read a short local arc instead of a long bus round-trip, so the core can still be floorplanned far from the
converters. It is the up-direction counterpart to the down-link ([RfLinkBridge](RfLinkBridge.md) /
[RfLink](RfLink.md)).

## Role in the system

```
  ReadoutDecoder ──res.valid level──▶ source() ── Flow(ReadoutResult) ── linkPipe ──▶ ReadoutResultSink ──▶ CPU
        (DSP region)                                  posted, up, no ack            (core region, local halt)
```

`ReadoutResult` carries the decoder's integrated point: the 1-bit discrimination `res` plus the integrated
I/Q (`real`/`imag`, `accWidth` bits each). See [ReadoutDecoder](../dsp/ReadoutDecoder.md) for what those
values mean.

## Why a posted up-`Flow`, not a bus read

The decoder result is the one CPU read that genuinely lives at the converter edge (the other RF reads —
`time`, `waitTimeCmp`, `fromHost` — are made core-local; see [ControlMemMaps](ControlMemMaps.md)). A
back-pressured halting read across a long bus would stall the CPU there-and-back for the whole round-trip,
defeating the point of moving the core away. Instead the decoder **pushes** the result up once per window
and the CPU halts on a **local** copy: the software contract (`res` read blocks until the integral settles)
is identical, but the halt is a short local arc. Only this one bundle ever travels back across the gap —
[ARCH](ARCH.md) §5.4.

## The two ends

### `ReadoutResultLink.source(...)` (DSP side)

Forwards the decoder's `res.valid` **as a level** (not an edge) with the current `res`/`real`/`imag`. The
carrier-triggered decoder already shapes that level exactly right — high from a window's settle until the
next window's `winStart` clears it, i.e. **low exactly while a fresh window integrates** — so the sink can
mirror it directly, with no edge-detect and no stale-beat bookkeeping.

### `ReadoutResultSink(...)` (core side)

**Mirrors** the decoder's `res.valid` level (through the link) into its own `valid` and latches
`{res, real, imag}` while that level is high. Its `mapping(factory)` contributes the readout result's read
map to a core-local [MemMapFiber](MemMapFiber.md): `res`@4 (HALTS the read locally until `valid`), `real`@8,
`imag`@12. There is **no arm and no consume** — a `res` read does not clear `valid`, so it is **idempotent**
(see [specs/new-readout-decoder](../../specs/new-readout-decoder/README.md) §2.4):

- **Halt.** Because the mirrored level is low exactly while a fresh window integrates, a `res` read that
  races a new window halts until it settles — the same local-halt contract as before, now with no arm.
- **Freshness is a software timing contract**, not a hardware clear. The level holds the *previous*
  window's result high through the `LEAD` gap between a `play` and the window opening, so software waits
  past the window's opening (`wait_until(now ≥ startTime + RQ_RO_LEAD)`) before reading — past `winStart`
  the stale level has dropped, so the halting read can only return the new window. `real`/`imag` are
  non-consuming latches read after `res`.

## Latency / timing

The up-path adds `linkPipe` plain `RegNext` stages (default 4), absorbed like the down-link. The sink read
and its halt are local single-cycle arcs. Pipe depth is a floorplan knob, never a timing-closure one.

## Configuration

`ReadoutResultSink(accWidth, resAddr = 4, realAddr = 8, imagAddr = 12)` — `accWidth` must match the
decoder's accumulator width; the three offsets keep the CPU-visible map reference-compatible.

## Verification

`riscq.soc.sim.ReadoutResultLinkSim` runs a real [ReadoutDecoder](../dsp/ReadoutDecoder.md) integrating a
tone over two carrier-`Flow` windows; `res.valid` is forwarded up as a level through `source`, pipelined,
and mirrored by the sink. The CPU (a `MasterAgent`) reads `res`/`real`/`imag` from the sink's **local** map
— the `res` read halts locally until the integral settles — and the values are checked **bit-exact** vs the
windowed-demod golden. The freshness contract is proven directly: a `res` read issued before the first
window settles halts and returns A; re-reading with no new window returns A again (**idempotent**); the
mirrored `valid` drops while window B integrates and rises on its settle; a read after waiting past B's
opening returns B (not the stale A). Swept over `linkPipe ∈ {0, 4, 16}` to show distance-tolerance.

```bash
mill runMain riscq.soc.sim.ReadoutResultLinkSim
```

## Related

- [ReadoutDecoder](../dsp/ReadoutDecoder.md) — produces `{res, real, imag}` and `res.valid`.
- [RfLink](RfLink.md) / [RfLinkBridge](RfLinkBridge.md) — the matching down-link.
- [ControlMemMaps](ControlMemMaps.md) — the other (core-local) RF reads.
- [ARCH](ARCH.md) §5.4 — the read-path split.
