# RiscqRfWithPulseTableFiber — one qubit core

**Source:** `src/riscq/soc/RiscqRfWithPulseTableFiber.scala` · **Package:** `riscq.soc` · **Type:** Area
(case class, instantiated per qubit by [`PulseTableSoc`](PulseTableSoc.md))

One qubit core: a RISC-V control core + its private I/D RAM + the CPU control block, plus a converter-edge
DSP datapath (two pulse drives, an envelope-shaped demod carrier, a readout decoder) reached over the
**narrow posted link**.
`PulseTableSoc` instantiates `qubitNum` of these. Read [`ARCH.md`](ARCH.md) for the link rationale.

## Structure

After the registered-boundary floorplan carve-out this is a **thin shell** around two pieces split at the
posted link's already-registered seam:

```
   ┌──────────── RiscvSoc (hard Component, pinned to X0) ────────────┐
   │  RiscqFiber core + I/D RAM + control block (time/fromHost)      │
   │  + RfLinkBridge (acks CPU stores locally) + ReadoutResultSink   │
   └──── cmd: Flow(RfCmd) ─┬────────────────────────  resultIn ◀─────┘
                           │ getPipe(linkPipe)                 │ getPipe(linkPipe)
        ┌──────────────────┴── posted (dspCd, datapath in X1–X5) ────┴──────────┐
        │  RfLink.demux ─┬─ gateChannel   (PulseDriveChannel, pulseNum=8) ─▶ DAC │
        │                ├─ roChannel     (PulseDriveChannel, pulseNum=1) ─▶ DAC │
        │                └─ demodChannel  (DemodChannel) ─▶ carrier Flow ┐       │
        │                   decoder       (ReadoutDecoder) ◀─ adc ──────┴▶ result up
        │                   (triggered by the carrier valid — no CPU sub-window)│
        │  pulseMemFiber / readoutMemFiber / demodMemFiber  (host envelope BRAMs)│
        └────────────────────────────────────────────────────────────────────────┘
```

- **`riscvSoc`** — the [`RiscvSoc`](RiscvSoc.md) hard `Component`: the timing-critical RISC-V core
  ([`RiscqFiber`](RiscqFiber.md)) + real BRAM/UltraRAM I/D RAM + the CPU-mapped control block + the
  [`RfLinkBridge`](RfLinkBridge.md) (which acks every CPU RF store locally in one cycle) + the
  [`ReadoutResultSink`](ReadoutResultLink.md). Its narrow registered I/O is `time`/`fromHost` in,
  `cmd: Flow(RfCmd)` out, `resultIn: Flow(ReadoutResult)` in (plus `iLoad`/`dTap` slave-IO). The
  `0x10000` RF window and `0x80000000` data-RAM maps are all inside it.
- **The shell** applies the `linkPipe` `RegNext` stages each way (`getPipe(riscvSoc.cmd, linkPipe)` down,
  `getPipe(upSrc, linkPipe)` up) and demuxes `cmd` to the per-generator channels. Everything past the
  pipe — the demux, the channels, the decoder, the envelope BRAMs, dac/adc — lives **here** (the parent),
  not in `RiscvSoc`, so the core can be floorplanned far from the converters.

### The RF datapath (`posted`, dspCd)

The piped `cmd` stream is fanned by [`RfLink.demux`](RfLink.md) to three 0x10000-wide sub-windows of the
RF address space (the fourth, `0x30000`, is reserved/unmapped — the decoder has no CPU-facing registers):

| Window | Channel | What |
|---|---|---|
| `0x00000` | `gateChannel` | [`PulseDriveChannel`](RfChannels.md), `pulseNum = 8` → gate-drive DAC |
| `0x10000` | `roChannel` | `PulseDriveChannel`, `pulseNum = 1` → readout-drive DAC |
| `0x20000` | `demodChannel` | [`DemodChannel`](RfChannels.md): a drive channel (`fire`@0 / `freq`@4 / `table`@0x10.. / `startTime`@0x4100) whose complex `carrier` Flow feeds **and triggers** the decoder — firing it is the readout |

The `gateChannel`/`roChannel` `PulseDriveChannel`s and the `demodChannel` `DemodChannel` all wrap a
[`PulseParamBuffer`](PulseParamBuffer.md) (DSP-side register file driven by the demuxed `Flow(RfCmd)`) + a
[`PulseGenerator`](../dsp/PulseGenerator.md). The buffer owns the per-buffer `startTime` (the software
contract from [`ARCH.md`](ARCH.md)); `io.timeBcast` takes the shared `time` broadcast. The two drive
generators run with `realOutput = true` — the DAC carries only the real lane, so the imaginary cone is tied
off inside the generator and synthesis prunes the dead DSPs (see [`SOC_TIPS.md`](SOC_TIPS.md) §7.7). The
demod runs with `realOutput = false` at `adcBatch` lanes: its complex `carrier` = the raw generator pulse
`Flow` (no `dcOffset` stage) feeds `decoder.io.carrier` through one register stage, so a scheduled,
envelope-weighted matched filter replaces the old free-running LO — and the carrier's `valid` **is** the
decoder's integration window ([specs/new-readout-decoder](../../specs/new-readout-decoder/README.md)): the
envelope batch `env[base+i]` weights integrated batch `startTime+i` by construction (verified bit-exact by
`DemodDecoderSim`).

The decoder result is latched on integration done and emitted on a [`ReadoutResultLink`](ReadoutResultLink.md)
up-`Flow`, piped `linkPipe` stages back into `RiscvSoc`'s core-local sink — so the CPU's halting `res`
read is a short local arc, not a round-trip across the gap.

### Host-writable complex pulse-envelope RAM

Three [`BramWriteFiber`](BramWriteFiber.md) banks (`pulseMemFiber` for the gate drive, `readoutMemFiber`
for the readout drive, `demodMemFiber` for the demod carrier) — **write-only** on the host side, since the
host only ever loads them (no read-back), which drops the read-reorder buffer and lets each 32-bit host
beat steer straight into a sub-word lane of the wide line (no fabric `WidthAdapter`). They are
host-written through the host→dsp CDC. The drive envelope line is
**512-bit complex** (`batchSize·2·w` = 16·2·16); the demod line is `adcBatch·2·w` = 128-bit (the ADC batch
is 4 lanes). Each bank stores an **interpolated** line — `gateEnvWidth = 512/gateInterp` (128 at interp 4),
`readoutEnvWidth = 512/readoutInterp` (32 at interp 16), `demodEnvWidth = 128/demodInterp` (32 at interp 4)
— to shrink the widest BRAM banks. On read, `expandEnv` reconstructs the full per-lane batch from the
interpolated line (each output lane reads the `m/interp`-th stored sample) over the bank's own lane count
(`batchSize` for the drives, `adcBatch` for the demod). Each interp factor must divide its bank's lane count.

### Host image load & test tap

`iMemPortArb` re-exposes `RiscvSoc`'s `iLoad` slave-IO as a fabric node so the toplevel's host AXI fabric
loads the program/data image into the BRAM slow port; this node carries the **host→dsp clock crossing**,
which lands on the fabric arc *outside* the hard `RiscvSoc` Component (so it stays out of the per-core
pblock). The fabric cannot cross a hard Component boundary, so a `bridgeLoad` helper wires the (narrower)
host master onto `RiscvSoc`'s generous fixed `iLoad` param by resizing the per-top-varying fields — which
keeps `RiscvSoc` byte-identical across tops (the floorplan-transfer requirement, see [`RiscvSoc`](RiscvSoc.md)).
`dMemTap`/`dMemPortDec` mirror this for the `withTestTap` sim path (null in the real SoC).

## Exported handles

For the rest of the SoC: `gatePulse` / `readoutPulse` (the `Flow(ComplexBatch)` DAC outputs), the two
envelope `MemReadPort`s, `decoderRd` (the decoder, for `robs`), `dac`/`adc` (real lane), and `startTime`
(the gate buffer's per-buffer schedule value, observed by sims).

## Key parameters

- **`plugins`** — the RISC-V plugin config (from `PulseTableSoc.coreParam`).
- **`linkPipe`** (default 4) — per-direction `RegNext` depth of the posted link.
- **`gatePulseNum`** (default 8) — gate-drive pulse-table depth; readout drive is fixed at 1.
- **`readoutInterp` / `gateInterp`** — envelope interpolation factors (must divide `batchSize`).
- **`readoutAccWidth`** — the decoder's accumulator width (32 = one-word readback; the window-length
  bound is the decoder's `maxWinLog2` no-overflow contract, enforced in software on the demod `dur`).
- **`time` / `fromHost`** — the shared batch-time replica + host mailbox, passed in from the toplevel.

## Verification

No standalone sim — verified through the toplevel sims (`PulseTableSocSim`, `PulseTableSocCpuSim`) and the
posted-link building-block sims (`PulseParamBufferSim`, `RfLinkBridgeSim`, `ReadoutResultLinkSim`). See
[`PulseTableSoc`](PulseTableSoc.md) for the commands.

## Related

- [`RiscvSoc`](RiscvSoc.md) — the hard, registered-boundary core unit this wraps.
- [`ARCH.md`](ARCH.md) — the posted-link architecture (why the split is at the registered seam).
- Posted link: [`RfLinkBridge`](RfLinkBridge.md) · [`RfLink`](RfLink.md) · [`ReadoutResultLink`](ReadoutResultLink.md).
- Datapath: [`RfChannels`](RfChannels.md) · [`PulseParamBuffer`](PulseParamBuffer.md) ·
  [`PulseGenerator`](../dsp/PulseGenerator.md) · [`ReadoutDecoder`](../dsp/ReadoutDecoder.md).
