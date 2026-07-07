# RiscvSoc — the hard per-core unit (registered-boundary floorplan target)

**Source:** `src/riscq/soc/RiscvSoc.scala` · **Package:** `riscq.soc` · **Type:** `Component`

## Purpose

`RiscvSoc` is the **timing-critical core unit** carved out of
[`RiscqRfWithPulseTableFiber`](RiscqRfWithPulseTableFiber.md): a hard SpinalHDL `Component` (i.e. its
own Verilog module, hence a clean per-instance pblock target) holding the RISC-V core, its real BRAM/URAM
instruction+data RAM, the CPU-mapped control block, the posted-link bridge, and the readout-result sink —
**and nothing else**. Everything past the posted link (the converter-edge DSP datapath) stays *out*, in
the parent.

It exists so the floorplan can pin each core into its own clock-region pblock and have that placement
**transfer** between tops: `RiscvSoc` is byte-for-byte identical in the real [`PulseTableSoc`](PulseTableSoc.md)
and in the dummy floorplan harnesses ([floorplan-harnesses](floorplan-harnesses.md)). See
[ARCH](ARCH.md) §6 for the two-region floorplan this enables.

## Role in the system

```
        host time/mailbox          posted RF writes
   time ─┐  fromHost ─┐                 cmd : master Flow(RfCmd) ─────────► DSP datapath (parent)
         ▼            ▼                                          ◄───────── resultIn : slave Flow(ReadoutResult)
  ┌────────────────── RiscvSoc (hard Component, riscqCd) ──────────────────┐
  │  RiscqFiber(core) ── iBus/dBus ─► I/D RAM (Bram|Uram, 2 RW ports)       │
  │                        │                                                │
  │                        ├─► MemMapFiber: TimeMemMap + HostMemMap + sink  │
  │                        └─► RfLinkBridge ─► cmd (out)                     │
  │  ReadoutResultSink ◄── resultIn (in)                                    │
  └── iLoad (MasterBus slave-IO, dsp clock) ◄── host image load (parent CDC)┘
      dTap (MasterBus slave-IO, withTestTap only) ◄── sim RF writes
```

The far side of `cmd` / `resultIn` / `iLoad` is the **only** thing the dummy floorplan top and the real
`PulseTableSoc` differ on. [`RiscqFiber`](RiscqFiber.md) is the bottom of this stack;
[`RiscqRfWithPulseTableFiber`](RiscqRfWithPulseTableFiber.md) is the parent that wraps `RiscvSoc` and
attaches the DSP datapath across the link.

## Design rationale

- **Why a hard `Component`, not just a fiber.** `tilelink.fabric` cannot cross a hard Component
  boundary (it elaborates by walking nodes), so to get a clean per-core Verilog module the only crossings
  must be concrete `tilelink.Bus` / `Flow` IO. That is exactly why the boundary is the **posted link**:
  the link is already narrow, one-way and registered, so making it the module edge costs nothing. The
  host image-load and the sim test-tap, which *do* need to reach the internal fabric, enter as
  `MasterBus` slave-IO ports (see [SOC_TIPS](SOC_TIPS.md) §1).
- **Registered on both sides of the link ⇒ placement transfers.** The bridge inside `RiscvSoc` plus the
  parent's `keepPipe`/`linkPipe` stages mean swapping the dummy anchor for the real datapath cannot
  change a core's *internal* timing — so the fmax measured against a dummy in
  [floorplan-harnesses](floorplan-harnesses.md) is faithful to the real SoC.
- **No clock crossing inside the hard Component.** `RiscvSoc` elaborates in its parent's clock domain
  (the dsp/batch clock); that implicit domain clocks the RAM, the instruction arbiter and the `iLoad`
  slave-IO. `riscqCd` is the *same clock* but with the core reset, and wraps only the CPU + control
  fibers. The host→dsp CDC lives on the parent's fabric arc that drives `iLoad`, keeping it **out** of
  the per-core pblock.
- **Generous, fixed host-load / test-tap masters.** `RiscvSoc.defaultHostLoadM2s` (4-bit source, 32-bit
  address, 256-byte size) is a deliberate *superset* of every parent's host fabric; the parent's
  `iMemLoad` bridge resizes its narrower fields up into this fixed param so `RiscvSoc` stays identical
  across tops (the floorplan-transfer requirement).
- **BRAM vs URAM I/D RAM.** Block RAM by default; UltraRAM with `useUram` (`pipeNum = 1` ⇒ read latency
  `NBPIPE+1 = 2`, matching BRAM + output register), so the fibers' `withOutReg` read-latency assumption
  holds either way. `mem` carries `KEEP_HIERARCHY` so the floorplan can pin it.

## Interface & configuration

Key construction params (`case class RiscvSoc`):

- `plugins`, `riscqCd` — the core's plugin list and its (core-reset) clock domain.
- `memDepth` / `memWidth` / `memOutReg` / `useUram` — the I/D RAM shape and technology.
- `rfAddrWidth` (18) — width of the posted `RfCmd` address (the RF window).
- `readoutAccWidth` (32) — width carried on the readout up-Flow / sink.
- `withTestTap` — add the sim-only `dTap` master into the data-bus decode so a testbench can schedule RF
  writes without a CPU program.

IO: `time`, `fromHost` (in); `cmd` (master `Flow(RfCmd)`); `resultIn` (slave `Flow(ReadoutResult)`);
`iLoad` and optional `dTap` (`MasterBus` slave-IO). The CPU RAM base is `0x80000000`; the RF window,
control block and result-sink addresses are decoded off the data bus (`RfLinkBridge` at `0x10000`, the
control block at `0`, the sink reads at `0x4200/4/8`).

## Verification

`RiscvSoc` is exercised end-to-end as part of the SoC sims — the register-driven sign-off
[`PulseTableSocSim`](PulseTableSoc.md) (drives `dTap`) and the CPU-in-the-loop
[`PulseTableSocCpuSim`](PulseTableSoc.md). Its constituents are unit-checked on their own:
[`RfLinkBridge`](RfLinkBridge.md), [`ReadoutResultLink`](ReadoutResultLink.md),
[`RamOnFabricSim`](RiscqFiber.md).

## Related

[RiscqRfWithPulseTableFiber](RiscqRfWithPulseTableFiber.md) · [PulseTableSoc](PulseTableSoc.md) ·
[RiscqFiber](RiscqFiber.md) · [RfLinkBridge](RfLinkBridge.md) · [ReadoutResultLink](ReadoutResultLink.md) ·
[ControlMemMaps](ControlMemMaps.md) · [floorplan-harnesses](floorplan-harnesses.md) · [ARCH](ARCH.md) ·
[SOC_TIPS](SOC_TIPS.md)
