# PulseTableSoc — multi-qubit control SoC toplevel

**Source:** `src/riscq/soc/PulseTableSoc.scala` · **Package:** `riscq.soc` · **Type:** Component
(`extends Zcu216Top`)

The multi-qubit control SoC for the Xilinx ZCU216/RFSoC (`xczu49dr-ffvf1760-2-e`). One shared 32-bit
batch-time counter drives `qubitNum` identical [`RiscqRfWithPulseTableFiber`](RiscqRfWithPulseTableFiber.md)
qubit cores; a 100 MHz host AXI bus loads programs / pulse tables / control while the faster `dspClk`
domain runs the real-time signal datapath. The RF tree inside each qubit is the **narrow posted link** —
read [`ARCH.md`](ARCH.md) first for why.

This doc covers the toplevel assembly. The per-qubit internals are in
[`RiscqRfWithPulseTableFiber`](RiscqRfWithPulseTableFiber.md); the board wrapper that drives the real
device ports is in [`Zcu216Top`](Zcu216Top.md).

## Role in the system

`PulseTableSoc` is the elaboration unit the Vivado flows synthesise. It owns nothing real itself beyond
the host bridge, the batch clock, the channel maps, and the readout trace — the qubit cores and the DSP
datapath are inside the per-qubit fibers, and the converter wiring is in the board wrapper.

```
  io.axi (100 MHz host) ──▶ Axi4ToTilelinkFiber ──▶ hostBus
                                                       ├─ per-core instruction RAM (direct)
                                                       ├─ per-core gate-drive envelope RAM (write-only, direct)
                                                       ├─ per-core readout-drive / demod envelope RAM (write-only, direct)
                                                       ├─ robs readout buffers (WidthAdapter)
                                                       └─ host control block (reset / fromHost / timeOffset)

  refTime+timeOffset ─▶ syncTime ─▶ time + per-core coreTime_i replicas ─▶ qubit cores (dspClk)

  qubit cores ──▶ dacMap (sum co-mapped channels, AdderTree) ──▶ io.dac
  io.adc ──▶ adcMap (fan-in) ──▶ qubit cores ;  pulse fire ──▶ robs trace
```

## Structure

**Host AXI bridge.** `Axi4ToTilelinkFiber(blockSize = 64)` converts `io.axi` to Tilelink and fans it to a
`hostBus`. `blockSize` covers the widest full-word transfer any slave negotiates (the `robs`
`WidthAdapter`'s 128-bit / 16-byte line); each fiber's decoder restricts the size down to what it supports.
Host fan-out targets, all derived from `SocMemoryMap` (offsets relative to each region bus): per-core
instruction RAM (32-bit, direct) and the three **write-only** envelope RAMs (gate / readout-drive / demod),
each a [`BramWriteFiber`](BramWriteFiber.md) whose `TileLinkMemWriteLogic` steers a 32-bit host beat into
the addressed sub-word lane of its wide line — so all three wire **direct** to their narrow region bus with
no `WidthAdapter`. The `robs` readout buffers (host-readable → still a read/write `BramFiber` + adapter) and
the host control block complete the fan-out.

**Shared batch clock, host-gated.** A free-running `refTime` (64-bit, gated by `riscqReset`) plus a
host-written `timeOffset` form `syncTime`; the low 32 bits become the `time` broadcast. Crucially each
core gets its own `coreTime_i` register fed from the *same* `syncTime` — value-identical every cycle (zero
skew), `EQUIVALENT_REGISTER_REMOVAL=NO` so Vivado cannot fold the replicas into one high-fanout net, and
the floorplan pins each replica to its core's band. The cores boot **held in reset**: `riscqReset` powers
up asserted and only releases when the host writes the control block (write `0x01` then `0x00` to
`riscqReset`), so nothing runs with uninitialised state before the first host reset pulse.

**Channel → converter maps.** `dacMap: Map[(core, channel), dacId]` and `adcMap: Map[core, adcId]` place
each logical channel on a physical converter. When several channels map to one DAC, the DAC word is the
**per-lane sum of their real parts**, built with [`AdderTree`](../dsp/DSP.md) (the sum wraps modulo `2^w`,
matching the QubiC reference — software keeps co-mapped channels within full-scale; dropping the
saturating clamp also takes its comparators off the converter-boundary path). A single mapped channel is a
trivial pass-through; an unmapped DAC is tied to 0. ADC inputs are buffered (real lane; `im := 0`) and
fanned to the mapped cores. `converterPipe` adds register stages on the long nets into/out of the RFDC
edge — extra latency, acceptable since fmax is soft. The default `SocChannelMap` (object in the same file)
gives each qubit's gate drive its own DAC `0..qubitNum-1`, and shares converter 14/15 for readout drive
and 0/1 for the ADC.

> Gotcha: `dacMap` is a `Map`. `dacMap.toList.collect { … (c, ch) }` must convert to a `List` **before**
> `.collect`, or collecting `(c, ch)` tuples rebuilds a `Map[c, ch]` and two channels of the same core
> collide on key `c` — the gate channel is silently dropped. See [`SOC_TIPS.md`](SOC_TIPS.md) §6.

**Readout trace (`robs`).** On any drive-pulse fire (`anyPulseValid`), a `BramFiber` buffer captures the
per-lane sum of all ADC inputs into a 32-bit-per-lane trace, addressed by a fire-incremented pointer, for
host read-back. Account for the `AdderTree` latency in the write timing (`fire := RegNext(anyPulseValid)`).

**Per-core hard boundary.** Each core's `RiscvSoc` carries `KEEP_HIERARCHY` so `opt_design` cannot merge
the identical cores' shared host-load logic into a MUXF7/F8 macro straddling two per-core pblocks — see
[`ARCH.md`](ARCH.md) (two-region floorplan) and [`SOC_TIPS.md`](SOC_TIPS.md) §7.

## Parameters that matter

- **`qubitNum`** — number of qubit cores (full ZCU216 build = 14).
- **`dacMap` / `adcMap`** — the channel→converter placement (use `SocChannelMap.dacMap/adcMap(qubitNum)`).
- **`linkPipe`** (default 4) — per-direction `RegNext` depth of the posted link; raise it to span a wider
  core↔converter floorplan (it only adds to the constant software lead-time `D`). See [`ARCH.md`](ARCH.md).
- **`vivado`** (default false) — when set, emits the FPGA IP-packager attributes (`X_INTERFACE_INFO`,
  `FREQ_HZ`) on the AXI/AXI-Stream ports and **renames the host clock `hostClk`/`hostRst`**. Keep it off
  for sims and the OOC bench, whose XDC constrains the un-renamed `clk` port (a renamed clock breaks the
  OOC bench). The attributes themselves are sim-neutral; only the rename bites.
- **`withTest`** (default false) — exposes each core's CPU data-bus decode to a second Tilelink master so
  a sim can schedule pulses without a CPU program. In the real SoC the CPU is the sole `dBus` master.
- **`coreParam`** — the RISC-V plugin config replicated across all cores; defaults to the verified
  timing-closure stack (every flag RVLS-bit-exact). See [`RISCV.md`](../riscv/RISCV.md).
- **`readoutInterp` / `gateInterp`** — envelope interpolation factors that shrink the widest BRAM banks.
- **`converterPipe`** — extra register stages on the DAC/ADC converter boundary.
- **dsp-fmax lever** (specs/dsp-fmax.md, default off / bit-exact, set per build in the config JSON):
  `adcPipe` (C2, the RFDC-edge ADC pipe depth, default 3). (The B1-alt param-buffer distributed RAM,
  the B2 dcOffset MAX_FANOUT cap and the B3 queue lean-pop are baked into `PulseParamBuffer`/`TimedQueue`;
  the C1 registered head (`regHead`) is a [TimedQueue](../dsp/TimedQueue.md)-level option, no longer
  plumbed through the SoC.)

## RTL generation

Two generator apps emit the toplevel Verilog (there is no plain `GenPulseTableSoc` object — pick the form
the flow needs):

```bash
# OOC fmax / floorplan flow — plain dspClk/clk ports, no IP-packager attrs.  Args: [qubitNum=14] [dir]
mill runMain riscq.soc.GenPulseTableSocOoc 14

# Vivado block-design flow — X_INTERFACE_INFO/FREQ_HZ attrs + ClockInterface.v (vivado = true).
mill runMain riscq.soc.GenPulseTableSocVivado 14
```

Both default to `qubitNum = 14` and the `SocChannelMap` layout; pass a smaller count for quick iteration.
`GenPulseTableSocVivado` also emits the `ClockInterface.v` clock-buffer wrapper.

## Verification

- `RamOnFabricSim` — the fabric-wired core (`RiscqFiber` over the memory fibers on a
  `DualClockRam` preloaded from an rv32 ELF): re-runs the `rv32ui-p` suite under RVLS lock-step, proving
  the core→fabric bridge + memory fibers + address decode.
- `PulseTableSocSim` — the assembled SoC, bus-driven (no ELF): an AXI host round-trip; a scheduled gate
  pulse propagating through the `dacMap` `AdderTree` to the mapped DAC with `robs` capture; and a VNA-style
  readout (matched ≫ detuned magnitude) over the full `io.adc → demod → integrate → read-back`. The
  readout is checked by magnitude (phase-invariant for a matched tone) rather than bit-exact, so it is
  robust to the bulk `io.adc`/`io.time` → integrator latency — see [`SOC_TIPS.md`](SOC_TIPS.md) §4.2.
- `PulseTableSocCpuSim` — CPU-in-the-loop: the RISC-V core runs `sw/pulse_sched.elf`, reads `time`, writes
  the gate buffer `startTime`, programs `table[0]` and fires — the pulse reaches the DAC with no test
  master (`withTest = false`, the CPU the sole `dBus` master, exactly as the real SoC).

```bash
mill runMain riscq.soc.sim.RamOnFabricSim
mill runMain riscq.soc.sim.PulseTableSocSim
mill runMain riscq.soc.sim.PulseTableSocCpuSim
```

OOC fmax is signed off by `riscq.soc.bench.PulseTableSocVivadoBench` (a SoC-specific two-clock XDC — tight
`dspClk`, loose `clk`, async-grouped; the generic single-clock bench would leave `dspClk` unconstrained,
see [`SOC_TIPS.md`](SOC_TIPS.md) §6). fmax is a soft constraint — recorded, not tuned.

## Related

- [`ARCH.md`](ARCH.md) — the posted-link architecture and the two-region floorplan (read first).
- [`RiscqRfWithPulseTableFiber`](RiscqRfWithPulseTableFiber.md) — one qubit core + its DSP datapath.
- [`Zcu216Top`](Zcu216Top.md) — the board wrapper + the `vivado-scripts/` flows.
- [`SOC_TIPS.md`](SOC_TIPS.md) — fabric / SpinalSim gotchas (read before SoC work).
- [`QUBIC_DATAPATH_COMPARISON.md`](QUBIC_DATAPATH_COMPARISON.md) — datapath vs the QubiC reference.
