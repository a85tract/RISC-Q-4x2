# `riscvsoc-bd/` — block-design implementation of the floorplan

This is the **block-design counterpart** of the out-of-context bench in [`../riscvsoc`](../riscvsoc).
Same SoC (`PulseTableSoc`, 14 cores), same **floorplan** (cores → X0 Y3–Y7 bands 3/row, datapath →
`X1Y0:X5Y7`), same recipe (`keepCoreHierarchy`, retiming, `df`+`1h`, `place ExtraNetDelay_high` /
`route AggressiveExplore`) — but built in a **real device context**: `PulseTableSoc` packaged as a user
IP in a Vivado block design alongside the **Zynq UltraScale+ PS + RF Data Converter + AXI SmartConnect**,
with real **ClockInterface BUFG** clocking from the board LVDS clocks and the real host-released reset.

**Purpose — evaluate the difference.** The OOC bench (`../riscvsoc`) reports `dspClk` WNS ≈ **−0.156 ns**
for the SoC *in isolation* with synthetic port clocks. This flow puts the identical SoC + floorplan into
the full ZCU216 image. The gap between the two WNS numbers is the **block-design / real-device penalty**:
the PS + RFDC sharing the die, BUFG clock insertion, the SmartConnect routing, and the fact that the SoC
synthesises **out-of-context as an IP child run** while place+route happen globally at the wrapper.

## How it differs from the OOC bench (what the comparison measures)

| | `../riscvsoc` (OOC) | `riscvsoc-bd` (this) |
|---|---|---|
| RTL | `GenPulseTableSocOoc` — `vivado=false`, plain `dspClk`/`clk` ports | `GenPulseTableSocJson <cfg> <dir> vivado` — `vivado=true`, `hostClk` + `X_INTERFACE`, `ClockInterface.v` |
| context | SoC alone, OOC | SoC IP + Zynq PS + RFDC + SmartConnect in a BD |
| clocks | `create_clock` on raw ports, async-grouped | board LVDS → `ClockInterface` BUFG → real clock tree |
| synth | one `synth_design -mode out_of_context -retiming` | BD wrapper `synth_1` + the SoC as an OOC IP child run (retiming pushed onto it via `RISCQ_IP_RETIMING`) |
| P&R | non-project `place_design`→`route_design` | project `impl_1` (`Performance_NetDelay_high` + post-route phys_opt) |
| floorplan | `../riscvsoc/pblocks-riscvsoc.tcl` (top-level names) | `pblocks-bd.tcl` (same floorplan, BD prefix `riscq_bd_i/top/inst/…` + PS/RFDC guard) |
| reset | synthetic | real host-released reset network |

Everything that *can* be held identical is — the divergences above are exactly the BD-context effects
being measured.

## Run

```bash
cd vivado-scripts/riscvsoc-bd
./build-riscvsoc-bd.sh                       # 14q, floorplan, synth + impl in the block design
RISCQ_QUBITS=3 ./build-riscvsoc-bd.sh    # smaller config (faster)
```

Outputs land in `<repo>/build/riscvsoc-bd/` (git-ignored): the generated RTL, `timing_impl.rpt` /
`util_impl.rpt` (the headline WNS/TNS + per-pblock utilisation), the BD, the `*.runs/` and `vivado.log`.
The driver echoes the impl WNS/TNS at the end. Compare that WNS to the `../riscvsoc` OOC number. Set
`RISCQ_PROJ_NAME=<name>` to build into `<repo>/build/<name>` instead (parallel designs).

## Files

This flow is **self-contained** — everything it needs lives in this directory:

| File | Role |
|---|---|
| `build-riscvsoc-bd.sh` | driver: generate BD RTL into the build dir, then launch Vivado on `flow-bd.tcl` with the floorplan + IP retiming |
| `flow-bd.tcl` | proc-free driver: sets config vars, then `source`s the `inc/*.tcl` steps |
| `inc/{config,create-project,package-ip,bd-build,bd-finalize,rfdc-config,rfdc-connect,run}.tcl` | the BD assembly: project, IP packaging, PS + RFDC + SmartConnect, synth + impl |
| `constraints-zcu216.xdc` | ZCU216 pin/clock constraints (added to the BD wrapper) |
| `pblocks-bd.tcl` | the floorplan ported into the block-design hierarchy |

The driver selects the floorplan + core retiming through two `inc/run.tcl` hooks:

- **`RISCQ_PBLOCK_TCL`** — the pre-place floorplan file (→ `pblocks-bd.tcl`).
- **`RISCQ_IP_RETIMING`** — set `GLOBAL_RETIMING on` for the SoC's OOC IP synth run (the BD wrapper's
  `synth_1` never reaches the cores), so the cores are synthesised with retiming like the OOC bench.

## Env knobs

Handled by `build-riscvsoc-bd.sh`: `RISCQ_VIVADO_BIN`, `RISCQ_QUBITS` (14), `RISCQ_SKIP_GEN`,
`RISCQ_PROJ_NAME` (`riscvsoc-bd`), `RISCQ_PLACE_DIRECTIVE` (`ExtraNetDelay_high`).

Read by `pblocks-bd.tcl`: `RISCQ_ROW` (3), `RISCQ_PERROW` (3), `RISCQ_CONFINE`
(`global`|`region`|`none`, default `global`), `RISCQ_BD_BASE` (`riscq_bd_i/top/inst`).

> RTL-level levers (`replicateTime`, `df`, `1h`, `linkPipe`, the unconditional `KEEP_HIERARCHY` on each
> core) are baked into the BD RTL by `GenPulseTableSocJson` (the `PulseTableSoc` defaults = the floorplan
> stack), as in the OOC flow. The qubit count and DAC/ADC maps come from the JSON config it reads.

## Caveats

- This needs the full ZCU216 BD IP set (Zynq PS, RF Data Converter, SmartConnect) — it is a heavier,
  longer run than the OOC bench, and produces a real `impl_1`.
- The impl strategy is `Performance_NetDelay_high` + **post-route** phys_opt (`inc/run.tcl`'s), vs the OOC
  bench's `place → phys_opt → route → phys_opt`. Close, not byte-identical — noted because it is one of
  the BD-vs-OOC differences, not a controlled variable.
