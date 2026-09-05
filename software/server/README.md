# software/server — the board side (RFSoC 4x2, PYNQ)

What runs ON the board, and how to put it there. The Python is the same `riscq` package as the
client (`../client/riscq/board/` is the server: `riscq.board.server` loads bundles, programs the
RF data converters, and executes the client's compiled kernels + register batches over RPC).

```
bits/<bundle>/           ready-made bitstreams: top.xsa (bit + hwh), params.json (the SoC
                         parameters the software must agree with), board.json (clocks, Nyquist
                         zones, the build's achieved PS clock), SHA256SUMS, PROVENANCE.md
wheels/                  Pyro5 + serpent wheels, so the board installs OFFLINE
board_setup.sh           run from your PC: copies client + bits + wheels to ~/riscq-4x2 on the board
start_server.sh          run on the board: starts riscq.board.server (needs sudo for the RF drivers)
requirements-board.txt   what the board needs beyond PYNQ
```

## Bundles

| bundle | DACs | verified |
|---|---|---|
| `rfsoc4x2-1q-fine` | gate (ch0) and readout (ch1) drives **summed onto DAC0**; ADC0 readout | co-sim + board: RX_DEMO PASS, waveform vs generator 0.4–0.5 % |
| `rfsoc4x2-2dac-fine` | gate → **DAC0**, readout → **DAC1**, ADC0 readout | co-sim + board: RX_DEMO PASS on the DAC1 readout path, DAC0 gate tone verified; timing-clean (WNS +0.032 ns) — details in its PROVENANCE.md |
| `rfsoc4x2-2dac-adcb` | gate → **DAC0**, readout → **DAC1**, **ADC1** readout (loop DAC1 → ADC1) | board: RX_DEMO PASS through DAC1 → ADC1; timing-clean (WNS +0.015 ns) — details in its PROVENANCE.md |
| `rfsoc4x2-2q-fine` | **two cores**: core 0 = dds 0/1 summed on **DAC1 (DAC_A)**, trace on **ADC1 (ADC_A)**; core 1 = dds 2/3 on **DAC0 (DAC_B)**, trace on **ADC0 (ADC_B)**; per-core traces, shared run origin; **MTS** (tile 230 distributes the DAC sample clock and clocks the fabric, latencies pinned at load, `"required": true`) | the demo notebook's bundle (both demos) — details in its PROVENANCE.md |

**Connector labels** (RealDigital RFSoC 4x2 reference manual: "ADCA and ADCB … tile 226, ADCC and ADCD …
tile 224; DACA in tile 230 and DACB in tile 228" — the letters run against the tile order; confirmed on
our board 2026-09-03 with a single cable): the SoC's **DAC0** is RFDC tile 228 =
the connector printed **DAC_B**; **DAC1** is tile 230 = **DAC_A**; **ADC0** is tile 226's first converter (xrfdc block 0, stream
`m20_axis`) = **ADC_B**; **ADC1** is its second converter (xrfdc block 1, stream `m22_axis`) = **ADC_A**. Bench wiring for the receive-side checks, in SoC numbering: on
`rfsoc4x2-1q-fine` loop **DAC0 → ADC0** (connectors DAC_B → ADC_B; both drives are on DAC0); on
`rfsoc4x2-2dac-fine` loop **DAC1 → ADC0** (DAC_A → ADC_B; the readout drive is on DAC1, the gate
drive on DAC0 is then not seen by the ADC). ADC0 is the core's readout ADC in those two bundles
(`adc_map [0]`); `rfsoc4x2-2dac-adcb` is the same 2-DAC design reading ADC1 instead (`adc_map [1]`),
so its readout loop is **DAC1 → ADC1** (connectors DAC_A → ADC_A). The two-core `rfsoc4x2-2q-fine`
uses both pairs: **DAC_A → ADC_A** (core 0) and **DAC_B → ADC_B** (core 1). Select a bundle with
`"bundle": "<name>"` in the device db.

Cable check: run the demo notebook's first experiment (or any `adc.gate` around a pulse) — a wrong
connector pair looks exactly like a dead board from software (bench note 2026-09-03: every digital
check passed while the ADC saw only noise for a whole day — the cable was on the A connectors while
the bundle used the ports behind the B connectors). If a verified bundle records only noise, check
the connector mapping above, then cables and seating, before suspecting software.

All four are the "fine" configuration: 0.254 ns envelope grid on both drive channels, 16 384
envelope lines, 32-bit frequency word, queue depth 8 (see `docs/hardware-contract.md`).

The three one-core bundles predate the 2026-09-04 trace-recorder fix (the RTL never wrote the LAST
batch of a recording): the final batch of any trace window they return is the previous run's sample.
Ignore that batch or rebuild them from the current RTL; `rfsoc4x2-2q-fine` has the fix.

**Multi-tile synchronization** (`rfsoc4x2-2q-fine`): the board runs the xrfdc MTS procedure at every
load — `board.json` `"mts": {"daclatency", "adclatency", "dac_tiles": 5, "adc_tiles": 4, "ref_tile": 2,
"required": true}`; DAC tiles 228 + 230 and ADC tile 226 are the synced set, tiles 224/225/227 stay
enabled idle in the bitstream because the converter IP offers ADC MTS only with tile 224 on — and
`info()` reports `mts_result` (0 = every synced tile at its target),
`mts_latencies` and `dsp_mhz` (the measured fabric clock, 491.52). PYNQ 3.0.1's `xrfdc` Python package
lacks the MTS bindings; `board_setup.sh` installs the patched wrapper from `xrfdc_mts/` (the RFSoC-MTS
project's patch; the stock files stay next to it as `*.orig-3.0.1`). The one-core bundles have
`"mts": null` (their tile clocks were never synchronized; they still load and run).

## Setup (once per board)

```bash
# your PC, from this directory; PYNQ default user/password xilinx/xilinx
ssh-copy-id xilinx@192.168.3.1                # key access (optional but convenient)
./board_setup.sh xilinx@192.168.3.1           # copies everything, installs the wheels offline
ssh -t xilinx@192.168.3.1 '~/riscq-4x2/start_server.sh'   # -t: sudo asks for its password
```

Then from the client: `device_db["core"] = {"type": "board", "host": "192.168.3.1",
"bundle": "rfsoc4x2-1q-fine"}` — `run_experiment` loads the bundle (~10 s the first time),
and everything else is the experiment code (`docs/artiq-interface.md`).

The PYNQ image itself comes from the board vendor (RFSoC-PYNQ for the RFSoC 4x2); the board
needs `xrfclk`/`xrfdc` (in the image) and a network path to the client.

## Security contract

The server RPC (Pyro5) has **no authentication**: whoever can reach the port can load
bitstreams and poke MMIO. `start_server.sh` binds to the board's own IP; keep the board on
the isolated point-to-point or lab network and do not forward the port.

## Adding a bundle

Build with `gateware/vivado-scripts/riscvsoc-bd/build-riscvsoc-bd.sh` (see `gateware/`), then:
`bits/<name>/top.xsa` (the flow's `PulseTableSoc.xsa`), `params.json` = the config JSON you
built from, `board.json` = copy of an existing one with `fclk0_mhz` set to the build's achieved
PS clock (Vivado report), `sha256sum * > SHA256SUMS`, and a `PROVENANCE.md` saying when/with
what/how verified. Re-run `board_setup.sh`; the name is what `device_db` references.
A bundle may carry its own LMX2594 register list (`"lmx_regs": "<file in the bundle dir>"`, TICS
Pro / xrfclk text format, R112 first): after xrfclk's default files the server programs both LMXs
from it with the datasheet sequence (reset, all registers, 10 ms, R0 again), and every load
guarantees that state. `rfsoc4x2-2q-fine` uses it for the LMX's phase-SYNC mode (see
`docs/hardware-contract.md`, "Clocks and re-locks"). Clock diagnostics over the same RPC:
`refclks(lmk, lmx, lmx_regs=None)` reprograms the LMK and both LMXs (plus a list),
`lmx_program("lmxdac" | "lmxadc", regs=None)` re-locks ONE LMX alone; reload the bundle afterwards.
`info()["refclks"]` reports what is programmed. `software/examples/lmx_relock_check.py` is the
clock-phase repeatability experiment built on them.
