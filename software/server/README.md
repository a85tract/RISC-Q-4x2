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

Bench wiring for the receive-side checks: on `rfsoc4x2-1q-fine` loop **DAC0 → ADC0** (both drives
are on DAC0); on `rfsoc4x2-2dac-fine` loop **DAC1 → ADC0** (the readout drive is on DAC1; the gate
drive on DAC0 is then not seen by the ADC). ADC0 is the core's readout ADC in both bundles
(`adc_map [0]`); using ADC1 instead is a config change (`adc_map [1]`) and hence another bitstream.

Quick cable test: `PYTHONPATH=software/client python software/examples/loopback_check.py --remote <board>
--bundle <bundle> --ch <0|1>` plays one tone on a drive channel and reports whether it reaches the
readout ADC (`TONE PRESENT` / `NO TONE`). Run it first after (re)cabling — an RF-path fault looks
exactly like a dead board from software (bench note 2026-09-03: every digital check passed while the
ADC saw only noise on every single-cable loopback; the cause was a broken SMA ground return on our
board, and a second cable between the other DAC/ADC pair restored the loop). If a verified bundle
shows NO TONE, suspect cables, connectors and grounds before software.

Both are the "fine" configuration: 0.254 ns envelope grid on both drive channels, 16 384
envelope lines, 32-bit frequency word, queue depth 8 (see `docs/hardware-contract.md`).

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
