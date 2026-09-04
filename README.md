# RISC-Q on the RFSoC 4x2

Pulse control for trapped-ion (and other) experiments on the AMD RFSoC 4x2 board, driven by the
RISC-Q PulseTableSoc (an on-FPGA RISC-V core scheduling a pulse-table DDS/envelope datapath),
with an **ARTIQ-shaped Python interface**: you write `EnvExperiment` classes with `@kernel`
`run()`, `with parallel:` / `with sequential:`, `dds.set(...)`, `dds.sw.pulse(...)`,
`adc.gate(...)` — RISC-Q's own scheduler, kernel compiler and firmware run underneath.

```python
from riscq.artiq_compat import *

class Rabi(EnvExperiment):
    def build(self):
        self.setattr_device("core"); self.setattr_device("ro_dds"); self.setattr_device("adc")

    @kernel
    def run(self):
        self.core.reset()
        with parallel:
            with sequential:
                self.ro_dds.set(82.0*MHz, phase=0.25, amplitude=0.4)
                self.ro_dds.sw.pulse(20*us)
            with sequential:
                self.adc.gate(30*us)

    def analyze(self):
        self.trace = self.adc.fetch_trace()

from configs.device_db_board import device_db      # or device_db_cosim: no hardware needed
exp = run_experiment(Rabi, device_db)
```

Verified end to end: the captured waveform of the reference two-pulse sequence matches the ideal
generator in bit-accurate co-simulation and on the board (`software/examples/artiq_api_demo.ipynb`,
run live and committed with its outputs): carrier phases within 0.5° per tone; the residual after
the delay/scale fit is bench-dependent (0.4–0.5 % rms on the one-core bundles' bench of 2026-08-31,
11 % on the 2026-09-04 bench whose DAC_A → ADC_A cable is lossy — see the bundle's PROVENANCE.md).

## Layout

| | |
|---|---|
| `docs/` | **start here**: [README](docs/README.md) quickstart, [the ARTIQ interface](docs/artiq-interface.md), [the explicit layer](docs/explicit-api.md), [the hardware contract](docs/hardware-contract.md) (grids, limits, every error) |
| `software/client/` | the `riscq` Python package (runs on your PC / in the docker image) + `Dockerfile` |
| `software/server/` | the board side: ready-made bitstream bundles in `bits/`, `board_setup.sh`, `start_server.sh` |
| `software/examples/` | the live demo notebook (two demos: the reference waveform on DAC_A vs ADC_A; four DDS channels on two DACs), device-db examples (`configs/`), the reference waveform generator and the hand-written reference scripts (`reference/`) |
| `sim/` | co-simulation (`riscq_sim`): the RTL under Verilator behind the same driver seam — everything runs without a board |
| `gateware/` | the RISC-Q hardware: SpinalHDL sources, `configs/` (SoC parameters), Vivado flow for the 4x2 |

## Three ways in

1. **No hardware — try it in co-simulation** (bit-accurate RTL; images: `client` 1.3 GB, `full` 2.3 GB):
   ```bash
   git submodule update --init --recursive      # SpinalHDL + rvls, needed to generate the RTL
   docker build -f software/client/Dockerfile --target full -t riscq-4x2:full .
   docker run -it --rm -v "$PWD":/work/RISC-Q -w /work/RISC-Q/software/examples riscq-4x2:full \
     python -m nbconvert --to notebook --execute --inplace artiq_api_demo.ipynb   # after switching its device_db to configs/device_db_cosim.py
   ```
   or use `device_db_cosim` from any script. The first start of a config generates the RTL and
   verilates it (a few minutes); then it is seconds.
2. **You have an RFSoC 4x2**: flash the vendor PYNQ image, then from `software/server/`:
   `./board_setup.sh xilinx@<board-ip>` and `ssh -t xilinx@<board-ip> '~/riscq-4x2/start_server.sh'`
   (see [software/server/README.md](software/server/README.md)). For the demo notebook loop
   DAC_A → ADC_A and DAC_B → ADC_B (the connector ↔ SoC-port mapping is in
   [software/server/README.md](software/server/README.md)); its first demo doubles as the cable
   check. Build the `client` image (RISC-V toolchain + Python) and run your experiments with the
   device dbs in `software/examples/configs/`.
3. **You want to change the gateware**: `gateware/` — Vivado 2024.1+ and the
   `vivado-scripts/riscvsoc-bd` flow (`RISCQ_BOARD=rfsoc4x2`, default config
   `configs/rfsoc4x2-1q-fine.json`), ~35 min a build; the co-sim
   verifies a new config before you synthesize it.

## Bundles shipped

| bundle | output mapping | status |
|---|---|---|
| `rfsoc4x2-1q-fine` | gate + readout drives summed on DAC0, ADC0 readout | board- and co-sim-verified (the demo notebook) |
| `rfsoc4x2-2dac-fine` | gate → DAC0, readout → DAC1, ADC0 readout | board- and co-sim-verified on both DACs (RX_DEMO PASS through DAC1, gate tone on DAC0), timing-clean (WNS +0.032 ns) — see its `PROVENANCE.md` |
| `rfsoc4x2-2dac-adcb` | gate → DAC0, readout → DAC1, ADC1 readout | the 2-DAC design reading ADC1 (loop DAC1 → ADC1); board-verified (RX_DEMO PASS), timing-clean (WNS +0.015 ns) — see its `PROVENANCE.md` |
| `rfsoc4x2-2q-fine` | **two cores**: dds 0/1 → DAC_A with its trace on ADC_A, dds 2/3 → DAC_B with its trace on ADC_B; one timeline, shared hardware time origin; **multi-tile synchronized** RF tiles (MTS required at load) | the demo notebook's bundle — see its `PROVENANCE.md` for the co-sim and board verification |

## Honest limits

* The ARTIQ interface is an *ARTIQ-syntax restricted subset*: `run()` is recorded once as
  Python and executed as **one** kernel; parallel arms must be `with sequential:`; no ARTIQ
  compiler, master or dashboard. Everything ARTIQ-shaped is the interaction layer only.
* One verified configuration family ("fine": 0.254 ns envelope grid, 32-bit frequency word);
  the scheduling limits (play spacing, queue depth, gate length, readout guard) are enforced by
  the planner and listed in the hardware contract.
* Across the two DACs the *timeline* is exact (one hardware time origin) and, on
  `rfsoc4x2-2q-fine`, the two DAC tiles are multi-tile synchronized (RF-tile latencies pinned to the
  bundle's recorded values at every load); the remaining connector-to-connector offset is the fixed
  board/cable path difference, which the notebook's second demo measures. The older one-core
  bundles are not synchronized.
* The board RPC is unauthenticated — isolated lab network only.
* Licensing of the upstream RISC-Q sources is being settled with its authors; until then this
  repository is not for redistribution.

Upstream: [Wu-Quantum-Application-System-Group/RISC-Q](https://github.com/Wu-Quantum-Application-System-Group/RISC-Q)
(branch `refactor`, kept as the `refactor` branch here for merges).
