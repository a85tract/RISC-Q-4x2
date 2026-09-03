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

Verified end to end: the captured waveform of the reference two-pulse sequence matches the
ideal generator to 0.4–0.5 % rms with carrier phases within 0.1° on the board and in
bit-accurate co-simulation (`software/examples/artiq_api_demo.ipynb`, run live).

## Layout

| | |
|---|---|
| `docs/` | **start here**: [README](docs/README.md) quickstart, [the ARTIQ interface](docs/artiq-interface.md), [the explicit layer](docs/explicit-api.md), [the hardware contract](docs/hardware-contract.md) (grids, limits, every error) |
| `software/client/` | the `riscq` Python package (runs on your PC / in the docker image) + `Dockerfile` |
| `software/server/` | the board side: ready-made bitstream bundles in `bits/`, `board_setup.sh`, `start_server.sh` |
| `software/examples/` | the live demo notebook, device-db examples (`configs/`), the verification script, the loopback cable check (`loopback_check.py`), the reference waveform generator |
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
   (see [software/server/README.md](software/server/README.md)). Loop a DAC into ADC0 for the
   receive-side demos and confirm the cable with `software/examples/loopback_check.py`. Build the `client` image (RISC-V toolchain + Python) and run your
   experiments with `device_db_board` / `device_db_2dac`.
3. **You want to change the gateware**: `gateware/` — Vivado 2024.1+ and the
   `vivado-scripts/riscvsoc-bd` flow (`RISCQ_BOARD=rfsoc4x2`, default config
   `configs/rfsoc4x2-1q-fine.json`), ~35 min a build; the co-sim
   verifies a new config before you synthesize it.

## Bundles shipped

| bundle | output mapping | status |
|---|---|---|
| `rfsoc4x2-1q-fine` | gate + readout drives summed on DAC0, ADC0 readout | board- and co-sim-verified (the demo notebook) |
| `rfsoc4x2-2dac-fine` | gate → DAC0, readout → DAC1, ADC0 readout | co-sim verified (RX_DEMO PASS), bitstream built and timing-clean (WNS +0.032 ns); board test pending — see its `PROVENANCE.md` |

## Honest limits

* The ARTIQ interface is an *ARTIQ-syntax restricted subset*: `run()` is recorded once as
  Python and executed as **one** kernel; parallel arms must be `with sequential:`; no ARTIQ
  compiler, master or dashboard. Everything ARTIQ-shaped is the interaction layer only.
* One verified configuration family ("fine": 0.254 ns envelope grid, 32-bit frequency word);
  the scheduling limits (play spacing, queue depth, gate length, readout guard) are enforced by
  the planner and listed in the hardware contract.
* The board RPC is unauthenticated — isolated lab network only.
* Licensing of the upstream RISC-Q sources is being settled with its authors; until then this
  repository is not for redistribution.

Upstream: [Wu-Quantum-Application-System-Group/RISC-Q](https://github.com/Wu-Quantum-Application-System-Group/RISC-Q)
(branch `refactor`, kept as the `refactor` branch here for merges).
