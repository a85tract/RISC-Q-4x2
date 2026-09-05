# RISC-Q on the RFSoC 4x2 — user documentation

An ARTIQ-feeling pulse API over the RISC-Q PulseTableSoc. You write experiments the way you
would for ARTIQ; RISC-Q's own scheduler, kernel compiler and firmware run underneath.

| document | what it covers |
|---|---|
| [bring-up.md](bring-up.md) | **first day with a board**: image, server install, how the FPGA gets programmed (bundles), client setup, wiring, running the demo notebook, the pitfalls we have met |
| [artiq-interface.md](artiq-interface.md) | **the layer you normally use**: `EnvExperiment`, `@kernel`, `device_db`, `run_experiment`, the devices (`dds` / `adc` / `demod`) — every call, signature and error |
| [explicit-api.md](explicit-api.md) | the lower `riscq.artiqapi` layer (explicit `Core` object): building timelines without the experiment class, `plan()` reports, `run()`, `RunResult` |
| [hardware-contract.md](hardware-contract.md) | what is exact and what is snapped: time grids, phase modes, the recording rules, scheduling limits, and every planner error you can hit |

## Quick start

```python
import sys
sys.path.insert(0, "software/client")   # + "sim" for co-simulation
from riscq.artiq_compat import *

class Rabi(EnvExperiment):
    def build(self):
        self.setattr_device("core")
        self.setattr_device("ro_dds")
        self.setattr_device("adc")

    @kernel
    def run(self):
        self.core.reset()
        with parallel:
            with sequential:                       # send
                self.ro_dds.set(82.0*MHz, phase=0.25, amplitude=0.4)
                self.ro_dds.sw.pulse(20*us)
            with sequential:                       # receive
                self.adc.gate(30*us)

    def analyze(self):
        self.trace = self.adc.fetch_trace()        # raw ADC samples (numpy int array)

device_db = {
    "core":   {"type": "board", "host": "192.168.3.1", "bundle": "rfsoc4x2-1q-fine"},
    "ro_dds": {"type": "dds", "channel": 1},
    "adc":    {"type": "adc"},
}
exp = run_experiment(Rabi, device_db)              # compile, execute on the board, analyze
```

The live end-to-end example is `../software/examples/artiq_api_demo.ipynb`, two experiments on the
two-core `rfsoc4x2-2q-fine` bundle: (1) the reference two-pulse waveform on DAC_A, captured on
ADC_A and compared point by point with `software/examples/reference/waveform_generator.py`;
(2) four DDS channels on one timeline — both DACs playing together, ADC_A and ADC_B recording —
showing how one `with parallel:` spans the two cores and how a phase difference between the DACs
is programmed and measured.

## Where things run

* **Board**: `device_db["core"] = {"type": "board", "host": "192.168.3.1", "bundle":
  "rfsoc4x2-1q-fine"}`. The board runs `riscq-board-server`; `run_experiment` uploads nothing
  but the compiled kernel + tables (the bitstream bundle is already in the board's store).
* **Co-simulation** (bit-accurate Verilator RTL, no hardware):
  `{"type": "cosim", "config": "gateware/configs/rfsoc4x2-1q-fine.json", "build":
  "sim/build/rfsoc4x2-1q-fine", "model": {"kind": "loopback", "src": 0, "dst": 0, "gain": 0.9,
  "delay": 5}}` — the optional `model` wires DAC→ADC so receive-side code sees the sent pulses.
* **Toolchain**: compiling a kernel needs the RISC-V cross toolchain, and the drivers need
  Pyro5 — both are in the docker images built from `software/client/Dockerfile`
  (`--target client`: toolchain + Python, for board users; `--target full`: adds Verilator, mill
  and cocotb for co-simulation). Run experiments and the notebook inside the image with your
  clone mounted at `/work/RISC-Q`; `PYTHONPATH` there already includes `software/client` and `sim`.

## Verification status

The notebook is the live check: demo 1 fails loudly if the capture does not match the generator
(a wrong cable pair looks like a dead board from software — check the connector mapping in
`software/server/README.md` first). Host-pure test suite: `software/client/tests/` (the ARTIQ
layers: `test_artiqapi.py` + `test_artiq_compat.py`, including the two-core planning, kernel
splitting and telemetry checks). The co-sim and board verification records of each bundle are in
its `software/server/bits/<bundle>/PROVENANCE.md`; the design notes and root-cause records (the
project journal) are kept with the maintainer's project notes, outside this repository.
