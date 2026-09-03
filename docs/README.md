# RISC-Q on the RFSoC 4x2 — user documentation

An ARTIQ-feeling pulse API over the RISC-Q PulseTableSoc. You write experiments the way you
would for ARTIQ; RISC-Q's own scheduler, kernel compiler and firmware run underneath.

| document | what it covers |
|---|---|
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

The live end-to-end example is `../software/examples/artiq_api_demo.ipynb` (4 cells: the experiment class, the
device db + run, and a point-by-point comparison of the captured waveform against
`software/examples/reference/waveform_generator.py` — last live shot: 0.48 % residual, carrier phases ≤ 0.11°).

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

The scripted pass/fail run is `software/examples/artiq_rx_demo.py` (`RX_DEMO: PASS` in co-sim
and on the board; final logs kept with the project notes). The bench smoke test is
`software/examples/loopback_check.py`: one tone on a drive channel, pass/fail on whether it reaches
the readout ADC — run it first after cabling. Host-pure test suite: 290 tests across `software/client/tests/` (55 of them cover the
ARTIQ layers: `test_artiqapi.py` + `test_artiq_compat.py`). The design notes and root-cause records (ARTIQ_API_PLAN.md, the project journal) are
kept with the maintainer's project notes, outside this repository.
