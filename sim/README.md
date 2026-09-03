# sim — co-simulation of the PulseTableSoc (`riscq_sim`)

The "board without a board": the SoC's RTL (generated from `../gateware` by mill/SpinalHDL) is
compiled with Verilator and driven by a cocotb bench that speaks the **same driver seam** as the
board server. Every layer above the driver — `riscq.run`, `riscq.artiqapi`, `riscq.artiq_compat`
— runs unchanged against it, bit-accurately. It is how this project verified the API before
touching hardware, and how a user without an RFSoC can try everything.

```python
from riscq_sim import cosim
drv = cosim.start("gateware/configs/rfsoc4x2-1q-fine.json", "sim/build/rfsoc4x2-1q-fine")
drv.sim.set_model({"kind": "loopback", "src": 0, "dst": 0, "gain": 0.9, "delay": 5})  # DAC0 -> ADC0
...   # A.run(drv, core, ...) / run_experiment(..., device_db with {"type": "cosim", ...})
cosim.stop(drv)
```

`cosim.start(config, build_dir)` regenerates the RTL only when the config JSON changed (sha
stamp), verilates only when the RTL changed, then runs the bench as a subprocess and returns a
`CosimDriver`. The first start of a new config takes a few minutes (and, in a fresh `full` image, mill downloads its
Scala dependencies once — network needed); later starts are seconds.

| what | where |
|---|---|
| `riscq_sim/cosim.py` | `start()`/`stop()`, the subprocess entry (`python -m riscq_sim.cosim <cfg> <build>`) |
| `riscq_sim/bench.py` | the cocotb bench: clocks, the register seam, DAC capture, ADC models, the RPC server |
| `riscq_sim/models.py` | ADC-seam models (`loopback`, quantum-model stubs) selected with `drv.sim.set_model` |
| `riscq_sim/rtl.py` | `mill runMain riscq.soc.GenPulseTableSocJson` in `../gateware` |
| `build/<config>/` | generated: `rtl/`, `sim_build/` (verilated model), logs — git-ignored |

## Requirements

Heavier than the client: Verilator (5.020+; verified on 5.032), a C++ toolchain, JDK 17 + mill
1.1.0 for the RTL generation, and Python < 3.14 with `cocotb==1.9.2`, `Pyro5`, `serpent`
(`requirements.txt`). All of it is in the `full` target of `software/client/Dockerfile`; that is
the supported way to run it. `PYTHONPATH` needs both `software/client` and `sim`.

## Loopback models and the two-DAC build

`set_model({"kind": "loopback", "src": <dac>, "dst": <adc>, "gain", "delay"})` wires one DAC
into one ADC. On the single-DAC bundles both drive channels are on DAC0 (`src: 0`); on
`rfsoc4x2-2dac-fine` the readout drive is on DAC1 (`src: 1`); on `rfsoc4x2-2dac-adcb` the readout
ADC is ADC1 (`dst: 1`) — see `software/examples/configs/device_db_*.py` and
`software/examples/artiq_rx_demo.py --loopback-src / --loopback-dst`.
