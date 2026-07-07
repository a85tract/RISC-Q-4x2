# riscq control software

The Python package for the riscq PulseTableSoc: kernel compiler (`riscq.lang`), build/run layers
(`riscq.build`/`riscq.run`), driver backends (`riscq.driver`), co-sim (`riscq.sim`), calibrations
(`riscq.cal`), and the ZCU216 board side (`riscq.board`). Design of record:
[specs/software/](../specs/software/README.md).

## Install

Host (dev machine): `pip install -e .[cal]` — scipy is the `cal` extra (fit helpers only);
the co-sim additionally needs the dev environment's cocotb + verilator + riscv-gcc.

## Board install (offline, spec 10 §2)

The ZCU216 ARM runs the same wheel with only numpy + Pyro5 (already-shipped PYNQ packages
`pynq`/`xrfclk`/`xrfdc` are used but never pip-installed). The board has no internet:

```bash
# on a connected machine (numpy is already on the PYNQ image)
pip download riscq Pyro5 serpent -d wheels/        # or: pip wheel . -w wheels/ from this dir
scp -r wheels/ xilinx@<board>:

# on the board
pip install --no-index --find-links wheels/ riscq
```

`tests/test_packaging.py` gates this in CI: the board module surface must import with only
numpy + Pyro5 installed.

## Board server

```bash
riscq-board-server [--bits ~/riscq-bits] [--bundle <name>] [--host 0.0.0.0] [--port 9091]
# → riscq board server @ PYRO:riscq.board@0.0.0.0:9091   (bundle: <name>)
```

Gateware reaches the board as a **bundle** (`top.xsa` + `params.json` + optional `board.json`)
uploaded over the same Pyro5 connection — `riscq.driver.remote.upload_bundle(drv, name, ...)`,
then `drv.board.load(name)`. See [specs/software/10-hardware-driver.md](../specs/software/10-hardware-driver.md).
