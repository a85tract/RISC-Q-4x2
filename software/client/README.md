# software/client — the `riscq` Python package

The control software that runs on your PC (or in the `client` docker image): the ARTIQ-shaped
interface (`riscq.artiq_compat`, `riscq.artiqapi`), the kernel compiler (`riscq.lang`,
`riscq.build`), the run layer (`riscq.run`), the driver backends (`riscq.driver`: board over
RPC, co-sim), calibrations (`riscq.cal`), and the board-side server (`riscq.board`, deployed to
the RFSoC 4x2 by `../server/board_setup.sh`). User documentation: `../../docs/`.

## Install

Supported: the docker images built from `Dockerfile` in this directory (see the top-level
README). Bare-metal alternative: Python 3.12, `pip install -r requirements.txt`, the RISC-V
toolchain on PATH (clang, lld, `riscv64-unknown-elf-objcopy/nm` names — see the Dockerfile's
toolchain stage), and `PYTHONPATH=software/client` (plus `sim` for co-simulation, which needs
`../../sim/requirements.txt` and Verilator/mill).

## Tests

From this directory:

```bash
python -m pytest tests -q --continue-on-collection-errors --ignore=tests/test_models.py
```

(290 host-pure tests; `test_models.py` needs an upstream `xcheck` config that is not in this
tree. Co-sim tests are skipped unless `--cosim`; they need the `full` image.)

## Layout

`riscq/` the package · `fw/` the kernel runtime header and start code the compiler stages ·
`tests/` · `pyproject.toml` (`pip install -e .`, extra `cal` = scipy) · `Dockerfile`.
