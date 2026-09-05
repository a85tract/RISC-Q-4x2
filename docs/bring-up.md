# Bring-up: from a boxed RFSoC 4x2 to the demo notebook

The end-to-end walkthrough for a new user or a new board: install the board side (the server),
set up the client, understand how the FPGA gets programmed, wire the loopbacks, run
`software/examples/artiq_api_demo.ipynb`, and recognise the failures we have already met. The
reference pages ([artiq-interface.md](artiq-interface.md), [hardware-contract.md](hardware-contract.md),
[../software/server/README.md](../software/server/README.md)) stay the source of truth for the API,
the limits and the bundle table; this page is the order of operations.

Verified 2026-09-04 on this exact path: PYNQ 3.0.1 image, server deployed with `board_setup.sh`,
the docker `client` image on a Windows PC, the notebook executed headless with the numbers quoted in
§6.

## 0. What you need

| | |
|---|---|
| board | RealDigital / AMD **RFSoC 4x2**, its power supply, a micro-SD card ≥ 16 GB |
| image | **PYNQ 3.0.1 for the RFSoC 4x2** (RFSoC-PYNQ release). Exactly 3.0.1: the multi-tile-sync wrapper that `board_setup.sh` installs is bound to that image's `libxrfdc.so`; any other version is refused and the two-core bundle will not load |
| network | either the board's **USB 3.0 composite micro-B port** to the PC with the supplied cable (PYNQ's USB network: the board is `192.168.3.1`, the PC gets `192.168.3.x`; the `PROG UART` micro-USB port next to the power switch carries UART and JTAG, no USB networking) or the RJ45 port on a lab LAN (the board takes a DHCP address; find it on the OLED display, the serial console or your DHCP server). All examples below use `192.168.3.1` |
| cables | **two identical SMA cables** for the notebook's loopbacks: DAC_A → ADC_A and DAC_B → ADC_B. Identical matters (§7) |
| PC | anything that runs Docker (Linux, macOS, Windows with Docker Desktop). The client needs a RISC-V cross compiler to build kernels; the docker image carries it |
| credentials | PYNQ's default login `xilinx` / `xilinx`; the same password for `sudo` on the board |

Nothing else is attached to the board: no JTAG, no Vivado on the PC — the bitstreams are shipped as
bundles in `software/server/bits/` and programmed by the board itself (§4).

## 1. Board: image, boot, login

1. Write the PYNQ 3.0.1 RFSoC 4x2 image to the SD card (Balena Etcher or `dd`) — the card that comes
   with the kit is preloaded with some PYNQ release, so check its version in step 3 and rewrite it if it
   is not 3.0.1. Set the `BOOT` switch (next to the SD slot) to SD, connect power and the network
   cable, power on. Boot takes about a minute; the 16x2 OLED display shows the IP address once the
   board is up. The `PROG UART` micro-USB port is a serial console (115200 baud) if you need to watch
   the boot.
2. `ssh xilinx@192.168.3.1` (password `xilinx`). Key access is convenient (the install script makes
   several ssh/rsync connections, each asking for the password otherwise): `ssh-keygen` if you have no
   key yet, then `ssh-copy-id xilinx@192.168.3.1` — from the same environment that will run
   `board_setup.sh`; in the docker image that means mounting your key (`-v ~/.ssh:/root/.ssh:ro`) or
   generating one inside. Password authentication works too, just more prompts.
3. Check the image: `/usr/local/share/pynq-venv/bin/python3 -c "import pynq; print(pynq.__version__)"`
   must print `3.0.1`.

The board's Jupyter (port 9090) is PYNQ's own and is not used by RISC-Q; leave it running or not.

## 2. Client on your PC

The client is the `riscq` Python package plus a RISC-V toolchain (every `run_experiment` compiles a
kernel with `riscv64-unknown-elf-clang`). Two ways:

**Docker (supported):** from the repository root,

```bash
docker build -f software/client/Dockerfile --target client -t riscq-4x2:client .
docker run -it --rm -v "$PWD":/work/RISC-Q riscq-4x2:client        # your clone mounted; PYTHONPATH is set
```

The image has Python 3.12, numpy, Pyro5, matplotlib, nbconvert, `ssh`/`rsync` (for §3) and the
toolchain (no `pip` — its venv is managed by `uv`, so extra packages go in with
`uv pip install --python /opt/venv312 <pkg>`; no `ping` either). The container reaches the board over the
host's network (Docker's default bridge routes to the host's interfaces, including PYNQ's USB network in
our Windows setup) — check with `ssh xilinx@192.168.3.1 hostname` from inside, or
`python -c "import socket; socket.create_connection(('192.168.3.1', 22), 3)"`.

**Bare metal:** Python 3.12, `pip install -r software/client/requirements.txt`, the RISC-V clang
toolchain on PATH under the names the Dockerfile's toolchain stage creates
(`riscv64-unknown-elf-clang`, `ld.lld`, `riscv64-unknown-elf-objcopy`, `-nm`), and
`PYTHONPATH=software/client`. `python -m pytest software/client/tests/test_build.py -q` proves the
toolchain works (it compiles a real kernel).

Windows notes: `board_setup.sh` needs `rsync`, which Git Bash lacks — run it from the docker image
or WSL. In Git Bash, `python` may be a Store alias that does nothing; use `py -3`. Git Bash rewrites
container paths in `docker exec` arguments (`/work/...` becomes `C:/Program Files/Git/work/...`);
prefix such commands with `MSYS_NO_PATHCONV=1`.

## 3. Deploy the server (once per board, again after every software update)

From `software/server/` (inside the docker image or any machine with ssh + rsync):

```bash
./board_setup.sh xilinx@192.168.3.1
ssh -t xilinx@192.168.3.1 '~/riscq-4x2/start_server.sh'      # -t: sudo asks for the password
```

`board_setup.sh` copies the client package (`~/riscq-4x2/client`), the bundles
(`~/riscq-4x2/bits/<bundle>/`), offline wheels and the MTS-capable `xrfdc` wrapper; installs Pyro5 +
serpent into PYNQ's venv without internet; replaces PYNQ's `xrfdc` Python wrapper (originals kept as
`*.orig-3.0.1`) after checking the image version; and lists the bundles it installed.

`start_server.sh` kills any old server and starts `riscq.board.server` as root (the FPGA, the RF
converters and the clock chips need it), bound to the board's first address (`hostname -I`), port
9091, logging to `~/riscq-4x2/board_server.log`. With both the USB link and Ethernet connected that
first address may not be the one your device db uses — pass it explicitly:
`~/riscq-4x2/start_server.sh --host 192.168.3.1`. A healthy start prints

```
riscq board server @ PYRO:riscq.board@192.168.3.1:9091   (bundle: None)
```

The server starts EMPTY — no bitstream is loaded until a client asks for a bundle (§4). From the PC:

```python
from riscq.driver import remote
drv = remote.RemoteDriver("192.168.3.1")
print(drv.board.bundles())          # {'rfsoc4x2-2q-fine': ['PROVENANCE.md', 'SHA256SUMS', 'board.json', ...], ...}
print(drv.board.info())             # {'bundle': None, ...}
drv.close()
```

The RPC has no authentication: keep the board on the USB link or an isolated lab LAN.

**After you change anything under `software/client/riscq/board/` (server or driver):** run
`board_setup.sh` again AND restart the server. A running server keeps the old code in memory; we
lost a day once believing a fix was live when the process predated it.

## 4. Programming the FPGA ("the bitfile")

There is no separate programming step. A **bundle** is a directory
`software/server/bits/<name>/` with `top.xsa` (the bitstream + hardware description),
`params.json` (the SoC parameters the software must agree with), `board.json` (clocks, Nyquist
zones, multi-tile-sync targets, the PS clock, optionally an LMX register list), `SHA256SUMS` and
`PROVENANCE.md`. The board programs itself when a bundle is **loaded**:

* `run_experiment(...)` loads the bundle named in `device_db["core"]["bundle"]` at the start of
  EVERY run (5–30 s; the notebook's two demos therefore load twice). The explicit layer
  (`riscq.artiqapi.run(drv, core, ...)`, [explicit-api.md](explicit-api.md)) reuses one loaded
  driver for many runs.
* Manually: `drv.board.load("rfsoc4x2-2q-fine")` returns `info()`.

What a load does, in order: (1) the first load in a server process programs the clock chips — the
LMK04828 (245.76 MHz) and both LMX2594 (491.52 MHz) from PYNQ's xrfclk files, then the bundle's own
LMX list if `board.json` names one (`rfsoc4x2-2q-fine` does; see "Clocks and re-locks" in
[hardware-contract.md](hardware-contract.md)) — later loads reprogram only if the bundle wants other
clocks; (2) downloads `top.xsa` with `pynq.Overlay`; (3) runs the xrfdc **multi-tile synchronization**
and compares the tile latencies with the values pinned in `board.json` — with `"required": true`
(the two-core bundle) a miss FAILS the load, nothing runs on unsynchronized converters; (4) sets
the Nyquist zones and verifies them by readback; (5) measures the fabric clock. `info()` shows the
result:

```
{'bundle': 'rfsoc4x2-2q-fine', 'xsa_sha': 'e7e3ae73…', 'mts_result': 0,
 'mts_latencies': ({0: 260, 2: 260}, {2: 88}), 'dsp_mhz': 491.5x,
 'refclks': (245.76, 491.52, {'lmxadc': '30345f5e…', 'lmxdac': '30345f5e…'})}
```

`mts_result` 0 and latencies (260, 260) / (88) are what the two-core bundle must show; `dsp_mhz` is
a 0.2 s measurement of the fabric clock and reads about 491.52 (491.49–491.64 in our logs); the
`refclks` shas say both LMXs run the bundle's list.

**Your own bitstream:** build it with `gateware/vivado-scripts/riscvsoc-bd` (see `gateware/`),
create `bits/<name>/` as described in [../software/server/README.md](../software/server/README.md)
("Adding a bundle"), re-run `board_setup.sh` (it rsyncs the whole `bits/` tree), and name it in the
device db. `riscq.driver.remote.upload_bundle(drv, name, xsa, params_json, board=dict)` pushes
`top.xsa` + `params.json` (+ `board.json`) over the RPC without ssh and works before any bundle is
loaded — but it carries no extra files such as an LMX list, and what it writes is owned by root
(the server runs as root) while `board_setup.sh` copies files as `xilinx`: a later `board_setup.sh`
rsync over an uploaded bundle can fail with a permission error until you
`sudo chown -R xilinx ~/riscq-4x2/bits` on the board. `board_setup.sh` is the usual way.

## 5. Wiring for the notebook

Two identical SMA cables: **DAC_A → ADC_A** and **DAC_B → ADC_B** (the letters printed on the
board). No attenuator: `amplitude=0.4` produces about ±6300 ADC codes on the 16-bit scale through
a short cable, `amplitude=1.0` about ±16000, both fine. The connector letters run AGAINST the SoC's tile order
(DAC_A is the SoC's DAC1 / tile 230, DAC_B is DAC0 / tile 228; ADC_A is tile 226's second converter,
ADC_B its first) — the bundles and the device dbs speak in connector letters, so you only need this if
you read the RTL or the one-core bundles' table in the server README.

## 6. Run the notebook

The notebook hard-codes `"host": "192.168.3.1"` in its two `device_db` cells — edit them if your
board has another address. It must run with `software/examples` as the working directory (it loads
`reference/waveform.npz` relatively and finds the client via `../client`).

Headless, inside the docker image with your clone mounted (3–4 minutes, two bundle loads included):

```bash
cd /work/RISC-Q/software/examples
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 artiq_api_demo.ipynb
```

Interactively: start the container with the port published (`docker run -it --rm -p 8888:8888
-v "$PWD":/work/RISC-Q riscq-4x2:client`), then inside it install JupyterLab (not preinstalled; the
image has no `pip`): `uv pip install --python /opt/venv312 jupyterlab`, run
`jupyter lab --ip 0.0.0.0 --no-browser --allow-root` from `software/examples` (the container runs as
root), open the printed URL on the PC, and execute the cells in order. A `--rm` container forgets the
install when it exits; drop `--rm` or reinstall next time.

What a good run prints (2026-09-04, identical 30 cm cables):

| cell | expected |
|---|---|
| demo 1 | `245760 ADC samples at 1.96608 GS/s = 125.0 us`; `delay D ≈ 190–200 ns` (your cables); `residual rms … (0.6 %)` — below 1 %; `carrier phase, capture vs generator` within ±0.5° at all three tones |
| demo 2 | `per-core telemetry [t1, entry, armed]` identical `t1` for cores 0 and 1; `board_info … 'mts_result': 0, 'mts_latencies': ({0: 260, 2: 260}, {2: 88})`; `envelope of ADC_A lags ADC_B by 0 samples`; `level ratio ADC_B / ADC_A ≈ 1.0 (±0.5 dB)`; `DAC_B - DAC_A carrier phase: … +90 → 90 + a few degrees`, the same few degrees on both pulses |

The few degrees are your two paths' delay difference (about 0.045°/MHz per 120 ps of cable
difference); re-measure them after a power cycle and at the frequency you use before folding them
into a `phase`.

## 7. Pitfalls we have already met

1. **Connector letters.** A verified bundle that records only noise is almost always the cable on
   the wrong pair — every digital check passes while the ADC sees nothing. Check §5 before anything
   else (we lost a day on this).
2. **A lossy or mismatched cable** shows as a lower level on one trace (the notebook prints the
   ratio), an envelope lag of a sample or more, and tens of degrees of "phase offset" that bend with
   frequency: 7 dB and +67° on one bench until the cable was replaced. Use two identical cables.
3. **`MTS did not reach its latency targets` / `MTS failed on a bundle that requires it`** — the load
   refused to hand over unsynchronized converters. Reload once; if it persists, restart the server
   (`start_server.sh`) so the clocks are reprogrammed, then reload. It has not happened on a healthy
   board in our runs; a converter tile that never reaches state 15 points at the clock chips or at
   a wrong PYNQ image.
4. **`no bundle loaded — call load(<bundle>) first`** — the server starts empty; every board RPC
   except `bundles()`, `info()`, `load()` and the bundle upload (`upload_bundle`) needs a loaded
   bundle.
5. **Connection refused / timeouts from Pyro** — the server is not running (`ssh` in and run
   `start_server.sh`; check `~/riscq-4x2/board_server.log`), or the board's IP changed (DHCP), or
   your container cannot reach the board's network.
6. **`Address already in use`** in the log — an old server is still bound; `start_server.sh` kills
   it first, or `sudo pkill -f riscq.board.server` by hand, then start again.
7. **Stale code on the board** — after editing the server/driver, `board_setup.sh` + restart (§3).
8. **PYNQ version** — `board_setup.sh` refuses to replace `xrfdc` on anything but 3.0.1; without the
   MTS wrapper the two-core bundle cannot load. Use the 3.0.1 image.
9. **First load after a power cycle or a fresh server** reprograms the clocks; the DAC→ADC timing
   can come back a few tens of ps different unless the bundle carries the LMX phase-SYNC list
   (`rfsoc4x2-2q-fine` does, older bundles do not). Re-measure the connector offset after such an
   event if you rely on it.
10. **Every `run_experiment` reloads the bitstream** (5–30 s). Fine for the notebook; for many short
    experiments use the explicit layer with one loaded driver.
11. **The scheduling limits are enforced, not silently rounded**: pulses closer than 3 batches on one
    channel, a 9th play within 0.8 µs on one channel, a readout followed too closely — the planner
    raises with a message listed in [hardware-contract.md](hardware-contract.md). Read the message; it
    names the batch numbers.
12. **The ADC trace records only while its readout channel (`dds` 1 or 3) fires**, and a silent gap
    in that channel would restart the recording at address 0 — so `adc.gate()` fills every gap inside
    its window with zero-amplitude plays automatically — lead-in, gaps and tail, an empty window too
    (both demos have a 5 µs gap and record straight through it). What you must respect: every pulse of
    that channel lies fully inside the gate or ends at least one batch before it — one that straddles
    an edge or comes after the gate is refused (`overlaps the adc gate ... boundary or follows it`) —
    and the fillers count as plays for the queue rules. The explicit layer's `fill_gaps` is the manual
    relative: it fills only the gaps between existing pulses ([explicit-api.md](explicit-api.md)).
13. **Docker on Windows**: path rewriting (`MSYS_NO_PATHCONV=1`), `python` vs `py -3`, and a container
    left with orphaned Verilator processes from co-simulation runs (load average 18, notebook cells
    timing out) — `pkill -f riscq_sim.cosim; pkill -x PulseTableSoc` inside the container.
14. **The notebook's last cell** (demo-2 analysis) correlates two 137 k-sample traces; it takes seconds on
    an idle machine and minutes on a loaded one. If nbconvert times out there, the machine is busy,
    not the board.

## 8. Where to look when something is off

* Board: `~/riscq-4x2/board_server.log` (the server's stdout — the pynq clock warning per load is
  normal), `drv.board.info()` (what is loaded, MTS, clocks), `pgrep -af riscq.board.server`.
* Client: the exception text — the planner's messages name channels and batches; the multi-core
  telemetry check names the core and the missing lead. An error raised on the board arrives as the
  same exception type with the server's traceback attached but not printed; to see it,
  `import Pyro5.errors` and, in the handler, `except Exception: print("".join(Pyro5.errors.get_pyro_traceback()))`.
* Bench: the notebook's demo 1 is the loopback check for DAC_A → ADC_A; for DAC_B → ADC_B swap the
  device db to `dds` channels 2/3 and `adc` channel 1, or read demo 2's level ratio.
* Clocks: `software/examples/lmx_relock_check.py` re-locks the clock chips and reports the
  loopback phase per lock (the experiment behind "Clocks and re-locks" in the hardware contract).
