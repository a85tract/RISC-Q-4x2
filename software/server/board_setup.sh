#!/usr/bin/env bash
# Install the RISC-Q board side onto an RFSoC 4x2 running PYNQ, from your PC, over ssh.
#
#   ./board_setup.sh [xilinx@192.168.3.1]
#
# What it does on the board (idempotent):
#   ~/riscq-4x2/client   <- software/client (the riscq package; the board runs riscq.board.server
#                           and the same riscq.run/riscq.map code the client uses)
#   ~/riscq-4x2/bits     <- software/server/bits/* (the bitstream bundles: top.xsa + params.json + board.json)
#   ~/riscq-4x2/wheels   <- software/server/wheels (Pyro5 + serpent, so the install works OFFLINE)
#   ~/riscq-4x2/xrfdc_mts <- software/server/xrfdc_mts (the PYNQ xrfdc wrapper WITH multi-tile-sync
#                           bindings, installed into the PYNQ venv; stock 3.0.1 cannot run MTS)
#   pip install --no-index Pyro5 serpent into PYNQ's venv, then start the server (start_server.sh).
#
# Needs: ssh key access to the board (PYNQ default user xilinx / password xilinx if you have not
# set one up: `ssh-copy-id xilinx@192.168.3.1`). The server needs sudo for the RFDC/clock drivers;
# the install and start_server.sh ask for the sudo password (PYNQ default: xilinx) — hence `ssh -t`.
set -euo pipefail

BOARD="${1:-xilinx@192.168.3.1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # software/server
CLIENT="$HERE/../client"
DEST='~/riscq-4x2'

echo "[board_setup] -> $BOARD:$DEST"
ssh "$BOARD" "mkdir -p $DEST/client $DEST/bits $DEST/wheels"
# the python package + kernel firmware header (no tests, no caches)
rsync -a --delete --exclude '__pycache__' --exclude 'build' --exclude 'tests' \
      "$CLIENT/riscq" "$CLIENT/fw" "$CLIENT/pyproject.toml" "$BOARD:$DEST/client/"
rsync -a --delete "$HERE/bits/" "$BOARD:$DEST/bits/"
rsync -a "$HERE/wheels/" "$BOARD:$DEST/wheels/"
rsync -a "$HERE/xrfdc_mts/" "$BOARD:$DEST/xrfdc_mts/"
scp -q "$HERE/start_server.sh" "$BOARD:$DEST/start_server.sh"

# interactive (-t) so sudo can prompt: PYNQ's venv is root-owned, the offline install needs it.
# PYNQ 3.0.1's xrfdc Python wrapper has no multi-tile-sync bindings (its C library has them): the
# RFSoC-MTS project's patched wrapper (xrfdc_mts/) replaces it, the originals stay as *.orig-3.0.1 —
# the 4x2 bundles run MTS at every load and refuse to run without it. The wrapper is bound to PYNQ 3.0.1's
# libxrfdc.so ABI (three-argument XRFdc_MultiConverter_Init), so any other PYNQ version is refused.
ssh -t "$BOARD" 'set -e; PY=/usr/local/share/pynq-venv/bin/python3; cd ~/riscq-4x2; \
  sudo $PY -m pip install --quiet --no-index --find-links wheels Pyro5 serpent; \
  $PY -c "import Pyro5, serpent, numpy; print(\"[board] Pyro5\", Pyro5.__version__, \"serpent\", serpent.__version__)"; \
  $PY -c "import pynq, sys; v = pynq.__version__; sys.exit(0 if v == \"3.0.1\" else print(\"[board] xrfdc_mts/ is the PYNQ 3.0.1 wrapper (its cdef matches that libxrfdc.so); this image is pynq\", v, \"- refusing to replace xrfdc\") or 1)"; \
  D=$(BOARD=RFSoC4x2 $PY -c "import xrfdc, os; print(os.path.dirname(xrfdc.__file__))"); \
  sudo cp -n $D/__init__.py $D/__init__.py.orig-3.0.1; sudo cp -n $D/xrfdc_functions.c $D/xrfdc_functions.c.orig-3.0.1; \
  sudo cp xrfdc_mts/__init__.py $D/__init__.py; sudo cp xrfdc_mts/xrfdc_functions.c $D/xrfdc_functions.c; sudo rm -rf $D/__pycache__; \
  BOARD=RFSoC4x2 $PY -c "import xrfdc; assert hasattr(xrfdc.RFdc, \"mts_dac\"); print(\"[board] xrfdc multi-tile sync bindings installed\")"; \
  chmod +x start_server.sh; echo "[board] bundles in the store:"; ls bits'

echo "[board_setup] done. Start the server:   ssh -t $BOARD '~/riscq-4x2/start_server.sh'"
echo "[board_setup] then point device_db['core'] at {'type': 'board', 'host': '<board ip>', 'bundle': 'rfsoc4x2-1q-fine'}"
