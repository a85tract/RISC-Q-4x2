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
#   pip install --no-index Pyro5 serpent into PYNQ's venv, then start the server (start_server.sh).
#
# Needs: ssh key access to the board (PYNQ default user xilinx / password xilinx if you have not
# set one up: `ssh-copy-id xilinx@192.168.3.1`). The server needs sudo for the RFDC/clock drivers;
# start_server.sh asks for the sudo password once (PYNQ default: xilinx).
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
scp -q "$HERE/start_server.sh" "$BOARD:$DEST/start_server.sh"

ssh "$BOARD" bash -s <<'REMOTE'
set -e
PY=/usr/local/share/pynq-venv/bin/python3
cd ~/riscq-4x2
# offline install of the RPC layer (PYNQ ships numpy already)
$PY -m pip install --quiet --no-index --find-links wheels Pyro5 serpent 2>/dev/null \
  || sudo -n $PY -m pip install --quiet --no-index --find-links wheels Pyro5 serpent
$PY -c "import Pyro5, serpent, numpy; print('[board] Pyro5', Pyro5.__version__, 'serpent', serpent.__version__)"
chmod +x start_server.sh
echo "[board] bundles in the store:"; ls bits
REMOTE

echo "[board_setup] done. Start the server:   ssh $BOARD '~/riscq-4x2/start_server.sh'"
echo "[board_setup] then point device_db['core'] at {'type': 'board', 'host': '<board ip>', 'bundle': 'rfsoc4x2-1q-fine'}"
