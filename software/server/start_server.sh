#!/usr/bin/env bash
# Start the RISC-Q board server on the RFSoC 4x2 (run ON the board; board_setup.sh installs it).
#
#   ssh -t xilinx@<board> '~/riscq-4x2/start_server.sh [--bundle rfsoc4x2-1q-fine] [--host <ip>]'
# (`ssh -t`: sudo needs a terminal to ask for the password)
#
# The server drives the FPGA (bitstream load, RF data-converter clocks, MMIO) so it needs root;
# PYNQ's default sudo password is "xilinx". It listens on the board's own IP by default — the
# RPC has no authentication, so keep the board on the isolated point-to-point / lab LAN and do
# NOT expose the port to a routed network (pass --host 0.0.0.0 only if you know why).
set -euo pipefail
cd ~/riscq-4x2
PY=/usr/local/share/pynq-venv/bin/python3
HOST="$(hostname -I | awk '{print $1}')"
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
sudo -v                                   # password once, up front (run this via `ssh -t`)
sudo pkill -f "riscq.board.server" 2>/dev/null || true   # the daemon runs as root
sleep 1
sudo env XILINX_XRT=/usr BOARD=RFSoC4x2 PYTHONPATH="$HOME/riscq-4x2/client" \
  setsid nohup "$PY" -u -m riscq.board.server --bits "$HOME/riscq-4x2/bits" --host "$HOST" "${ARGS[@]}" \
  > board_server.log 2>&1 < /dev/null &
sleep 4
NEWPID="$(pgrep -n -f 'python3 -u -m riscq.board.server' || true)"
if [ -n "$NEWPID" ] && sudo kill -0 "$NEWPID" 2>/dev/null && ! grep -qi "error\|Traceback\|address already in use" board_server.log; then
  echo "[board] server pid $NEWPID up on $HOST (log: ~/riscq-4x2/board_server.log)"
  tail -3 board_server.log
else
  echo "[board] server did not start cleanly:"; tail -20 board_server.log; exit 1
fi
