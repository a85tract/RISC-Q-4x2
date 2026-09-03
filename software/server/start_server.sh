#!/usr/bin/env bash
# Start the RISC-Q board server on the RFSoC 4x2 (run ON the board; board_setup.sh installs it).
#
#   ~/riscq-4x2/start_server.sh [--bundle rfsoc4x2-1q-fine] [--host <ip>]
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
pkill -f "riscq.board.server" 2>/dev/null || true
sudo -v                                   # ask for the password once, up front
sudo env XILINX_XRT=/usr BOARD=RFSoC4x2 PYTHONPATH="$HOME/riscq-4x2/client" \
  setsid nohup "$PY" -u -m riscq.board.server --bits "$HOME/riscq-4x2/bits" --host "$HOST" "${ARGS[@]}" \
  > board_server.log 2>&1 < /dev/null &
sleep 3
if pgrep -f "riscq.board.server" >/dev/null; then
  echo "[board] server up on $HOST (log: ~/riscq-4x2/board_server.log)"
  tail -3 board_server.log
else
  echo "[board] server did not start:"; tail -20 board_server.log; exit 1
fi
