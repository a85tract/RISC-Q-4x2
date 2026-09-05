"""RemoteDriver: Pyro5 proxy to the board server (spec 10 §6). Mirrors CosimDriver — the 4
Driver methods proxy 1:1, `.remote` is set UNCONDITIONALLY (run.setup/rerun route server-side,
one RPC per batch, so LAN latency never multiplies per register poke), `.board` exposes the
RFDC ops + bundle store. No `.sim` attribute: sim-only operations don't exist on hardware, and
poll_done (only ever executed server-side) uses its hardware branch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import Pyro5.api

from riscq.driver.cosim import _RemoteExtras, _to_bytes

CHUNK = 4 * 1024 * 1024   # bundle-upload chunk size (spec 10 §4: <= 4 MB, one in flight)


class _BoardExtras:
    """The board-ops + store surface of the server (spec 10 §5), thin delegates."""

    def __init__(self, proxy: Pyro5.api.Proxy):
        self._proxy = proxy

    def get_params(self) -> str:
        """The loaded bundle's SocParams JSON (the config handshake, spec 04 §2)."""
        return self._proxy.get_params()

    def info(self) -> dict:
        return self._proxy.info()

    def mts(self, daclatency: int = 260, adclatency: int = 60) -> int:
        return self._proxy.mts(daclatency, adclatency)

    def refclks(self, lmk_freq: float, lmx_freq: float | None = None, lmx_regs=None) -> None:
        """Reprogram the LMK and both LMXs; `lmx_regs` (the ints of an LMX2594 TICS list, R112 first)
        is programmed into both LMXs afterwards. Reload the bundle next: the RF tiles lost their clock."""
        self._proxy.refclks(lmk_freq, lmx_freq, list(lmx_regs) if lmx_regs else None)

    def lmx_program(self, which: str, regs=None) -> str:
        """Diagnostic: reprogram ONE LMX ("lmxdac" / "lmxadc") from `regs` (None: xrfclk's own list),
        the LMK and the other LMX untouched. Reload the bundle next."""
        return self._proxy.lmx_program(str(which), list(regs) if regs else None)

    def adc_nyquist_zone(self, n: int) -> None:
        self._proxy.adc_nyquist_zone(int(n))

    def dac_nyquist_zone(self, tile: int, block: int, n: int) -> None:
        self._proxy.dac_nyquist_zone(int(tile), int(block), int(n))

    def dacvop(self, tile: int, block: int, uA: int) -> None:
        self._proxy.dacvop(int(tile), int(block), int(uA))

    def bundles(self) -> dict:
        return self._proxy.bundles()

    def load(self, bundle: str, download: bool = True) -> dict:
        """Construct/replace the server's PynqDriver from a stored bundle (full bring-up)."""
        return self._proxy.load(str(bundle), bool(download))


class RemoteDriver:
    """Driver over Pyro5 to the board server. `host` is a hostname/IP (+ `port`) or a full
    PYRO: uri."""

    def __init__(self, host: str, port: int = 9091):
        uri = host if host.startswith("PYRO:") else f"PYRO:riscq.board@{host}:{port}"
        self._proxy = Pyro5.api.Proxy(uri)
        self.remote = _RemoteExtras(self._proxy)
        self.board = _BoardExtras(self._proxy)

    def read32(self, addr: int) -> int:
        return self._proxy.read32(int(addr))

    def write32(self, addr: int, value: int) -> None:
        self._proxy.write32(int(addr), int(value) & 0xFFFFFFFF)

    def read_block(self, addr: int, nbytes: int) -> bytes:
        return _to_bytes(self._proxy.read_block(int(addr), int(nbytes)))

    def write_block(self, addr: int, data: bytes) -> None:
        self._proxy.write_block(int(addr), bytes(data))

    def close(self) -> None:
        self._proxy._pyroRelease()


def upload_bundle(drv: RemoteDriver, name: str, xsa: str | Path, params_json: str | Path,
                  board: dict | None = None) -> None:
    """Chunk a bundle's files up to the server's store (spec 10 §4): top.xsa + params.json
    (+ board.json when `board` is given). Activate it with drv.board.load(name)."""
    files = [("top.xsa", Path(xsa).read_bytes()),
             ("params.json", Path(params_json).read_bytes())]
    if board is not None:
        files.append(("board.json", json.dumps(board, indent=2).encode()))
    for filename, data in files:
        drv._proxy.store_begin(name, filename, len(data), hashlib.sha256(data).hexdigest())
        for off in range(0, len(data), CHUNK):
            drv._proxy.store_chunk(data[off:off + CHUNK])
        drv._proxy.store_end()
