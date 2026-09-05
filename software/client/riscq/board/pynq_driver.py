"""PynqDriver: the 4-method Driver over the ZCU216 AXI window + the RFDC bring-up ops
(spec 10 §3). The ONLY module that imports pynq/xrfclk/xrfdc — it only ever runs on the board;
the server imports it lazily inside load(). The RFDC operations are reproduced from two working
references — QubiC's PLInterface and qcal-riscq's RiscqPlInterface (this gateware's previous
driver, same board) — and stay unverified until M6 hardware bring-up (spec 10 §7)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import struct
import time
from pathlib import Path

import pynq
import xrfclk
import xrfdc  # noqa: F401 — registers the RFdc driver so overlay.rf_data_converter binds

log = logging.getLogger(__name__)

AXI_BASE = 0x8000_0000   # riscvsoc-bd flow: bd-build.tcl assign_bd_address -offset
AXI_SIZE = 0x1000_0000   # ... -range

# board.json defaults (spec 10 §4); tile/block keys are "tile,block" strings
BOARD_DEFAULTS = {
    "lmk_freq": 500.25,                             # qcal-riscq's proven value on this board
    "lmx_freq": None,
    "adc_nyquist": 1,
    "dac_nyquist": {"default": 2},
    "dac_current": {},
    # multi-tile sync: latency targets (QubiC's ZCU216 values; re-pinned at bring-up), the tile masks
    # of the DAC / ADC sync groups (RFSoC 4x2: 0b0101 / 0b0100), the reference tile, and whether a
    # miss must FAIL load() ("required": the 4x2 two-core bundles) instead of being logged
    "mts": {"daclatency": 260, "adclatency": 60, "dac_tiles": 0xF, "adc_tiles": 0xF,
            "ref_tile": 2, "required": False},
    "fclk0_mhz": None,                              # pin pynq Clocks.fclk0_mhz after download (RFSoC 4x2: 100)
    # a bundle's own LMX2594 register list (xrfclk TICS format, R112..R0; the server resolves the file
    # to the ints): programmed into BOTH LMXs after xrfclk's defaults. The 4x2 MTS bundle uses it for
    # the LMX's phase-SYNC mode (VCO_PHASE_SYNC = 1, N / IncludedDivide): measured 2026-09-04, without
    # it a full clock (re)programming lands the DAC->ADC timing in one of two states 25-60 ps apart,
    # with it the state repeats within 3 ps (docs/hardware-contract.md, "Clocks and re-locks")
    "lmx_regs": None,
}

_refclks_state = None    # (lmk, lmx, {lmx name: sha of its programmed list, None = xrfclk's}) as last
                         # programmed — once per server process, not per load (spec 10 §3.2); a load
                         # reprograms only when the bundle wants something else. None while unknown.


def _lmx_spidevs() -> dict[str, str]:
    """{device-tree name: /dev/spidevB.C} of the LMX2594s (RFSoC 4x2: lmxdac = spi0.1, lmxadc = spi0.2);
    xrfclk has bound spidev to them by the time this runs."""
    out = {}
    for dev in Path("/sys/bus/spi/devices").glob("*"):
        node = dev / "of_node"
        if (node / "compatible").read_bytes().rstrip(b"\0") != b"ti,lmx2594":
            continue
        name = (node / "name").read_bytes().rstrip(b"\0").decode()
        out[name] = "/dev/" + next((dev / "spidev").iterdir()).name
    if set(out) != {"lmxdac", "lmxadc"}:
        raise RuntimeError(f"expected the RFSoC 4x2's two LMX2594s (lmxdac, lmxadc) on SPI, found {sorted(out)}")
    return out


def _write_lmx(spidev: str, regs) -> None:
    """Program one LMX2594 with the datasheet's sequence (SNAS696C 7.5.1): RESET on, RESET off, every
    register R112 -> R0 as listed, 10 ms, then R0 again (FCAL_EN runs the VCO calibration). 24-bit
    frames, as xrfclk writes them."""
    regs = [int(v) & 0xFFFFFF for v in regs]
    assert len(regs) == 113 and regs[0] >> 16 == 112 and regs[-1] >> 16 == 0, "expect R112..R0"
    with open(spidev, "rb+", buffering=0) as f:
        for v in (0x000002, 0x000000, *regs):
            f.write(struct.pack(">I", v)[1:])
        time.sleep(0.01)
        f.write(struct.pack(">I", regs[-1])[1:])


def _regs_sha(regs) -> str | None:
    if not regs:
        return None
    return hashlib.sha256(",".join(str(int(v)) for v in regs).encode()).hexdigest()


class PynqDriver:
    """MMIO Driver + overlay + RFDC ops. Construction IS bring-up, in the reference order:
    ref clocks -> overlay download -> MMIO -> MTS -> Nyquist zones -> DAC VOP."""

    def __init__(self, xsa: str, params_json: str, board: dict | None = None,
                 download: bool = True):
        # pynq's Xrt-backed EmbeddedDevice does `asyncio.get_event_loop()` when it is first
        # constructed (XrtDevice.__init__, triggered by the Overlay below). The board server
        # runs load() in a Pyro5 worker thread, which on Python 3.10+ has no implicit loop — give
        # it one so the probe doesn't raise "no current event loop in thread ...". We never run
        # the loop (the driver polls, it never waits on interrupts); the probe just needs the
        # call to succeed. No-op on the main thread, which already has a loop.
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        cfg = {**BOARD_DEFAULTS, **(board or {})}
        self.params_text = Path(params_json).read_text()

        # a load guarantees the clock state the bundle was validated with: program when this process
        # has not yet, or when what is programmed (frequencies, or either LMX's list) differs from the
        # bundle's. lmx_program() records what it programmed per LMX, so one LMX re-locked with the
        # bundle's own list survives a load (the re-lock experiment relies on that) and one re-locked
        # with another list is restored.
        sha = _regs_sha(cfg["lmx_regs"])
        if (_refclks_state is None or _refclks_state[:2] != (cfg["lmk_freq"], cfg["lmx_freq"])
                or any(v != sha for v in _refclks_state[2].values())):
            self.refclks(cfg["lmk_freq"], cfg["lmx_freq"], cfg["lmx_regs"])
        self.refclks_state = _refclks_state
        log.info(f"loading overlay: {xsa}")
        self.overlay = pynq.Overlay(str(xsa), download=download)
        if cfg["fclk0_mhz"]:
            # boards whose PS preset cannot hit the requested pl_clk0 exactly (RFSoC 4x2:
            # 96968727 Hz achieved for a 100 MHz request) get re-pinned after download.
            pynq.Clocks.fclk0_mhz = cfg["fclk0_mhz"]
            log.info(f"pl_clk0 pinned to {pynq.Clocks.fclk0_mhz} MHz")
        self.rfdc = self.overlay.rf_data_converter
        self.mmio = pynq.MMIO(AXI_BASE, AXI_SIZE)

        # Auto-MTS: board.json "mts": null skips it at bring-up (run drv.board.mts() by hand instead;
        # mts_result stays None = "not run"). A miss is logged + surfaced via info() — or, with
        # "required": true (the 4x2 two-core bundles, whose DAC_A <-> DAC_B alignment depends on it),
        # it aborts load() so nothing runs on an unsynchronized converter.
        self.mts_latencies = None
        if cfg["mts"]:
            mts_cfg = {**BOARD_DEFAULTS["mts"], **cfg["mts"]}
            required = bool(mts_cfg.pop("required"))
            try:
                self.mts_result = self.mts(**mts_cfg)
            except RuntimeError as e:
                if required:
                    raise RuntimeError(f"MTS failed on a bundle that requires it: {e}") from e
                log.error(f"auto-MTS failed, continuing (set board.json \"mts\": null to skip): {e}")
                self.mts_result = 1
            if required and self.mts_result != 0:
                raise RuntimeError(
                    f"MTS did not reach its latency targets ({mts_cfg}); measured dac/adc "
                    f"latencies {self.mts_latencies} — this bundle requires multi-tile sync")
        else:
            self.mts_result = None
        log.info(f"mts: {self.mts_result} latencies: {self.mts_latencies}")
        zones = cfg["dac_nyquist"]
        dac_set, dac_skipped = {}, []
        for tile in range(4):
            for block in range(4):
                n = zones.get(f"{tile},{block}", zones.get("default", 2))
                try:
                    self.dac_nyquist_zone(tile, block, n)
                    dac_set[f"{tile},{block}"] = int(self.rfdc.dac_tiles[tile].blocks[block].NyquistZone)
                except Exception:  # absent tile/block on partial-converter boards (RFSoC 4x2)
                    dac_skipped.append(f"{tile},{block}")
        bad = {k: v for k, v in dac_set.items() if v != zones.get(k, zones.get("default", 2))}
        if bad:
            raise RuntimeError(f"DAC Nyquist zone readback mismatch {bad} (asked {zones})")
        log.info(f"dac nyquist set {dac_set}; absent blocks {dac_skipped}")
        self.adc_nyquist_zone(cfg["adc_nyquist"])
        for tileblock, uA in cfg["dac_current"].items():
            tile, block = (int(x) for x in tileblock.split(","))
            self.dacvop(tile, block, uA)

    # ── the Driver protocol over pynq.MMIO (a numpy uint32 view of the /dev/mem mmap) ──

    def _check(self, addr: int, nbytes: int = 4) -> None:
        if addr % 4:
            raise ValueError(f"unaligned address {addr:#x}")
        if addr < 0 or addr + nbytes > AXI_SIZE:
            raise ValueError(f"[{addr:#x}, {addr + nbytes:#x}) outside the AXI window "
                             f"(size {AXI_SIZE:#x})")

    def read32(self, addr: int) -> int:
        self._check(addr)
        return self.mmio.read(addr)

    def write32(self, addr: int, value: int) -> None:
        self._check(addr)
        self.mmio.write(addr, int(value) & 0xFFFFFFFF)

    def read_block(self, addr: int, nbytes: int) -> bytes:
        self._check(addr, nbytes)
        # Word-at-a-time single-beat reads (like read32/write_block). A numpy slice read
        # (self.mmio.array[a:b].tobytes()) issues a multi-beat/wide AXI burst that this gateware's
        # BRAM host read port DECERRs -> SIGBUS ("Bus error"); the PL only handles 32-bit beats.
        nwords = -(-nbytes // 4)
        buf = b"".join((self.mmio.read(addr + 4 * i) & 0xFFFFFFFF).to_bytes(4, "little")
                       for i in range(nwords))
        return buf[:nbytes]

    def write_block(self, addr: int, data: bytes) -> None:
        data = bytes(data)
        if len(data) % 4:
            raise ValueError(f"write_block length {len(data)} is not a multiple of 4 "
                             "(pynq MMIO word-loops the payload)")
        self._check(addr, len(data))
        self.mmio.write(addr, data)

    # ── RFDC ops, reproduced verbatim from the references (spec 10 §3.3 / §7) ──

    def refclks(self, lmk_freq: float, lmx_freq: float | None = None, lmx_regs=None) -> None:
        """Program the LMK and both LMXs from xrfclk's files, then — given `lmx_regs`, the ints of an
        LMX2594 TICS list (R112 first) — every LMX again from that list with the datasheet sequence
        (SNAS696C 7.5.1). Recorded in `refclks_state` (and info()). The bundle's board.json "lmx_regs"
        and the re-lock experiment (software/examples/lmx_relock_check.py) both come through here."""
        global _refclks_state
        _refclks_state = self.refclks_state = None      # unknown until everything below has succeeded
        if lmx_freq is None:
            xrfclk.set_ref_clks(lmk_freq=lmk_freq)
        else:
            xrfclk.set_ref_clks(lmk_freq=lmk_freq, lmx_freq=lmx_freq)
        devs = _lmx_spidevs()
        if lmx_regs:
            for name, dev in sorted(devs.items()):
                _write_lmx(dev, lmx_regs)
                log.info(f"{name} ({dev}): programmed the LMX list {_regs_sha(lmx_regs)[:12]}")
        _refclks_state = self.refclks_state = (lmk_freq, lmx_freq, {n: _regs_sha(lmx_regs) for n in devs})
        log.info(f"ref clocks: {self.refclks_state}")

    def lmx_program(self, which: str, regs=None) -> str:
        """Diagnostic: reprogram ONE LMX (`lmxdac` / `lmxadc`) from `regs` (None: xrfclk's own list for
        the current lmx_freq), the LMK and the other LMX untouched — the clock-phase repeatability
        experiment. Reload the bundle afterwards: the tiles lost their refclk. Returns the spidev used."""
        global _refclks_state
        dev = _lmx_spidevs()[which]
        if _refclks_state is None:
            raise RuntimeError("the clocks have not been programmed in this process (load a bundle first)")
        lmk_freq, lmx_freq, per_lmx = _refclks_state
        if regs is None:
            regs = xrfclk.xrfclk._Config["lmx2594"][float(lmx_freq)]
        _refclks_state = self.refclks_state = None      # unknown while the SPI write is in flight
        _write_lmx(dev, regs)
        sha = None if regs is xrfclk.xrfclk._Config["lmx2594"][float(lmx_freq)] else _regs_sha(regs)
        _refclks_state = self.refclks_state = (lmk_freq, lmx_freq, {**per_lmx, which: sha})
        log.info(f"{which} ({dev}): reprogrammed alone -> {self.refclks_state}")
        return dev

    def config_mts(self, daclatency: int = -1, adclatency: int = -1, dac_tiles: int = 0xF,
                   adc_tiles: int = 0xF, ref_tile: int = 2) -> None:
        for mts_cfg, tiles in ((self.rfdc.mts_dac_config, dac_tiles), (self.rfdc.mts_adc_config, adc_tiles)):
            mts_cfg.RefTile = ref_tile   # the clock-owning tile (the references' choice; RFSoC 4x2: DAC 230)
            mts_cfg.Tiles = tiles        # bitmask of the tiles in the sync group
            mts_cfg.SysRef_Enable = 1
        self.rfdc.mts_dac_config.Target_Latency = daclatency
        self.rfdc.mts_adc_config.Target_Latency = adclatency

    def mts(self, daclatency: int = 260, adclatency: int = 60, dac_tiles: int = 0xF,
            adc_tiles: int = 0xF, ref_tile: int = 2) -> int:
        """Two-pass MTS (QubiC): free sync to measure the latencies; if all tiles agree and are
        within target, re-sync pinned to the targets. 0 iff every measured latency == target.
        Raises (XRFdc_MultiConverter_Sync) if a tile never reaches the started state — run it by
        hand via drv.board.mts() to see the full converter error. The latencies of the synced tiles
        are kept in `mts_latencies` ({tile: latency} for DAC and ADC) for info() / PROVENANCE."""
        self._wait_tiles_started(dac_tiles, adc_tiles)
        self.config_mts(-1, -1, dac_tiles, adc_tiles, ref_tile)
        self.rfdc.mts_dac()
        self.rfdc.mts_adc()
        dac_lat, adc_lat = self._mts_latencies(dac_tiles, adc_tiles)
        log.info(f"mts free sync: dac={dac_lat} adc={adc_lat}")
        dv, av = list(dac_lat.values()), list(adc_lat.values())
        if (all(l == dv[0] for l in dv) and dv[0] <= daclatency
                and all(l == av[0] for l in av) and av[0] <= adclatency):
            self.config_mts(daclatency, adclatency, dac_tiles, adc_tiles, ref_tile)
            self.rfdc.mts_dac()
            self.rfdc.mts_adc()
            dac_lat, adc_lat = self._mts_latencies(dac_tiles, adc_tiles)
            log.info(f"mts pinned to dac {daclatency} / adc {adclatency}: dac={dac_lat} adc={adc_lat}")
        self.mts_latencies = (dac_lat, adc_lat)
        return 0 if (all(l == daclatency for l in dac_lat.values())
                     and all(l == adclatency for l in adc_lat.values())) else 1

    def _wait_tiles_started(self, dac_tiles: int, adc_tiles: int, timeout_s: float = 5.0) -> None:
        """Block until every tile in the two sync masks reports TileState 15 (started). The tiles
        power up on their own after the bitstream lands, the clock-distribution receivers last; a
        warm re-download of the same bitstream reached XRFdc_MultiConverter_Sync before ADC tile 226
        was up ("ADC tile 2 in Multi-Tile group not started", 2026-09-04)."""
        import time

        def states():
            st = self.rfdc.IPStatus
            out = {}
            for kind, mask in (("DAC", dac_tiles), ("ADC", adc_tiles)):
                tiles = st[f"{kind}TileStatus"] if isinstance(st, dict) else getattr(st, f"{kind}TileStatus")
                for i in range(4):
                    if mask >> i & 1:
                        t = tiles[i]
                        out[f"{kind}{i}"] = int(t["TileState"] if isinstance(t, dict) else t.TileState)
            return out

        deadline = time.monotonic() + timeout_s
        while True:
            s = states()
            if all(v == 15 for v in s.values()):
                log.info(f"tiles started: {s}")
                return
            if time.monotonic() > deadline:
                raise RuntimeError(f"RF tiles not started within {timeout_s} s (TileState 15 = started): {s}")
            time.sleep(0.02)

    def _mts_latencies(self, dac_tiles: int = 0xF, adc_tiles: int = 0xF) -> tuple[dict, dict]:
        return ({i: int(self.rfdc.mts_dac_config.Latency[i]) for i in range(4) if dac_tiles >> i & 1},
                {i: int(self.rfdc.mts_adc_config.Latency[i]) for i in range(4) if adc_tiles >> i & 1})

    def adc_nyquist_zone(self, n: int) -> None:
        done, skipped = [], []
        for tile in range(4):        # rfdc-config.tcl enables every slice on the ZCU216 (full 4x4)
            for block in range(4):
                try:
                    self.rfdc.adc_tiles[tile].blocks[block].NyquistZone = n
                    if int(self.rfdc.adc_tiles[tile].blocks[block].NyquistZone) != n:
                        raise RuntimeError(f"ADC {tile},{block} Nyquist zone readback != {n}")
                    done.append(f"{tile},{block}")
                except RuntimeError as e:
                    # the xrfdc wrapper raises RuntimeError for BOTH a failed call and an absent block
                    # ("ADC 0 block 1 not available in XRFdc_SetNyquistZone": the 4x2's dual-ADC tiles
                    # have blocks 0 and 2 only) — the absent block is skipped, anything else is loud
                    if "not available" not in str(e):
                        raise
                    skipped.append(f"{tile},{block}")
                except Exception:  # absent tile/block on partial-converter boards (RFSoC 4x2)
                    skipped.append(f"{tile},{block}")
        log.info(f"adc nyquist {n} set on {done}; absent blocks {skipped}")

    def dac_nyquist_zone(self, tile: int, block: int, n: int) -> None:
        self.rfdc.dac_tiles[tile].blocks[block].NyquistZone = n

    def dacvop(self, tile: int, block: int, uA: int) -> None:
        log.info(f"dac vop: tile {tile} block {block} -> {uA} uA")
        self.rfdc.dac_tiles[tile].blocks[block].SetDACVOP(uA)
