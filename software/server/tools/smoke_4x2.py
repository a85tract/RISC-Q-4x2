"""RISC-Q refactor (PulseTableSoc) RFSoC 4x2 board smoke — the gated bring-up ladder.

Runs ON the board (PYNQ 3.0.1 venv, as root), with the deployed riscq package on PYTHONPATH:
    sudo env XILINX_XRT=/usr PYTHONPATH=/home/xilinx/riscq-refactor/software RISCQ_STAGE=digital \
        /usr/local/share/pynq-venv/bin/python3 -u smoke_4x2.py
RISCQ_STAGE: clocks | overlay | probe | digital   (each stage includes the previous ones)
RISCQ_BIT:    bitstream path   (default /home/xilinx/riscq-refactor/PulseTableSoc.bit)
RISCQ_PARAMS: SocParams JSON   (default /home/xilinx/riscq-refactor/rfsoc4x2-1q.json)

Why the gates exist (three PS wedges on this bench, dev-line port): every host access into the
SoC's dsp domain is answered only if dspClk (= RFDC clk_dac0, DAC tile 228's output) is running
and dspRst is released. A transaction issued before that is lost forever, the host bus is dead
from then on -> AXI stall -> the A53 cannot even be halted -> physical power cycle.
So: reference clocks -> overlay -> RFDC tile status via xrfdc -> dspAlive counter must advance
-> only then any dsp-domain MMIO. Single-word MMIO only (no numpy slices into MMIO.array).
"""
import os
import sys
import time

BIT    = os.environ.get('RISCQ_BIT', '/home/xilinx/riscq-refactor/PulseTableSoc.bit')
PARAMS = os.environ.get('RISCQ_PARAMS', '/home/xilinx/riscq-refactor/rfsoc4x2-1q.json')
STAGE  = os.environ.get('RISCQ_STAGE', 'probe')
T0 = time.time()


def banner(s):
    print(f'--- [{time.time()-T0:6.1f}s] {s} ---', flush=True)


def say(s):
    print(f'    {s}', flush=True)


def fail(code, msg):
    print('FAIL:', msg)
    print('RISCQ_4X2_SMOKE: FAIL', flush=True)
    sys.exit(code)


# ---------------------------------------------------------------- 0. memory map -----------
from pathlib import Path
from riscq.map import SocMap, SocParams

m = SocMap(SocParams.from_json(Path(PARAMS).read_text()))
HOSTCTRL = m.host_ctrl           # host-clock domain block (riscqReset @0, liveness @+0x100/704)
RAM_BYTES = m.params.mem_depth * 4
say(f'map: region_size={m.region_size:#x} host_ctrl={HOSTCTRL:#x} core RAM {RAM_BYTES//1024} KiB')

# ---------------------------------------------------------------- 1. reference clocks ----
import xrfclk
import xrfdc                     # noqa: F401 — before Overlay(): registers the RFDC driver
from pynq import Overlay, MMIO, Clocks

banner('clocks: LMK 245.76 MHz -> LMX 491.52 MHz tile refs')
# lmx_freq MUST be passed: xrfclk defaults to the ZCU111 plan (lmx 409.6). With the LMX left at
# 409.6 the tile PLL (x16) lands outside its VCO range, never locks (power-up state 7), clk_dac0
# never exists and the first dsp-domain MMIO wedges the PS — three times on this bench.
xrfclk.set_ref_clks(lmk_freq=245.76, lmx_freq=491.52)
time.sleep(0.2)
if STAGE == 'clocks':
    print('RISCQ_4X2_SMOKE: STAGE clocks done')
    sys.exit(0)

# ---------------------------------------------------------------- 2. overlay ---------------
banner(f'overlay download: {BIT}')
ol = Overlay(BIT)
Clocks.fclk0_mhz = 96.968727  # the TIMING-ANALYZED pl_clk0 (BD achieved value; Codex final-review finding 5)
say(f'pl_clk0 pinned: {Clocks.fclk0_mhz:.3f} MHz')
say(f'ip_dict: {list(ol.ip_dict.keys())}')
if 'rf_data_converter' not in ol.ip_dict:
    fail(2, 'this bitstream has no RFDC AXI-Lite — refusing (no way to gate on the tiles)')
rfdc = ol.rf_data_converter
if type(rfdc).__name__ == 'DefaultIP':
    fail(2, 'RFDC bound as DefaultIP (xrfdc not imported before Overlay?)')
if STAGE == 'overlay':
    print('RISCQ_4X2_SMOKE: STAGE overlay done')
    sys.exit(0)

# ---------------------------------------------------------------- 3. tile status gate ------
banner('RFDC tile status (xrfdc IPStatus)')


def tile_report():
    st = rfdc.IPStatus
    out = {}
    for kind, key in (('DAC', 'DACTileStatus'), ('ADC', 'ADCTileStatus')):
        for t, s in enumerate(st[key]):
            if not s['IsEnabled']:
                continue
            out[(kind, t)] = s
            say(f'{kind} tile {t}: state={s["TileState"]:2d} powerup={s["PowerUpState"]} '
                f'pll={s["PLLState"]} blocks=0x{s["BlockStatusMask"]:x}')
    return out


def tile_ok(st, kind, t):
    s = st.get((kind, t))
    return bool(s) and s['PowerUpState'] == 1 and s['PLLState'] == 1


need = [('DAC', 0), ('DAC', 2), ('ADC', 2)]           # 228 (dspClk source), 230, 226
st = tile_report()
if not all(tile_ok(st, *k) for k in need):
    say('not all tiles up; StartUp() on the missing ones, re-read after 1 s')
    for kind, t in need:
        if not tile_ok(st, kind, t):
            tiles = rfdc.dac_tiles if kind == 'DAC' else rfdc.adc_tiles
            try:
                tiles[t].StartUp()
            except Exception as e:
                say(f'StartUp {kind}{t}: {e}')
    time.sleep(1.0)
    st = tile_report()
if not tile_ok(st, 'DAC', 0):
    fail(3, 'DAC tile 0 (dspClk source) not powered up / PLL locked — NOT touching the SoC')

# ---------------------------------------------------------------- 4. liveness counters -----
BASE = 0x80000000
mmio = MMIO(BASE, m.region_size * 8)


def wr(off, val):
    mmio.write(off, val & 0xFFFFFFFF)


def rd(off):
    return mmio.read(off)


banner('liveness counters in the host-control block')
h0 = rd(HOSTCTRL + 0x104); time.sleep(0.001); h1 = rd(HOSTCTRL + 0x104)
say(f'hostAlive: {h0:#010x} -> {h1:#010x}  {"ALIVE" if h1 != h0 else "DEAD"}')
d0 = rd(HOSTCTRL + 0x100); time.sleep(0.001); d1 = rd(HOSTCTRL + 0x100)
say(f'dspAlive : {d0:#010x} -> {d1:#010x}  {"ALIVE" if d1 != d0 else "DEAD"}')
if h1 == h0:
    fail(4, 'hostAlive counter does not advance (host-control block unreachable?)')
if d1 == d0:
    fail(5, 'dspAlive counter does not advance: dspClk/dspRst not live — NOT touching the dsp domain')
if STAGE == 'probe':
    print('RISCQ_4X2_SMOKE: STAGE probe done (dsp domain is live)')
    sys.exit(0)

# ---------------------------------------------------------------- 5. digital ---------------
banner('hold riscq core in reset (power-up default is asserted; write 1 anyway)')
wr(HOSTCTRL, 1)

banner(f'core-0 RAM ({RAM_BYTES//1024} KiB, dspClk domain) write/readback, single words')
N = min(1024, RAM_BYTES // 4)


def pattern(i, rnd):
    return ((i * 0x9E3779B1) ^ (rnd * 0x5DEECE66D)) & 0xFFFFFFFF


nbad_total = 0
for rnd in range(2):
    t = time.time()
    for i in range(N):
        wr(4 * i, pattern(i, rnd))
    tw_ = time.time() - t
    bad = [(i, rd(4 * i)) for i in range(N)]
    bad = [(i, v) for i, v in bad if v != pattern(i, rnd)]
    tr_ = time.time() - t - tw_
    say(f'RAM {N}-word pattern round {rnd}: {"PASS" if not bad else "FAIL"} ({len(bad)} bad); '
        f'{tw_*1e6/N:.1f} us/write, {tr_*1e6/N:.1f} us/read')
    for i, v in bad[:8]:
        again = rd(4 * i)
        say(f'   word {i:4d} @ {4*i:#07x}: got {v:#010x} expected {pattern(i, rnd):#010x} '
            f'reread {again:#010x} ({"persistent" if again != pattern(i, rnd) else "transient"})')
    nbad_total += len(bad)

ok = nbad_total == 0
print('RISCQ_4X2_SMOKE:', 'PASS' if ok else 'FAIL', '(stage digital)', flush=True)
sys.exit(0 if ok else 6)
