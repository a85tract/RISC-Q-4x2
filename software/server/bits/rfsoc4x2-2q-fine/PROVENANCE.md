# rfsoc4x2-2q-fine

Two cores on one timeline: core 0 = dds 0 (gate) + dds 1 (readout) summed on **DAC1 = connector DAC_A**,
its raw trace from **ADC1 = connector ADC_A**; core 1 = dds 2 + dds 3 on **DAC0 = DAC_B**, trace from
**ADC0 = ADC_B** (`dac_map [[1, 1], [0, 0]]`, `adc_map [1, 0]`, `rob_per_core`, `run_origin`). "Fine"
parameters as the one-core bundles (interp 2/2, 16384 envelope lines, 65536-batch traces per core, 32-bit
frequency word, queue depth 8). Multi-tile synchronized RF tiles: DAC tile 230 (DAC_A) owns the sample
PLL and distributes 7.86432 GHz to tile 228 (DAC_B) through 229; ADC tile 226 (both connectors) keeps its
PLL and distributes to 225/224/227, which are enabled idle (the converter IP offers RF-ADC MTS only with
tile 224 on; the SYSREF distribution chain runs through every tile). The fabric clock is tile 230's own
output clock (`RISCQ_DSPCLK=tile`, the flow default); the PL SYSREF is synchronized into it for the
converter IP. That fabric clock is a deliberate deviation from the RFSoC-MTS reference, which clocks
its user logic from an MMCM on the LMK's PL clock: MTS aligns the converters' latency and re-aligns the
tile clocks at load, so the SoC is held in reset and left untouched until `load()` has finished (the
server enforces the order); the MMCM variant is kept in the flow but did not close timing (below).

Built 2026-09-04 with Vivado 2026.1 from `gateware/configs/rfsoc4x2-2q-fine.json` (= params.json),
project `rfsoc4x2-2q-fine-mts-t`, implementation run impl_3 (strategy Performance_NetDelay_high, place
ExtraNetDelay_high, phys_opt AggressiveExplore after placement and after routing, route
AggressiveExplore): WNS +0.020 ns, WHS +0.010 ns, TNS/THS 0, 265401 endpoints, every net routed
(`timing_impl.txt` here, the head of Vivado's routed summary); DRC (`drc_impl.rpt`): one RTSTAT-10 warning (a net with no routable loads), no errors. top.xsa
sha256 e7e3ae73f93e53e56c7c765160cacb56c4ee106eb65bcf5357f83bfc444e4ad3 (what the board server reports as
`xsa_sha` after load). board.json: MTS `daclatency 260` / `adclatency 88` (DAC tiles 228 + 230, ADC tile
226, RefTile 2), `"required": true`; fclk0_mhz 96.968727. The same netlist clocked from an MMCM on the
LMK PL clock (the RFSoC-MTS reference structure, `RISCQ_DSPCLK=mmcm`) has 14 ps more clock uncertainty
and stopped between WNS -0.001 and -0.13 ns in 17 implementation runs; it is not shipped.

RTL fix carried by this bundle (2026-09-04): the trace recorder now writes the LAST batch of a recording
(upstream wrote addresses 0, 0, 1, ..., N-2 for an N-batch fire, so the final batch of every window read
back as the previous run's sample). The one-core bundles predate it.

Verification:
- co-simulation (Verilator, two loopbacks DAC1->ADC1 / DAC0->ADC0, `sim/cosim2q_check.py`, one clean
  end-to-end run on this RTL): PASS 11/11 — ion-trap reference; per-core isolation incl. the last batch;
  distinct tones per core; full-scale sign; full trace depth to the last sample; one origin with
  asymmetric kernels (same tone -> identical traces, half a turn -> inverted); phase modes with a hop;
  the 32-bit batch clock wrapping inside a run; the 1-core build's trace within one 16-bit phase LSB
  (max diff 3 codes).
- board (2026-09-04, DAC_A -> ADC_A and DAC_B -> ADC_B cabled): MTS at every load — free sync measured
  DAC latencies {228: 256, 230: 256} / ADC {226: 80} identically on three loads, then pinned to 260 / 88:
  `mts_result` 0 and latencies {228: 260, 230: 260} / {226: 88} on three further loads; `dsp_mhz`
  491.57-491.59 (the tile clock). A warm re-load runs MTS only after every tile in the sync masks reports
  TileState 15 (`PynqDriver._wait_tiles_started`; without it the second load failed with "ADC tile 2 in
  Multi-Tile group not started").
- `software/examples/artiq_api_demo.ipynb` executed live (2026-09-04, two identical cables on the two
  loops, the bundle's SYNC-mode LMX list programmed — see the clock item below): demo 1 (the ion-trap
  reference waveform on DAC_A, ADC_A capture vs `reference/waveform.npz`) — delay fit D = 192.6 ns,
  carrier phases capture vs generator -0.24 / +0.28 / +0.00 deg at 83.765 / 80.235 / 82 MHz, in-pulse
  residual rms 0.60 %. Demo 2 (four dds on one timeline, DAC_A and DAC_B): both cores report the same
  origin and telemetry; the envelope cross-correlation lag between the two loopback traces is 0 samples;
  level ratio ADC_B / ADC_A 0.98 (-0.2 dB); DAC_B - DAC_A carrier phase +1.39 deg at 83.765 MHz (asked 0)
  and +1.28 deg at 82 MHz (asked +90 -> +91.28) — the connector / cable path difference of this bench, a
  few tens of ps. The same notebook run with xrfclk's default LMX list (earlier the same evening) gave
  D = 192.487 ns, 0.62 %, -0.26 / +0.26 / -0.00 deg and +1.55 / +1.44 deg: the SYNC list moves the whole
  DAC -> ADC delay by +128 ps (one LMX VCO period), nothing else. Measured with the same ABSOLUTE tone on
  dds 1 and dds 3 (path_phase_vs_freq, default list): B - A = -1.4 / +1.4 / +3.2 / +4.6 / +5.9 deg at 41 /
  82 / 123 / 164 / 205 MHz, envelope lag 0 samples at every frequency, levels within 3 %.
  Earlier the same day, with the cable set the board came with, the A loop delivered 4922 / 2664 / 1020 /
  153 / 1606 codes at those frequencies against a flat ~6000 on the B loop (a -32 dB notch near 164 MHz,
  a 575 MHz pickup, 7 dB less level at 82 MHz) and the demo read +67 deg and an 11 % residual; swapping
  both cables for identical ones removed all of it — so that was the cable, confirmed. The offset was
  measured within one session; re-measure after a power cycle before relying on it.

- Clock (re)programming and the LMX2594 phase-SYNC list (2026-09-04 evening, `software/examples/
  lmx_relock_check.py`, 4 locks per arm; per lock the 82 MHz ABSOLUTE tone on dds 1 -> ADC_A and dds 3 -> ADC_B,
  each path's carrier phase against the generator, envelope lag, levels, MTS latencies, measured dsp clock):
  with PYNQ's xrfclk LMX list, re-locking one LMX alone (LMK untouched) repeats within 0.05 deg (lmxdac:
  spreads A 0.05 / B 0.02 deg; lmxadc: 0.04 / 0.06), but reprogramming the whole chain (LMK + both LMXs,
  `refclks`) lands in one of two states — A 69.4 / 68.6, B 70.8 / 69.0 deg, alternating — spreads A 0.77 /
  B 1.83 / B-A 1.06 deg (25-60 ps at 82 MHz); MTS reached 260 / 88 and the dsp clock read 491.6 MHz every
  time. With this bundle's `lmx2594-491.52-sync.txt` (xrfclk's LMX2594_491.52.txt with R0 0x241C -> 0x641C,
  VCO_PHASE_SYNC = 1, and R36 PLL_N 320 -> 80 for the included /4; TI SNAS696C 7.3.10 category 1: fOUT is an
  integer multiple of OSCin, integer mode, CHDIV 16 -> no SYNC pulse needed; derived by
  `software/examples/clocks/make_lmx_sync_list.py`) the whole-chain reprogramming repeats: A 65.61 / 65.66 /
  65.60 / 65.62, B 66.93 / 66.99 / 66.87 / 66.87 deg -> spreads 0.06 / 0.12 / 0.08 deg; single re-locks
  0.03 / 0.09 (lmxdac) and 0.01 / 0.12 (lmxadc). The SYNC state sits at a fixed ~3 deg other absolute phase
  than the default list's states. Loads take 4-22 s in either state (`lmx_load_timing` probe). So the
  board.json `"lmx_regs"` entry is part of this bundle: the server programs the list into both LMXs after
  xrfclk's defaults with the datasheet sequence (reset on/off, R112..R0, 10 ms, R0 again) and every load
  guarantees that state (`info()["refclks"]` reports (245.76, 491.52, {lmxdac: sha, lmxadc: sha}) with the
  sha256 of the programmed values, 30345f5ecd6415a3... for this list; the file itself is in SHA256SUMS).
  The notebook below was re-executed in this state. Not covered: real power cycles, scope probing of the 491.52 MHz edges,
  the LMX's own lock-detect readback (no SPI read path from PYNQ; the evidence of lock is MTS + the
  reproducible phase). Earlier in the evening a first version of this experiment reported the SYNC list as
  ineffective: its whole-chain arm had loaded a bundle WITHOUT `"lmx_regs"`, and a load restores the bundle's
  clock state, so xrfclk's list had been re-applied before each measurement — the arm was re-run against the
  bundle carrying the list (`lmx_sync_sync_clocks.log`).

Evidence files next to this note: `timing_impl.txt` (the head of Vivado's routed timing summary:
design summary, clocks, worst paths), `drc_impl.rpt`, `cosim2q_check.log` (the 11/11 suite run).
Source: the working tree that this bundle is committed with; the generated Verilog used for the
build has sha256 7b6e75431a5c9b03ae32eaf95cfb9c50661c4c756df71eae764dd6f67bcbb4b1 (PulseTableSoc.v, `GenPulseTableSocJson rfsoc4x2-2q-fine.json ... vivado`),
the flow files sha256 6402b8f2fde9c6540c3fadd3f388c0c7064b38ef81f16a661e35c97d35936eba (bd-build-4x2.tcl, rfdc-config-4x2.tcl, constraints-rfsoc4x2.xdc,
constraints-rfsoc4x2-late.xdc, bd-finalize.tcl, concatenated in that order).
