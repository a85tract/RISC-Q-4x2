# rfsoc4x2-2dac-fine

The two-DAC variant of `rfsoc4x2-1q-fine`: identical SoC parameters (interp 2/2, 16384 envelope
lines, rob 65536, 32-bit frequency word, queue depth 8) with `dac_map [[0, 1]]` — the gate drive
(ch0) on DAC0 and the readout drive (ch1) on DAC1, ADC0 as the readout. Removing the two-input
DAC combine shortens the DAC pipeline by two dsp cycles (SocMap.dac_pipe 4 -> 2, mirrored by the
RTL's dacAlignStages), which the software derives from the same config.

Built 2026-09-03 with Vivado 2026.1 from `gateware/configs/rfsoc4x2-2dac-fine.json` (= params.json);
top.xsa sha256 b09c3de4e1400c160bb27a8d2c3c8df768dcfc74a15a72c7fcda99ed591938ce (what the board server reports as xsa_sha after load); timing: WNS 0.032 ns, TNS 0.000 ns (timing_attempt3.rpt);
board.json is the board-level clock/Nyquist config shared with rfsoc4x2-1q-fine (the PS clock is a
board preset the flow asserts).

Implementation history (Vivado 2026.1, all from the same synthesized netlist):
- attempt 1 (flow default Performance_NetDelay_high, no extra constraints): write_bitstream refused —
  DRC CASC-31 (the demod envelope RAM's 16-deep RAMB36 cascade placed across a clock-region row
  break) and WNS -0.078 ns on the trace-RAM write staging -> RAMB36 paths.
- attempt 2 (Performance_ExplorePostRoutePhysOpt + place ExtraTimingOpt + post-route phys_opt): a
  bitstream was written but WNS -0.127 ns (97 endpoints; worst path envReader address -> envelope
  RAM enable) — REJECTED, not shipped.
- attempt 3 (back to Performance_NetDelay_high + constraints-rfsoc4x2-late.xdc: hard RAM-only pblock
  pinning the demod RAM to CLOCKREGION_X1Y3, MAX_FANOUT_MODE CLOCK_REGION / FORCE_MAX_FANOUT on the
  trace-write staging FFs and the envReader address FDCEs): WNS +0.032 ns, TNS 0, no DRC — this xsa.

Verification:
- co-simulation (Verilator, loopback DAC1 -> ADC0): `software/examples/artiq_rx_demo.py --cosim
  --config gateware/configs/rfsoc4x2-2dac-fine.json --loopback-src 1` -> RX_DEMO: PASS
  (IQ ratio constant to 0.004 deg / 0.01 %, demod +90 -> +90.001 deg, tone +90 -> hw -89.999 /
  host -89.997 deg, res = sign(real); the capture shows the readout tone alone, as it should).
- board (2026-09-03, bundle loaded by the server = xsa sha above), BOTH DACs:
  - gate path ch0 -> DAC0 (DAC_A): 82 MHz tone at the ADC0 loopback, 6084 codes for amplitude 0.4
    (`software/examples/loopback_check.py --bundle rfsoc4x2-2dac-fine --ch 0`, cable DAC_A -> ADC_A);
    the full ion-trap sequence shows the gate tone alone on DAC_A at half the summed single-DAC
    amplitude (6052 vs 11956 codes), as it must once the two drives are on separate DACs.
  - readout path ch1 -> DAC1 (DAC_B): loopback 5867 codes at 82.00 MHz (cable DAC_B -> ADC_A), and
    `artiq_rx_demo.py --remote --bundle rfsoc4x2-2dac-fine` -> RX_DEMO: PASS — IQ ratio constant to
    0.01 % / 0.002 deg across the three cases, demod +90 deg -> hw +89.993 deg, tone +90 deg -> hw
    -90.008 / host -90.007 deg, res = sign(real). Trace part A shows the readout tone alone (6268
    codes), the gate on DAC_A being out of the recorded ADC0.
  - bench note: our board has a broken SMA ground return, so every loopback needs a second cable
    between the other DAC/ADC pair (here DAC_A -> ADC_B) — see software/server/README.md. That is
    a property of this board, not of the bundle.
