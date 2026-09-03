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

Verification:
- co-simulation (Verilator, loopback DAC1 -> ADC0): `software/examples/artiq_rx_demo.py --cosim
  --config gateware/configs/rfsoc4x2-2dac-fine.json --loopback-src 1` -> RX_DEMO: PASS
  (IQ ratio constant to 0.004 deg / 0.01 %, demod +90 -> +90.001 deg, tone +90 -> hw -89.999 /
  host -89.997 deg, res = sign(real); the capture shows the readout tone alone, as it should).
- board: pending — needs the loopback cable on DAC1 -> ADC0 (this file is updated when done).
