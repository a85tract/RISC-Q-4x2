# rfsoc4x2-2dac-adcb

`rfsoc4x2-2dac-fine` with the readout ADC moved to **ADC1 (connector ADC_A)**: identical SoC parameters
(interp 2/2, 16384 envelope lines, rob 65536, 32-bit frequency word, queue depth 8), `dac_map
[[0, 1]]` (gate ch0 -> DAC0 = connector DAC_B, readout ch1 -> DAC1 = connector DAC_A) and `adc_map [1]`. The generated RTL
differs from the 2dac-fine RTL in exactly the four lines that pick `io_adc_1_payload` instead of
`io_adc_0_payload`; the block design, RFDC configuration and constraints are the same. Use it when
the loopback / readout cable is on connector ADC_A (DAC_A -> ADC_A is then the readout loop).

Built 2026-09-03 with Vivado 2026.1 from `gateware/configs/rfsoc4x2-2dac-adcb.json` (= params.json);
top.xsa sha256 c64812569c9d9abd4572e87ae352531f271ebc9f72c130d7c8bc5d3a63019cfe (what the board
server reports as xsa_sha after load); timing: WNS +0.015 ns, WHS +0.005 ns, TNS/THS 0, 157178
endpoints (riscq_bd_wrapper_timing_summary_routed.rpt); DRC: only DSP48 pipelining advisories
(DPIP-2 / DPOP-3 / DPOP-4), no errors. board.json = the shared 4x2 clock/Nyquist config.

Implementation history (same synthesized netlist):
- attempt 1 (build-riscvsoc-bd.sh defaults: Performance_NetDelay_high strategy but the script's
  `ExtraNetDelay_high` placer directive, late constraints applied): WNS -0.070 ns, TNS -2.46 ns
  after post-route phys_opt — an xsa was written by the flow and REJECTED (kept out of the repo).
- attempt 2 (the recipe that closed 2dac-fine: Performance_NetDelay_high with its own placer
  directive, phys_opt + post-route phys_opt, same late constraints): WNS +0.015 ns — this xsa.

Verification:
- board (2026-09-03, readout loop DAC1 -> ADC1 = connectors DAC_A -> ADC_A): `loopback_check.py --bundle rfsoc4x2-2dac-adcb --ch 1`
  -> 6018 codes at 82.00 MHz; `artiq_rx_demo.py --remote --bundle rfsoc4x2-2dac-adcb` -> RX_DEMO:
  PASS — IQ ratio |r| = 0.9998 constant to 0.01 % / 0.002 deg over the three cases, demod +90 deg
  -> hw +89.984 deg, tone +90 deg -> hw -90.016 / host -90.016 deg, res = sign(real); trace part A
  = the readout tone alone (6532 codes). The DAC0 gate path of THIS bitstream was not observed
  (DAC0 = connector DAC_B was cabled to ADC_B, which this bundle does not record); the same gate
  datapath is verified on rfsoc4x2-2dac-fine.
- co-simulation (Verilator, loopback DAC1 -> ADC1): `artiq_rx_demo.py --cosim --config
  gateware/configs/rfsoc4x2-2dac-adcb.json --loopback-src 1 --loopback-dst 1` -> RX_DEMO: PASS
  (demod +90 -> +90.002 deg, tone +90 -> hw -90.000 / host -89.997 deg, res = sign(real); the
  capture shows the readout tone through ADC1).
