# rfsoc4x2-1q-fine

Built 2026-08-27 with Vivado 2026.1 from gateware/configs/rfsoc4x2-1q-fine.json (params.json is that
file), board-verified (MS_COMPARE PASS, RX_DEMO PASS on the RFSoC 4x2, physical DAC0->ADC0 loopback).
Both drive channels (gate ch0, readout ch1) are summed onto DAC0; ADC0 is the readout.
xsa sha256 4310d07dc687a1ad73b631c73501effb41e426c084a5f97f4ce41cdf95f2c4b1 is what the board
server reports as xsa_sha after load. board.json fclk0_mhz is this build's achieved PS clock.
