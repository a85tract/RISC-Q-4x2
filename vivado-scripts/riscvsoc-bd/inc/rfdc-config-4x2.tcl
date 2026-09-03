# RF Data Converter configuration for the RealDigital RFSoC 4x2 (XCZU48DR).
#
# Transplanted VERBATIM from the dev-line board-verified rfdc-4x2.tcl (itself from the QubiC 4x2
# port: analog SMA loopback S8 PASS — linearity 2.00x/step, phase -90.06 deg for a 90 deg shift,
# 983x off-tone rejection), including the staged parameter application and post-apply assertions.
# Physical mapping (board-measured via the vendor base overlay):
#     s00_axis -> DAC tile 228 slice 0 -> DAC_A SMA        (TOP DAC0_AXIS)
#     s20_axis -> DAC tile 230 slice 0 -> DAC_B SMA        (TOP DAC1_AXIS)
#     m20_axis <- ADC tile 226 slice 0 <- ADC_A SMA        (TOP ADC0_AXIS)
#     m22_axis <- ADC tile 226 slice 2 <- ADC_B SMA        (TOP ADC1_AXIS)
# ADC_A/B are on tile 226, NOT 224; tile 224's factory-default slices must be explicitly disabled
# or its clock pins dangle (BD 41-758). No CLK104: tile refs are 491.52 MHz from the on-board
# LMX2594s (programmed at runtime via xrfclk — lmx_freq=491.52 is MANDATORY), each used tile runs
# its own PLL. MTS off: cross-tile DAC phase determinism is NOT claimed until measured.

proc _rfdc_set {args} {
  if {[catch {set_property -dict $args [get_bd_cells rf_data_converter]} err]} {
    error "rfdc-config-4x2: failed to apply $args : $err"
  }
}
_rfdc_set CONFIG.Axiclk_Freq {100}
# stage 1: enable tiles
_rfdc_set CONFIG.ADC_Slice00_Enable {false} CONFIG.ADC_Slice02_Enable {false} CONFIG.ADC2_Enable {1} CONFIG.ADC2_Link_Coupling {0} CONFIG.ADC2_PLL_Enable {true} CONFIG.DAC0_Enable {1} CONFIG.DAC0_Link_Coupling {0} CONFIG.DAC0_PLL_Enable {true} CONFIG.DAC2_Enable {1} CONFIG.DAC2_Link_Coupling {0} CONFIG.DAC2_PLL_Enable {true}
# stage 2: enable slices
_rfdc_set CONFIG.ADC_Slice20_Enable {true} CONFIG.ADC_Slice22_Enable {true} CONFIG.DAC_Slice00_Enable {true} CONFIG.DAC_Slice20_Enable {true}
# stage 3: datapath modes
_rfdc_set CONFIG.ADC_Decimation_Mode20 {1} CONFIG.ADC_Mixer_Type20 {1} CONFIG.ADC_Coarse_Mixer_Freq20 {3} CONFIG.ADC_Dither20 {false} CONFIG.ADC_OBS20 {0} CONFIG.ADC_Decimation_Mode22 {1} CONFIG.ADC_Mixer_Type22 {1} CONFIG.ADC_Coarse_Mixer_Freq22 {3} CONFIG.ADC_Dither22 {false} CONFIG.ADC_OBS22 {0} CONFIG.DAC_Interpolation_Mode00 {1} CONFIG.DAC_Mixer_Type00 {1} CONFIG.DAC_Coarse_Mixer_Freq00 {3} CONFIG.DAC_Mode00 {3} CONFIG.DAC_Nyquist00 {0} CONFIG.DAC_Interpolation_Mode20 {1} CONFIG.DAC_Mixer_Type20 {1} CONFIG.DAC_Coarse_Mixer_Freq20 {3} CONFIG.DAC_Mode20 {3} CONFIG.DAC_Nyquist20 {0}
# stage 4: sampling rates
_rfdc_set CONFIG.ADC2_Sampling_Rate {1.96608} CONFIG.DAC0_Sampling_Rate {7.86432} CONFIG.DAC2_Sampling_Rate {7.86432}
# stage 5: AXIS samples per word
_rfdc_set CONFIG.ADC_Data_Width20 {4} CONFIG.ADC_Data_Width22 {4}
# stage 6: tile PLLs
_rfdc_set CONFIG.ADC2_Clock_Dist {0} CONFIG.ADC2_Clock_Source {2} CONFIG.ADC2_Multi_Tile_Sync {false} CONFIG.ADC2_PLL_Enable {true} CONFIG.DAC0_Clock_Dist {0} CONFIG.DAC0_Clock_Source {4} CONFIG.DAC0_Multi_Tile_Sync {false} CONFIG.DAC0_PLL_Enable {true} CONFIG.DAC2_Clock_Dist {0} CONFIG.DAC2_Clock_Source {6} CONFIG.DAC2_Multi_Tile_Sync {false} CONFIG.DAC2_PLL_Enable {true}
# stage 7: reference clocks
_rfdc_set CONFIG.ADC2_Refclk_Freq {491.520} CONFIG.DAC0_Refclk_Freq {491.520} CONFIG.DAC2_Refclk_Freq {491.520}
# stage 8: tile output clocks
_rfdc_set CONFIG.ADC2_Outclk_Freq {122.880} CONFIG.DAC0_Outclk_Freq {491.520} CONFIG.DAC2_Outclk_Freq {491.520}

# --- verify the effective configuration against the board-validated operating point. A silently
# --- ignored or clamped parameter must fail the build HERE, not hours later in implementation.
proc _rfdc_expect {param expected} {
  set actual [get_property CONFIG.$param [get_bd_cells rf_data_converter]]
  if {$actual != $expected} {
    error "rfdc-config-4x2 assertion failed: CONFIG.$param is '$actual', expected '$expected'"
  }
  puts "rfdc-config-4x2 OK: CONFIG.$param = $actual"
}
_rfdc_expect ADC2_Fabric_Freq 491.520
_rfdc_expect ADC2_Sampling_Rate 1.96608
_rfdc_expect DAC0_Fabric_Freq 491.520
_rfdc_expect DAC0_Sampling_Rate 7.86432
_rfdc_expect DAC0_Outclk_Freq 491.520
_rfdc_expect DAC2_Fabric_Freq 491.520
_rfdc_expect DAC2_Sampling_Rate 7.86432
_rfdc_expect DAC2_Outclk_Freq 491.520
_rfdc_expect ADC_Data_Width20 4
