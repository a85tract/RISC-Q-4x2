# RF Data Converter configuration for the RealDigital RFSoC 4x2 (XCZU48DR) — multi-tile synchronized.
#
# Physical mapping (RealDigital RFSoC 4x2 reference manual — DACA tile 230, DACB tile 228, ADCA/ADCB
# tile 226 blocks 1/0 — confirmed on the board 2026-09-03):
#     s00_axis -> DAC tile 228 slice 0 -> DAC_B SMA        (TOP DAC0_AXIS)
#     s20_axis -> DAC tile 230 slice 0 -> DAC_A SMA        (TOP DAC1_AXIS)
#     m20_axis <- ADC tile 226 slice 0 <- ADC_B SMA        (TOP ADC0_AXIS)
#     m22_axis <- ADC tile 226 slice 2 <- ADC_A SMA        (TOP ADC1_AXIS)
# ADC_A/B are on tile 226, NOT 224; tile 224's factory-default slices must be explicitly disabled
# or its clock pins dangle (BD 41-758). No CLK104: tile refs are 491.52 MHz from the on-board
# LMX2594s (programmed at runtime via xrfclk — lmx_freq=491.52 is MANDATORY).
#
# MULTI-TILE SYNCHRONIZATION (the upstream ZCU216 RISC-Q and QubiC both run MTS; the earlier 4x2 builds
# left it off and paid with a per-power-cycle DAC_A <-> DAC_B latency offset). Same recipe as the Xilinx
# RFSoC-MTS reference design for this board (boards/RFSoC4x2/build_mts/mts.tcl):
#   * ONE DAC tile owns the PLL and distributes its 7.86432 GHz sample clock to the others — tile 230
#     (DAC2, DAC_A) is the source (Clock_Dist 2), tile 228 (DAC0, DAC_B) takes it (PLL off, Clock_Source 6
#     = DAC tile 2, "Refclk" = the distributed 7864.32 MHz), and tile 229 (DAC1) is enabled ONLY to forward
#     the clock along the distribution network (slice 10 on, its stream left idle, MTS group excludes it).
#     Both DAC_A and DAC_B then run off one physical sample clock — no LMX-to-LMX phase in between.
#   * ADC tile 226 (ADC2, both blocks) keeps its own PLL and DISTRIBUTES its sample clock to the other
#     three RF-ADC tiles (225, 224, 227: PLL off, one idle slice each, streams dropped) — the reference's
#     tile set. The IP greys ADC Multi_Tile_Sync out ("attempt to modify the value of disabled parameter
#     ADC2_Multi_Tile_Sync ... ignored") unless tile 224 is enabled (probe 2026-09-04: 226 alone, 226+225,
#     226+227 all refused; 226+225+224 and all four accepted), and distribution may not hop over a disabled
#     tile ("Tile hopping is not allowed"), so 225 must be on for 224 to receive. SYSREF reaches the ADC
#     tiles over the on-chip SYSREF distribution chain that xrfdc programs into every tile
#     (XRFdc_MTS_Sysref_Dist writes 224..227) — keeping all of them powered, as the reference does, keeps
#     that chain intact. MTS pins 226's latency; the software mask syncs tile 2 only.
#   * Multi_Tile_Sync on every enabled tile; the analog SYSREF (7.68 MHz from the LMK, DAC_230_SYSREF pins)
#     comes in on sysref_in, and the PL copy (SYS_REF_FPGA, AP18/AR18) is synchronized into dspClk and fed
#     to user_sysref_adc/dac (bd-build-4x2.tcl).
#   * The fabric clock (dspClk) is selected by RISCQ_DSPCLK in bd-build-4x2.tcl: "tile" (default, shipped)
#     = the source tile's own clk_dac2 through a BUFG (0.035 ns of timing uncertainty; the SoC sits in
#     reset while MTS re-aligns the tile clock at load), "mmcm" = the LMK's PL clock (FPGA_REFCLK_IN
#     122.88 MHz) x4 through an MMCM (the reference design's structure, 0.049 ns).
# Software: riscq.board.pynq_driver.mts() (xrfdc MultiConverter_Sync, DAC tiles 0b0101, ADC tiles 0b0100,
# RefTile 2), pinned to the latencies recorded in the bundle's board.json. The reference design syncs
# ADC tiles 0b0101 (224 + 226) because it uses all four ADC connectors; this design's two ADCs are both
# on tile 226, so only that tile is in the ADC group. The reference's MMCM reset / lock check before the
# sync belongs to its PL-clock MMCM and has no counterpart in the tile-clock variant.

proc _rfdc_set {args} {
  if {[catch {set_property -dict $args [get_bd_cells rf_data_converter]} err]} {
    error "rfdc-config-4x2: failed to apply $args : $err"
  }
}
_rfdc_set CONFIG.Axiclk_Freq {100}
# ONE dict, as the RFSoC-MTS reference does: the IP validates every set_property call as a whole, and a
# distributed clock tree cannot be built piecemeal (enabling DAC229's slice before its clock source is
# known is rejected: "DAC228 or DAC230 must distribute a clock that can be used by DAC229").
# an RF-ADC tile that only receives the distributed clock: PLL off, clock from tile 226, one idle slice
proc _rfdc_adc_receiver {n} {
  return [list \
    CONFIG.ADC${n}_Enable {1} CONFIG.ADC${n}_Link_Coupling {0} CONFIG.ADC${n}_PLL_Enable {false} \
    CONFIG.ADC${n}_Clock_Dist {0} CONFIG.ADC${n}_Clock_Source {2} CONFIG.ADC${n}_Sampling_Rate {1.96608} \
    CONFIG.ADC${n}_Refclk_Freq {1966.080} CONFIG.ADC${n}_Outclk_Freq {122.880} CONFIG.ADC${n}_Multi_Tile_Sync {true} \
    CONFIG.ADC_Slice${n}0_Enable {true} CONFIG.ADC_Data_Width${n}0 {4} CONFIG.ADC_Decimation_Mode${n}0 {1} \
    CONFIG.ADC_Mixer_Type${n}0 {1} CONFIG.ADC_Coarse_Mixer_Freq${n}0 {3} CONFIG.ADC_Dither${n}0 {false} CONFIG.ADC_OBS${n}0 {0}]
}
_rfdc_set \
  CONFIG.ADC_Slice02_Enable {false} \
  CONFIG.ADC2_Enable {1} CONFIG.ADC2_Link_Coupling {0} CONFIG.ADC2_PLL_Enable {true} CONFIG.ADC2_Clock_Dist {2} CONFIG.ADC2_Clock_Source {2} \
  CONFIG.ADC2_Sampling_Rate {1.96608} CONFIG.ADC2_Refclk_Freq {491.520} CONFIG.ADC2_Outclk_Freq {122.880} CONFIG.ADC2_Multi_Tile_Sync {true} \
  CONFIG.ADC_Slice20_Enable {true} CONFIG.ADC_Slice22_Enable {true} CONFIG.ADC_Data_Width20 {4} CONFIG.ADC_Data_Width22 {4} \
  CONFIG.ADC_Decimation_Mode20 {1} CONFIG.ADC_Mixer_Type20 {1} CONFIG.ADC_Coarse_Mixer_Freq20 {3} CONFIG.ADC_Dither20 {false} CONFIG.ADC_OBS20 {0} \
  CONFIG.ADC_Decimation_Mode22 {1} CONFIG.ADC_Mixer_Type22 {1} CONFIG.ADC_Coarse_Mixer_Freq22 {3} CONFIG.ADC_Dither22 {false} CONFIG.ADC_OBS22 {0} \
  {*}[_rfdc_adc_receiver 0] {*}[_rfdc_adc_receiver 1] {*}[_rfdc_adc_receiver 3] \
  CONFIG.DAC2_Enable {1} CONFIG.DAC2_Link_Coupling {0} CONFIG.DAC2_PLL_Enable {true} CONFIG.DAC2_Clock_Dist {2} CONFIG.DAC2_Clock_Source {6}   CONFIG.DAC2_Sampling_Rate {7.86432} CONFIG.DAC2_Refclk_Freq {491.520} CONFIG.DAC2_Outclk_Freq {491.520} CONFIG.DAC2_Multi_Tile_Sync {true}   CONFIG.DAC0_Enable {1} CONFIG.DAC0_Link_Coupling {0} CONFIG.DAC0_PLL_Enable {false} CONFIG.DAC0_Clock_Dist {0} CONFIG.DAC0_Clock_Source {6}   CONFIG.DAC0_Sampling_Rate {7.86432} CONFIG.DAC0_Refclk_Freq {7864.320} CONFIG.DAC0_Outclk_Freq {491.520} CONFIG.DAC0_Multi_Tile_Sync {true}   CONFIG.DAC1_Enable {1} CONFIG.DAC1_Link_Coupling {0} CONFIG.DAC1_PLL_Enable {false} CONFIG.DAC1_Clock_Dist {0} CONFIG.DAC1_Clock_Source {6}   CONFIG.DAC1_Sampling_Rate {7.86432} CONFIG.DAC1_Refclk_Freq {7864.320} CONFIG.DAC1_Outclk_Freq {491.520} CONFIG.DAC1_Multi_Tile_Sync {true}   CONFIG.DAC_Slice00_Enable {true} CONFIG.DAC_Slice10_Enable {true} CONFIG.DAC_Slice20_Enable {true}   CONFIG.DAC_Interpolation_Mode00 {1} CONFIG.DAC_Mixer_Type00 {1} CONFIG.DAC_Coarse_Mixer_Freq00 {3} CONFIG.DAC_Mode00 {3} CONFIG.DAC_Nyquist00 {0}   CONFIG.DAC_Interpolation_Mode10 {1} CONFIG.DAC_Mixer_Type10 {1} CONFIG.DAC_Coarse_Mixer_Freq10 {3} CONFIG.DAC_Mode10 {3} CONFIG.DAC_Nyquist10 {0}   CONFIG.DAC_Interpolation_Mode20 {1} CONFIG.DAC_Mixer_Type20 {1} CONFIG.DAC_Coarse_Mixer_Freq20 {3} CONFIG.DAC_Mode20 {3} CONFIG.DAC_Nyquist20 {0}

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
_rfdc_expect ADC2_Multi_Tile_Sync true
_rfdc_expect ADC2_Clock_Dist 2
foreach _t {0 1 3} {
  _rfdc_expect ADC${_t}_Clock_Source 2
  _rfdc_expect ADC${_t}_PLL_Enable false
  _rfdc_expect ADC${_t}_Multi_Tile_Sync true
}
_rfdc_expect DAC0_Fabric_Freq 491.520
_rfdc_expect DAC0_Sampling_Rate 7.86432
_rfdc_expect DAC0_Outclk_Freq 491.520
_rfdc_expect DAC0_Clock_Source 6
_rfdc_expect DAC0_PLL_Enable false
_rfdc_expect DAC0_Multi_Tile_Sync true
_rfdc_expect DAC1_Clock_Source 6
_rfdc_expect DAC2_Fabric_Freq 491.520
_rfdc_expect DAC2_Sampling_Rate 7.86432
_rfdc_expect DAC2_Outclk_Freq 491.520
_rfdc_expect DAC2_Clock_Dist 2
_rfdc_expect DAC2_PLL_Enable true
_rfdc_expect DAC2_Multi_Tile_Sync true
_rfdc_expect ADC_Data_Width20 4
