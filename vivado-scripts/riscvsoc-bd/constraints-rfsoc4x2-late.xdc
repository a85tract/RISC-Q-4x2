# RFSoC 4x2: applied at IMPLEMENTATION only, PROCESSING_ORDER LATE — these clocks are created by
# IP constraints (the PS's clk_pl_0 and the RFDC's per-tile RFDAC*/RFADC* clocks) and do not exist
# when synthesis reads the ordinary constraint file.
#
# Two facts, both measured on this board during the dev-line port before this file was written:
#   1. Without the group, clk_pl_0 <-> RFDAC0_CLK cross-domain paths are timed as synchronous
#      2.035 ns paths and fail at WNS -1.672. Every such crossing in the SoC goes through
#      SpinalHDL BufferCC / gray-coded StreamFifoCC, so asynchronous is the truth, not a waiver.
#   2. RFDAC0_CLK is a PRIMARY clock created by the RFDC IP on the tile's fabric-clock output —
#      it is NOT generated from the dac0_clk_clk_p board clock, so an -include_generated_clocks
#      bucket does NOT capture it. The tile clocks must be named explicitly.
set_clock_groups -asynchronous \
  -group [get_clocks clk_pl_0] \
  -group [get_clocks -quiet {RFDAC*_CLK RFADC*_CLK dac0_clk_clk_p dac2_clk_clk_p adc2_clk_clk_p}]
