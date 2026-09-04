# RFSoC 4x2, RISCQ_DSPCLK=mmcm only, implementation-only / LATE (added by bd-finalize.tcl): the MMCM's
# output clock (dspClk) is asynchronous to every other domain. Keep this a plain command — XDC files take
# no if/foreach (Designutils 20-1307).
set_clock_groups -asynchronous \
  -group [get_clocks -include_generated_clocks pl_clk_clk_p] \
  -group [get_clocks -quiet {clk_pl_0 RFDAC*_CLK RFADC*_CLK dac2_clk_clk_p adc2_clk_clk_p pl_sysref_clk_p}]
