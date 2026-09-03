# RFSoC 4x2 (xczu48dr) physical + timing constraints for the PulseTableSoc block-design wrapper.
# Board-verified structure (dev-line RISC-Q build 9 WNS +0.012 and the QubiC 4x2 port WNS +0.059).
#
# The three RF tile reference clocks are 491.52 MHz (period 2.034505 ns) from the on-board
# LMX2594 synthesisers. 2.034 ns is used: rounding down is conservative for setup analysis.
# hostClk (PS pl_clk0) and dspClk (RFDC clk_dac0 output) are derived clocks; Vivado propagates
# them from the PS IP and the RFDC IP respectively, so nothing else needs a create_clock here.
# The async clock groups CANNOT live here: the PS / RFDC IP clocks do not exist at synthesis —
# they are in constraints-rfsoc4x2-late.xdc (implementation-only, PROCESSING_ORDER LATE).
create_clock -period 2.034 -name dac0_clk_clk_p [get_ports dac0_clk_clk_p]
create_clock -period 2.034 -name dac2_clk_clk_p [get_ports dac2_clk_clk_p]
create_clock -period 2.034 -name adc2_clk_clk_p [get_ports adc2_clk_clk_p]
