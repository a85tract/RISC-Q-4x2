# RFSoC 4x2, RISCQ_DSPCLK=mmcm only (added by bd-finalize.tcl): the LMK04828's PL clock — FPGA_REFCLK_IN,
# 122.88 MHz LVDS on AN11/AP11 — is the dsp MMCM's input.
set_property -dict {PACKAGE_PIN AN11 IOSTANDARD LVDS DIFF_TERM_ADV TERM_100} [get_ports pl_clk_clk_p]
set_property -dict {PACKAGE_PIN AP11 IOSTANDARD LVDS DIFF_TERM_ADV TERM_100} [get_ports pl_clk_clk_n]
create_clock -period 8.138 -name pl_clk_clk_p [get_ports pl_clk_clk_p]
