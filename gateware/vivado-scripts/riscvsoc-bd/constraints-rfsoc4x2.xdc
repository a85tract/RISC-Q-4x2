# RFSoC 4x2 (xczu48dr) physical + timing constraints for the PulseTableSoc block-design wrapper.
#
# Clocks (see bd-build-4x2.tcl): the RF tile reference clocks are 491.52 MHz (period 2.034505 ns) from the
# on-board LMX2594s (2.034 is used: rounding down is conservative for setup); the PL SYSREF copy
# (SYS_REF_FPGA, 7.68 MHz LVDS on AP18/AR18) feeds the converter IP user_sysref through a synchronizer.
# dspClk is either the MTS source tile output clock clk_dac2 (RISCQ_DSPCLK=tile) or the LMK PL clock
# (FPGA_REFCLK_IN 122.88 MHz on AN11/AP11) x4 through the dsp MMCM (RISCQ_DSPCLK=mmcm) — the latter has
# its own files, constraints-rfsoc4x2-mmcm*.xdc, that bd-finalize.tcl adds for that variant only, since
# XDC files take no if/foreach (Designutils 20-1307). hostClk (PS pl_clk0) and the IP-created clocks are
# propagated by Vivado; the async clock groups CANNOT live here (those clocks do not exist at synthesis):
# they are in constraints-rfsoc4x2-late.xdc (implementation-only, PROCESSING_ORDER LATE).
# the PL clock (RISCQ_DSPCLK=mmcm only) is constrained in constraints-rfsoc4x2-mmcm.xdc, which
# bd-finalize.tcl adds for that variant: XDC files take no control flow (Designutils 20-1307).
set_property -dict {PACKAGE_PIN AP18 IOSTANDARD LVDS DIFF_TERM_ADV TERM_100} [get_ports pl_sysref_clk_p]
set_property -dict {PACKAGE_PIN AR18 IOSTANDARD LVDS DIFF_TERM_ADV TERM_100} [get_ports pl_sysref_clk_n]
# the PL SYSREF is a 7.68 MHz level captured by an xpm_cdc synchronizer, not a clock for logic
create_clock -period 130.208 -name pl_sysref_clk_p [get_ports pl_sysref_clk_p]

# only the clock-owning tiles have an LMX reference port (DAC0 runs on DAC2's distributed sample clock)
create_clock -period 2.034 -name dac2_clk_clk_p [get_ports dac2_clk_clk_p]
create_clock -period 2.034 -name adc2_clk_clk_p [get_ports adc2_clk_clk_p]
