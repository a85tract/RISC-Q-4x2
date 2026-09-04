# ===========================================================================================
# RFSoC 4x2 (xczu48dr) block-design assembly: Zynq PS + user top IP + proc_sys_resets +
# RF Data Converter (2 DAC / 2 ADC, multi-tile synchronized) + AXI SmartConnect + the MTS clock tree.
#
# Clocking: hostClk is the PS pl_clk0; dspClk (491.52 MHz = 7.86432 GSPS / 16 samples per beat) drives the
# SoC and every converter stream and is selected by RISCQ_DSPCLK (see below): "tile" (default, the shipped
# rfsoc4x2-2q-fine) = the MTS source tile's own output clock clk_dac2, which MTS aligns to SYSREF at load
# while the SoC sits in reset; "mmcm" = the LMK04828's PL clock (FPGA_REFCLK_IN, 122.88 MHz LVDS on
# AN11/AP11) x4 through an MMCM, the Xilinx RFSoC-MTS reference structure (14 ps more clock uncertainty,
# which cost this design its timing closure). The PL SYSREF (SYS_REF_FPGA, 7.68 MHz LVDS on AP18/AR18) is
# synchronized into dspClk and fed to the converter IP's user_sysref_adc/dac in both variants.
# Publishes $ZYNQ_PS / $TOP / $PS_RST / $DSP_RST / $AXI_CONNECT / $RFDC for bd-finalize.tcl.
# ===========================================================================================

create_bd_design -dir $BUILD_DIR/bd $BD_NAME

# ---- Zynq UltraScale+ PS: pl_clk0 (100 MHz requested) / pl_resetn0 / M_AXI_HPM0_LPD ----
# apply_board_preset without 4x2 board files applies defaults; that is fine because the PS is
# actually configured by the PYNQ image's FSBL at boot — only the PL-facing choices matter here.
set ZYNQ_PS [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.* zynq_ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} \
  [get_bd_cells zynq_ps]
set_property -dict [list \
  CONFIG.PSU__USE__M_AXI_GP0 {0} \
  CONFIG.PSU__USE__M_AXI_GP1 {0} \
  CONFIG.PSU__USE__M_AXI_GP2 {1} \
  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
] [get_bd_cells zynq_ps]

# ---- user top IP ----
set TOP [create_bd_cell -type ip -vlnv user.org:user:${TOP_MODULE}:1.0 top]

# ---- proc_sys_reset per clock domain ----
set PS_RST  [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 ps_rst]
set DSP_RST [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 dsp_rst]
connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] \
  [get_bd_pins ps_rst/ext_reset_in] [get_bd_pins dsp_rst/ext_reset_in]
connect_bd_net [get_bd_pins dsp_rst/peripheral_reset] [get_bd_pins $TOP/dspRst]
connect_bd_net [get_bd_pins ps_rst/peripheral_reset]  [get_bd_pins $TOP/hostRst]

# ---- the MTS clock tree: PL clock -> MMCM -> dspClk; PL SYSREF -> synchronizer -> user_sysref ----
# RISCQ_DSPCLK selects the fabric clock: "tile" (default) = the MTS source tile's own output clock clk_dac2
# through a BUFG — the pre-MTS arrangement and the classic MTS examples' choice; timing uncertainty
# 0.035 ns; the SoC must sit in reset while MTS re-aligns the tile clock, which the load protocol
# guarantees (no dsp-domain access before load() has finished). "mmcm" = the RFSoC-MTS reference
# structure (the LMK's PL clock x4): 0.049 ns of uncertainty, and on this design 17 implementation runs
# stopped between -0.001 and -0.13 ns of WNS; kept as an option.
set DSPCLK [expr {[info exists ::env(RISCQ_DSPCLK)] ? $::env(RISCQ_DSPCLK) : "tile"}]
if {$DSPCLK ni {mmcm tile}} { error "bd-4x2: RISCQ_DSPCLK must be mmcm or tile, not $DSPCLK" }
puts "bd-4x2: dspClk source = $DSPCLK"
if {$DSPCLK eq "mmcm"} {
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 pl_clk
set_property -dict [list CONFIG.FREQ_HZ 122880000] [get_bd_intf_ports pl_clk]
set MMCM [create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 dsp_mmcm]
set_property -dict [list \
  CONFIG.PRIM_SOURCE {Differential_clock_capable_pin} \
  CONFIG.PRIM_IN_FREQ {122.88} \
  CONFIG.CLKIN1_JITTER_PS {1.0} \
  CONFIG.PRIMITIVE {MMCM} \
  CONFIG.JITTER_SEL {Min_O_Jitter} \
  CONFIG.MMCM_BANDWIDTH {HIGH} \
  CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {491.52} \
  CONFIG.CLKOUT1_DRIVES {Buffer} \
  CONFIG.USE_LOCKED {true} \
  CONFIG.USE_RESET {false} \
  CONFIG.USE_PHASE_ALIGNMENT {true} \
] $MMCM
connect_bd_intf_net [get_bd_intf_ports pl_clk] [get_bd_intf_pins $MMCM/CLK_IN1_D]
# the achieved output must be exactly the SoC's dsp frequency: 122.88 x 4 = 491.52 is an exact MMCM
# ratio (M 8 / D 1 / O 2, VCO 983.04 MHz); the wizard publishes the achieved value on the clk_out1 pin
# and validate_bd_design cross-checks every FREQ_HZ against the consumers (the TOP's dspClk, the
# converter streams), so a slip would fail the build loudly. Report it here for the log.
set _mmcm_out [get_property CONFIG.FREQ_HZ [get_bd_pins $MMCM/clk_out1]]
puts "bd-4x2: dsp MMCM clk_out1 FREQ_HZ = ${_mmcm_out} (want ${DSP_FREQ})"
if {$_mmcm_out ne "" && $_mmcm_out != $DSP_FREQ} {
  error "bd-4x2: the dsp MMCM achieves ${_mmcm_out} Hz, not ${DSP_FREQ} Hz — pick an exact multiplier for the 122.88 MHz PL clock"
}
}

create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 pl_sysref
set_property -dict [list CONFIG.FREQ_HZ 7680000] [get_bd_intf_ports pl_sysref]
set SYSREF_BUF [create_bd_cell -type ip -vlnv xilinx.com:ip:util_ds_buf:2.2 pl_sysref_ibuf]
set_property -dict [list CONFIG.C_BUF_TYPE {IBUFDS}] $SYSREF_BUF
connect_bd_intf_net [get_bd_intf_ports pl_sysref] [get_bd_intf_pins $SYSREF_BUF/CLK_IN_D]
set SYSREF_CDC [create_bd_cell -type ip -vlnv xilinx.com:ip:xpm_cdc_gen:1.0 pl_sysref_sync]
set_property -dict [list \
  CONFIG.CDC_TYPE {xpm_cdc_single} \
  CONFIG.DEST_SYNC_FF {2} \
  CONFIG.INIT_SYNC_FF {true} \
  CONFIG.SRC_INPUT_REG {false} \
  CONFIG.WIDTH {1} \
] $SYSREF_CDC
connect_bd_net [get_bd_pins $SYSREF_BUF/IBUF_OUT] [get_bd_pins $SYSREF_CDC/src_in]

# ---- RF Data Converter: staged config + post-apply assertions (rfdc-config-4x2.tcl) ----
set RFDC [create_bd_cell -type ip -vlnv xilinx.com:ip:usp_rf_data_converter:2.6 rf_data_converter]
source $INC/rfdc-config-4x2.tcl
# the dspClk source pin: the MMCM output, or the clock-owning DAC tile's fabric clock (491.52 MHz)
if {$DSPCLK eq "mmcm"} {
  set DSPCLK_PIN [get_bd_pins $MMCM/clk_out1]
} else {
  set DSPCLK_PIN [get_bd_pins rf_data_converter/clk_dac2]
  puts "bd-4x2: dspClk = rf_data_converter/clk_dac2 (FREQ_HZ [get_property CONFIG.FREQ_HZ $DSPCLK_PIN])"
}

# ---- analog / sysref / sample-clock external ports (4x2 dual-tile pair naming on this part) ----
# vout10 is tile 229's idle DAC (the clock-forwarding tile); vin0_01 / vin1_01 / vin3_01 are the idle
# ADCs of tiles 224 / 225 / 227 (clock receivers that make ADC MTS selectable): brought out so the port
# set is complete.
foreach _p {vout00 vout10 vout20} {
  create_bd_intf_port -mode Master -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 $_p
  connect_bd_intf_net [get_bd_intf_pins rf_data_converter/$_p] [get_bd_intf_ports $_p]
}
foreach _p {vin0_01 vin1_01 vin2_01 vin2_23 vin3_01} {
  create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 $_p
  connect_bd_intf_net [get_bd_intf_pins rf_data_converter/$_p] [get_bd_intf_ports $_p]
}
create_bd_intf_port -mode Slave -vlnv xilinx.com:display_usp_rf_data_converter:diff_pins_rtl:1.0 sysref_in
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/sysref_in] [get_bd_intf_ports sysref_in]
# only the clock-owning tiles still take an LMX reference (DAC0 runs on DAC2's distributed clock)
foreach _c {dac0_clk dac2_clk adc2_clk} {
  if {[llength [get_bd_intf_pins -quiet rf_data_converter/$_c]]} {
    create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 $_c
    connect_bd_intf_net [get_bd_intf_pins rf_data_converter/$_c] [get_bd_intf_ports $_c]
    puts "bd-4x2: reference clock port $_c"
  } else {
    puts "bd-4x2: no $_c pin on the converter IP (tile clocked by distribution) — port not created"
  }
}
connect_bd_net [get_bd_pins $SYSREF_CDC/dest_out] \
  [get_bd_pins rf_data_converter/user_sysref_adc] [get_bd_pins rf_data_converter/user_sysref_dac]

# ---- converter streams (SMA map per the RealDigital manual, confirmed 2026-09-03: DAC_B=228s0, DAC_A=230s0,
#      ADC_B=226 s0 (m20), ADC_A=226 s2 (m22) — the 4x2 letters run opposite to the tile order) ----
connect_bd_intf_net [get_bd_intf_pins $TOP/DAC0_AXIS] [get_bd_intf_pins rf_data_converter/s00_axis]
connect_bd_intf_net [get_bd_intf_pins $TOP/DAC1_AXIS] [get_bd_intf_pins rf_data_converter/s20_axis]
connect_bd_intf_net [get_bd_intf_pins $TOP/ADC0_AXIS] [get_bd_intf_pins rf_data_converter/m20_axis]
connect_bd_intf_net [get_bd_intf_pins $TOP/ADC1_AXIS] [get_bd_intf_pins rf_data_converter/m22_axis]
# s10_axis (tile 229, forwarding only) stays idle: no source, tvalid low, the DAC outputs zero;
# m00/m10/m30_axis (tiles 224/225/227, ADC clock receivers) stay unconnected: their samples are dropped

# ---- AXI SmartConnect: PS HPM0_LPD -> { top S_AXIS @0x8000_0000, rfdc s_axi @0x9000_0000 } ----
# The second master is what gives software ANY visibility of the RFDC (tile status, PLL lock,
# StartUp/Reset, MTS): without it the dsp domain can be dead with no way to know, and the first host
# MMIO into it wedges the PS — the original bring-up failure on this board.
set AXI_CONNECT [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect]
set_property CONFIG.NUM_SI 1 $AXI_CONNECT
set_property CONFIG.NUM_MI 2 $AXI_CONNECT
set_property CONFIG.NUM_CLKS {2} $AXI_CONNECT
set_property CONFIG.HAS_ARESETN {0} $AXI_CONNECT
connect_bd_intf_net [get_bd_intf_pins $AXI_CONNECT/M00_AXI]   [get_bd_intf_pins $TOP/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins $AXI_CONNECT/M01_AXI]   [get_bd_intf_pins rf_data_converter/s_axi]
connect_bd_intf_net [get_bd_intf_pins zynq_ps/M_AXI_HPM0_LPD] [get_bd_intf_pins $AXI_CONNECT/S00_AXI]

# ---- clocks ----
# hostClk = PS pl_clk0: PS AXI master, both smartconnect clock pins, PS reset, TOP host side,
# RFDC s_axi. dspClk = the MMCM's 491.52 MHz: TOP dsp side, dsp reset (held until the MMCM locks),
# the SYSREF synchronizer (its src_clk too: unused with SRC_INPUT_REG off, but validate_bd_design
# wants every clock pin driven — BD 41-758), and every converter stream clock (s0 + s1 + s2,
# m0 + m1 + m2 + m3).
connect_bd_net [get_bd_pins zynq_ps/pl_clk0] \
  [get_bd_pins zynq_ps/maxihpm0_lpd_aclk] \
  [get_bd_pins $AXI_CONNECT/aclk] [get_bd_pins $AXI_CONNECT/aclk1] \
  [get_bd_pins ps_rst/slowest_sync_clk] \
  [get_bd_pins $TOP/hostClk] \
  [get_bd_pins rf_data_converter/s_axi_aclk]
connect_bd_net $DSPCLK_PIN \
  [get_bd_pins $TOP/dspClk] \
  [get_bd_pins dsp_rst/slowest_sync_clk] \
  [get_bd_pins $SYSREF_CDC/src_clk] [get_bd_pins $SYSREF_CDC/dest_clk] \
  [get_bd_pins rf_data_converter/s0_axis_aclk] \
  [get_bd_pins rf_data_converter/s1_axis_aclk] \
  [get_bd_pins rf_data_converter/s2_axis_aclk] \
  [get_bd_pins rf_data_converter/m0_axis_aclk] \
  [get_bd_pins rf_data_converter/m1_axis_aclk] \
  [get_bd_pins rf_data_converter/m2_axis_aclk] \
  [get_bd_pins rf_data_converter/m3_axis_aclk]
if {$DSPCLK eq "mmcm"} {
  connect_bd_net [get_bd_pins $MMCM/locked] [get_bd_pins dsp_rst/dcm_locked]
}
connect_bd_net [get_bd_pins dsp_rst/peripheral_aresetn] \
  [get_bd_pins rf_data_converter/s0_axis_aresetn] \
  [get_bd_pins rf_data_converter/s1_axis_aresetn] \
  [get_bd_pins rf_data_converter/s2_axis_aresetn] \
  [get_bd_pins rf_data_converter/m0_axis_aresetn] \
  [get_bd_pins rf_data_converter/m1_axis_aresetn] \
  [get_bd_pins rf_data_converter/m2_axis_aresetn] \
  [get_bd_pins rf_data_converter/m3_axis_aresetn]
connect_bd_net [get_bd_pins ps_rst/peripheral_aresetn] [get_bd_pins rf_data_converter/s_axi_aresetn]

# pl_clk0's ACHIEVED frequency can differ from the 100 MHz request (PS PLL divider granularity),
# and the packaged IP + BD must carry the achieved number (validate_bd_design cross-checks every
# FREQ_HZ). config.tcl's board preset pins HOST_FREQ to the board-measured achieved value; this
# assertion fails the build LOUDLY if the PS ever achieves something else.
set _plfreq [get_property CONFIG.FREQ_HZ [get_bd_pins zynq_ps/pl_clk0]]
if {$_plfreq != $HOST_FREQ} {
  error "bd-4x2: pl_clk0 achieved ${_plfreq} Hz != HOST_FREQ ${HOST_FREQ} Hz — update the rfsoc4x2 HOST_FREQ preset in inc/config.tcl (or RISCQ_HOST_FREQ) to the achieved value; the packaged IP must carry it"
}

# ---- addresses: TOP at 0x8000_0000 (256 MB window, SoC maps 128 MB), RFDC at 0x9000_0000 ----
assign_bd_address -offset 0x80000000 -range 0x10000000 \
  -target_address_space [get_bd_addr_spaces zynq_ps/Data] [get_bd_addr_segs $TOP/S_AXIS/reg0] -force
assign_bd_address -offset 0x90000000 -range 0x00040000 \
  -target_address_space [get_bd_addr_spaces zynq_ps/Data] [get_bd_addr_segs rf_data_converter/s_axi/Reg] -force
assign_bd_address
