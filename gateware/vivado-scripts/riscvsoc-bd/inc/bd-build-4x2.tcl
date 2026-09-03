# ===========================================================================================
# RFSoC 4x2 (xczu48dr) block-design assembly: Zynq PS + user top IP + proc_sys_resets +
# RF Data Converter (2 DAC / 2 ADC) + AXI SmartConnect.
#
# No ClockInterface here — the 4x2 exposes no PL LVDS clock-pair pins for it. hostClk is the PS
# pl_clk0; dspClk is the RFDC clk_dac0 output (491.52 MHz = 7.86432 GSPS / 16 samples per beat).
# Wiring transplanted from the dev-line board-verified utils/{zynq-ps-4x2, axi-smart-connect-4x2,
# rfdc-4x2}.tcl (archive bundle @ f915e1b; loopback + ion-trap waveform PASS on this bench).
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

# ---- RF Data Converter: staged config + post-apply assertions (rfdc-config-4x2.tcl) ----
set RFDC [create_bd_cell -type ip -vlnv xilinx.com:ip:usp_rf_data_converter:2.6 rf_data_converter]
source $INC/rfdc-config-4x2.tcl

# ---- analog / sysref / sample-clock external ports (4x2 dual-tile pair naming on this part) ----
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vout00
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vout20
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vin2_01
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vin2_23
create_bd_intf_port -mode Slave -vlnv xilinx.com:display_usp_rf_data_converter:diff_pins_rtl:1.0 sysref_in
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 dac0_clk
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 dac2_clk
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 adc2_clk
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vout00]    [get_bd_intf_ports vout00]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vout20]    [get_bd_intf_ports vout20]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vin2_01]   [get_bd_intf_ports vin2_01]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vin2_23]   [get_bd_intf_ports vin2_23]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/sysref_in] [get_bd_intf_ports sysref_in]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/dac0_clk]  [get_bd_intf_ports dac0_clk]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/dac2_clk]  [get_bd_intf_ports dac2_clk]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/adc2_clk]  [get_bd_intf_ports adc2_clk]

# ---- converter streams (SMA map measured 2026-09-03 with a single cable: DAC_B=228s0, DAC_A=230s0,
#      ADC_B=226 s0 (m20), ADC_A=226 s2 (m22) — the 4x2 letters run opposite to the tile order) ----
connect_bd_intf_net [get_bd_intf_pins $TOP/DAC0_AXIS] [get_bd_intf_pins rf_data_converter/s00_axis]
connect_bd_intf_net [get_bd_intf_pins $TOP/DAC1_AXIS] [get_bd_intf_pins rf_data_converter/s20_axis]
connect_bd_intf_net [get_bd_intf_pins $TOP/ADC0_AXIS] [get_bd_intf_pins rf_data_converter/m20_axis]
connect_bd_intf_net [get_bd_intf_pins $TOP/ADC1_AXIS] [get_bd_intf_pins rf_data_converter/m22_axis]

# ---- AXI SmartConnect: PS HPM0_LPD -> { top S_AXIS @0x8000_0000, rfdc s_axi @0x9000_0000 } ----
# The second master is what gives software ANY visibility of the RFDC (tile status, PLL lock,
# StartUp/Reset): without it the dsp domain can be dead with no way to know, and the first host
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
# RFDC s_axi. dspClk = RFDC clk_dac0: TOP dsp side, dsp reset, every used converter stream clock
# (s0+s2+m2 on one clock with per-domain resets is the structure both board-verified builds used).
connect_bd_net [get_bd_pins zynq_ps/pl_clk0] \
  [get_bd_pins zynq_ps/maxihpm0_lpd_aclk] \
  [get_bd_pins $AXI_CONNECT/aclk] [get_bd_pins $AXI_CONNECT/aclk1] \
  [get_bd_pins ps_rst/slowest_sync_clk] \
  [get_bd_pins $TOP/hostClk] \
  [get_bd_pins rf_data_converter/s_axi_aclk]
connect_bd_net [get_bd_pins rf_data_converter/clk_dac0] \
  [get_bd_pins $TOP/dspClk] \
  [get_bd_pins dsp_rst/slowest_sync_clk] \
  [get_bd_pins rf_data_converter/s0_axis_aclk] \
  [get_bd_pins rf_data_converter/s2_axis_aclk] \
  [get_bd_pins rf_data_converter/m2_axis_aclk]
connect_bd_net [get_bd_pins dsp_rst/peripheral_aresetn] \
  [get_bd_pins rf_data_converter/s0_axis_aresetn] \
  [get_bd_pins rf_data_converter/s2_axis_aresetn] \
  [get_bd_pins rf_data_converter/m2_axis_aresetn]
connect_bd_net [get_bd_pins ps_rst/peripheral_aresetn] [get_bd_pins rf_data_converter/s_axi_aresetn]

set_property -dict [list CONFIG.FREQ_HZ $DSP_FREQ] [get_bd_pins $TOP/dspClk]

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
