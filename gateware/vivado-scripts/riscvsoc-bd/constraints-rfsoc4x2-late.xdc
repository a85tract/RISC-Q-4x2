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
# Since the MTS clock tree (2026-09-04) the SoC's dspClk is the dsp_mmcm output generated from the LMK's
# PL clock (pl_clk_clk_p): the whole PulseTableSoc, the converter streams and the SYSREF synchronizer
# live in that domain; the converter IP's own tile clocks cross into it only inside the IP's FIFOs. The
# PL SYSREF (pl_sysref_clk_p) is a level sampled by an xpm_cdc synchronizer.
set_clock_groups -asynchronous \
  -group [get_clocks clk_pl_0] \
  -group [get_clocks -quiet {RFDAC*_CLK RFADC*_CLK dac0_clk_clk_p dac2_clk_clk_p adc2_clk_clk_p}] \
  -group [get_clocks -quiet pl_sysref_clk_p]
# The MMCM domain (RISCQ_DSPCLK=mmcm only) gets its group from constraints-rfsoc4x2-mmcm-late.xdc,
# added by bd-finalize.tcl for that variant — XDC files take no if/foreach (Designutils 20-1307;
# an `if` here silently dropped the group on 2026-09-04 and three runs timed hostClk <-> dspClk
# paths synchronously, WNS -1.7 ns).

# ---- demod envelope RAM: keep its cascaded RAMB36 chain inside ONE clock region ----------------
# The 16 RAMB36E2 of riscqCores_0's demod envelope RAM (env_depth 16384 x 32) are depth-cascaded.
# Left to the placer, the rfsoc4x2-2dac-fine build put the chain across a clock-region row break
# and write_bitstream refused with DRC CASC-31 (2026-09-03). The board-verified rfsoc4x2-1q-fine
# build had all 16 in CLOCKREGION_X1Y3 (22 RAMB36 sites there) — pin them there. RAM-only, hard
# pblock: no LUT/FF is constrained, so the timing impact is the placement the good build already had.
create_pblock p_demod0_ram
add_cells_to_pblock [get_pblocks p_demod0_ram] \
  [get_cells -hier -filter {REF_NAME == RAMB36E2 && NAME =~ *riscqArea_riscqCores_0_demodMemFiber_rams_0*}]
resize_pblock [get_pblocks p_demod0_ram] -add {CLOCKREGION_X1Y3:CLOCKREGION_X1Y3}
set_property IS_SOFT FALSE [get_pblocks p_demod0_ram]
# A second core (rfsoc4x2-2q-*): its 16 cascaded demod RAMB36 get the row below (X1Y2 — the 1q-fine
# build had only 8 of its readout-envelope RAMB36 there, so the region has room); the pattern matches
# nothing on a 1-core design and the empty pblock is harmless.
create_pblock p_demod1_ram
add_cells_to_pblock [get_pblocks p_demod1_ram] \
  [get_cells -hier -quiet -filter {REF_NAME == RAMB36E2 && NAME =~ *riscqArea_riscqCores_1_demodMemFiber_rams_0*}]
resize_pblock [get_pblocks p_demod1_ram] -add {CLOCKREGION_X1Y2:CLOCKREGION_X1Y2}
set_property IS_SOFT FALSE [get_pblocks p_demod1_ram]

# ---- trace ("robs") RAM write fanout: enable if the DSP-clock WNS goes negative -----------------
# The posted-link write staging registers (_zz_io_port0_write_reg / _zz_io_port0_wdata_reg) drive
# the trace BRAM write ports across many clock regions; the 2-DAC placement lost ~120 ps there
# (WNS -0.078 vs +0.045 on the 1q build). Latency-neutral physical replication (UG904):
# ENABLED 2026-09-03: attempt 2 (ExtraTimingOpt placer + post-route phys_opt) still failed at
# WNS -0.127 ns on these very paths, so the placer is told to replicate them per clock region.
set_property MAX_FANOUT_MODE CLOCK_REGION [get_nets -of_objects [get_pins -of_objects \
  [get_cells -hier -filter {REF_NAME == FDRE && (NAME =~ *_zz_io_port0_write_reg* || NAME =~ *_zz_io_port0_wdata_reg*)}] \
  -filter {REF_PIN_NAME == Q}]]
set_property FORCE_MAX_FANOUT 4 [get_nets -of_objects [get_pins -of_objects \
  [get_cells -hier -filter {REF_NAME == FDRE && (NAME =~ *_zz_io_port0_write_reg* || NAME =~ *_zz_io_port0_wdata_reg*)}] \
  -filter {REF_PIN_NAME == Q}]]

# ---- envelope-reader address -> deep envelope RAM (16-deep RAMB36 cascades): same medicine -----
# 2-DAC attempt 2's worst path was posted_gateChannel/pulseGenerator_3/envReader/addrReg_reg[12]
# -> pulseMemFiber_rams_0/.../ENARDEN: one address register enabling many cascaded RAMB36. Ask the
# placer to replicate these per clock region as well (latency-neutral).
set_property MAX_FANOUT_MODE CLOCK_REGION [get_nets -of_objects [get_pins -of_objects \
  [get_cells -hier -filter {REF_NAME =~ FD* && NAME =~ *envReader/addrReg_reg*}] -filter {REF_PIN_NAME == Q}]]
set_property FORCE_MAX_FANOUT 8 [get_nets -of_objects [get_pins -of_objects \
  [get_cells -hier -filter {REF_NAME =~ FD* && NAME =~ *envReader/addrReg_reg*}] -filter {REF_PIN_NAME == Q}]]

# ---- trace recorder write enable -> 16 trace banks per core: same medicine (2026-09-04) -----------
# The 2026-09-04 recorder fix moved the write enable to riscqArea_trace<N>_fire_regNext_regNext; in the
# MTS builds one of its replicas -> robs_1_rams_4 ENBWREN was among the last failing paths (1.26 ns of
# route, -0.27 ns skew). Replicate it per clock region like the other RAM enables (latency-neutral).
set_property MAX_FANOUT_MODE CLOCK_REGION [get_nets -quiet -of_objects [get_pins -quiet -of_objects   [get_cells -hier -quiet -filter {REF_NAME =~ FD* && NAME =~ *_fire_regNext_regNext_reg*}] -filter {REF_PIN_NAME == Q}]]
set_property FORCE_MAX_FANOUT 4 [get_nets -quiet -of_objects [get_pins -quiet -of_objects   [get_cells -hier -quiet -filter {REF_NAME =~ FD* && NAME =~ *_fire_regNext_regNext_reg*}] -filter {REF_PIN_NAME == Q}]]
