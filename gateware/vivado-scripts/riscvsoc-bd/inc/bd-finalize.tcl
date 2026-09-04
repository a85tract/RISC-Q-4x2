# ---- Validate, generate the HDL wrapper, add constraints, set the top ------------------------------
validate_bd_design

make_wrapper -files [get_files $BD_NAME.bd] -top -import -force
generate_target all [get_files $BD_NAME.bd]
close_bd_design $BD_NAME

set_property top ${BD_NAME}_wrapper [current_fileset]

if {[file exists $SCRIPT_DIR/constraints-${BOARD}.xdc]} {
  add_files -fileset constrs_1 -norecurse $SCRIPT_DIR/constraints-${BOARD}.xdc
}
# Late constraints (async clock groups over IP-created clocks): those clocks do not exist at
# synthesis, so the file is implementation-only and ordered LAST.
if {[file exists $SCRIPT_DIR/constraints-${BOARD}-late.xdc]} {
  add_files -fileset constrs_1 -norecurse $SCRIPT_DIR/constraints-${BOARD}-late.xdc
  set_property USED_IN {implementation} [get_files $SCRIPT_DIR/constraints-${BOARD}-late.xdc]
  set_property PROCESSING_ORDER LATE   [get_files $SCRIPT_DIR/constraints-${BOARD}-late.xdc]
}
# RISCQ_DSPCLK=mmcm (4x2): the PL-clock pins/clock and the MMCM domain's async group live in their own
# files because XDC takes no control flow — Tcl decides here which variant's files join the project.
if {$BOARD eq "rfsoc4x2" && [info exists DSPCLK] && $DSPCLK eq "mmcm"} {
  add_files -fileset constrs_1 -norecurse $SCRIPT_DIR/constraints-rfsoc4x2-mmcm.xdc
  add_files -fileset constrs_1 -norecurse $SCRIPT_DIR/constraints-rfsoc4x2-mmcm-late.xdc
  set_property USED_IN {implementation} [get_files $SCRIPT_DIR/constraints-rfsoc4x2-mmcm-late.xdc]
  set_property PROCESSING_ORDER LATE   [get_files $SCRIPT_DIR/constraints-rfsoc4x2-mmcm-late.xdc]
  puts "bd-finalize: MMCM variant constraints added"
}
update_compile_order -fileset sources_1
