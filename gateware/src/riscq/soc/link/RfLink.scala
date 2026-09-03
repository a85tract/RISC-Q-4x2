package riscq.soc.link

import spinal.core._

/**
 * Posted-link bundles for the narrow core→DSP RF link. The link replaces the wide, bidirectional
 * `tilelink.fabric` decode tree between a RISC-V core and its converter-edge DSP with three pipelined,
 * unidirectional, timing-insensitive `Flow`s — so the core can be floorplanned far from the converters
 * (every link stage is a plain `RegNext`, absorbed by the lead-time scheduler).
 */

/**
 * One posted RF register write crossing the link **downstream** (core → DSP). A narrow, ordered
 * `Flow` payload: the byte `address` (offset within the core's RF window) plus the 32-bit write
 * `data`. The far-side demux routes it to the addressed channel, which decodes the field exactly as
 * the old `SlaveFactory` did (16-bit fields packed at bit 16). Posted ⇒ no ack on this path; the
 * `RfLinkBridge` next to the core terminates the CPU's TileLink D channel locally.
 */
case class RfCmd(addrWidth: Int) extends Bundle {
  val address = UInt(addrWidth bits)
  val data    = Bits(32 bits)
}
