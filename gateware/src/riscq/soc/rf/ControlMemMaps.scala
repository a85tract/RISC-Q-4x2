package riscq.soc.rf

import spinal.core._
import spinal.lib.bus.tilelink
import spinal.lib.bus.misc.SingleMapping

/**
 * CPU-mapped control-block register fragments contributed to a `MemMapFiber` (`addMapping`) — ported
 * from the RISC-Q reference (`riscq.soc.Misc`).
 *
 * [[TimeMemMap]] exposes the SoC batch-time and a wait-compare the control software spins on:
 *   - `time`@0xbff8        — current batch time (registered copy of the external time);
 *   - `timeCmp`@0x4000     — software-written compare value;
 *   - `waitTimeCmp`@0x4008 — a read that **halts** until `time + delay ≥ timeCmp`, i.e. the CPU blocks
 *                            until the wall clock catches up to the scheduled instant;
 *   - `runOrigin`@0x4010   — (run_origin builds) the shared run origin latched at the reset release.
 */
case class TimeMemMap(externalTime: UInt, runOrigin: Option[UInt] = None, signedWait: Boolean = false)
    extends Area {
  val timeCmp = Reg(UInt(32 bit)) init 0
  val time    = RegNext(externalTime)

  def mapping(factory: tilelink.SlaveFactory): Unit = {
    val timeAddr = 0xbff8
    factory.read(time, timeAddr)

    val timeCmpAddr = 0x4000
    factory.readAndWrite(timeCmp, timeCmpAddr)
    val delay = 3

    val waitTimeAddr = timeCmpAddr + 8
    // "not yet due": the upstream unsigned compare, or (`signedWait`, the run_origin builds) the same
    // wrap-safe signed difference TimedQueue uses — a deadline just past the 32-bit wrap of the batch
    // clock must still block instead of falling through (a run whose origin sits within a lead of the
    // wrap would otherwise publish DONE while its pulses are still queued).
    val waitTimeCmp  = RegNext(if (signedWait) (time + delay - timeCmp).msb else time + delay < timeCmp)
    factory.read(waitTimeCmp, waitTimeAddr)
    factory.onReadPrimitive(SingleMapping(waitTimeAddr), haltSensitive = false, null) {
      when(waitTimeCmp) {
        factory.readHalt()
      }
    }
    // run_origin builds: the batch time latched at the reset release (+ lead), identical on every core —
    // a multi-core kernel set's common t1 (`runOrigin`@0x4010, read-only).
    runOrigin.foreach(o => factory.read(o, timeCmpAddr + 0x10))
  }
}
