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
 *                            until the wall clock catches up to the scheduled instant.
 */
case class TimeMemMap(externalTime: UInt) extends Area {
  val timeCmp = Reg(UInt(32 bit)) init 0
  val time    = RegNext(externalTime)

  def mapping(factory: tilelink.SlaveFactory): Unit = {
    val timeAddr = 0xbff8
    factory.read(time, timeAddr)

    val timeCmpAddr = 0x4000
    factory.readAndWrite(timeCmp, timeCmpAddr)
    val delay = 3

    val waitTimeAddr = timeCmpAddr + 8
    val waitTimeCmp  = RegNext(time + delay < timeCmp)
    factory.read(waitTimeCmp, waitTimeAddr)
    factory.onReadPrimitive(SingleMapping(waitTimeAddr), haltSensitive = false, null) {
      when(waitTimeCmp) {
        factory.readHalt()
      }
    }
  }
}
