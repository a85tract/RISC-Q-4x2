package riscq.soc.link

import spinal.core._
import spinal.lib._
import spinal.lib.bus.tilelink
import spinal.lib.bus.misc.SingleMapping

/**
 * Readout decoder result, carried **upstream** (DSP → core) on the link's second narrow posted `Flow`.
 * The decoder's integrated point — the 1-bit discrimination `res` plus the integrated I/Q
 * (`real`/`imag`) — travels up into a near-core [[ReadoutResultSink]] the CPU polls locally, so the
 * halting `res` read is a short local arc instead of a long bus round-trip.
 */
case class ReadoutResult(accWidth: Int) extends Bundle {
  val res  = Bool()
  val real = SInt(accWidth bits)
  val imag = SInt(accWidth bits)
}

object ReadoutResultLink {
  /**
   * DSP-side source: forward the decoder's `res.valid` **as a level** (not an edge) with the current
   * `res`/`real`/`imag`. The carrier-triggered decoder already shapes `res.valid` exactly right — high
   * from a window's settle until the next window's `winStart` clears it, i.e. low exactly while a fresh
   * window integrates — so the sink can mirror it directly: no edge-detect, no stale-beat bookkeeping.
   */
  def source(resValid: Bool, res: Bool, real: SInt, imag: SInt, accWidth: Int): Flow[ReadoutResult] = {
    val out = Flow(ReadoutResult(accWidth))
    out.valid        := resValid
    out.payload.res  := res
    out.payload.real := real
    out.payload.imag := imag
    out
  }
}

/**
 * Core-side readout-result register — the local end of the up-`Flow`. It **mirrors** the decoder's
 * `res.valid` level (carried up the link) into a `valid` flag the CPU's local `SlaveFactory` halts the
 * `res` read on, and latches `{res, real, imag}` while that level is high. Because the level is low
 * exactly during the next window's integration, a `res` read that races a fresh window halts until it
 * settles — the same local-halt contract as before, now with no `arm` and an **idempotent** read.
 *
 * Freshness is a software timing contract (not a hardware clear): the level holds the *previous*
 * window's result high through the `LEAD` gap between a `play` and the window opening, so software waits
 * past the window's opening (`waitUntil(now ≥ startTime + LEAD_RO)`) before reading — past `winStart`
 * the stale level has dropped, so the halting read can only return the new window. See
 * `specs/new-readout-decoder` §2.4 / §3.2.
 *
 * @param resultIn the up-`Flow` from the DSP-side [[ReadoutResultLink.source]]
 */
case class ReadoutResultSink(
    accWidth: Int,
    resAddr: Int  = 4,
    realAddr: Int = 8,
    imagAddr: Int = 12
) extends Area {
  val resultIn = Flow(ReadoutResult(accWidth))

  val valid = RegNext(resultIn.valid) init False   // mirror the decoder's res.valid level (link-delayed)
  val res   = Reg(Bool()) init False
  val real  = Reg(SInt(accWidth bits)) init 0
  val imag  = Reg(SInt(accWidth bits)) init 0
  when(resultIn.valid) {                            // latch the settled point while the level is high
    res  := resultIn.payload.res
    real := resultIn.payload.real
    imag := resultIn.payload.imag
  }

  /** Local read map (the readout decoder's `res`@4 / `real`@8 / `imag`@12, but served here with a
   *  **local** halt instead of a bus round-trip). Contribute to a `MemMapFiber`. */
  def mapping(factory: tilelink.SlaveFactory): Unit = {
    factory.read(res, resAddr)
    factory.onReadPrimitive(SingleMapping(resAddr), haltSensitive = false, null) {
      when(!valid)(factory.writeHalt()) // res read HALTS until the integral has settled (locally)
    }
    factory.read(real, realAddr)
    factory.read(imag, imagAddr)
  }
}
