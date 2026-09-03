package riscq.dsp.pulse

import spinal.core._
import spinal.lib._
import riscq.dsp._

/**
 * Carrier batch generator:
 *
 * {{{
 *   carrier[k](t) = amp · exp(iπ·gPhase) · phasor[k] ,   gPhase = freq·(N·t) + phase  (mod 2)
 *               = amp·Amax · exp(iπ·(freq·(N·t + k) + phase))     (with phasor[k] = Amax·exp(iπ f k))
 * }}}
 *
 * '''Intra-batch factorization''': one CORDIC computes the per-batch, time-dependent factor
 * `amp·exp(iπ·gPhase)` and `N` `ComplexMul`s combine it with the static per-lane phasors. The
 * '''time-product phase''' `gPhase = freq·((t·N) mod 2^w) + phase` is recomputed from absolute time
 * every cycle and truncated to SF(w): `(a·(b mod 2^w)) mod 2^w = a·b mod 2^w`, and one wrap of the
 * truncated factor is a whole number of turns, so the phase is '''exact mod 2π''' — pulses are
 * phase-coherent across arbitrary time gaps and channels, the property qubit control needs.
 * '''Amplitude rides the CORDIC `x` input''' — no separate amplitude multiply.
 *
 * Operand-magnitude contract for the `ComplexMul` saturation: `|amp·exp| ≤ amp < 1` and
 * `|phasor[k]| < 1`, so the carrier product never saturates; the overall carrier scale is
 * `amp·(Amax·2^-(w-1))²` — a pure gain the golden model includes.
 *
 * `io.carrier` is a plain (always-valid) batch — validity is the duration gate's job. Inputs
 * are held in parameter registers (`RegNextWhen` on each Flow); `io.phasors` is snapshotted on valid.
 * Per-input latencies are exported as sums of sub-module latencies (no hard-coded literals).
 */
case class CarrierBatchGenerator(batchSize: Int, dataWidth: Int, timeWidth: Int, correctGain: Boolean = true,
    saturate: Boolean = true, freqWidth: Int = 0) extends Component {
  require(isPow2(batchSize), "batchSize must be a power of two (time·N is an exact left shift)")
  val N     = batchSize
  val w     = dataWidth
  val log2N = log2Up(N)
  val amax  = (BigInt(1) << (w - 1)) - 1

  /** Frequency-word width (M7b). `0` (default) = `dataWidth`, the historic SF(16) word. A wider word
   *  keeps the SAME physical MSB weight — the CORDIC still takes a `w`-bit angle, now sliced from the
   *  TOP of the product — so a legacy `F16 << (fw - w)` write reproduces the narrow behaviour bit for
   *  bit, and the extra low bits become fractional frequency (fw = 32: 1.83 Hz instead of 120 kHz). */
  val fw = if (freqWidth <= 0) dataWidth else freqWidth
  require(fw >= dataWidth, s"freqWidth $fw must be at least dataWidth $dataWidth")
  // The phase is exact mod 2π only while `(t·N) mod 2^fw` loses nothing to the time counter's own
  // wrap: the counter must carry at least the fw bits the product consumes. Otherwise the
  // fractional phase would jump when `time` wraps.
  require(timeWidth + log2Up(batchSize) >= fw,
    s"timeWidth $timeWidth + log2(N) ${log2Up(batchSize)} must cover freqWidth $fw")

  /** Extra pipeline stages INSIDE the freq×time product, and the ONE place that number is written.
   *  Measured OOC on xczu48dr at 2.035 ns (`m7b-bench`, effective 28×28 / 30×30 sliced products):
   *  flat misses by ~296 ps; 1–2 stages change nothing (the critical path is inside the multiplier,
   *  not after it); 3 stages close it at +0.389 ns; a 4th buys 0–3 ps. Every latency export below
   *  derives from this constant, so the queue lead times track it automatically. */
  val extraMulLatency = if (fw > dataWidth) 3 else 0

  val io = new Bundle {
    val time    = in port UInt(timeWidth bits)
    val amp     = slave port Flow(SInt(w bits))
    val phase   = slave port Flow(SInt(w bits))
    val freq    = slave port Flow(SInt(fw bits))
    val phasors = slave port Flow(ComplexBatch(N, w))
    val carrier = out port ComplexBatch(N, w)
  }

  // Held parameter registers. No reset init: each is write-before-read — popped from its TimedQueue at
  // lead time, before the (duration-gated) carrier is ever observed at the DAC — so the reset value is
  // never seen. Dropping the inits keeps these FFs out of the async-reset group, which packs denser and
  // lightens the reset net for the replicated instances. Do not restore.
  val ampReg   = RegNextWhen(io.amp.payload, io.amp.valid)
  val phaseReg = RegNextWhen(io.phase.payload, io.phase.valid)
  val freqReg  = RegNextWhen(io.freq.payload, io.freq.valid)
  // Phasor snapshot, taken on a BUFFERED enable. `io.phasors.valid` (= phasorGen `!regen`) is the
  // snapshot enable for all N·2 lanes — one cross-module net fanning out to N·2·w = 512 phRe/phIm FF
  // clock-enables. Rather than drive 512 clock-enables off a single combinational net, register the
  // enable (`phValBuf`, max_fanout-capped so the tool can replicate it beside the lanes) and delay the
  // payload one matching stage (`phIn`) so the snapshot stays aligned. Cost: the batch lands one cycle
  // later, folded into `phasorLatency` below (so PulseGenerator's freq-phasor lead time re-derives
  // automatically, with no literals). No reset init on the snapshot/payload: write-before-read, never
  // observed before the first `freq` write regenerates a real batch (the pulse is duration-gated to
  // zero), so the boot value is irrelevant.
  val phValBuf = RegNext(io.phasors.valid) init False    // buffered snapshot enable (drives the 512 FFs)
  phValBuf.addAttribute("max_fanout", 32)
  val phIn     = RegNext(io.phasors.payload)             // payload delayed to match the buffered enable
  val phRe     = Vec.fill(N)(Reg(SInt(w bits)))
  val phIm     = Vec.fill(N)(Reg(SInt(w bits)))
  when(phValBuf)(for (k <- 0 until N) {
    phRe(k) := phIn(k).re
    phIm(k) := phIn(k).im
  })

  // phase pipeline → gPhase (all truncating-wrap = exact phase wrap):
  // (t·N) mod 2^fw, SF(fw). The `+ extraMulLatency` PRE-COMPENSATES the deeper product: without it
  // the phase emitted at a given output cycle would correspond to a time `extraMulLatency` batches
  // EARLIER than in the narrow build, i.e. the same startTime/freq/phase would come out rotated by
  // −extraMulLatency·N·freq. With it, the absolute-time phase law `phase = freq·N·t + phase` holds
  // unchanged at every width, so software keeps ONE formula and a legacy seated word is equivalent
  // cycle for cycle, not merely bit for bit. At extraMulLatency = 0 this is the original expression.
  val batchTime = RegNext((((if (extraMulLatency == 0) io.time else io.time + extraMulLatency)
                            << log2N).resize(fw bits)).asSInt)
  // `extraMulLatency` registers give retiming material to pull INTO the DSP cascade (see above);
  // at fw = dataWidth there are none and this is the historic single-register product.
  val timePhase = (0 until extraMulLatency)
    .foldLeft(RegNext(freqReg * batchTime))((s, _) => RegNext(s))
  // The phase is the TOP `w` bits of the product. At fw = w that IS the low-w truncation, written
  // in the original inline form so the narrow build's Verilog stays bit-identical (a named `val`
  // here would rename the wire and move the non-regression hash for no logical reason).
  val gPhase    = RegNext((if (fw == w) timePhase.resize(w bits)
                           else timePhase(fw - 1 downto fw - w)) + phaseReg)  // + phase, mod 2^w

  // One CORDIC: amp·exp(iπ·gPhase), amplitude on the x input. When correctGain = false the
  // gain stage is dropped — software must prescale amp by 1/K so K·(amp/K) = amp.
  // resetValid = false on both the CORDIC and the ComplexMuls: their rsp.valid is unused
  // (payload read at fixed latency; the carrier feeds the duration-gated pulse output), so the reset-free
  // valid chains shed the global reset and infer SRLs.
  val cordic = Cordic(CordicParams(xyWidth = w, zWidth = w, correctGain = correctGain, saturate = saturate,
                                   resetValid = false))
  cordic.io.cmd.valid         := True
  cordic.io.cmd.payload.xy.re := ampReg
  cordic.io.cmd.payload.xy.im := S(0, w bits)
  cordic.io.cmd.payload.z     := gPhase

  // Broadcast register (N-lane fanout) then N ComplexMuls against the static phasors.
  val cBase = RegNext(cordic.io.rsp.payload)
  val muls  = Array.fill(N)(ComplexMul(w, saturate, resetValid = false))
  for (k <- 0 until N) {
    muls(k).io.cmd.valid     := True
    muls(k).io.cmd.payload.a := cBase
    muls(k).io.cmd.payload.b.re := phRe(k)
    muls(k).io.cmd.payload.b.im := phIm(k)
    io.carrier(k) := muls(k).io.rsp.payload
  }

  // exported per-input latencies (param-Flow fire → carrier), each a sum of sub-module latencies.
  private def tail: Int = cordic.latency + 1 + ComplexMul.latency(saturate) // cordic + broadcast + ComplexMul
  def ampLatency: Int    = 1 + tail            // ampReg → cordic.xy
  def phaseLatency: Int  = 2 + tail            // phaseReg → gPhase → cordic.z
  def freqLatency: Int   = 3 + extraMulLatency + tail  // freqReg → timePhase(+extra) → gPhase → cordic.z
  def phasorLatency: Int = 2 + ComplexMul.latency(saturate) // phValBuf + snapshot → ComplexMul.b
  def timeLatency: Int   = 3 + extraMulLatency + tail  // time → batchTime → timePhase(+extra) → gPhase
  // The freq and time paths are the same physical depth (they meet at gPhase); the freq queue's lead
  // uses that depth directly.
  require(freqLatency == timeLatency, "freq and time paths must have equal latency into gPhase")
  /** Batches the time input is PRE-ADVANCED by (see `batchTime`). A consumer that predicts the
   *  emitted phase from absolute time — the golden model — must subtract this from `timeLatency`:
   *  the extra product stages are already cancelled, so the phase REFERENCE is unchanged. */
  def timePhaseOffset: Int = extraMulLatency
}
