package riscq.soc

import spinal.core._
import spinal.core.sim._
import spinal.core.fiber.Fiber
import spinal.lib._
import spinal.lib.misc.plugin.Hostable
import spinal.lib.bus.tilelink
import spinal.lib.bus.tilelink.fabric.{Node, MasterBus}
import spinal.lib.bus.misc.SizeMapping
import riscq.memory.{Bram, HalfUram, Uram}
import riscq.soc.fabric.{RiscqFiber, TileLinkCpuMemFiber, MemMapFiber}
import riscq.soc.rf.TimeMemMap
import riscq.soc.link.{RfLinkBridge, ReadoutResult, ReadoutResultSink, RfCmd}

/**
 * The **timing-critical core unit** carved out of [[RiscqRfWithPulseTableFiber]] for the
 * registered-boundary floorplan: a hard SpinalHDL `Component` (Verilog module ⇒ a clean per-instance
 * pblock target) holding the RISC-V core + its real BRAM I/D RAM + the CPU-mapped control block + the
 * posted-link bridge + the readout-result sink — and nothing else. Everything past the posted link (the
 * converter-edge DSP datapath) stays OUT, in the parent.
 *
 * IO boundary — narrow and **registered** on both sides of the posted link:
 *   - `time`                     : shared batch-time broadcast (in);
 *   - `cmd : master Flow(RfCmd)`  : posted RF writes out of the [[RfLinkBridge]] → the DSP datapath;
 *   - `resultIn : slave Flow(ReadoutResult)` : the readout result back from the DSP → [[ReadoutResultSink]];
 *   - `iLoad`  (a [[MasterBus]] slave-IO, implicit dsp clock) : the program/data image load into the
 *     BRAM's slow port; the parent drives it *across* the host→dsp CDC, so this slave-IO is already in
 *     the dsp domain and the crossing lives outside this hard Component (see
 *     `RiscqRfWithPulseTableFiber.iMemLoad`);
 *   - `dTap`   (a [[MasterBus]] slave-IO, riscqCd; `withTestTap` only) : a second master into the CPU
 *     data-bus decode so a sim can schedule RF writes without a CPU program (the test harness).
 *
 * Because the fabric cannot cross a hard Component boundary, the only crossings are concrete
 * `tilelink.Bus`/`Flow` IO; the host-load + test taps enter as `MasterBus` slave-IO ports. The far side
 * of `cmd`/`resultIn`/`iLoad` is the *only* thing the dummy floorplan top vs the real [[PulseTableSoc]]
 * differ on — `RiscvSoc` itself is byte-for-byte identical (the floorplan-transfer claim).
 *
 * Faithfulness: this reproduces the exact internal wiring [[RiscqRfWithPulseTableFiber]] had before the
 * carve-out; the readout sink still carries `{res, real, imag}` on three read addresses (narrowing it to
 * 32 bits is deferred so `PulseTableSocSim`, which reads real/imag, stays bit-exact).
 */
case class RiscvSoc(
    plugins: Seq[Hostable],
    riscqCd: ClockDomain,
    timeWidth: Int = 32,
    readoutAccWidth: Int = 32,
    memDepth: Int = 1024,
    memWidth: Int = 32,
    memOutReg: Boolean = true,
    useUram: Boolean = true,
    rfAddrWidth: Int = 18,
    // test harness: add a second master into the CPU data-bus decode (a sim drives it directly).
    withTestTap: Boolean = false,
    // run_origin builds: an `origin` input (the SoC's reset-release time latch) read at CTRL 0x4010, and
    // the wrap-safe signed wait_until compare. Off = the upstream Component, byte for byte.
    runOrigin: Boolean = false,
    signedWait: Boolean = false,
    // host-load master params (the BRAM slow-port image load). Must match what the parent's host fabric
    // presents to `iLoad` (source-id width included), so keep it a single shared definition.
    hostLoadM2s: tilelink.M2sParameters = RiscvSoc.defaultHostLoadM2s,
    testTapM2s: tilelink.M2sParameters = RiscvSoc.defaultTestTapM2s
) extends Component {
  val memOffset = 0x80000000L

  // ── IO boundary ──
  val time     = in  port UInt(timeWidth bits)        // shared batch-time broadcast
  val origin   = runOrigin generate (in port UInt(timeWidth bits))   // shared run origin (run_origin builds)
  val cmd      = master port Flow(RfCmd(rfAddrWidth)) // posted RF writes (RfLinkBridge) → DSP
  val resultIn = slave  port Flow(ReadoutResult(readoutAccWidth)) // readout result ← DSP → sink

  def getPipe[T <: Data](data: T, cycles: Int): T = {
    var res = data
    for (_ <- 0 until cycles) {
      res = RegNext(res)
      res.addAttribute("DONT_TOUCH")
    }
    res
  }

  // ── CPU + its private instruction/data RAM ──
  // RiscvSoc elaborates in its parent's clock domain (the dsp clock) — that implicit `ClockDomain.current`
  // is the single clock for the BRAM, the instruction arbiter and the `iLoad` slave-IO. `riscqCd` (same
  // clock, core reset) is passed in and wraps only the CPU + control fibers below.
  val riscqFiber = riscqCd(RiscqFiber(plugins))
  // CPU instruction/data RAM: block RAM by default, UltraRAM with `useUram`. Both expose the same two
  // true-dual-port read/write ports (port0/port1, here single-clock in the implicit dsp domain).
  // Only DEEP memories (memDepth > 4096) use a HalfUram: it packs two 32-bit words per 64-bit UltraRAM
  // row, so the array costs half the URAM primitives of a 1-word-per-row Uram (32-bit only — hence the
  // memWidth guard). At or below one URAM's 4096-deep primitive a plain Uram already fits in a single
  // URAM, so HalfUram's combinational read-half mux would only add datapath delay for no primitive
  // saving — use a plain Uram there. Both URAM forms share the same read latency (uramPipeNum + 2).
  // Read latency differs and the fibers must be told exactly: Bram = 1 + outReg (2); the URAM template
  // chains memreg + NBPIPE pipes + the dout register, so pipeNum = 1 is 3 — one MORE than Bram+outReg
  // (HalfUram's half-mux is combinational, so it inherits that exact latency — Uram and HalfUram match).
  // (Feeding the fibers latency 2 with the URAM made every read — CPU load, fetch, host readback —
  // return the *previous* read's data; caught by the M0 co-sim contract test + PulseTableSocCpuSim.)
  val uramPipeNum = 1
  val (mem, memPort0, memPort1, memReadLatency) = if (useUram) {
    if (memDepth > 4096) {
      require(memWidth == 32, s"HalfUram-backed CPU RAM is 32-bit only, got memWidth=$memWidth")
      val uram = HalfUram(addressWidth = log2Up(memDepth), pipeNum = uramPipeNum)
      (uram, uram.io.port0, uram.io.port1, uramPipeNum + 2)
    } else {
      val uram = Uram(Bits(memWidth bits), addressWidth = log2Up(memDepth), pipeNum = uramPipeNum)
      (uram, uram.io.port0, uram.io.port1, uramPipeNum + 2)
    }
  } else {
    val bram = Bram(Bits(memWidth bits), depth = memDepth,
      fastCd = ClockDomain.current, slowCd = ClockDomain.current, outReg = memOutReg)
    (bram, bram.io.port0, bram.io.port1, 1 + memOutReg.toInt)
  }
  mem.setName("mem").addAttribute("KEEP_HIERARCHY", "TRUE")

  // The data bus already carries the posted-store adapter inside RiscqFiber (on the DataMemBus, ahead of
  // its Tilelink bridge), so the fabric just decodes the resulting Tilelink master.
  val dMemPortDec = riscqCd(Node())
  dMemPortDec at 0 of riscqFiber.dBus
  val dBusFiber = riscqCd(TileLinkCpuMemFiber(memPort0, latency = memReadLatency))
  dBusFiber.up at memOffset of dMemPortDec

  // optional test tap: a second master into the data-bus decode (the sim drives dTap.node.bus directly).
  val dTap = withTestTap generate riscqCd(new MasterBus(testTapM2s))
  if (withTestTap) dMemPortDec at 0 of dTap.node

  // instruction fetch + host image-load share the BRAM slow port through an arbiter; the host enters on
  // the `iLoad` MasterBus slave-IO. iLoad lives in the implicit (dsp) clock domain — same as the arbiter
  // / BRAM slow port — so there is NO clock crossing inside this hard Component: the parent puts the
  // host→dsp CDC on the fabric arc that drives iLoad (see `RiscqRfWithPulseTableFiber.iMemLoad`),
  // keeping it out of the per-core pblock.
  val iLoad = new MasterBus(hostLoadM2s)
  val iMemPortArb = Node()
  iMemPortArb at memOffset of riscqFiber.iBus
  iMemPortArb at 0 of iLoad.node
  val iBusFiber = TileLinkCpuMemFiber(memPort1, latency = memReadLatency)
  iBusFiber.up at 0 of iMemPortArb

  // ── control block: time (+ the core-local readout-result sink, added after the bridge) ──
  val memMapFiber = riscqCd(MemMapFiber(addressWidth = 22, dataWidth = 32))
  val ctrlTime    = riscqCd(getPipe(time, 1))
  val timeMemMap  = TimeMemMap(ctrlTime, runOrigin = if (runOrigin) Some(riscqCd(getPipe(origin, 1))) else None,
    signedWait = signedWait)
  memMapFiber.addMapping(timeMemMap.mapping)

  // ── posted-link bridge + core-local readout-result sink (riscqCd) ──
  val posted = riscqCd { new Composite(this, "posted") {
    val bridge = RfLinkBridge(rfAddrWidth)
    bridge.up at SizeMapping(0x10000, 4 << 16) of dMemPortDec
    bridge.up.setUpConnection(a = StreamPipe.FULL, d = StreamPipe.FULL)

    val sink = ReadoutResultSink(readoutAccWidth, resAddr = 0x4200, realAddr = 0x4204, imagAddr = 0x4208)
    sink.resultIn << resultIn                       // up-link result arrives from the DSP (parent); the
                                                    // sink mirrors the decoder's res.valid level (no arm)
  } }

  cmd << posted.bridge.cmd                          // posted RF writes leave for the DSP datapath

  // finish the control block: add the local result-sink read map, then connect the bus.
  memMapFiber.addMapping(posted.sink.mapping)        // res@0x4200 / real@0x4204 / imag@0x4208 (local)
  memMapFiber.up at SizeMapping(0, 1 << 16) of dMemPortDec
  memMapFiber.up.setUpConnection(a = StreamPipe.FULL, d = StreamPipe.FULL)
}

object RiscvSoc {
  /** Host image-load master (BRAM slow port). Deliberately **generous** (4-bit source, 32-bit address,
   *  256-byte size ⇒ 4-bit size field) so it is a superset of every parent's host fabric (e.g. the
   *  PulseTableSoc AXI window, which negotiates different narrower params);
   *  the parent's `iMemLoad` bridge resizes its narrower fields up into this fixed param, so `RiscvSoc`
   *  stays byte-for-byte identical across tops (the floorplan-transfer requirement). */
  def defaultHostLoadM2s = tilelink.M2sParameters(
    addressWidth = 32, dataWidth = 32,
    masters = List(tilelink.M2sAgent(name = null, mapping = List(tilelink.M2sSource(
      id = SizeMapping(0, 16),
      emits = tilelink.M2sTransfers(
        get = tilelink.SizeRange.upTo(0x100), putFull = tilelink.SizeRange.upTo(0x100),
        putPartial = tilelink.SizeRange.upTo(0x100)))))))

  /** Test-tap master (a sim's MasterAgent drives this, bridged in from the parent like `iLoad`).
   *  Generous param (same shape as the host load) so the parent's test master resizes up into it. */
  def defaultTestTapM2s = defaultHostLoadM2s
}
