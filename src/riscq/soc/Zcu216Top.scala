package riscq.soc

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import riscq.misc.{Axi4VivadoHelper, Axi4StreamVivadoHelper, VivadoClkHelper}

/**
 * SoC toplevel scaffolding for the RFSoC (ZCU216-class) part — the agentic counterpart of the RISC-Q
 * reference `Zcu216Top` / `RiscqZcu216SocPorts` / `RiscqZcu216MemoryMap`. When `vivado` is off the FPGA
 * IP-packaging attributes (`X_INTERFACE_INFO`, `FREQ_HZ`, clocking blackboxes) are omitted — these ports
 * are plain `Bool`/`Axi4`/`Stream`.
 */

/**
 * Host-AXI address map, **derived** from the per-window sizes rather than the reference's scattered
 * literals. One equal region per window-class; per-core sub-windows are `coreStride` apart. CPU-visible
 * (per-core `dBus`) offsets live in [[RiscqRfWithPulseTableFiber]]; this is only the host-AXI side, which
 * can be laid out freely.
 */
case class SocMemoryMap(
    qubitNum: Int,
    coreMemBytes: Int,        // bytes of instruction RAM visible per core
    pulseMemBytes: Int,       // bytes of gate-drive pulse-envelope RAM visible per core
    readoutEnvBytes: Int,     // bytes of readout-drive (interpolated) envelope RAM visible per core
    demodEnvBytes: Int,       // bytes of demod-carrier (interpolated) envelope RAM visible per core
    readoutBufBytes: Int      // bytes of readout buffers visible per core
) {
  private def pow2ceil(x: Int): Int = 1 << log2Up(x)
  val coreStride       = pow2ceil(coreMemBytes)
  val pulseStride      = pow2ceil(pulseMemBytes)
  val readoutEnvStride = pow2ceil(readoutEnvBytes)
  val demodEnvStride   = pow2ceil(demodEnvBytes)
  val readoutStride    = pow2ceil(readoutBufBytes)

  // six equal top-level regions; each holds qubitNum strided sub-windows.
  val regionSize = pow2ceil(Seq(coreStride, pulseStride, readoutEnvStride, demodEnvStride, readoutStride).max * qubitNum)

  val coreMemBase    = 0
  val pulseMemBase   = 1 * regionSize
  val readoutEnvBase = 2 * regionSize
  val demodEnvBase   = 3 * regionSize
  val readoutBufBase = 4 * regionSize
  val hostCtrlBase   = 5 * regionSize

  def coreMemOffset(core: Int)    = coreMemBase + core * coreStride
  def pulseMemOffset(core: Int)   = pulseMemBase + core * pulseStride
  def readoutEnvOffset(core: Int) = readoutEnvBase + core * readoutEnvStride
  def demodEnvOffset(core: Int)   = demodEnvBase + core * demodEnvStride
  def readoutBufOffset(core: Int) = readoutBufBase + core * readoutStride
}

/**
 * External SoC ports. When `vivado` is set the AXI / AXI-Stream ports carry the Vivado
 * `X_INTERFACE_INFO` attributes (interface names `S_AXIS` / `DAC{i}_AXIS` / `ADC{i}_AXIS`) the IP packager
 * needs to bundle them into bus interfaces; on the default they are plain scalar ports, as the
 * functional sims/benches expect.
 */
case class RiscqZcu216SocPorts(
    dacNum: Int = 16, adcNum: Int = 16, dacBatch: Int = 16, adcBatch: Int = 16, dataWidth: Int = 16,
    vivado: Boolean = false
) extends Bundle {
  val dspClk = in Bool ()
  val dspRst = in Bool ()
  val axi    = slave(Axi4(Axi4Config(addressWidth = 32, dataWidth = 32, idWidth = 2)))
  val dac    = List.fill(dacNum)(master port Stream(Bits(dacBatch * dataWidth bits)))
  val adc    = List.fill(adcNum)(slave port Stream(Bits(adcBatch * dataWidth bits)))

  // DAC always streams; ADC is always accepted (free-running converters).
  dac.foreach(_.valid := True)
  adc.foreach(_.ready := True)

  if (vivado) {
    Axi4VivadoHelper.addInference(axi, "S_AXIS")
    dac.zipWithIndex.foreach { case (d, id) => Axi4StreamVivadoHelper.addStreamInference(d, s"DAC${id}_AXIS") }
    adc.zipWithIndex.foreach { case (a, id) => Axi4StreamVivadoHelper.addStreamInference(a, s"ADC${id}_AXIS") }
  }
}

/**
 * Abstract toplevel: owns the two clock domains (the host AXI clock, taken as the implicit clock
 * domain, and the external `dspClk`/`dspRst` converter clock) and the external ports.
 */
abstract class Zcu216Top(
    dacNum: Int = 16, adcNum: Int = 16, dacBatch: Int = 16, adcBatch: Int = 16, dataWidth: Int = 16,
    vivado: Boolean = false
) extends Component {
  val io = RiscqZcu216SocPorts(dacNum, adcNum, dacBatch, adcBatch, dataWidth, vivado)

  val hostCd = ClockDomain.current
  io.dspClk.setName("dspClk")
  io.dspRst.setName("dspRst")
  val dspCd = ClockDomain(io.dspClk, io.dspRst)

  // Vivado IP-packaging: name the host clock/reset `hostClk`/`hostRst` and tag both domains with their
  // target frequencies so the block design's clock inference + CDC analysis work. Default off so the
  // single-clock OOC bench (which constrains the un-renamed `clk`) is undisturbed.
  if (vivado) {
    hostCd.renamePulledWires("hostClk", "hostRst")
    VivadoClkHelper.addInference(hostCd.readClockWire, hostCd.readResetWire, 100000000L)
    VivadoClkHelper.addInference(dspCd.readClockWire, io.dspRst, 500000000L)
  }
}
