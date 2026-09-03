"""Loopback check: does the DAC of a drive channel reach the bundle's readout ADC?

Plays one tone on a drive channel while recording the ADC, then reports the strongest spectral
line near the tone. Bundle-agnostic (the short gate fits every bundle's trace RAM), so it is the
first thing to run after cabling a board — before suspecting anything in software.

  PYTHONPATH=software/client python software/examples/loopback_check.py --remote 192.168.3.1 \
      --bundle rfsoc4x2-2dac-fine --ch 1       # the readout drive = DAC1 on the 2-DAC bundle

Wiring per bundle: software/server/README.md.
"""
import argparse
import sys

import numpy as np

from riscq import artiqapi as A
from riscq.driver.remote import RemoteDriver
from riscq.map import SocMap, SocParams


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remote", required=True, help="board server address")
    ap.add_argument("--bundle", default="rfsoc4x2-1q-fine")
    ap.add_argument("--ch", type=int, default=1, help="drive channel: 0 = gate, 1 = readout")
    ap.add_argument("--freq-mhz", type=float, default=82.0)
    ap.add_argument("--amplitude", type=float, default=0.4)
    ap.add_argument("--work", default="loopback_check_work")
    ap.add_argument("--out", default=None, help="save the trace (npz)")
    a = ap.parse_args()

    drv = RemoteDriver(a.remote)
    drv.board.load(a.bundle)
    m = SocMap(SocParams.from_json(drv.board.get_params()))
    p = m.params
    print(f"loaded {a.bundle}: drive ch{a.ch} -> DAC{p.dac_map[0][a.ch]}, readout ADC{p.adc_map[0]}",
          flush=True)

    core = A.Core(m)
    dds, adc = A.DDSChannel(core, a.ch, "probe"), A.ADCChannel(core)
    with A.parallel(core):
        with A.branch(core):
            A.delay(core, 2 * A.us)
            dds.set(a.freq_mhz * A.MHz, phase=0.0, amplitude=a.amplitude)
            dds.sw.pulse(20 * A.us)
        with A.branch(core):
            adc.gate(25 * A.us)
    res = A.run(drv, core, a.work, doc=f"loopback check ch{a.ch}")
    tr = adc.fetch_trace().astype(np.float64)
    fs, n = float(res.fs), tr.size

    spec = np.abs(np.fft.rfft(tr * np.hanning(n))) / n * 4      # Hann-corrected single-sided amplitude
    f = np.fft.rfftfreq(n, 1 / fs)
    f0 = a.freq_mhz * 1e6
    band = (f > f0 - 20e6) & (f < f0 + 20e6)
    k = int(np.argmax(np.where(band, spec, 0)))
    floor = float(np.median(spec[band]))
    present = spec[k] > 100 and spec[k] > 20 * floor
    print(f"trace: {n} samples @ {fs / 1e6:.2f} MS/s, max|s| = {np.abs(tr).max():.0f}, rms = {tr.std():.1f}")
    print(f"strongest line within 20 MHz of {a.freq_mhz} MHz: {f[k] / 1e6:.2f} MHz, {spec[k]:.0f} codes "
          f"(band median {floor:.1f})")
    print("LOOPBACK:", "TONE PRESENT" if present
          else "NO TONE - check the cable and SMA seating of this DAC -> ADC pair")
    if a.out:
        np.savez(a.out, trace=tr, fs=fs)
    return 0 if present else 1


if __name__ == "__main__":
    sys.exit(main())
