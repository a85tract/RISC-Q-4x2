"""Batched calibration kernels (spec 08 §2.1, spec 09 §4): one run returns a WHOLE sweep. Each
calibration family has one ~20-line kernel that walks the sweep on a fixed-period grid — the sweep is
a loop on the core, not a host loop of short runs (the design of record, README principle 3 / spec 06
§2).

Every swept knob is an affine sequence (`x_i = x0 + i·dx`), so the kernel COMPUTES it on-core instead
of walking a host-preloaded input Array (spec 09): code knobs (amp/freq codes) accumulate a Q16 pair
`(x0q, dxq)` written to the register RAW — the integer code sits in `data[31:16]`, the low fraction
is ignored by hardware (spec 12), so there is no `>> 16` extraction; time/phase knobs accumulate a
plain int pair (phase pre-seated by the host). `out` is the only Array left. The knob advance sits at the END of the point: rows realize `[x0, x0+dx, x0+2dx, ...]`. There is
no warm-up row — the cold-first-read guard it once absorbed was root-caused and retired (spec 11).

Every shot fires on a fixed `period` grid whose idle head is the T1 relax reset (the model has no
auto-reset, spec-M4 B0), so all readouts land at the same time-referenced demod-LO phase — no
straddled-reset dependence (spec 08 §2.2). A compile-time `mode` binding folds the shot tail (spec 08
§2.1 modes table): COUNTS accumulates the hardware-classified bit per point (exact, self-normalised
populations — no |0> reference needed); RAW writes per-shot IQ through a cursor (classifier work / the
poor-separation fallback); k_vna sums per-point IQ integrals coherently (iqsum, the matched-pair VNA).
Every measurement plays the channel-1 readout drive (ro["meas"]) covering the demod window — mandatory
on hardware and the projective co-sim model's window trigger; firing the demod IS the readout (its
`dur` is the integration window). read_res() is called first every shot (it HALTS until the integral
settles) so read_real/read_imag latch THIS shot; the demod carrier is issued LAST so the run-invariant
demod, not the variable drive, is the trailing posted store (spec 08 §2.2).

`prep` is a scalar per-run param (spec 09 §1): the readout cals that need both prep states run the same
resident program twice (prep=0 / prep=1) through the setup/rerun layer. The virtual-Z in k_ramsey is a
channel phaseOffset (spec 09 B1), NOT a pulse-table `phase` rewrite, so the calibrated x90 table phase
survives the run. k_rabi's n-gate train is one `play` + N-1 bare `fire`s — B0 auto-advances the
channel `startTime` by the fired slot's `dur`, so a contiguous train needs no per-pulse time math.
"""

from riscq.cal.base import SEP
from riscq.lang import Array, ParamTable, kernel
from riscq.map import READOUT_LEAD

COUNTS = 0
RAW = 1


@kernel
def k_rabi(gate: ParamTable, ro: ParamTable, demod: ParamTable, out: Array, npts: int, shots: int,
           period: int, ngates: int, code: int, mode: int, a0q: int, daq: int, prep: int):
    """Batched Rabi: sweep the X90 amplitude on-core (Q16 pair a0q/daq; realized code = aq >> 16,
    written raw — spec 12),
    `shots` shots/point on a fixed grid. `prep` gates the drive; the n-gate train is one `play` + N-1
    bare `fire`s (B0 startTime auto-advance). COUNTS → out[i] += classified bit; RAW → per-shot IQ
    (out sized 2·npts·shots), cursor k."""
    init_pulse_params(demod.pulses)  # noqa: F821
    set_freq(demod, code)  # noqa: F821
    init_pulse_params(ro.pulses)  # noqa: F821
    set_freq(ro, ro.freq)  # noqa: F821
    init_pulse_params(gate.pulses)  # noqa: F821
    set_freq(gate, gate.freq)  # noqa: F821
    d = gate["x90"].dur  # noqa: F821
    t_ro = now() + period  # noqa: F821  first grid slot (idle head resets to |0>)
    aq = a0q
    if mode == RAW:
        k = 0
    for i in range(npts):
        set_amp(gate, gate["x90"], aq)  # noqa: F821  raw Q16 accumulator: integer amp code sits in
        #                                              data[31:16], the fraction is ignored by HW (spec 12)
        for s in range(shots):
            if prep == 1:
                play(gate, gate["x90"], t_ro - SEP - ngates * d)  # noqa: F821  first gate of the train
                for g in range(ngates - 1):
                    fire(gate, gate["x90"])  # noqa: F821  B0: startTime auto-advances by dur
            play(ro, ro["meas"], t_ro)  # noqa: F821     readout drive (window trigger; hardware)
            play(demod, demod["sq"], t_ro)  # noqa: F821  demod carrier LAST; firing it IS the readout
            wait_until(t_ro + READOUT_LEAD)  # noqa: F821
            if mode == COUNTS:
                out[i] += read_res()  # noqa: F821  self-normalised population count
            else:
                read_res()  # noqa: F821            HALT until settled, then latch fresh IQ
                out[k] = read_real()  # noqa: F821
                out[k + 1] = read_imag()  # noqa: F821
                k = k + 2
            t_ro = t_ro + period  # noqa: F821  next grid slot
        aq = aq + daq


@kernel
def k_ramsey(gate: ParamTable, ro: ParamTable, demod: ParamTable, out: Array, npts: int, shots: int,
             period: int, code: int, mode: int, w0: int, dw: int, p0: int, dp: int):
    """Batched Ramsey (covers Frequency / Phase / T2): per point X90 — wait w — virtual-Z(phi) — X90,
    `shots` shots/point on a fixed grid (`period` sized for the longest wait). The wait (w0/dw,
    batches) is computed on-core; the virtual-Z phase (p0/dp) is a host-pre-seated pair accumulated
    in the seated domain (spec 12) — the virtual-Z is a channel phaseOffset (B1), so the calibrated
    x90 table phase is UNTOUCHED. Same COUNTS/RAW fold as
    k_rabi."""
    init_pulse_params(demod.pulses)  # noqa: F821
    set_freq(demod, code)  # noqa: F821
    init_pulse_params(ro.pulses)  # noqa: F821
    set_freq(ro, ro.freq)  # noqa: F821
    init_pulse_params(gate.pulses)  # noqa: F821
    set_freq(gate, gate.freq)  # noqa: F821
    d = gate["x90"].dur  # noqa: F821
    t_ro = now() + period  # noqa: F821
    w = w0
    phi = p0
    if mode == RAW:
        k = 0
    for i in range(npts):
        for s in range(shots):
            ta = t_ro - SEP - d          # 2nd X90 start
            set_phase_offset(gate, 0)  # noqa: F821  frame 0 for the 1st X90 (captured at fire)
            play(gate, gate["x90"], ta - w - d)  # noqa: F821  1st X90 (`wait` w earlier)
            set_phase_offset(gate, phi)  # noqa: F821  virtual-Z on the 2nd X90; table phase untouched
            play(gate, gate["x90"], ta)  # noqa: F821  2nd X90
            play(ro, ro["meas"], t_ro)  # noqa: F821
            play(demod, demod["sq"], t_ro)  # noqa: F821
            wait_until(t_ro + READOUT_LEAD)  # noqa: F821
            if mode == COUNTS:
                out[i] += read_res()  # noqa: F821
            else:
                read_res()  # noqa: F821
                out[k] = read_real()  # noqa: F821
                out[k + 1] = read_imag()  # noqa: F821
                k = k + 2
            t_ro = t_ro + period  # noqa: F821
        w = w + dw
        phi = phi + dp


@kernel
def k_t1(gate: ParamTable, ro: ParamTable, demod: ParamTable, out: Array, npts: int, shots: int,
         period: int, code: int, mode: int, d0: int, dd: int, prep: int):
    """Batched T1 (and the raw-IQ shots kernel): per point (prep) X180 to |1> ending `dly` batches
    before the readout; `dly` is computed on-core (d0/dd). T1 cal: prep=1, sweep the delay, COUNTS.
    Raw-IQ clusters (ReadoutCalibration / ReadoutFidelity): prep=0 → |0>, prep=1 → |1> (one run each,
    dd=0), RAW."""
    init_pulse_params(demod.pulses)  # noqa: F821
    set_freq(demod, code)  # noqa: F821
    init_pulse_params(ro.pulses)  # noqa: F821
    set_freq(ro, ro.freq)  # noqa: F821
    init_pulse_params(gate.pulses)  # noqa: F821
    set_freq(gate, gate.freq)  # noqa: F821
    t_ro = now() + period  # noqa: F821
    dly = d0
    if mode == RAW:
        k = 0
    for i in range(npts):
        for s in range(shots):
            if prep == 1:
                play(gate, gate["x180"], t_ro - dly - gate["x180"].dur)  # noqa: F821
            play(ro, ro["meas"], t_ro)  # noqa: F821
            play(demod, demod["sq"], t_ro)  # noqa: F821
            wait_until(t_ro + READOUT_LEAD)  # noqa: F821
            if mode == COUNTS:
                out[i] += read_res()  # noqa: F821
            else:
                read_res()  # noqa: F821
                out[k] = read_real()  # noqa: F821
                out[k + 1] = read_imag()  # noqa: F821
                k = k + 2
            t_ro = t_ro + period  # noqa: F821
        dly = dly + dd


@kernel
def k_vna(ro: ParamTable, demod: ParamTable, out: Array, npts: int, shots: int, period: int,
          sh: int, c0q: int, dcq: int):
    """Batched VNA (Separation) in iqsum mode: retune the readout drive and demod as a MATCHED PAIR
    (`set_freq(ro, cq)` raw Q16 + `set_freq(demod, (4*c)<<16)`, the ADC code is 4× the DAC code seated
    into data[31:16] — spec 12) over an on-core Q16 frequency sweep (c0q/dcq; realized code
    c = cq >> 16) and coherently sum `shots` per-point IQ
    integrals (>> sh headroom). No qubit prep — a |0> readout sweep. The retune is scheduled a full
    `period` (≫ LEAD) ahead of its play (spec 08 §2.2, B1)."""
    init_pulse_params(demod.pulses)  # noqa: F821
    init_pulse_params(ro.pulses)  # noqa: F821
    t_ro = now() + period  # noqa: F821
    cq = c0q
    j = 0
    for i in range(npts):
        c = cq >> 16
        set_freq(ro, cq)  # noqa: F821          raw Q16 accumulator: DAC code in data[31:16] (spec 12)
        set_freq(demod, (4 * c) << 16)  # noqa: F821  ADC code = 4x the rounded DAC code, seated (spec 12)
        for s in range(shots):
            play(ro, ro["meas"], t_ro)  # noqa: F821
            play(demod, demod["sq"], t_ro)  # noqa: F821
            wait_until(t_ro + READOUT_LEAD)  # noqa: F821
            read_res()  # noqa: F821  HALT until settled
            out[j] += read_real() >> sh  # noqa: F821    coherent per-point complex sum
            out[j + 1] += read_imag() >> sh  # noqa: F821
            t_ro = t_ro + period  # noqa: F821
        j = j + 2
        cq = cq + dcq
