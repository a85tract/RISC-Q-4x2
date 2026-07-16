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
poor-separation fallback); IQSUM (k_vna only) sums per-point IQ integrals coherently (the matched-pair
VNA). k_vna folds RAW too — the same frequency sweep with a |1> prep, so two reruns give both prep
states' clusters at every point (Separation's two-state SNR, spec 13 §5).
Every measurement plays the channel-1 readout drive (ro["meas"]) covering the demod window — mandatory
on hardware and the projective co-sim model's window trigger; firing the demod IS the readout (its
`dur` is the integration window). The demod opens `ddly` batches after the drive (a compile-time
binding: the config's ADC round-trip delay, spec 13 §3 — 0 in co-sim). read_res() is called first every
shot (it HALTS until the integral settles) so read_real/read_imag latch THIS shot; the demod carrier is
issued LAST so the run-invariant demod, not the variable drive, is the trailing posted store (spec 08 §2.2).

`prep` is a scalar per-run param (spec 09 §1): the readout cals that need both prep states run the same
resident program twice (prep=0 / prep=1) through the setup/rerun layer. The virtual-Z in k_ramsey and
k_phase is a channel phaseOffset (spec 09 B1), NOT a pulse-table `phase` rewrite, so the calibrated x90
table phase survives the run. k_rabi's n-gate train is one `play` + N-1 bare `fire`s — B0 auto-advances
the channel `startTime` by the fired slot's `dur`, so a contiguous train needs no per-pulse time math.

THE X90 FRAME BRACKET (spec 13 §7). qcal's X90 is virtualz(vz0) . FAST_DRAG . virtualz(vz1): part of
its frame correction lands before the pulse and part after, so the pulse's axis sits BETWEEN the two
advances. EVERY X90 play in EVERY kernel is exactly that — `set_phase_offset(f + vz0); play/fire;
f += vz0 + vz1` with `f` an on-core frame rebuilt each shot — so the pair Phase calibrates
(`qubit/{q}/x90/vz`) is actually played by the other cals too, not only by k_phase. The kernels bind
the pair as two compile-time seated phase words: `vz0` and `vzsum` = vz0 + vz1 (base.x90_vz; the
seated domain of spec 12, wrapping mod 2^32 = mod one turn). A config with no pair (every co-sim one)
binds 0/0 and the bracket is a no-op. The config's X pulse carries no pair, so its branches play bare.
"""

from riscq.cal.base import SEP, X90
from riscq.lang import Array, ParamTable, kernel
from riscq.map import READOUT_LEAD

COUNTS = 0
RAW = 1
IQSUM = 2

Y180_X90 = 0    # k_phase's compile-time `seq` fold: qcal's two Phase sequences (spec 13 §6)
X180_Y90 = 1


@kernel
def k_rabi(gate: ParamTable, ro: ParamTable, demod: ParamTable, out: Array, npts: int, shots: int,
           period: int, ngates: int, code: int, mode: int, ddly: int, prep_gate: int, vz0: int,
           vzsum: int, a0q: int, daq: int, prep: int, herald: int, hoff: int):
    """Batched Rabi: sweep the CALIBRATED GATE's amplitude on-core (Q16 pair a0q/daq; realized code =
    aq >> 16, written raw — spec 12), `shots` shots/point on a fixed grid. `prep` gates the drive; the
    n-gate train is one `play` + N-1 bare `fire`s (B0 startTime auto-advance). COUNTS → out[i] +=
    classified bit; RAW → per-shot IQ (out sized 2·npts·shots), cursor k.

    `prep_gate` is the COMPILE-TIME binding of qcal's `gate=` knob (spec 13 §7): X90 → the train is the
    config's X90, each play carrying its frame bracket (vz0/vzsum, see the module docstring); X → the
    train is the config's own X pulse, played bare. The dead branch is eliminated before slot
    resolution, so an X90 sweep needs no "x" slot in the table.

    `herald` is a COMPILE-TIME binding (spec 13 §8): 0 → today's shot tail, byte-identical; 1 → a
    pre-sequence herald read at `t_ro − hoff` gates the shot on the qubit being in |0>, and `out`
    carries interleaved (count, kept) pairs (P = count/kept). Heralding folds with COUNTS only."""
    init_pulse_params(demod.pulses)  # noqa: F821
    set_freq(demod, code)  # noqa: F821
    init_pulse_params(ro.pulses)  # noqa: F821
    set_freq(ro, ro.freq)  # noqa: F821
    init_pulse_params(gate.pulses)  # noqa: F821
    set_freq(gate, gate.freq)  # noqa: F821
    if prep_gate == X90:
        d = gate["x90"].dur  # noqa: F821
    else:
        d = gate["x"].dur  # noqa: F821
    t_ro = now() + period  # noqa: F821  first grid slot (idle head resets to |0>)
    aq = a0q
    if mode == RAW:
        k = 0
    for i in range(npts):
        if prep_gate == X90:
            set_amp(gate, gate["x90"], aq)  # noqa: F821  raw Q16 accumulator: integer amp code sits in
            #                                              data[31:16], the fraction ignored by HW (spec 12)
        else:
            set_amp(gate, gate["x"], aq)  # noqa: F821
        for s in range(shots):
            if herald == 1:
                play(ro, ro["meas"], t_ro - hoff)  # noqa: F821         pre-sequence herald read
                play(demod, demod["sq"], t_ro - hoff + ddly)  # noqa: F821
                wait_until(t_ro - hoff + ddly + READOUT_LEAD)  # noqa: F821
                h = read_res()  # noqa: F821
            else:
                h = 0
            if h == 0:                 # herald passed (qubit in |0>) — or heralding off (h folds to 0)
                if prep == 1:
                    if prep_gate == X90:
                        f = vz0                  # the frame is rebuilt every shot: frame(0) + vz0
                        set_phase_offset(gate, f)  # noqa: F821
                        play(gate, gate["x90"], t_ro - SEP - ngates * d)  # noqa: F821  1st gate of the train
                        for g in range(ngates - 1):
                            f = f + vzsum        # frame += vz0 + vz1, then the next play's + vz0
                            set_phase_offset(gate, f)  # noqa: F821
                            fire(gate, gate["x90"])  # noqa: F821  B0: startTime auto-advances by dur
                    else:
                        play(gate, gate["x"], t_ro - SEP - ngates * d)  # noqa: F821  the config's X, bare
                        for g in range(ngates - 1):
                            fire(gate, gate["x"])  # noqa: F821
                play(ro, ro["meas"], t_ro)  # noqa: F821     readout drive (window trigger; hardware)
                play(demod, demod["sq"], t_ro + ddly)  # noqa: F821  demod LAST; firing it IS the readout
                wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821
                if mode == COUNTS:
                    if herald == 1:
                        out[2 * i] = out[2 * i] + read_res()  # noqa: F821    kept-shot |1> count
                        out[2 * i + 1] = out[2 * i + 1] + 1  # noqa: F821     kept-shot denominator
                    else:
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
             period: int, code: int, mode: int, ddly: int, vz0: int, vzsum: int, w0: int, dw: int,
             p0: int, dp: int, herald: int, hoff: int):
    """Batched Ramsey (covers Frequency / T2): per point X90 — wait w — virtual-Z(phi) — X90,
    `shots` shots/point on a fixed grid (`period` sized for the longest wait). The wait (w0/dw,
    batches) is computed on-core; the virtual-Z phase (p0/dp) is a host-pre-seated pair accumulated
    in the seated domain (spec 12) — the virtual-Z is a channel phaseOffset (B1), so the calibrated
    x90 table phase is UNTOUCHED. Same COUNTS/RAW fold as k_rabi.

    The swept virtual-Z phi is an Rz BETWEEN the two X90s, so it COMPOSES with each X90's own frame
    bracket rather than replacing it: the 1st X90 fires at frame(0) + vz0, the frame then advances by
    vzsum (the gate's pair) and by phi (the sweep), so the 2nd fires at vzsum + phi + vz0.

    `herald` is a COMPILE-TIME binding (spec 13 §8): 0 → today's shot tail, byte-identical; 1 → a
    pre-sequence herald read at `t_ro − hoff` gates the shot on the qubit being in |0>, and `out`
    carries interleaved (count, kept) pairs (P = count/kept). Heralding folds with COUNTS only."""
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
            if herald == 1:
                play(ro, ro["meas"], t_ro - hoff)  # noqa: F821         pre-sequence herald read
                play(demod, demod["sq"], t_ro - hoff + ddly)  # noqa: F821
                wait_until(t_ro - hoff + ddly + READOUT_LEAD)  # noqa: F821
                h = read_res()  # noqa: F821
            else:
                h = 0
            if h == 0:                 # herald passed (qubit in |0>) — or heralding off (h folds to 0)
                ta = t_ro - SEP - d          # 2nd X90 start
                set_phase_offset(gate, vz0)  # noqa: F821  1st X90: frame(0) + vz0 (captured at fire)
                play(gate, gate["x90"], ta - w - d)  # noqa: F821  1st X90 (`wait` w earlier)
                set_phase_offset(gate, phi + vz0 + vzsum)  # noqa: F821  2nd X90: (frame vzsum + the swept
                #                                    virtual-Z phi) + vz0; the table phase stays untouched
                play(gate, gate["x90"], ta)  # noqa: F821  2nd X90
                play(ro, ro["meas"], t_ro)  # noqa: F821
                play(demod, demod["sq"], t_ro + ddly)  # noqa: F821  demod opens `ddly` after the drive
                wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821
                if mode == COUNTS:
                    if herald == 1:
                        out[2 * i] = out[2 * i] + read_res()  # noqa: F821    kept-shot |1> count
                        out[2 * i + 1] = out[2 * i + 1] + 1  # noqa: F821     kept-shot denominator
                    else:
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
         period: int, code: int, mode: int, ddly: int, vz0: int, vzsum: int, d0: int, dd: int,
         prep: int, prep_gate: int, herald: int, hoff: int):
    """Batched T1 (and the raw-IQ shots kernel): per point (prep) drive the qubit to |1> ending `dly`
    batches before the readout; `dly` is computed on-core (d0/dd). T1 cal: prep=1, sweep the delay,
    COUNTS. Raw-IQ clusters (ReadoutCalibration / ReadoutFidelity): prep=0 → |0>, prep=1 → |1> (one run
    each, dd=0), RAW.

    `prep_gate` is a COMPILE-TIME binding folding to qcal's two |1> preps (spec 13 §4): X90 → two X90
    plays (one `play` + one bare `fire`, contiguous by B0's startTime auto-advance), each carrying its
    frame bracket (vz0/vzsum — the module docstring); X → one play of the config's own X pulse, bare.
    The dead branch is eliminated before slot resolution, so an X90 prep needs no "x" slot in the
    table.

    `herald` is a COMPILE-TIME binding (spec 13 §8): 0 → today's shot tail, byte-identical (the whole
    herald branch and the (count, kept) split fold away). 1 → a pre-sequence readout at `t_ro − hoff`
    finds the qubit; only if it reads |0> (`h == 0`) does the shot run and count, and `out` carries
    interleaved (count, kept) pairs so the host divides P = count/kept. Heralding folds with COUNTS
    only; the RAW readout cals bind herald=0."""
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
            if herald == 1:
                play(ro, ro["meas"], t_ro - hoff)  # noqa: F821         pre-sequence herald read
                play(demod, demod["sq"], t_ro - hoff + ddly)  # noqa: F821
                wait_until(t_ro - hoff + ddly + READOUT_LEAD)  # noqa: F821
                h = read_res()  # noqa: F821
            else:
                h = 0
            if h == 0:                 # herald passed (qubit in |0>) — or heralding off (h folds to 0)
                if prep == 1:
                    if prep_gate == X90:   # X90·X90: one play + one bare fire (B0 startTime auto-advance)
                        set_phase_offset(gate, vz0)  # noqa: F821          X90 #1: frame(0) + vz0
                        play(gate, gate["x90"], t_ro - dly - 2 * gate["x90"].dur)  # noqa: F821
                        set_phase_offset(gate, vz0 + vzsum)  # noqa: F821  X90 #2: frame(vzsum) + vz0
                        fire(gate, gate["x90"])  # noqa: F821
                    else:                  # the config's own X pulse, played once (no frame pair)
                        play(gate, gate["x"], t_ro - dly - gate["x"].dur)  # noqa: F821
                play(ro, ro["meas"], t_ro)  # noqa: F821
                play(demod, demod["sq"], t_ro + ddly)  # noqa: F821  demod opens `ddly` after the drive
                wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821
                if mode == COUNTS:
                    if herald == 1:
                        out[2 * i] = out[2 * i] + read_res()  # noqa: F821    kept-shot |1> count
                        out[2 * i + 1] = out[2 * i + 1] + 1  # noqa: F821     kept-shot denominator
                    else:
                        out[i] += read_res()  # noqa: F821
                else:
                    read_res()  # noqa: F821
                    out[k] = read_real()  # noqa: F821
                    out[k + 1] = read_imag()  # noqa: F821
                    k = k + 2
            t_ro = t_ro + period  # noqa: F821
        dly = dly + dd


@kernel
def k_vna(gate: ParamTable, ro: ParamTable, demod: ParamTable, out: Array, npts: int, shots: int,
          period: int, sh: int, ddly: int, mode: int, prep_gate: int, vz0: int, vzsum: int, c0q: int,
          dcq: int, prep: int):
    """Batched VNA: retune the readout drive and demod as a MATCHED PAIR (`set_freq(ro, cq)` raw Q16 +
    `set_freq(demod, (4*c)<<16)`, the ADC code is 4× the DAC code seated into data[31:16] — spec 12)
    over an on-core Q16 frequency sweep (c0q/dcq; realized code c = cq >> 16). The retune is scheduled
    a full `period` (≫ LEAD) ahead of its play (spec 08 §2.2, B1). Two `mode` folds:

      IQSUM — no prep, coherently sum `shots` per-point IQ integrals (>> sh headroom): a |0> magnitude
              sweep (the ADC-tone datapath probe of test_batch);
      RAW   — per-shot IQ through a cursor (out sized 2·npts·shots), with the |1> prep gated by the
              runtime `prep` scalar: run it twice (prep=0, prep=1 — two reruns of the ONE resident
              program) and the host has both prep states' clusters at every frequency, which is what
              Separation's two-state cluster SNR needs (spec 13 §5).

    `prep_gate` folds qcal's two |1> preps exactly as k_t1 does (X90·X90 / the config's own X). In
    IQSUM mode the gate table is reached only from dead branches, so it needs no binding at all."""
    init_pulse_params(demod.pulses)  # noqa: F821
    init_pulse_params(ro.pulses)  # noqa: F821
    if mode == RAW:                  # compile-time: the IQSUM |0> sweep never touches the gate channel
        init_pulse_params(gate.pulses)  # noqa: F821
        set_freq(gate, gate.freq)  # noqa: F821
    t_ro = now() + period  # noqa: F821
    cq = c0q
    k = 0
    for i in range(npts):
        c = cq >> 16
        set_freq(ro, cq)  # noqa: F821          raw Q16 accumulator: DAC code in data[31:16] (spec 12)
        set_freq(demod, (4 * c) << 16)  # noqa: F821  ADC code = 4x the rounded DAC code, seated (spec 12)
        for s in range(shots):
            if mode == RAW:
                if prep == 1:
                    if prep_gate == X90:   # X90·X90: one play + one bare fire (B0 auto-advance)
                        set_phase_offset(gate, vz0)  # noqa: F821          X90 #1: frame(0) + vz0
                        play(gate, gate["x90"], t_ro - SEP - 2 * gate["x90"].dur)  # noqa: F821
                        set_phase_offset(gate, vz0 + vzsum)  # noqa: F821  X90 #2: frame(vzsum) + vz0
                        fire(gate, gate["x90"])  # noqa: F821
                    else:                  # the config's own X pulse, played once (no frame pair)
                        play(gate, gate["x"], t_ro - SEP - gate["x"].dur)  # noqa: F821
            play(ro, ro["meas"], t_ro)  # noqa: F821
            play(demod, demod["sq"], t_ro + ddly)  # noqa: F821  demod opens `ddly` after the drive
            wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821
            read_res()  # noqa: F821  HALT until settled
            if mode == RAW:
                out[k] = read_real()  # noqa: F821       per-shot IQ (point-major cursor)
                out[k + 1] = read_imag()  # noqa: F821
                k = k + 2
            else:
                out[2 * i] += read_real() >> sh  # noqa: F821    coherent per-point complex sum
                out[2 * i + 1] += read_imag() >> sh  # noqa: F821
            t_ro = t_ro + period  # noqa: F821
        cq = cq + dcq


@kernel
def k_ro_amp(gate: ParamTable, ro: ParamTable, demod: ParamTable, out: Array, npts: int, shots: int,
             period: int, code: int, ddly: int, prep_gate: int, vz0: int, vzsum: int, a0q: int,
             daq: int, prep: int, herald: int, hoff: int):
    """Batched readout-drive AMPLITUDE sweep (qcal's Fidelity knob, spec 13 §5), COUNTS mode: prep the
    qubit (the runtime `prep` scalar; `prep_gate` folds X90·X90 / the config's X, as in k_t1), then
    sweep the READOUT drive's amplitude on-core — a Q16 pair (a0q/daq) accumulated into the ro slot's
    amp, exactly as k_rabi sweeps the gate's — and accumulate the hardware-classified bit per point.

    Run it TWICE (prep=0, prep=1: two reruns of the one resident program) and the two count arrays are
    the confusion DIAGONAL under the hardware discriminator — whose demod phase ReadoutCalibration
    fixed and which is NOT retrained per point (no train-on-test). npts=1/daq=0 is the single-amplitude
    confusion program (ReadoutFidelity / Window).

    `herald` is a COMPILE-TIME binding (spec 13 §8): 0 → today's shot tail, byte-identical; 1 → a
    pre-prep herald read at `t_ro − hoff` gates the shot on the qubit being in |0>, and `out` carries
    interleaved (count, kept) pairs (P = count/kept). qcal's transpiler heralds EVERY circuit,
    confusion circuits included. The herald read plays ro["meas"] too, so it runs at the point's
    SWEPT amplitude — the same readout the shot it gates uses."""
    init_pulse_params(demod.pulses)  # noqa: F821
    set_freq(demod, code)  # noqa: F821
    init_pulse_params(ro.pulses)  # noqa: F821
    set_freq(ro, ro.freq)  # noqa: F821
    init_pulse_params(gate.pulses)  # noqa: F821
    set_freq(gate, gate.freq)  # noqa: F821
    t_ro = now() + period  # noqa: F821
    aq = a0q
    for i in range(npts):
        set_amp(ro, ro["meas"], aq)  # noqa: F821  raw Q16 accumulator: the amp code sits in data[31:16]
        for s in range(shots):
            if herald == 1:
                play(ro, ro["meas"], t_ro - hoff)  # noqa: F821         pre-prep herald read
                play(demod, demod["sq"], t_ro - hoff + ddly)  # noqa: F821
                wait_until(t_ro - hoff + ddly + READOUT_LEAD)  # noqa: F821
                h = read_res()  # noqa: F821
            else:
                h = 0
            if h == 0:                 # herald passed (qubit in |0>) — or heralding off (h folds to 0)
                if prep == 1:
                    if prep_gate == X90:
                        set_phase_offset(gate, vz0)  # noqa: F821          X90 #1: frame(0) + vz0
                        play(gate, gate["x90"], t_ro - SEP - 2 * gate["x90"].dur)  # noqa: F821
                        set_phase_offset(gate, vz0 + vzsum)  # noqa: F821  X90 #2: frame(vzsum) + vz0
                        fire(gate, gate["x90"])  # noqa: F821
                    else:
                        play(gate, gate["x"], t_ro - SEP - gate["x"].dur)  # noqa: F821
                play(ro, ro["meas"], t_ro)  # noqa: F821    the swept-amplitude measurement tone
                play(demod, demod["sq"], t_ro + ddly)  # noqa: F821  demod opens `ddly` after the drive
                wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821
                if herald == 1:
                    out[2 * i] = out[2 * i] + read_res()  # noqa: F821    kept-shot |1> count
                    out[2 * i + 1] = out[2 * i + 1] + 1  # noqa: F821     kept-shot denominator
                else:
                    out[i] += read_res()  # noqa: F821    the hardware discriminator's bit
            t_ro = t_ro + period  # noqa: F821
        aq = aq + daq


@kernel
def k_phase(gate: ParamTable, ro: ParamTable, demod: ParamTable, out: Array, npts: int, shots: int,
            period: int, code: int, ddly: int, seq: int, hpi: int, p0: int, dp: int, herald: int,
            hoff: int):
    """Batched X90 PHASE calibration (qcal's two sequences, spec 13 §6), COUNTS mode. A compile-time
    `seq` binding folds to one of qcal's circuits (single_qubit.py:862-963) — three back-to-back X90s
    with an Rz(±pi/2) frame rotation dropped in at a different place:

      Y180_X90:  Rz(+pi/2) X90 X90 Rz(-pi/2) X90   (the first two X90s make a Y180)
      X180_Y90:  X90 X90 Rz(+pi/2) X90            (the first two make an X180, the last a Y90)

    An `Rz` is a channel FRAME advance (`set_phase_offset`, the virtual-Z of spec 09 B1); `hpi` is
    pi/2 as a seated phase word. X180_Y90's trailing Rz(-pi/2) is qcal's frame bookkeeping AFTER the
    last pulse — unobservable in a z-basis measurement, and the frame is rebuilt every shot — so it is
    not emitted.

    THE FRAME CONTRACT. qcal's X90 is virtualz(vz0) . FAST_DRAG . virtualz(vz1): part of the frame
    correction lands before the pulse and part after, so the pulse's axis sits BETWEEN the two
    advances. Every X90 play here is exactly that — `set_phase_offset(f + vz0); play; f += vz0 + vz1`
    with `f` an on-core frame — and the swept knob IS the virtual-Z: qcal sweeps one phase and writes
    it to BOTH slots (single_qubit.py:1081), so vz0 = vz1 = phi here. phi is computed on-core from the
    host-seated pair (p0/dp, spec 12) and the pulse table's own axis `phase` (the FAST_DRAG's, which
    this calibration does NOT touch) rides underneath it in the slot.

    `herald` is a COMPILE-TIME binding (spec 13 §8): 0 → today's shot tail, byte-identical; 1 → a
    pre-sequence herald read at `t_ro − hoff` gates the shot on the qubit being in |0>, and `out`
    carries interleaved (count, kept) pairs (P = count/kept) — qcal heralds both Phase sequences."""
    init_pulse_params(demod.pulses)  # noqa: F821
    set_freq(demod, code)  # noqa: F821
    init_pulse_params(ro.pulses)  # noqa: F821
    set_freq(ro, ro.freq)  # noqa: F821
    init_pulse_params(gate.pulses)  # noqa: F821
    set_freq(gate, gate.freq)  # noqa: F821
    d = gate["x90"].dur  # noqa: F821
    t_ro = now() + period  # noqa: F821
    phi = p0
    for i in range(npts):
        for s in range(shots):
            if herald == 1:
                play(ro, ro["meas"], t_ro - hoff)  # noqa: F821         pre-sequence herald read
                play(demod, demod["sq"], t_ro - hoff + ddly)  # noqa: F821
                wait_until(t_ro - hoff + ddly + READOUT_LEAD)  # noqa: F821
                h = read_res()  # noqa: F821
            else:
                h = 0
            if h == 0:                 # herald passed (qubit in |0>) — or heralding off (h folds to 0)
                ta = t_ro - SEP - 3 * d       # the contiguous 3-X90 train starts here
                if seq == Y180_X90:
                    f = hpi                   # Rz(+pi/2)
                else:
                    f = 0
                set_phase_offset(gate, f + phi)  # noqa: F821  X90 #1: the frame + vz0
                play(gate, gate["x90"], ta)  # noqa: F821
                f = f + 2 * phi                  # frame += vz0 + vz1
                set_phase_offset(gate, f + phi)  # noqa: F821  X90 #2
                play(gate, gate["x90"], ta + d)  # noqa: F821
                f = f + 2 * phi
                if seq == Y180_X90:
                    f = f - hpi               # Rz(-pi/2)
                else:
                    f = f + hpi               # Rz(+pi/2)
                set_phase_offset(gate, f + phi)  # noqa: F821  X90 #3
                play(gate, gate["x90"], ta + 2 * d)  # noqa: F821
                play(ro, ro["meas"], t_ro)  # noqa: F821
                play(demod, demod["sq"], t_ro + ddly)  # noqa: F821  demod opens `ddly` after the drive
                wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821
                if herald == 1:
                    out[2 * i] = out[2 * i] + read_res()  # noqa: F821    kept-shot |1> count
                    out[2 * i + 1] = out[2 * i + 1] + 1  # noqa: F821     kept-shot denominator
                else:
                    out[i] += read_res()  # noqa: F821
            t_ro = t_ro + period  # noqa: F821
        phi = phi + dp
