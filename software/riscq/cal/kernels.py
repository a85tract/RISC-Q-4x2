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
table phase survives the run.

DEEP GATE TRAINS (spec 14 F1). An n-gate train walks a FIXED `step`-batch grid (host-computed,
base.train_step) and is PACED: gate g is pushed only after gate g − TRAIN_AHEAD has started, so the
depth-4 param queues never overflow (an unpaced train silently drops everything past the 4th gate).
Each gate is placed explicitly at its own `t` rather than riding B0's startTime auto-advance, because
`step` exceeds the pulse length whenever the pulse is too short for the core to keep up (base's
TRAIN_STEP); when it does not, `step` IS the length and the train is back-to-back as before.

THE X90 FRAME BRACKET (spec 13 §7). qcal's X90 is virtualz(vz0) . FAST_DRAG . virtualz(vz1): part of
its frame correction lands before the pulse and part after, so the pulse's axis sits BETWEEN the two
advances. EVERY X90 play in EVERY kernel is exactly that — `set_phase_offset(f + vz0); play/fire;
f += vz0 + vz1` with `f` an on-core frame rebuilt each shot — so the pair Phase calibrates
(`qubit/{q}/x90/vz`) is actually played by the other cals too, not only by k_phase. The kernels bind
the pair as two compile-time seated phase words: `vz0` and `vzsum` = vz0 + vz1 (base.x90_vz; the
seated domain of spec 12, wrapping mod 2^32 = mod one turn). A config with no pair (every co-sim one)
binds 0/0 and the bracket is a no-op. The config's X pulse carries no pair, so its branches play bare.
"""

from riscq.cal.base import SEP, TRAIN_AHEAD, X90
from riscq.lang import Array, ParamTable, kernel
from riscq.map import LEAD, READOUT_LEAD

COUNTS = 0
RAW = 1
IQSUM = 2

Y180_X90 = 0    # k_phase's compile-time `seq` fold: qcal's two Phase sequences (spec 13 §6)
X180_Y90 = 1
X90_X_X90 = 2   # ... and its gate='X' circuit, which sweeps the X's own axis (spec 14 F1)

CONTROL = 0     # k_jazz / k_cz_* compile-time `role` fold: which core of the pair (spec two-qubit/01)
TARGET = 1
COUPLER = 2     # the coupler core plays the CZ drive; its readout is never fired (spec 01 §1)

FREQ = 0        # k_cz_pop's compile-time `knob` fold: which coupler-pulse field the sweep advances
DUR = 1
AMP = 2

ACTIVE = 0      # k_cz_local's compile-time `role` fold: the qubit that Ramseys (its own local phase)
SPECTATOR = 1   # k_cz_local: the partner, prepped |0>/|1> (COUPLER = 2, as elsewhere)

COUPLER_FORM = 0  # k_cz_* compile-time `form` fold (spec 04 §4.1): a dedicated coupler core plays
DRIVE_FORM = 1    # the CZ tone / BOTH qubit cores retune f_GE -> f_CZ and play their own line


@kernel
def k_rabi(gate: ParamTable, ro: ParamTable, demod: ParamTable, out: Array, npts: int, shots: int,
           period: int, ngates: int, step: int, code: int, mode: int, ddly: int, prep_gate: int,
           vz0: int, vzsum: int, a0q: int, daq: int, prep: int, herald: int, hoff: int):
    """Batched Rabi: sweep the CALIBRATED GATE's amplitude on-core (Q16 pair a0q/daq; realized code =
    aq >> 16, written raw — spec 12), `shots` shots/point on a fixed grid. `prep` gates the drive; the
    n-gate train walks the paced `step` grid of the module docstring (spec 14 F1). COUNTS → out[i] +=
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
                    # the LAST gate ends a clean SEP before the readout; `step` only spaces the ones
                    # before it, so a single-gate sweep is placed exactly as it was pre-pacing
                    t = t_ro - SEP - (ngates - 1) * step - d
                    tp = t - TRAIN_AHEAD * step        # the pace mark the 2nd gate is pushed against
                    if prep_gate == X90:
                        f = vz0                  # the frame is rebuilt every shot: frame(0) + vz0
                        set_phase_offset(gate, f)  # noqa: F821
                        play(gate, gate["x90"], t)  # noqa: F821
                        for g in range(ngates - 1):
                            tp = tp + step
                            wait_until(tp)  # noqa: F821  gate g−TRAIN_AHEAD has started (and popped)
                            t = t + step
                            f = f + vzsum        # frame += vz0 + vz1, then the next play's + vz0
                            set_phase_offset(gate, f)  # noqa: F821
                            play(gate, gate["x90"], t)  # noqa: F821
                    else:
                        play(gate, gate["x"], t)  # noqa: F821  the config's X, bare
                        for g in range(ngates - 1):
                            tp = tp + step
                            wait_until(tp)  # noqa: F821
                            t = t + step
                            play(gate, gate["x"], t)  # noqa: F821
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
def k_rpe_echo(gate: ParamTable, ro: ParamTable, demod: ParamTable, out: Array, shots: int,
               period: int, depth: int, tail: int, step: int, code: int, ddly: int, hpi: int,
               vz0: int, vzsum: int, herald: int, hoff: int):
    """RPE's INTERLEAVED X90 echo — qcal's gate='X90' axis circuits (spec 14 F5), COUNTS mode. One
    point, `shots` shots on the fixed `period` grid; `out[0]` is the |1> count.

    The circuit repeats qcal's echo block `depth` times and then plays `tail` more X90s:

        block  =  Z90 · X90 · X90 · Z90 · Z90 · X90 · X90 · Z90

    Every Z90 is a VIRTUAL Z — a frame advance of π/2 (`hpi`, a seated phase word), no pulse — so a
    block costs four X90 plays. The two X90 pairs are played a full π apart in the frame, which
    echoes the rotation ANGLE away (an amplitude error cancels) while the tilt of the rotation axis
    out of the drive plane ACCUMULATES. That is what makes the sequence read the X90's drive phase
    relative to its virtual-Z frame rather than its amplitude — the knob `qubit/{q}/x90/vz` sets.
    The block's frame advances sum to a full turn, so it leaves the frame where it found it.

    It is emitted as 2·depth halves of `Z90 · X90 · X90 · Z90`: the block IS two of those, and the
    Z90 · Z90 in its middle is just where two halves meet.

    `tail` reads the two BALANCED quadratures. Two trailing X90s are an X180, which maps the Bloch
    vector (x, y, z) → (x, −y, −z) and so reads exactly the opposite z-basis outcome: tail 0/2 are
    the cos pair, tail 1/3 the sin pair. Reading both closes is what keeps a fringe not centred on
    ½ (T1 decay before the readout) out of the angle — the same four-close trick RPEAmplitude plays
    on the direct train.

    `depth`/`tail`/`step` are compile-time bindings but the block loop is a RUNTIME `for`, so depth
    costs wall time, not instruction memory. The 4·depth + tail plays walk the paced `step` grid of
    the module docstring (spec 14 F1) — every play, including the first, is pushed only once the
    play TRAIN_AHEAD before it has started. `herald` folds exactly as in k_rabi."""
    init_pulse_params(demod.pulses)  # noqa: F821
    set_freq(demod, code)  # noqa: F821
    init_pulse_params(ro.pulses)  # noqa: F821
    set_freq(ro, ro.freq)  # noqa: F821
    init_pulse_params(gate.pulses)  # noqa: F821
    set_freq(gate, gate.freq)  # noqa: F821
    d = gate["x90"].dur  # noqa: F821
    n = 4 * depth + tail                       # every X90 the shot plays (compile-time)
    t_ro = now() + period  # noqa: F821  first grid slot (idle head resets to |0>)
    for s in range(shots):
        if herald == 1:
            play(ro, ro["meas"], t_ro - hoff)  # noqa: F821         pre-sequence herald read
            play(demod, demod["sq"], t_ro - hoff + ddly)  # noqa: F821
            wait_until(t_ro - hoff + ddly + READOUT_LEAD)  # noqa: F821
            h = read_res()  # noqa: F821
        else:
            h = 0
        if h == 0:                 # herald passed (qubit in |0>) — or heralding off (h folds to 0)
            # the LAST play ends a clean SEP before the readout; t/tp start one step SHORT of the
            # first play so every play runs the same paced body (the first wait is already past)
            t = t_ro - SEP - n * step - d
            tp = t - TRAIN_AHEAD * step        # the pace mark each play is pushed against
            f = 0                              # the frame, rebuilt every shot
            for b in range(2 * depth):         # a half-block: Z90 · X90 · X90 · Z90
                f = f + hpi                    # Z90
                tp = tp + step
                wait_until(tp)  # noqa: F821   the play TRAIN_AHEAD back has started (and popped)
                t = t + step
                set_phase_offset(gate, f + vz0)  # noqa: F821
                play(gate, gate["x90"], t)  # noqa: F821
                f = f + vzsum                  # frame += vz0 + vz1
                tp = tp + step
                wait_until(tp)  # noqa: F821
                t = t + step
                set_phase_offset(gate, f + vz0)  # noqa: F821
                play(gate, gate["x90"], t)  # noqa: F821
                f = f + vzsum + hpi            # frame += the pair, then the closing Z90
            for g in range(tail):              # the balanced close: 0/2 cos, 1/3 sin
                tp = tp + step
                wait_until(tp)  # noqa: F821
                t = t + step
                set_phase_offset(gate, f + vz0)  # noqa: F821
                play(gate, gate["x90"], t)  # noqa: F821
                f = f + vzsum
            play(ro, ro["meas"], t_ro)  # noqa: F821     readout drive (window trigger; hardware)
            play(demod, demod["sq"], t_ro + ddly)  # noqa: F821  demod LAST; firing it IS the readout
            wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821
            if herald == 1:
                out[0] = out[0] + read_res()  # noqa: F821    kept-shot |1> count
                out[1] = out[1] + 1  # noqa: F821             kept-shot denominator
            else:
                out[0] += read_res()  # noqa: F821  self-normalised population count
        t_ro = t_ro + period  # noqa: F821  next grid slot


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
            period: int, code: int, ddly: int, seq: int, hpi: int, vz0: int, vzsum: int, p0: int,
            dp: int, herald: int, hoff: int):
    """Batched gate PHASE calibration (qcal's Phase circuits, spec 13 §6 / spec 14 F1), COUNTS mode. A
    compile-time `seq` binding folds to one of qcal's circuits (single_qubit.py:862-963):

      Y180_X90:  Rz(+pi/2) X90 X90 Rz(-pi/2) X90   (the first two X90s make a Y180)
      X180_Y90:  X90 X90 Rz(+pi/2) X90            (the first two make an X180, the last a Y90)
      X90_X_X90: X90 X X90                        (qcal's gate='X' circuit)

    An `Rz` is a channel FRAME advance (`set_phase_offset`, the virtual-Z of spec 09 B1); `hpi` is
    pi/2 as a seated phase word. X180_Y90's trailing Rz(-pi/2) is qcal's frame bookkeeping AFTER the
    last pulse — unobservable in a z-basis measurement, and the frame is rebuilt every shot — so it is
    not emitted.

    X90_X_X90 sweeps a DIFFERENT knob from the two X90 circuits: the two X90s carry the config's own
    calibrated frame bracket (vz0/vzsum), and the swept phi tilts only the X's AXIS — the X pulse's
    own `phase`, which is what qcal's `Phase(gate='X')` writes (its param is the X pulse entry's
    kwargs/phase). X90 · X(axis phi) · X90 is a 2pi rotation that returns to |0> only when the X sits
    on the X90s' axis, so P(1) is cosinusoidal in phi with its MINIMUM at the calibrated value.

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
                if seq == X90_X_X90:      # gate='X': X90 · X(axis phi) · X90, the X90s' own bracket
                    xd = gate["x"].dur  # noqa: F821
                    ta = t_ro - SEP - 2 * d - xd
                    set_phase_offset(gate, vz0)  # noqa: F821       X90 #1: frame(0) + vz0
                    play(gate, gate["x90"], ta)  # noqa: F821
                    set_phase_offset(gate, vzsum + phi)  # noqa: F821  the X, tilted by the swept axis
                    play(gate, gate["x"], ta + d)  # noqa: F821
                    set_phase_offset(gate, vzsum + vz0)  # noqa: F821  X90 #2, back on the X90 axis
                    play(gate, gate["x90"], ta + d + xd)  # noqa: F821
                else:
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


# ── EF-subspace kernels (spec two-qubit/01 §4.1) ────────────────────────────────────────────────
# The EF calibrations drive the {|1>, |2>} transition on the SAME gate channel as GE, ONE carrier at a
# time: a GE pi first populates |1>, then the carrier is retuned to f_ef for the EF drive. Both switch
# the carrier TWICE per shot — set_freq(ge) tagged at the GE-prep start, set_freq(ef) at the EF-drive
# start (each set_freq is queued against the buffer's CURRENT startTime, so the explicit set_start that
# precedes it pins the retune's effective time; SOC_TIPS §5). A full LEAD gap separates the GE-prep end
# from the EF drive so the ef phasor regen (leadFreqP = linkPipe+52 < LEAD) never overruns the tail of
# the GE pulse. Both are RAW-only: |1> vs |2> discrimination is host-side (the 3-level ClassifierN reads
# P(|2>) off the clusters), the hardware res sign cannot separate them. `ge_freq`/`ef_freq` are SEATED
# carrier words (units.freq_to_code); the GE prep is a two-X90 pi carrying its frame bracket (vz0/vzsum,
# the module docstring), the EF drive plays in a fresh 0 frame (no EF virtual-Z cal in this scope).


@kernel
def k_ef_rabi(gate: ParamTable, ro: ParamTable, demod: ParamTable, out: Array, npts: int, shots: int,
              period: int, ngates: int, step: int, code: int, ddly: int, ge_freq: int, ef_freq: int,
              vz0: int, vzsum: int, a0q: int, daq: int):
    """Batched EF Rabi: prep |1> (GE pi), retune to f_ef, then sweep the EF X90 amplitude on-core (Q16
    pair a0q/daq; realized code = aq >> 16 written raw, spec 12), `shots` shots/point on a fixed grid.
    RAW mode: per-shot IQ through a cursor (out sized 2·npts·shots) — the host reads P(|2>), which peaks
    at an EF pi (|1>->|2>). The n-gate train walks the paced `step` grid (module docstring, spec 14 F1),
    all in one 0 frame so the gates add coherently (qcal's repetition amplifies an EF-amplitude error)."""
    init_pulse_params(demod.pulses)  # noqa: F821
    set_freq(demod, code)  # noqa: F821
    init_pulse_params(ro.pulses)  # noqa: F821
    set_freq(ro, ro.freq)  # noqa: F821
    init_pulse_params(gate.pulses)  # noqa: F821
    ge = gate["x90"].dur  # noqa: F821
    ef = gate["ef"].dur  # noqa: F821
    t_ro = now() + period  # noqa: F821  first grid slot (idle head resets to |0>)
    aq = a0q
    k = 0
    for i in range(npts):
        set_amp(gate, gate["ef"], aq)  # noqa: F821  raw Q16 accumulator (integer code in data[31:16])
        for s in range(shots):
            t_ef = t_ro - SEP - (ngates - 1) * step - ef   # EF train start (its last gate flush at SEP)
            t_ge = t_ef - LEAD - 2 * ge           # GE prep start (LEAD gap before the EF drive)
            set_start(gate, t_ge)  # noqa: F821     pin the GE retune's effective time to the prep start
            set_freq(gate, ge_freq)  # noqa: F821
            set_phase_offset(gate, vz0)  # noqa: F821          GE X90 #1: frame(0) + vz0
            play(gate, gate["x90"], t_ge)  # noqa: F821
            set_phase_offset(gate, vz0 + vzsum)  # noqa: F821  GE X90 #2: frame(vzsum) + vz0
            fire(gate, gate["x90"])  # noqa: F821  B0: startTime auto-advances by dur
            set_start(gate, t_ef)  # noqa: F821     pin the EF retune to the EF-drive start
            set_freq(gate, ef_freq)  # noqa: F821
            set_phase_offset(gate, 0)  # noqa: F821  fresh 0 frame for the EF train
            fire(gate, gate["ef"])  # noqa: F821     EF X90 #1 at t_ef (startTime already set)
            t = t_ef
            tp = t_ef - TRAIN_AHEAD * step
            for g in range(ngates - 1):
                tp = tp + step
                wait_until(tp)  # noqa: F821  gate g−TRAIN_AHEAD has started (and popped)
                t = t + step
                play(gate, gate["ef"], t)  # noqa: F821  the paced EF train
            play(ro, ro["meas"], t_ro)  # noqa: F821     readout drive (window trigger)
            play(demod, demod["sq"], t_ro + ddly)  # noqa: F821  demod opens `ddly` after the drive
            wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821
            read_res()  # noqa: F821            HALT until settled, then latch fresh IQ
            out[k] = read_real()  # noqa: F821
            out[k + 1] = read_imag()  # noqa: F821
            k = k + 2
            t_ro = t_ro + period  # noqa: F821  next grid slot
        aq = aq + daq


@kernel
def k_ef_ramsey(gate: ParamTable, ro: ParamTable, demod: ParamTable, out: Array, npts: int,
                shots: int, period: int, code: int, ddly: int, ge_freq: int, ef_freq: int, vz0: int,
                vzsum: int, w0: int, dw: int, p0: int, dp: int):
    """Batched EF Ramsey (covers EF Frequency): prep |1> (GE pi), retune to f_ef, then EF X90 — wait w —
    virtual-Z(phi) — EF X90, `shots` shots/point on a fixed grid. Same wait/virtual-Z structure as
    k_ramsey: the wait (w0/dw, batches) is computed on-core; the detuning is the host-seated phase pair
    (p0/dp) accumulated in the seated domain (spec 12) — phi at wait w_i is 16·detune·w_i, so the fringe
    runs at |delta + applied| off the model's EF reference, exactly the GE V-fit. RAW mode: the host
    reads P(|2>) off the 3-level clusters. The two EF X90s play in a fresh 0 frame, the swept phi as the
    Rz between them (the EF X90's own frame bracket is out of scope here — no EF Phase cal)."""
    init_pulse_params(demod.pulses)  # noqa: F821
    set_freq(demod, code)  # noqa: F821
    init_pulse_params(ro.pulses)  # noqa: F821
    set_freq(ro, ro.freq)  # noqa: F821
    init_pulse_params(gate.pulses)  # noqa: F821
    ge = gate["x90"].dur  # noqa: F821
    ef = gate["ef"].dur  # noqa: F821
    t_ro = now() + period  # noqa: F821
    w = w0
    phi = p0
    k = 0
    for i in range(npts):
        for s in range(shots):
            t_ef2 = t_ro - SEP - ef               # 2nd EF X90 start
            t_ef1 = t_ef2 - w - ef                # 1st EF X90 start (`wait` w earlier)
            t_ge = t_ef1 - LEAD - 2 * ge          # GE prep start (LEAD gap before the EF drive)
            set_start(gate, t_ge)  # noqa: F821     pin the GE retune to the prep start
            set_freq(gate, ge_freq)  # noqa: F821
            set_phase_offset(gate, vz0)  # noqa: F821          GE X90 #1: frame(0) + vz0
            play(gate, gate["x90"], t_ge)  # noqa: F821
            set_phase_offset(gate, vz0 + vzsum)  # noqa: F821  GE X90 #2: frame(vzsum) + vz0
            fire(gate, gate["x90"])  # noqa: F821
            set_start(gate, t_ef1)  # noqa: F821     pin the EF retune to the EF-drive start
            set_freq(gate, ef_freq)  # noqa: F821
            set_phase_offset(gate, 0)  # noqa: F821  1st EF X90: fresh 0 frame
            play(gate, gate["ef"], t_ef1)  # noqa: F821
            set_phase_offset(gate, phi)  # noqa: F821  2nd EF X90: the swept virtual-Z detuning phi
            play(gate, gate["ef"], t_ef2)  # noqa: F821
            play(ro, ro["meas"], t_ro)  # noqa: F821
            play(demod, demod["sq"], t_ro + ddly)  # noqa: F821  demod opens `ddly` after the drive
            wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821
            read_res()  # noqa: F821
            out[k] = read_real()  # noqa: F821
            out[k + 1] = read_imag()  # noqa: F821
            k = k + 2
            t_ro = t_ro + period  # noqa: F821
        w = w + dw
        phi = phi + dp


@kernel
def k_ef_phase(gate: ParamTable, ro: ParamTable, demod: ParamTable, out: Array, npts: int,
               shots: int, period: int, code: int, ddly: int, ge_freq: int, ef_freq: int, vz0: int,
               vzsum: int, seq: int, hpi: int, p0: int, dp: int):
    """Batched EF gate PHASE calibration — qcal's `Phase(subspace='EF')` (spec 04 §2 / X4, spec 14
    §3.3): prep |1> (GE π, the k_ef_* retune mechanism), retune to f_ef, then one of qcal's circuits
    ON THE EF GATE (its EF variants only prepend that GE π) — the compile-time `seq` binding folds to

      Y180_X90 / X180_Y90:  the three-EF-X90 sequences of the GE k_phase, `hpi` the seated π/2 Rz,
                            and the swept phi IS the EF X90's virtual-Z pair (vz0 = vz1 = phi —
                            exactly the GE k_phase contract: the sweep REPLACES the stored pair,
                            which the class's relative_phase knob centres the sweep on instead);
      X90_X_X90:            qcal's gate='X' circuit, EF-X90 · EF-X · EF-X90 (slot "efx"), where the
                            swept phi tilts only the EF X's own AXIS. The two EF X90s play in a
                            fresh 0 frame, so they sit on their own stored axis and phi is measured
                            RELATIVE to it; the composite is a 2π rotation inside {|1>, |2>} that
                            returns to |1> only on alignment, so P(|2>) is MINIMAL there.

    The EF segment starts from a fresh 0 frame; the GE prep carries the GE bracket (vz0/vzsum). RAW
    mode: per-shot IQ through a cursor (out sized 2·npts·shots) — the host reads P(|2>) off the
    3-level clusters. The retune's deterministic frame slip offsets EVERY EF pulse's axis equally,
    which conjugates out of a z-basis population (Rz(δ)·U·Rz(−δ) on an Rz eigenstate), so it moves
    neither the two sequences' crossing nor the X circuit's minimum. The shot is 5 gate pulses — one
    over the param-queue depth — so the last EF pulse is paced past the prep's pop (the CZ module
    comment's queue contract)."""
    init_pulse_params(demod.pulses)  # noqa: F821
    set_freq(demod, code)  # noqa: F821
    init_pulse_params(ro.pulses)  # noqa: F821
    set_freq(ro, ro.freq)  # noqa: F821
    init_pulse_params(gate.pulses)  # noqa: F821
    ge = gate["x90"].dur  # noqa: F821
    ef = gate["ef"].dur  # noqa: F821
    t_ro = now() + period  # noqa: F821
    phi = p0
    k = 0
    for i in range(npts):
        for s in range(shots):
            if seq == X90_X_X90:
                efx = gate["efx"].dur  # noqa: F821
                t_ef = t_ro - SEP - 2 * ef - efx  # the EF-X90 · EF-X · EF-X90 train start
            else:
                t_ef = t_ro - SEP - 3 * ef        # the contiguous 3-EF-X90 train start
            t_ge = t_ef - LEAD - 2 * ge           # GE prep start (LEAD gap before the EF train)
            set_start(gate, t_ge)  # noqa: F821     pin the GE retune to the prep start
            set_freq(gate, ge_freq)  # noqa: F821
            set_phase_offset(gate, vz0)  # noqa: F821          GE X90 #1: frame(0) + vz0
            play(gate, gate["x90"], t_ge)  # noqa: F821
            set_phase_offset(gate, vz0 + vzsum)  # noqa: F821  GE X90 #2: frame(vzsum) + vz0
            fire(gate, gate["x90"])  # noqa: F821
            set_start(gate, t_ef)  # noqa: F821     pin the EF retune to the train start
            set_freq(gate, ef_freq)  # noqa: F821
            if seq == X90_X_X90:                  # gate='X': the swept phi is the EF X's OWN axis
                set_phase_offset(gate, 0)  # noqa: F821  fresh 0 frame: EF X90 #1 on its stored axis
                fire(gate, gate["ef"])  # noqa: F821     at t_ef (startTime already set)
                set_phase_offset(gate, phi)  # noqa: F821  the EF X, tilted by the swept axis
                fire(gate, gate["efx"])  # noqa: F821    (B0 startTime auto-advance)
                wait_until(t_ge + 2 * ge)  # noqa: F821  pace the depth-4 queues (below)
                set_phase_offset(gate, 0)  # noqa: F821  EF X90 #2, back on the X90s' own axis
                fire(gate, gate["ef"])  # noqa: F821
            else:
                if seq == Y180_X90:
                    f = hpi                       # Rz(+π/2)
                else:
                    f = 0
                set_phase_offset(gate, f + phi)  # noqa: F821  EF X90 #1: the frame + vz0(= phi)
                fire(gate, gate["ef"])  # noqa: F821     at t_ef (startTime already set)
                f = f + 2 * phi                   # frame += vz0 + vz1 (= 2·phi)
                set_phase_offset(gate, f + phi)  # noqa: F821  EF X90 #2 (B0 startTime auto-advance)
                fire(gate, gate["ef"])  # noqa: F821
                f = f + 2 * phi
                if seq == Y180_X90:
                    f = f - hpi                   # Rz(−π/2)
                else:
                    f = f + hpi                   # Rz(+π/2)
                wait_until(t_ge + 2 * ge)  # noqa: F821  pace the depth-4 queues (the CZ module
                #                             comment): the prep has popped, so the 5th pulse fits
                set_phase_offset(gate, f + phi)  # noqa: F821  EF X90 #3
                fire(gate, gate["ef"])  # noqa: F821
            play(ro, ro["meas"], t_ro)  # noqa: F821
            play(demod, demod["sq"], t_ro + ddly)  # noqa: F821  demod opens `ddly` after the drive
            wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821
            read_res()  # noqa: F821
            out[k] = read_real()  # noqa: F821
            out[k + 1] = read_imag()  # noqa: F821
            k = k + 2
            t_ro = t_ro + period  # noqa: F821
        phi = phi + dp


# ── JAZZ: residual-ZZ Ramsey (spec two-qubit/01 §4.3) ───────────────────────────────────────────
# The first two-qubit measurement: X90s + idles only, no coupler/EF drive. Each core of the pair runs
# ONE role of the SAME kernel on the SAME fixed grid (the shared batch clock is the only sync, as in
# every multi-core cal). The TARGET does a Hahn-echo Ramsey — X90 · idle w · [pi on both] · idle w ·
# Rz(phi) · close X90/Y90 — while the CONTROL preps |0>/|1> and echoes in lock-step. A pi on BOTH
# qubits at the midpoint (BIRD) refocuses each qubit's own detuning but preserves the control-
# conditional ZZ, so the target fringe frequency shifts by the ZZ between the control states; the host
# fits f(control=1) − f(control=0) = ZZ (zz.py:43-288). The four sequences per delay (C0/C1 × X90/Y90
# close) are RUNTIME reruns of one resident image: `prep` (control |0>/|1>) and `quad` (X90/Y90 close)
# are scalars, so nothing recompiles between them.


@kernel
def k_jazz(gate: ParamTable, ro: ParamTable, demod: ParamTable, out: Array, npts: int, shots: int,
           period: int, code: int, ddly: int, role: int, vz0: int, vzsum: int, hpi: int, w0: int,
           dw: int, p0: int, dp: int, prep: int, quad: int):
    """Batched JAZZ (residual-ZZ Ramsey). `role` (COMPILE-TIME) folds to one core's sequence; the
    half-wait w (batches, w0/dw) is the on-core swept knob (full echo delay t = 2w), and the virtual
    detuning phi = 2*pi*detuning*t is the host-seated phase pair (p0/dp = 32*dc*w step, dc a DAC-rate
    detuning code) accumulated in the seated domain (spec 12) — the same Rz-as-phaseOffset mechanism as
    k_ramsey. COUNTS mode: the host reads P(|1>) per point and fits the fringe.

    TARGET: X90 prep · idle w · echo (X90 X90 = pi) · idle w · Rz(phi) · close, the close an X90
    (`quad` == 0, in-phase I) or a Y90 (`quad` == 1: X90 + hpi = pi/2, out-of-phase Q). CONTROL: prep
    |1> (X90 X90, gated by the runtime `prep`) or |0> (nothing) · idle w · echo (X90 X90) — no close,
    it only conditions the target's ZZ. Both echo at the SAME absolute t_echo (the midpoint pi is
    simultaneous) and read out on the shared grid at t_ro. Every X90 carries the frame bracket
    (vz0/vzsum; a co-sim config with no pair binds 0/0 and it is a no-op)."""
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
    for i in range(npts):
        for s in range(shots):
            t_echo = t_ro - SEP - 3 * d - w      # echo start; the target close ends at t_ro − SEP
            f = 0                                # the on-core X90 frame, rebuilt each shot
            if role == TARGET:
                set_phase_offset(gate, f + vz0)  # noqa: F821          prep X90: frame(0) + vz0
                play(gate, gate["x90"], t_echo - w - d)  # noqa: F821  (idle w before the echo)
                f = f + vzsum
                set_phase_offset(gate, f + vz0)  # noqa: F821          echo X90 #1
                play(gate, gate["x90"], t_echo)  # noqa: F821
                f = f + vzsum
                set_phase_offset(gate, f + vz0)  # noqa: F821          echo X90 #2 (B0 auto-advance)
                fire(gate, gate["x90"])  # noqa: F821
                f = f + vzsum
                if quad == 1:                    # Y90 close: X90 + pi/2 (the out-of-phase quadrature)
                    set_phase_offset(gate, f + vz0 + phi + hpi)  # noqa: F821
                else:                            # X90 close: the in-phase quadrature
                    set_phase_offset(gate, f + vz0 + phi)  # noqa: F821
                play(gate, gate["x90"], t_echo + 2 * d + w)  # noqa: F821  (idle w + Rz(phi) then close)
            else:                                # CONTROL: prep |1>/|0>, echo with the target — no close
                if prep == 1:
                    set_phase_offset(gate, f + vz0)  # noqa: F821       |1> prep: X90 X90 (pi)
                    play(gate, gate["x90"], t_echo - w - 2 * d)  # noqa: F821
                    f = f + vzsum
                    set_phase_offset(gate, f + vz0)  # noqa: F821
                    fire(gate, gate["x90"])  # noqa: F821
                    f = f + vzsum
                set_phase_offset(gate, f + vz0)  # noqa: F821           echo X90 #1
                play(gate, gate["x90"], t_echo)  # noqa: F821
                f = f + vzsum
                set_phase_offset(gate, f + vz0)  # noqa: F821           echo X90 #2
                fire(gate, gate["x90"])  # noqa: F821
            play(ro, ro["meas"], t_ro)  # noqa: F821     readout drive (window trigger; both cores)
            play(demod, demod["sq"], t_ro + ddly)  # noqa: F821  demod LAST; firing it IS the readout
            wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821
            out[i] += read_res()  # noqa: F821  self-normalised |1> count (the target's fringe)
            t_ro = t_ro + period  # noqa: F821
        w = w + dw
        phi = phi + dp


# ── CZ calibration (spec two-qubit/01 §4.4-4.5; two-qubit-drive form per spec 04 §4.1) ──────────
# Two gate forms share the three kernels through a compile-time `form` fold:
#
# COUPLER_FORM — the CZ is PARAMETRIC: a coupler-flux tone at f_CZ = |f_11 − f_02| Rabi-flops
# |11>↔|02>, and a full 2π round trip leaves |11> with the conditional π phase that IS the gate (the
# model's coupling, spec 01 §6). THREE compile-time roles on the shared grid: the two qubit cores
# (CONTROL/TARGET) prep + read out through their own demod/ADC lanes (frequency-multiplexed on the
# shared readout DAC), and the COUPLER core plays the CZ drive and NEVER fires its readout (spec 01
# §1) — it only walks the grid in lock-step. The COUPLER carries the swept knob; the qubit cores bind
# a dead 0 sweep. All timing math uses `xd` (the X90 dur, passed in) so the coupler role — whose gate
# table has only a "cz" slot, no "x90" — computes the same grid without referencing a missing slot.
#
# DRIVE_FORM — the fixed-frequency two-qubit-drive CZ (X6Y3, spec 04 §1): NO coupler core; both qubit
# cores play the CZ tone on their OWN gate channel at the common in-band `CZ/freq`, simultaneously
# (control line phase 0, target line the calibrated RELATIVE phase — baked in each core's "cz" slot).
# The tone shares the qubit's single gate NCO, so each core retunes f_GE -> f_CZ before the CZ
# segment and back after — the k_ef_* mechanism: `set_start(t)` pins each retune's effective time, a
# full LEAD gap separates a retuned play from the segment before it (phasor regen, leadFreqP <
# LEAD), and the segment timings are FIXED on the grid so the retune's deterministic frame slip
# (Δf · segment) is constant — it lands in the LocalPhases virtual-Z by design (spec 04 §1). The
# swept knob binds on BOTH cores in LOCKSTEP (same x0/dx pair): a FREQ sweep is the CZ-segment
# retune word itself, DUR/AMP are `set_dur`/`set_amp` on both cores' cz slot; `fcz` is the config
# f_CZ (seated word) the non-FREQ sweeps retune to.
#
# EF SANDWICH (spec 04 §1 / X4, the (5,6)/(6,7) X6Y3 pairs): the pair's CZ acts in the SHELVED
# manifold — string-reference pre/post-pulses play one member's EF X around the tones, |1>→|2>
# before and |2>→|1> after. Three more compile-time bindings on the drive-form kernels: `sw`=1 on
# the SHELF core folds in the two EF X segments (retune f_GE → f_EF → play "ef" → retune f_CZ →
# tone → retune f_EF → play "ef" → retune f_GE, every retuned play a full LEAD after the previous
# segment); `fef` is the shelf's seated EF carrier; `tail` = LEAD + the EF X length is the extra
# segment between the tone(s) and the close/readout region, bound on BOTH cores (0 on plain pairs)
# so the two lines still fire at the SAME absolute slot — the partner core just idles through the
# shelf's EF segments. All segments stay fixed on the grid, so the extra retunes' frame slips are
# constant and land in the LocalPhases virtual-Z like the plain form's (spec 04 §1).
#
# QUEUE-DEPTH PACING (X4 finding): every PulseGenerator parameter sits behind a depth-4 TimedQueue
# on the POSTED RF link — a push while the FIFO is full is silently DROPPED (no backpressure
# reaches the core). The deployed RegHead impl adds one head-stage slot (5 in flight happens to
# fit), but the Shadow/Srl* congestion variants have no head stage, so the portable contract is
# ≤ 4 scheduled-ahead entries per param queue. The shelf core's shot is 5 pulses (2 prep X90 +
# EF X + cz + EF X), so each sandwich fold issues its un-shelving block only after a `wait_until`
# that drains the cz train — by then the earlier entries have popped (a pulse pops `lead` BEFORE
# its start) and the queues never hold more than 4. The `ngates` CZ trains are paced the same way
# (base.TRAIN_AHEAD, spec 14 F1), which retires the old ngates ≤ 4−prep cap (spec 04 §5).


@kernel
def k_cz_pop(gate: ParamTable, ro: ParamTable, demod: ParamTable, out: Array, npts: int, shots: int,
             period: int, code: int, ddly: int, role: int, knob: int, form: int, vz0: int,
             vzsum: int, xd: int, czmax: int, fcz: int, fef: int, sw: int, tail: int, x0: int,
             dx: int):
    """Batched CZ resonance / return sweep (spec 01 §4.4). Prep |11> on both qubits (X90 X90 = π each),
    play the swept CZ pulse, read out — so the CONTROL's |1> population DIPS as |11>→|02> (control
    1→0) and RETURNS as the round trip closes. A compile-time `knob` folds the CZ-pulse field the
    on-core pair (x0/dx) advances per point:

      FREQ — `set_freq(gate, xk)` the carrier (Q16 raw, realized code xk>>16): P(11) dips at f_CZ;
      DUR  — `set_dur(gate, cz, xk)` the slot's dur field (plain batches): truncates the fixed
             max-length (`czmax`) envelope's flat top → max P(11) return at a full round trip;
      AMP  — `set_amp(gate, cz, xk)` (Q16 raw): max P(11) return at a full 2π.

    COUPLER_FORM: the COUPLER role carries the sweep, the qubit roles bind a dead 0 pair. The coupler
    pulse is placed at the FIXED `t_ro − SEP − czmax` (the max sweep length), so the qubit prep timing
    is constant across a DUR sweep — a shorter dur just leaves an idle tail before readout.

    DRIVE_FORM (spec 04 §4.1): NO coupler role — this qubit core preps its own |1> (the |11> half),
    retunes f_GE → f_CZ (`set_start` + `set_freq`, a LEAD gap after the prep for the phasor regen),
    fires its OWN cz line at the fixed `t_ro − SEP − tail − czmax`, and reads out; the FREQ sweep IS
    the CZ-segment retune word, DUR/AMP hit this core's cz slot — the same pair bound LOCKSTEP on
    both cores. The cz plays in a fresh 0 frame: the slot phase carries the line's (relative) phase.
    `sw`=1 (the EF-sandwich shelf core, the module comment) folds the shelving EF X in after the
    prep and the un-shelving one at `t_ro − SEP − ef` (`tail` = LEAD + ef on both cores; 0 plain).

    COUNTS mode; the host reads the CONTROL core's P(1) (a clean 2-level dip — |02> has control |0>;
    the target's |2> is not 2-level separable)."""
    init_pulse_params(demod.pulses)  # noqa: F821
    set_freq(demod, code)  # noqa: F821
    init_pulse_params(ro.pulses)  # noqa: F821
    set_freq(ro, ro.freq)  # noqa: F821
    init_pulse_params(gate.pulses)  # noqa: F821
    set_freq(gate, gate.freq)  # noqa: F821       coupler: the f_CZ seed carrier; qubit: the GE carrier
    t_ro = now() + period  # noqa: F821
    xk = x0
    for i in range(npts):
        if form == DRIVE_FORM:                    # the slot knobs advance on BOTH cores, LOCKSTEP
            if knob == DUR:
                set_dur(gate, gate["cz"], xk)  # noqa: F821  swept flat-top length (plain batches)
            elif knob == AMP:
                set_amp(gate, gate["cz"], xk)  # noqa: F821  swept amp (Q16 raw) — both lines together
        elif role == COUPLER:
            if knob == FREQ:
                set_freq(gate, xk)  # noqa: F821          swept carrier (Q16 raw, spec 12)
            elif knob == DUR:
                set_dur(gate, gate["cz"], xk)  # noqa: F821  swept flat-top length (plain batches)
            else:
                set_amp(gate, gate["cz"], xk)  # noqa: F821  swept amp (Q16 raw)
        for s in range(shots):
            if form == DRIVE_FORM:                # this qubit: prep |1>, retune, fire OWN cz line
                t_cz = t_ro - SEP - tail - czmax
                if sw == 1:                       # shelf core: prep · LEAD · EF X · LEAD · tone
                    t_ge = t_cz - 2 * LEAD - gate["ef"].dur - 2 * xd  # noqa: F821
                else:
                    t_ge = t_cz - LEAD - 2 * xd   # GE prep ends a full LEAD before the retuned play
                set_start(gate, t_ge)  # noqa: F821     pin the GE retune to the prep start
                set_freq(gate, gate.freq)  # noqa: F821
                set_phase_offset(gate, vz0)  # noqa: F821          |1> prep: X90 X90 (π)
                play(gate, gate["x90"], t_ge)  # noqa: F821
                set_phase_offset(gate, vz0 + vzsum)  # noqa: F821
                fire(gate, gate["x90"])  # noqa: F821  B0 startTime auto-advance → contiguous π
                if sw == 1:                       # retune f_GE → f_EF, shelve |1> → |2>
                    set_start(gate, t_cz - LEAD - gate["ef"].dur)  # noqa: F821
                    set_freq(gate, fef)  # noqa: F821
                    set_phase_offset(gate, 0)  # noqa: F821
                    fire(gate, gate["ef"])  # noqa: F821
                set_start(gate, t_cz)  # noqa: F821     pin the CZ retune to the tone start
                if knob == FREQ:
                    set_freq(gate, xk)  # noqa: F821    the swept carrier — LOCKSTEP on both cores
                else:
                    set_freq(gate, fcz)  # noqa: F821   the config f_CZ (seated word)
                set_phase_offset(gate, 0)  # noqa: F821  fresh frame: the slot phase is the line's phase
                fire(gate, gate["cz"])  # noqa: F821     this core's half of the two-tone CZ
                if sw == 1:                       # retune f_CZ → f_EF, un-shelve |2> → |1>
                    wait_until(t_ge + 2 * xd)  # noqa: F821  pace the depth-4 queues (module comment):
                    #                              the prep has popped, so this 5th pulse fits
                    set_start(gate, t_ro - SEP - gate["ef"].dur)  # noqa: F821
                    set_freq(gate, fef)  # noqa: F821
                    set_phase_offset(gate, 0)  # noqa: F821
                    fire(gate, gate["ef"])  # noqa: F821
                play(ro, ro["meas"], t_ro)  # noqa: F821
                play(demod, demod["sq"], t_ro + ddly)  # noqa: F821
                wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821
                out[i] += read_res()  # noqa: F821
            elif role == COUPLER:
                play(gate, gate["cz"], t_ro - SEP - czmax)  # noqa: F821  the CZ drive (no readout fired)
                wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821       stay on the shared grid
            else:
                set_phase_offset(gate, vz0)  # noqa: F821          |11> prep: X90 X90 (π) on this qubit
                play(gate, gate["x90"], t_ro - SEP - czmax - 2 * xd)  # noqa: F821
                set_phase_offset(gate, vz0 + vzsum)  # noqa: F821
                fire(gate, gate["x90"])  # noqa: F821  B0 startTime auto-advance → contiguous π
                play(ro, ro["meas"], t_ro)  # noqa: F821     readout drive (window trigger; both qubits)
                play(demod, demod["sq"], t_ro + ddly)  # noqa: F821  demod LAST; firing it IS the readout
                wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821
                out[i] += read_res()  # noqa: F821  self-normalised |1> count (control dips at resonance)
            t_ro = t_ro + period  # noqa: F821
        xk = xk + dx


@kernel
def k_cz_cond(gate: ParamTable, ro: ParamTable, demod: ParamTable, out: Array, npts: int, shots: int,
              period: int, code: int, ddly: int, role: int, knob: int, form: int, ngates: int,
              vz0: int, vzsum: int, hpi: int, zi: int, iz: int, xd: int, czd: int, fcz: int,
              fef: int, sw: int, tail: int, x0: int, dx: int, prep: int, quad: int):
    """Batched CZ CONDITIONALITY tomography (spec 01 §4.5): a target Ramsey `Y90 · cz^ngates · close`
    with the control in |0> (`prep`=0) or |1> (`prep`=1), the close a Y90 (`quad`=0, the X-quadrature)
    or an X90 (`quad`=1, the Y-quadrature). A conditional-π CZ rotates the target's Bloch vector by π
    ONLY when the control is |1>, so the host's R = √((ΔP0_X)² + (ΔP0_Y)²) between the two control
    states → 1 at a perfect CZ (spec 01 §4.5). `prep`/`quad` are runtime scalars → the FOUR sequences
    are reruns of one resident image (as k_jazz). `quad` 2/3 close at −Y90/−X90 — each the π-opposite
    of its 0/1 partner, so a (0, 2) or (3, 1) rerun pair reads BALANCED quadratures whose fringe
    centre divides out of the angle (spec 14 F5 finding 1; the CZRPE ladders). The TARGET/CONTROL
    roles are caller-assigned per core, so the same image also serves the RPE (3, 1) rung — the
    Ramsey on the physical control with the target prepped |1> (`_cz_cond_progs(ramsey=...)`).

    Each `cz` is the CZ drive plus the virtual-Z local-phase corrections (`zi` on the control frame,
    `iz` on the target frame — 0 until LocalPhases, spec 01 §4.6, calibrates them). The swept knob
    advances the CZ carrier (FREQ, cz.Frequency) or amp (AMP, cz.Amplitude); the four-sequence
    machinery also serves cz.RelativePhase, which binds a DEAD sweep and host-paces the target's cz
    slot phase between reruns (spec 04 §4.3).

    COUPLER_FORM: the COUPLER fires `ngates` back-to-back CZ pulses (error amplification for
    `ngates`>1) and carries the sweep; the qubit roles bind a dead 0 pair.

    DRIVE_FORM (spec 04 §4.1): no coupler role — each qubit core retunes f_GE → f_CZ ONCE before the
    train, fires `ngates` back-to-back cz slots (B0 startTime auto-advance), and retunes back once
    (the target for its close, after a LEAD gap for the GE phasor regen; the prep likewise ends a
    LEAD before the retuned train). The sweep pair binds LOCKSTEP on both cores; the cz train plays
    in a fresh 0 frame (the slot phase carries each line's phase). `sw`=1 (the EF-sandwich shelf
    core, the module comment) folds the shelving EF X in between prep and train and the un-shelving
    one between train and close (`tail` = LEAD + ef on both cores; 0 plain).

    Every X90/Y90 carries its frame bracket (vz0/vzsum); a Y90 is the X90 + `hpi` (π/2). Both qubit
    cores read out on the shared grid (the target's counts are the measurement; the control's are
    ignored)."""
    init_pulse_params(demod.pulses)  # noqa: F821
    set_freq(demod, code)  # noqa: F821
    init_pulse_params(ro.pulses)  # noqa: F821
    set_freq(ro, ro.freq)  # noqa: F821
    init_pulse_params(gate.pulses)  # noqa: F821
    set_freq(gate, gate.freq)  # noqa: F821
    t_ro = now() + period  # noqa: F821
    xk = x0
    for i in range(npts):
        if form == DRIVE_FORM:
            if knob == AMP:
                set_amp(gate, gate["cz"], xk)  # noqa: F821  both lines together, LOCKSTEP
        elif role == COUPLER:
            if knob == FREQ:
                set_freq(gate, xk)  # noqa: F821
            else:
                set_amp(gate, gate["cz"], xk)  # noqa: F821
        for s in range(shots):
            t_close = t_ro - SEP - xd             # the target's close X90/Y90 start
            if form == DRIVE_FORM:                # both cores: prep (GE) · cz^n (f_CZ) · [close (GE)]
                t_cz0 = t_close - LEAD - tail - ngates * czd   # LEAD: GE phasor regen before the close
                if sw == 1:
                    t_pre = t_cz0 - LEAD - gate["ef"].dur  # noqa: F821  the shelving EF X start
                else:
                    t_pre = t_cz0                 # plain: the prep ends LEAD before the train itself
                f = 0                             # the on-core X90 frame, rebuilt each shot
                if role == TARGET:
                    set_start(gate, t_pre - LEAD - xd)  # noqa: F821  pin the GE retune to the prep
                    set_freq(gate, gate.freq)  # noqa: F821
                    set_phase_offset(gate, f + vz0 + hpi)  # noqa: F821  Y90 (target prep)
                    play(gate, gate["x90"], t_pre - LEAD - xd)  # noqa: F821
                    f = f + vzsum
                else:                             # CONTROL: prep |0>/|1>
                    if prep == 1:
                        set_start(gate, t_pre - LEAD - 2 * xd)  # noqa: F821
                        set_freq(gate, gate.freq)  # noqa: F821
                        set_phase_offset(gate, f + vz0)  # noqa: F821  X (π) = X90 X90
                        play(gate, gate["x90"], t_pre - LEAD - 2 * xd)  # noqa: F821
                        f = f + vzsum
                        set_phase_offset(gate, f + vz0)  # noqa: F821
                        fire(gate, gate["x90"])  # noqa: F821
                        f = f + vzsum
                if sw == 1:                       # retune f_GE → f_EF, shelve |1> → |2>
                    set_start(gate, t_pre)  # noqa: F821
                    set_freq(gate, fef)  # noqa: F821
                    set_phase_offset(gate, 0)  # noqa: F821
                    fire(gate, gate["ef"])  # noqa: F821
                set_start(gate, t_cz0)  # noqa: F821    ONE retune to f_CZ before the train
                if knob == FREQ:
                    set_freq(gate, xk)  # noqa: F821    the swept carrier — LOCKSTEP on both cores
                else:
                    set_freq(gate, fcz)  # noqa: F821
                set_phase_offset(gate, 0)  # noqa: F821  fresh frame: the slot phase is the line's phase
                t = t_cz0
                tp = t_cz0 - TRAIN_AHEAD * czd
                fire(gate, gate["cz"])  # noqa: F821     cz #1 at t_cz0
                for g in range(ngates - 1):
                    tp = tp + czd
                    wait_until(tp)  # noqa: F821  pace the train (spec 14 F1): cz g−TRAIN_AHEAD has started
                    t = t + czd
                    play(gate, gate["cz"], t)  # noqa: F821  back-to-back (czd ≫ the pacing floor)
                wait_until(t)  # noqa: F821  the train has drained — the tail's pushes fit the depth-4
                #                queues (the shelf core's shot is 5 pulses; the module comment)
                if sw == 1:                       # retune f_CZ → f_EF, un-shelve |2> → |1>
                    set_start(gate, t_close - LEAD - gate["ef"].dur)  # noqa: F821
                    set_freq(gate, fef)  # noqa: F821
                    set_phase_offset(gate, 0)  # noqa: F821
                    fire(gate, gate["ef"])  # noqa: F821  ends LEAD before the retuned close
                if role == TARGET:
                    for g in range(ngates):
                        f = f + iz                # target frame += IZ per CZ (the local-phase correction)
                    set_start(gate, t_close)  # noqa: F821  ONE retune back to GE for the close
                    set_freq(gate, gate.freq)  # noqa: F821
                    if quad == 0:                 # X-seq: close Y90 (in-phase)
                        set_phase_offset(gate, f + vz0 + hpi)  # noqa: F821
                    elif quad == 1:               # Y-seq: close X90 (out-of-phase quadrature)
                        set_phase_offset(gate, f + vz0)  # noqa: F821
                    elif quad == 2:               # X-seq balanced partner: close −Y90 (spec 14 F5)
                        set_phase_offset(gate, f + vz0 - hpi)  # noqa: F821
                    else:                         # Y-seq balanced partner: close −X90
                        set_phase_offset(gate, f + vz0 + hpi + hpi)  # noqa: F821
                    play(gate, gate["x90"], t_close)  # noqa: F821
                else:
                    for g in range(ngates):
                        f = f + zi                # control frame += ZI per CZ (bookkeeping)
                play(ro, ro["meas"], t_ro)  # noqa: F821
                play(demod, demod["sq"], t_ro + ddly)  # noqa: F821
                wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821
                out[i] += read_res()  # noqa: F821  target: P(1) of the branch (control: ignored)
            elif role == COUPLER:
                t_cz0 = t_close - ngates * czd    # the first CZ pulse start
                t = t_cz0
                tp = t_cz0 - TRAIN_AHEAD * czd
                play(gate, gate["cz"], t)  # noqa: F821  the CZ train (contiguous, explicit start each)
                for g in range(ngates - 1):
                    tp = tp + czd
                    wait_until(tp)  # noqa: F821  pace the train (spec 14 F1)
                    t = t + czd
                    play(gate, gate["cz"], t)  # noqa: F821
                wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821  stay on the grid, no readout
            else:
                t_cz0 = t_close - ngates * czd    # the first CZ pulse start
                f = 0                             # the on-core X90 frame, rebuilt each shot
                if role == TARGET:
                    set_phase_offset(gate, f + vz0 + hpi)  # noqa: F821  Y90 #1 (target prep)
                    play(gate, gate["x90"], t_cz0 - xd)  # noqa: F821
                    f = f + vzsum
                    for g in range(ngates):
                        f = f + iz                # target frame += IZ per CZ (the local-phase correction)
                    if quad == 0:                 # X-seq: close Y90 (in-phase)
                        set_phase_offset(gate, f + vz0 + hpi)  # noqa: F821
                    elif quad == 1:               # Y-seq: close X90 (out-of-phase quadrature)
                        set_phase_offset(gate, f + vz0)  # noqa: F821
                    elif quad == 2:               # X-seq balanced partner: close −Y90 (spec 14 F5)
                        set_phase_offset(gate, f + vz0 - hpi)  # noqa: F821
                    else:                         # Y-seq balanced partner: close −X90
                        set_phase_offset(gate, f + vz0 + hpi + hpi)  # noqa: F821
                    play(gate, gate["x90"], t_close)  # noqa: F821
                else:                             # CONTROL: prep |0>/|1>, no close — only conditions
                    if prep == 1:
                        set_phase_offset(gate, f + vz0)  # noqa: F821  X (π) = X90 X90, ends at t_cz0
                        play(gate, gate["x90"], t_cz0 - 2 * xd)  # noqa: F821
                        f = f + vzsum
                        set_phase_offset(gate, f + vz0)  # noqa: F821
                        fire(gate, gate["x90"])  # noqa: F821
                        f = f + vzsum
                    for g in range(ngates):
                        f = f + zi                # control frame += ZI per CZ (bookkeeping)
                play(ro, ro["meas"], t_ro)  # noqa: F821
                play(demod, demod["sq"], t_ro + ddly)  # noqa: F821
                wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821
                out[i] += read_res()  # noqa: F821  target: P(1) of the tomography branch (control: ignored)
            t_ro = t_ro + period  # noqa: F821
        xk = xk + dx


@kernel
def k_cz_local(gate: ParamTable, ro: ParamTable, demod: ParamTable, out: Array, npts: int, shots: int,
               period: int, code: int, ddly: int, role: int, form: int, vz0: int, vzsum: int,
               hpi: int, xd: int, czd: int, fcz: int, fef: int, sw: int, tail: int, p0: int,
               dp: int, sp: int):
    """Batched CZ LOCAL-PHASE Ramsey (spec 01 §4.6). One qubit (`role`=ACTIVE) runs a Ramsey around ONE
    CZ — `Y90 · cz · Rz(φ) · Y90` — while the partner (SPECTATOR) sits in |0> or |1> (the runtime `sp`
    scalar). The swept virtual-Z φ (host-seated p0/dp pair) is inserted on the ACTIVE frame between the
    CZ and the closing Y90; the host cosine-fits the ACTIVE P(1) vs φ, and the φ at the extremum is the
    single-qubit Z the CZ leaves on the active qubit for that spectator state (spec 01 §4.6).
    LocalPhases runs it with the ACTIVE role on the control (→ ZI) and on the target (→ IZ), each with
    sp=0/1, and writes the mean of the two spectator branches.

    COUPLER_FORM: the COUPLER role fires the one CZ drive. DRIVE_FORM (spec 04 §4.1): no coupler —
    BOTH pair cores retune f_GE → `fcz` and fire their OWN cz line (the two-tone gate needs both), the
    ACTIVE around its Ramsey (a LEAD gap after the prep and before the retuned close, the k_ef_*
    phasor-regen contract; the fixed timings make the retune's frame slip constant — it lands in THIS
    calibration's vz by design), the SPECTATOR after its |0>/|1> prep. `sw`=1 (the EF-sandwich shelf
    core, the module comment) folds the shelving/un-shelving EF X around the tone on either role
    (`tail` = LEAD + ef on both cores; 0 plain) — the extra retunes' slips land in this vz too.

    Only the ACTIVE qubit reads out (its Ramsey IS the measurement, the window trigger); the SPECTATOR
    and COUPLER stay on the shared grid without firing their readout (spec 01 §1). Every X90/Y90 carries
    its frame bracket (vz0/vzsum); a Y90 is the X90 + `hpi` (π/2)."""
    init_pulse_params(demod.pulses)  # noqa: F821
    set_freq(demod, code)  # noqa: F821
    init_pulse_params(ro.pulses)  # noqa: F821
    set_freq(ro, ro.freq)  # noqa: F821
    init_pulse_params(gate.pulses)  # noqa: F821
    set_freq(gate, gate.freq)  # noqa: F821
    t_ro = now() + period  # noqa: F821
    phi = p0
    for i in range(npts):
        for s in range(shots):
            t_close = t_ro - SEP - xd             # the ACTIVE close Y90 start
            if form == DRIVE_FORM:                # the pair plays its own two-tone CZ; no coupler
                t_cz = t_close - LEAD - tail - czd   # LEAD: GE phasor regen before the retuned close
                if sw == 1:
                    t_pre = t_cz - LEAD - gate["ef"].dur  # noqa: F821  the shelving EF X start
                else:
                    t_pre = t_cz                  # plain: the prep ends LEAD before the tone itself
                f = 0                             # the on-core X90 frame, rebuilt each shot
                if role == ACTIVE:
                    set_start(gate, t_pre - LEAD - xd)  # noqa: F821  pin the GE retune to the prep
                    set_freq(gate, gate.freq)  # noqa: F821
                    set_phase_offset(gate, f + vz0 + hpi)  # noqa: F821  Y90 #1
                    play(gate, gate["x90"], t_pre - LEAD - xd)  # noqa: F821
                    f = f + vzsum
                    if sw == 1:                   # retune f_GE → f_EF, shelve |1> → |2>
                        set_start(gate, t_pre)  # noqa: F821
                        set_freq(gate, fef)  # noqa: F821
                        set_phase_offset(gate, 0)  # noqa: F821
                        fire(gate, gate["ef"])  # noqa: F821
                    set_start(gate, t_cz)  # noqa: F821     pin the CZ retune to the tone start
                    set_freq(gate, fcz)  # noqa: F821
                    set_phase_offset(gate, 0)  # noqa: F821  fresh frame: slot phase = the line's phase
                    fire(gate, gate["cz"])  # noqa: F821     this core's half of the two-tone CZ
                    if sw == 1:                   # retune f_CZ → f_EF, un-shelve |2> → |1>
                        wait_until(t_pre - LEAD)  # noqa: F821  pace the depth-4 queues (module
                        #                             comment): the prep popped, the 5th pulse fits
                        set_start(gate, t_close - LEAD - gate["ef"].dur)  # noqa: F821
                        set_freq(gate, fef)  # noqa: F821
                        set_phase_offset(gate, 0)  # noqa: F821
                        fire(gate, gate["ef"])  # noqa: F821  ends LEAD before the retuned close
                    set_start(gate, t_close)  # noqa: F821   retune back to GE for the close
                    set_freq(gate, gate.freq)  # noqa: F821
                    set_phase_offset(gate, f + vz0 + hpi + phi)  # noqa: F821  Rz(φ) + close Y90
                    play(gate, gate["x90"], t_close)  # noqa: F821
                    play(ro, ro["meas"], t_ro)  # noqa: F821
                    play(demod, demod["sq"], t_ro + ddly)  # noqa: F821
                    wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821
                    out[i] += read_res()  # noqa: F821  the ACTIVE Ramsey fringe vs φ
                else:                             # SPECTATOR: prep |0>/|1>, fire ITS half of the tone
                    if sp == 1:
                        set_start(gate, t_pre - LEAD - 2 * xd)  # noqa: F821  pin the GE retune
                        set_freq(gate, gate.freq)  # noqa: F821
                        set_phase_offset(gate, f + vz0)  # noqa: F821  X (π), ends LEAD before the tone
                        play(gate, gate["x90"], t_pre - LEAD - 2 * xd)  # noqa: F821
                        f = f + vzsum
                        set_phase_offset(gate, f + vz0)  # noqa: F821
                        fire(gate, gate["x90"])  # noqa: F821
                    if sw == 1:                   # retune f_GE → f_EF, shelve |1> → |2>
                        set_start(gate, t_pre)  # noqa: F821
                        set_freq(gate, fef)  # noqa: F821
                        set_phase_offset(gate, 0)  # noqa: F821
                        fire(gate, gate["ef"])  # noqa: F821
                    set_start(gate, t_cz)  # noqa: F821
                    set_freq(gate, fcz)  # noqa: F821
                    set_phase_offset(gate, 0)  # noqa: F821
                    fire(gate, gate["cz"])  # noqa: F821
                    if sw == 1:                   # retune f_CZ → f_EF, un-shelve |2> → |1>
                        wait_until(t_pre - LEAD)  # noqa: F821  pace the depth-4 queues (module
                        #                             comment): the prep popped, the 5th pulse fits
                        set_start(gate, t_close - LEAD - gate["ef"].dur)  # noqa: F821
                        set_freq(gate, fef)  # noqa: F821
                        set_phase_offset(gate, 0)  # noqa: F821
                        fire(gate, gate["ef"])  # noqa: F821
                    wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821
            else:                                 # COUPLER_FORM: the coupler plays the one drive
                t_cz0 = t_close - czd             # the single CZ pulse start
                f = 0                             # the on-core X90 frame, rebuilt each shot
                if role == COUPLER:
                    play(gate, gate["cz"], t_cz0)  # noqa: F821  the CZ drive
                    wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821
                elif role == ACTIVE:
                    set_phase_offset(gate, f + vz0 + hpi)  # noqa: F821  Y90 #1
                    play(gate, gate["x90"], t_cz0 - xd)  # noqa: F821  (ends at the CZ start)
                    f = f + vzsum
                    set_phase_offset(gate, f + vz0 + hpi + phi)  # noqa: F821  Rz(φ) + close Y90
                    play(gate, gate["x90"], t_close)  # noqa: F821
                    play(ro, ro["meas"], t_ro)  # noqa: F821
                    play(demod, demod["sq"], t_ro + ddly)  # noqa: F821
                    wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821
                    out[i] += read_res()  # noqa: F821  the ACTIVE Ramsey fringe vs φ
                else:                             # SPECTATOR: prep |0>/|1>, hold through the CZ
                    if sp == 1:
                        set_phase_offset(gate, f + vz0)  # noqa: F821  X (π) = X90 X90, ends at t_cz0
                        play(gate, gate["x90"], t_cz0 - 2 * xd)  # noqa: F821
                        f = f + vzsum
                        set_phase_offset(gate, f + vz0)  # noqa: F821
                        fire(gate, gate["x90"])  # noqa: F821
                    wait_until(t_ro + ddly + READOUT_LEAD)  # noqa: F821
            t_ro = t_ro + period  # noqa: F821
        phi = phi + dp
