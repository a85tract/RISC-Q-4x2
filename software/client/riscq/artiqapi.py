"""An ARTIQ-shaped API over RISC-Q: same names, same semantics, RISC-Q's compiler underneath.

The experimenter writes physical units on a timeline, exactly as in ARTIQ:

    core = Core(soc_map)
    ch0, ch1 = DDSChannel(core, 0), DDSChannel(core, 1)

    with parallel(core):                       # both tones start together
        with branch(core):
            ch0.set(83.765*MHz, phase=0.0);  ch0.sw.pulse(100*us)
        with branch(core):
            ch1.set(80.235*MHz, phase=0.5);  ch1.sw.pulse(100*us)
    delay(core, 5*us)
    ch1.set(82*MHz, phase=0.25, phase_mode=PHASE_MODE_ABSOLUTE)
    ch1.sw.pulse(20*us)

Everything this project had to do by hand stays inside: the 2.0345 ns batch grid, the envelope line
layout with its sub-batch leading/trailing zeros, the reserved-line allocation, the long-pulse
chunking that keeps a reserved line out of a wrapping traversal, the TIME_TO_PULSE = 36 pipeline
convention and the phase-register arithmetic.

MULTI-CORE BUILDS (qubit_num > 1): still ONE timeline. DDS channel 2k is hardware core k's gate
drive and 2k+1 its readout drive (their DAC is the build's dac_map); ADCChannel(core, k) is core k's
raw trace (it records while dds 2k+1 fires, from the ADC in adc_map[k]) and DemodChannel(core, k) its
IQ readout. run() compiles one kernel per hardware core from the same plan. Internally every event
carries the flat channel id 3*k + local (local 0 gate, 1 readout, 2 demod), which is the plain
0/1/2 of the single-core builds.

THE CONTRACT — what is exact, what is snapped (stated the way `units.freq_word` states its own):
  * `mu` is the timeline's ARITHMETIC QUANTUM: one DAC sample (127.157 ps on a 491.52 MHz, 16-lane
    build). `seconds_to_mu` rounds to nearest. It is NOT the finest realizable pulse edge.
  * the CARRIER PHASE at the instant you asked for is exact to the phase register's 0.0055 deg
    (0.19 ps of equivalent time shift at 82 MHz): a phase shift of a pure tone IS an exact time
    shift, so snapping the envelope costs no phase accuracy;
  * BOTH ENVELOPE EDGES snap to the channel's envelope grid — 16 / samples_per_line DAC samples,
    i.e. 0.254 ns when a line holds 8 samples, 2.035 ns when it holds 1. `PulseEvent` reports
    `start_error_ps` and `end_error_ps` against what you asked for.

PHASE MODES — `artiq.coredevice.ad9910` semantics, anchored at the `set()` timestamp as in ARTIQ:
  PHASE_MODE_CONTINUOUS  the accumulator is not reset: the phase continues across a frequency
                         change without a jump, so where you land after a hop depends on how long
                         you spent at the other frequency (not reproducible).
  PHASE_MODE_ABSOLUTE    phi(t) = p + (t - t')f with t' the instant of the `set()` call: `phase` is
                         the phase AT THAT INSTANT, history irrelevant.
  PHASE_MODE_TRACKING    phi(t) = p + (t - T)f with T a global fiducial (the sequence origin here):
                         each frequency has its own metronome running since T and `phase` is an
                         offset from it, so frequency hops are phase-reproducible ("coherent").

THE ARITHMETIC. Hardware law (`golden.pulse_window`): the envelope batch `t` plays the carrier of
`t - TIME_TO_PULSE`, so at absolute DAC sample n the output phase is `W*(n - 576) + P`. Write the
intended carrier as `phi(n) = W*n + C`; then uniformly

    P = C + W*576                       and        C = C_const + C_k * t1_dac

where `t1_dac` is the sequence origin, a RUNTIME value (`now()` + a setup lead). Per mode, with
`n_set` the `set()` instant measured from the origin:

    TRACKING    C = p - W*t1_dac                     -> C_const = p,                      C_k = -W
    ABSOLUTE    C = p - W*(t1_dac + n_set)           -> C_const = p - W*n_set,            C_k = -W
    CONTINUOUS  C2 = phi_prev(n_set) + p - W2*(t1_dac + n_set)
                                                     -> C_const = C1_const + (W1-W2)*n_set + p
                                                        C_k = C1_k + W1 - W2 = -W2   (telescopes)

so `C_k = -W` in every mode and the generated kernel always emits `P = Pconst - W*(t1*16)`.
Checked to <= 1 LSB16 against `golden.phase16` by `software/tests/test_artiqapi.py`.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import numpy as np

from riscq.map import LEAD, READOUT_LEAD, SocMap
from riscq.pulses import Pulse, envelopes, golden, units

# ── artiq.language.units ─────────────────────────────────────────────────────────────────────────
s = 1.0
ms = 1e-3
us = 1e-6
ns = 1e-9
Hz = 1.0
kHz = 1e3
MHz = 1e6
GHz = 1e9

# ── artiq.coredevice.ad9910 phase modes ──────────────────────────────────────────────────────────
PHASE_MODE_CONTINUOUS = 0
PHASE_MODE_ABSOLUTE = 1
PHASE_MODE_TRACKING = 2

_MODE_NAME = {PHASE_MODE_CONTINUOUS: "CONTINUOUS", PHASE_MODE_ABSOLUTE: "ABSOLUTE",
              PHASE_MODE_TRACKING: "TRACKING"}
_M32 = 1 << 32
_TTP16 = golden.TIME_TO_PULSE * 16          # 576 DAC samples
# The decoder finishes the integral at the window's END and the result crosses linkPipe = 4
# registered stages; READOUT_LEAD (48 batches ~ 98 ns) upper-bounds that comfortably and is the
# constant the hand-written kernels already trust, so it doubles as the settle tail here.
RESULT_TAIL = READOUT_LEAD
_READOUT_MAX_WIN_LOG2 = 14                  # RTL no-overflow contract: window <= 2^14 batches


def _i32(v: int) -> int:
    """Seated words go out as int32 (32768 << 16 is -2^31, not +2^31)."""
    return (int(v) + (1 << 31)) % _M32 - (1 << 31)


# ── channel ids: user-facing dds index 2k/2k+1 <-> internal flat id 3k + local (0 gate, 1 readout,
#    2 demod) of hardware core k; on a single-core build the flat id IS the local one ──
def _dds_flat(m: SocMap, index: int) -> int:
    n = 2 * m.params.qubit_num
    if not 0 <= int(index) < n:
        raise ValueError(
            f"dds channel {index} does not exist: a {m.params.qubit_num}-core build has {n} dds "
            f"channels (2k = core k's gate drive, 2k+1 = core k's readout drive); IQ readout is "
            "DemodChannel, not a dds")
    return 3 * (int(index) // 2) + int(index) % 2


def _label(f: int, m: SocMap) -> str:
    """Human name of a flat channel id: ch<dds index> or demod<core>."""
    k, local = divmod(int(f), 3)
    if local == 2:
        return "demod" if m.params.qubit_num == 1 else f"demod{k}"
    return f"ch{2 * k + local}"


# ── the timeline ─────────────────────────────────────────────────────────────────────────────────

class Core:
    """The timeline cursor and the machine unit.

    `mu` = one DAC sample: the timeline's arithmetic quantum, finer than any realizable envelope
    edge, so `seconds_to_mu` never hides a grid the hardware does not have. Edge snapping happens
    later, per channel, and is reported.
    """

    def __init__(self, soc_map: SocMap):
        self.m = soc_map
        self.f_dac = 16 * soc_map.params.dsp_freq_hz
        self._cursor = 0
        self._cursor_s = 0.0        # diagnostic: the un-rounded request, for the error report
        self._parallel = None
        self.sets: list[ToneSet] = []
        self.events: list[PulseEvent] = []
        self.trace_gates: list["TraceGate"] = []
        self.last_result: "RunResult | None" = None

    # -- artiq.language.core --
    def now_mu(self) -> int:
        return self._cursor

    def at_mu(self, t: int) -> None:
        self._cursor = int(t)
        self._cursor_s = int(t) / self.f_dac

    def delay_mu(self, dt: int) -> None:
        self._cursor += int(dt)
        self._cursor_s += int(dt) / self.f_dac

    def delay(self, dt: float) -> None:
        self._cursor += self.seconds_to_mu(dt)
        self._cursor_s += dt

    def seconds_to_mu(self, t: float) -> int:
        return int(round(t * self.f_dac))

    def mu_to_seconds(self, mu: int) -> float:
        return mu / self.f_dac

    # -- not ARTIQ: the honest error report needs the un-rounded request --
    def at_s(self, t: float) -> None:
        self._cursor = self.seconds_to_mu(t)
        self._cursor_s = t

    def clear(self) -> None:
        """Drop the recorded sequence. NOT ARTIQ's `core.reset()`, which touches hardware."""
        self._cursor = 0
        self._cursor_s = 0.0
        self.sets.clear()
        self.events.clear()
        self.trace_gates.clear()
        self.last_result = None


# module-level timeline verbs, for ARTIQ's hand feel: `delay(core, 5*us)`
def now_mu(core: Core) -> int:
    return core.now_mu()


def at_mu(core: Core, t: int) -> None:
    core.at_mu(t)


def delay(core: Core, dt: float) -> None:
    core.delay(dt)


def delay_mu(core: Core, dt: int) -> None:
    core.delay_mu(dt)


class _ParallelTracker:
    """Rewinds the cursor to the block start for each branch, remembers where each branch ended."""

    def __init__(self, core, start):
        self.core, self.start, self.ends = core, start, []

    def branch_begin(self):
        self.core._cursor, self.core._cursor_s = self.start

    def branch_end(self):
        self.ends.append((self.core._cursor, self.core._cursor_s))


@contextlib.contextmanager
def sequential(core: Core):
    """ARTIQ's `with sequential:` — statements advance the cursor (the default here)."""
    yield core


@contextlib.contextmanager
def parallel(core: Core):
    """ARTIQ's `with parallel:` — every `with branch(core):` inside starts at the same instant and
    the cursor afterwards is the LATEST end among them."""
    start = (core._cursor, core._cursor_s)
    tracker = _ParallelTracker(core, start)
    prev, core._parallel = core._parallel, tracker
    try:
        yield core
    finally:
        core._parallel = prev
        core._cursor, core._cursor_s = (max(tracker.ends, key=lambda e: e[0])
                                        if tracker.ends else start)


@contextlib.contextmanager
def branch(core: Core):
    """One arm of a `with parallel(core):` block."""
    tr = core._parallel
    if tr is None:
        yield core
        return
    tr.branch_begin()
    try:
        yield core
    finally:
        tr.branch_end()


# ── one DDS channel ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToneSet:
    """One `set()` call: a state change on a channel at an instant, whether or not a pulse follows.

    CONTINUOUS chains through these, not through the pulses — `set(TRACKING); set(CONTINUOUS);
    pulse()` must see the first set as its predecessor even though it played nothing."""
    channel: int                  # flat id 3*core + local (see the module docstring)
    set_mu: int
    set_s: float                  # un-rounded: the phase is anchored here, not on the mu grid
    frequency: float
    phase_turns: float
    amplitude: float
    phase_mode: int
    is_filler: bool = False       # a capture filler disturbs neither the phase chain nor the regs
    is_demod: bool = False        # a demod tone uses the ADC-rate word law (units.demod_freq_word)
    phase_const: int = 0          # filled in by plan()
    freq_word: int = 0


@dataclass
class PulseEvent:
    """One scheduled pulse, in both the user's units and the hardware's."""
    channel: int                  # flat id 3*core + local (see the module docstring)
    set_index: int                # which ToneSet this pulse plays
    set_mu: int                   # the `set()` instant (ARTIQ's t' for ABSOLUTE / CONTINUOUS)
    set_s: float                  # the same instant UN-ROUNDED — what the phase is anchored to
    start_mu: int                 # the requested pulse start, in mu
    dur_mu: int
    requested_start_s: float      # un-rounded, for the error report
    requested_dur_s: float
    frequency: float
    phase_turns: float
    amplitude: float
    phase_mode: int
    # filled in by plan()
    start_dac: int = 0
    end_dac: int = 0
    batch: int = 0
    lead_zeros: int = 0
    trail_zeros: int = 0
    dur_batches: int = 0
    freq_word: int = 0
    phase_const: int = 0          # the kernel emits  P = phase_const - freq_word * (t1 * 16)
    amp_code: int = 0
    env_line: int = 0
    tail_line: int = -1           # -1 when the last batch is full
    is_demod: bool = False        # this pulse IS an integration window (its play is the readout)
    result_slot: int = -1         # index into the `out` results array (3 words per readout)

    def realized_start_ns(self, core: Core) -> float:
        return self.start_dac / core.f_dac * 1e9

    def start_error_ps(self, core: Core) -> float:
        return (self.start_dac / core.f_dac - self.requested_start_s) * 1e12

    def realized_dur_ns(self, core: Core) -> float:
        return (self.end_dac - self.start_dac) / core.f_dac * 1e9

    def end_error_ps(self, core: Core) -> float:
        return (self.end_dac / core.f_dac
                - (self.requested_start_s + self.requested_dur_s)) * 1e12


class _Switch:
    """`dds.sw` — the RF switch, as in ARTIQ (`urukul_ch.sw.pulse(t)`)."""

    def __init__(self, dds: "DDSChannel"):
        self._dds = dds

    def pulse(self, duration: float) -> None:
        self._dds._emit(self._dds.core.seconds_to_mu(duration), duration)

    def pulse_mu(self, duration_mu: int) -> None:
        self._dds._emit(int(duration_mu), int(duration_mu) / self._dds.core.f_dac)


class DDSChannel:
    """One RF drive channel, shaped like `artiq.coredevice.ad9910.AD9910`.

    `index` counts drive channels across the build's cores: 2k is hardware core k's gate drive,
    2k+1 its readout drive (on a 1-core build simply 0 and 1). The physical DAC of each is the
    build's dac_map; the demod carrier is never a dds — see DemodChannel."""

    def __init__(self, core: Core, index: int, name: str | None = None):
        self.core = core
        self.index = int(index)
        self.name = name or f"ch{index}"
        self.flat = _dds_flat(core.m, self.index)       # raises on an out-of-range index
        self.core_index = self.flat // 3
        info = core.m.channel(self.flat % 3)
        self.samples_per_line = info.samples_per_line
        self.step = 16 // self.samples_per_line     # DAC samples per stored envelope sample
        self.sw = _Switch(self)
        self._phase_mode = PHASE_MODE_TRACKING
        self._tone = None                           # (f, p, a, mode, set_mu)

    # -- artiq.coredevice.ad9910 --
    def set_phase_mode(self, phase_mode: int) -> None:
        if phase_mode not in _MODE_NAME:
            raise ValueError(f"unknown phase mode {phase_mode!r}")
        self._phase_mode = phase_mode

    def set(self, frequency: float, phase: float = 0.0, amplitude: float = 1.0,
            phase_mode: int | None = None) -> None:
        """Set the tone AT THE CURRENT CURSOR INSTANT — a state change that does not advance time,
        as in ARTIQ. `phase` is in TURNS."""
        mode = self._phase_mode if phase_mode is None else phase_mode
        if mode not in _MODE_NAME:
            raise ValueError(f"unknown phase mode {mode!r}")
        if not 0.0 <= amplitude <= 1.0:
            raise ValueError(f"amplitude {amplitude} outside [0, 1]")
        self.core.sets.append(ToneSet(
            channel=self.flat, set_mu=self.core.now_mu(), set_s=self.core._cursor_s,
            frequency=float(frequency), phase_turns=float(phase), amplitude=float(amplitude),
            phase_mode=mode))
        self._tone = len(self.core.sets) - 1

    def pulse(self, duration: float) -> None:
        """Convenience alias for `dds.sw.pulse(duration)` (ARTIQ gates an RF switch, not the DDS)."""
        self.sw.pulse(duration)

    def pulse_mu(self, duration_mu: int) -> None:
        self.sw.pulse_mu(duration_mu)

    # -- internal --
    def _emit(self, dur_mu: int, dur_s: float) -> None:
        if self._tone is None:
            raise RuntimeError(f"{self.name}.pulse() before {self.name}.set()")
        t = self.core.sets[self._tone]
        self.core.events.append(PulseEvent(
            channel=self.flat, set_index=self._tone, set_mu=t.set_mu, set_s=t.set_s,
            start_mu=self.core.now_mu(), dur_mu=int(dur_mu),
            requested_start_s=self.core._cursor_s, requested_dur_s=dur_s,
            frequency=t.frequency, phase_turns=t.phase_turns, amplitude=t.amplitude,
            phase_mode=t.phase_mode))
        self.core.delay_mu(int(dur_mu))


# ── the receive side ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TraceGate:
    """One raw-ADC recording window (the "robs" trace BRAM)."""
    start_mu: int
    dur_mu: int
    requested_start_s: float
    requested_dur_s: float
    core_index: int = 0           # the hardware core whose trace (and readout drive) this is
    batch0: int = 0               # filled in by plan(): first recorded batch
    batches: int = 0


class ADCChannel:
    """The raw ADC trace recorder, shaped like ARTIQ's TTL input gate.

    `gate(duration)` schedules a recording window at the cursor and ADVANCES the cursor (exactly
    `TTLInOut.gate_rising` semantics — put it in a `with parallel(core): with branch(core):` to
    record concurrently with the pulses, as one gates a counter in ARTIQ), returning the end mu.

    `index` is the hardware core whose trace this is: it records the ADC in the build's adc_map[k]
    while that core's readout drive (dds 2k+1) fires. On a 1-core build there is only index 0.

    What the hardware really does (owned here, invisible to the user): the trace BRAM records only
    while the core's readout-drive channel fires, and its write address RESETS whenever it stops —
    so the planner makes that channel fire CONTIGUOUSLY across the gate by inserting amplitude-0
    fillers, refuses its pulses OUTSIDE the gate (they would restart the trace), allows at most ONE
    gate per trace per run, snaps the gate outward to whole batches, and checks it fits rob_depth.
    After `run()`, `fetch_trace()` returns the samples (int32, fs = ADC_BATCH x dsp clock). On a
    build without per-core traces (rob_per_core off) the ONE trace stores, per lane, the SUM over
    the mapped physical ADCs and records on ANY core's readout fire — so it is offered only as
    index 0 of a 1-core build there."""

    def __init__(self, core: Core, index: int = 0, name: str = "adc"):
        self.core = core
        self.name = name
        self.index = int(index)
        p = core.m.params
        if not 0 <= self.index < p.qubit_num:
            raise ValueError(f"adc {index} does not exist: a {p.qubit_num}-core build has traces "
                             f"0..{p.qubit_num - 1}")
        if p.qubit_num > 1 and not p.rob_per_core:
            raise ValueError(
                "this multi-core build has ONE shared trace (rob_per_core off): it sums every mapped "
                "ADC and records on any core's readout fire, so it is not a per-channel adc")
        self.core_index = self.index

    def gate(self, duration: float) -> int:
        return self.gate_mu(self.core.seconds_to_mu(duration))

    def gate_mu(self, duration_mu: int) -> int:
        self.core.trace_gates.append(TraceGate(
            start_mu=self.core.now_mu(), dur_mu=int(duration_mu),
            requested_start_s=self.core._cursor_s,
            requested_dur_s=int(duration_mu) / self.core.f_dac, core_index=self.core_index))
        self.core.delay_mu(int(duration_mu))
        return self.core.now_mu()

    def fetch_trace(self):
        """The last run's trace (EdgeCounter's fetch_count naming)."""
        r = self.core.last_result
        cr = None if r is None else r.cores.get(self.core_index)
        if cr is None or cr.trace is None:
            raise RuntimeError(f"no trace for {self.name}: run() the sequence first (with an "
                               f"{self.name}.gate window)")
        return cr.trace


class DemodChannel:
    """The IQ readout channel (logical channel 2), shaped like ARTIQ's EdgeCounter/Sampler.

    `set(frequency, ...)` takes the RF frequency in Hz — the matched ADC-rate LO word
    (`demod_freq_word`, the 4x law) never reaches the user. `gate(duration)` plays the demod pulse
    at the cursor: on this hardware that IS the readout — the pulse's window is the integration
    window — and it queues one IQ result; the cursor advances by the duration. Results come back
    in order after `run()` via `fetch()` (EdgeCounter: multiple gates may be queued).

    PHASE contract (honest): `phase` rotates the returned IQ by exactly the requested turns
    (relative phase — verified in co-sim). The ABSOLUTE IQ angle contains one fixed, per-build
    pipeline offset (the demod path's time alignment is not established by any golden model);
    calibrate it empirically, as `cal/` does with its demod phase knob.

    The matched-filter envelope is RECTANGULAR in v1 (the demod envelope RAM is tiled with the
    unit sample); host-provided complex weights are future work.

    TIMING contract (from riscq.h): reading a result HALTS the core until the window's integral
    settles, so the planner refuses any event that starts within READOUT_LEAD + LEAD batches after
    an IQ gate's end — space the schedule out, as the cal kernels' `period` grid does."""

    def __init__(self, core: Core, index: int = 0, name: str = "demod"):
        self.core = core
        self.index = int(index)                 # the hardware core whose readout this is
        p = core.m.params
        if not 0 <= self.index < p.qubit_num:
            raise ValueError(f"demod {index} does not exist: a {p.qubit_num}-core build has readouts "
                             f"0..{p.qubit_num - 1}")
        self.core_index = self.index
        self.flat = 3 * self.index + 2          # the core's demod carrier channel
        self.name = name
        self._phase_mode = PHASE_MODE_TRACKING
        self._tone = None

    def set_phase_mode(self, phase_mode: int) -> None:
        if phase_mode not in _MODE_NAME:
            raise ValueError(f"unknown phase mode {phase_mode!r}")
        self._phase_mode = phase_mode

    def set(self, frequency: float, phase: float = 0.0, amplitude: float = 1.0,
            phase_mode: int | None = None) -> None:
        mode = self._phase_mode if phase_mode is None else phase_mode
        if mode not in _MODE_NAME:
            raise ValueError(f"unknown phase mode {mode!r}")
        if not 0.0 <= amplitude <= 1.0:
            raise ValueError(f"amplitude {amplitude} outside [0, 1]")
        self.core.sets.append(ToneSet(
            channel=self.flat, set_mu=self.core.now_mu(), set_s=self.core._cursor_s,
            frequency=float(frequency), phase_turns=float(phase), amplitude=float(amplitude),
            phase_mode=mode, is_demod=True))
        self._tone = len(self.core.sets) - 1

    def gate(self, duration: float) -> int:
        return self.gate_mu(self.core.seconds_to_mu(duration))

    def gate_mu(self, duration_mu: int) -> int:
        if self._tone is None:
            raise RuntimeError(f"{self.name}.gate() before {self.name}.set()")
        t = self.core.sets[self._tone]
        if -(-int(duration_mu) // 16) > (1 << _READOUT_MAX_WIN_LOG2):
            raise ValueError(
                f"integration window of {-(-int(duration_mu) // 16)} batches exceeds the "
                f"accumulator's no-overflow contract (2^{_READOUT_MAX_WIN_LOG2} = "
                f"{1 << _READOUT_MAX_WIN_LOG2} batches ~ 33 us)")
        slot = sum(1 for e in self.core.events if e.is_demod and e.channel == self.flat)
        self.core.events.append(PulseEvent(
            channel=self.flat, set_index=self._tone, set_mu=t.set_mu, set_s=t.set_s,
            start_mu=self.core.now_mu(), dur_mu=int(duration_mu),
            requested_start_s=self.core._cursor_s,
            requested_dur_s=int(duration_mu) / self.core.f_dac,
            frequency=t.frequency, phase_turns=t.phase_turns, amplitude=t.amplitude,
            phase_mode=t.phase_mode, is_demod=True, result_slot=slot))
        self.core.delay_mu(int(duration_mu))
        return self.core.now_mu()

    def fetch_iq(self):
        """The next queued readout, in gate order: a record with .res (hardware sign bit),
        .real/.imag (32-bit integrals) and .iq (complex)."""
        r = self.core.last_result
        cr = None if r is None else r.cores.get(self.core_index)
        if cr is None or not len(cr.res):
            raise RuntimeError(f"no readout results for {self.name}: run() a sequence with "
                               f"{self.name}.gate windows first")
        if getattr(self, "_fetch_result", None) is not r:   # a NEW run: start from its first result
            self._fetch_result, self._fetch_cursor = r, 0
        k = self._fetch_cursor
        if k >= len(cr.res):
            raise RuntimeError(f"all {len(cr.res)} queued results already fetched")
        self._fetch_cursor = k + 1
        import types
        return types.SimpleNamespace(res=int(cr.res[k]), real=int(cr.real[k]),
                                     imag=int(cr.imag[k]), iq=complex(cr.iq[k]))


def _fill_trace_window(core: Core, gate: TraceGate) -> int:
    """Make the gate's core's readout drive fire contiguously over the gate window (batch-granular),
    inserting amplitude-0 fillers for the lead-in, the holes, and the tail. Returns the number of
    fillers added."""
    ro = 3 * gate.core_index + 1                      # that core's readout-drive channel
    name = _label(ro, core.m)
    b0 = gate.start_mu // 16
    b1 = -(-(gate.start_mu + gate.dur_mu) // 16)
    evs = sorted((e for e in core.events if e.channel == ro), key=lambda e: e.start_mu)
    step = 16 // core.m.channel(1).samples_per_line

    def batches_of(e):
        # EXACTLY plan()'s arithmetic: sub-batch leading edge, whole-batch trailing edge
        sd = int(round(e.start_mu / step) * step)
        ed = 16 * int(round((e.start_mu + e.dur_mu) / 16))
        return sd // 16, max(ed // 16, sd // 16 + 1)

    inside = []
    for e in evs:
        eb0, eb1 = batches_of(e)
        if eb1 < b0:
            continue          # ends with a break before the gate: the refire overwrites its trace
        if eb0 >= b0 and eb1 <= b1:
            inside.append(e)
            continue
        raise RuntimeError(
            f"{name} pulse at batches [{eb0}, {eb1}) overlaps the adc gate [{b0}, {b1}) boundary or "
            f"follows it: firing there would merge with or restart the recording — keep {name} "
            "pulses fully inside the gate, or ended at least one batch before it")
    evs = inside

    freq = evs[0].frequency if evs else 0.0
    spans, cur = [], b0
    for e in evs:
        eb0, eb1 = batches_of(e)
        if eb0 > cur:
            spans.append((cur, eb0))
        cur = max(cur, eb1)
    if cur < b1:
        spans.append((cur, b1))

    for lo, hi in spans:
        start_mu, dur_mu = lo * 16, (hi - lo) * 16
        core.sets.append(ToneSet(
            channel=ro, set_mu=start_mu, set_s=start_mu / core.f_dac, frequency=freq,
            phase_turns=0.0, amplitude=0.0, phase_mode=PHASE_MODE_TRACKING, is_filler=True))
        core.events.append(PulseEvent(
            channel=ro, set_index=len(core.sets) - 1, set_mu=start_mu,
            set_s=start_mu / core.f_dac, start_mu=start_mu, dur_mu=dur_mu,
            requested_start_s=start_mu / core.f_dac, requested_dur_s=dur_mu / core.f_dac,
            frequency=freq, phase_turns=0.0, amplitude=0.0, phase_mode=PHASE_MODE_TRACKING))
    gate.batch0, gate.batches = b0, b1 - b0
    return len(spans)


# ── the scheduler: physical units -> hardware parameters ─────────────────────────────────────────

@dataclass
class Schedule:
    """The planned sequence: every event on the hardware's grids, residuals in the carrier phase,
    the leftovers reported."""
    core: Core
    events: list[PulseEvent]
    env_lines: dict[int, dict[tuple, int]] = field(default_factory=dict)  # ch -> {(lead,trail):line}
    chunks: dict[int, list[int]] = field(default_factory=dict)            # event index -> lengths
    first_line: int = 0                                                   # full lines start here

    def report(self) -> str:
        c = self.core
        out = [f"{'ch':>6} {'start asked':>13} {'start got':>13} {'err':>8} "
               f"{'dur asked':>12} {'dur got':>12} {'end err':>8} "
               f"{'freq':>13} {'phase':>7} {'mode':>10}"]
        for e in self.events:
            out.append(
                f"{_label(e.channel, c.m):>6} {e.requested_start_s*1e9:>12.4f}n {e.realized_start_ns(c):>12.4f}n "
                f"{e.start_error_ps(c):>+7.1f}p {e.requested_dur_s*1e9:>11.4f}n "
                f"{e.realized_dur_ns(c):>11.4f}n {e.end_error_ps(c):>+7.1f}p "
                f"{units.word_to_freq(e.freq_word, c.m.params)/(4 if e.is_demod else 1)/1e6:>12.6f}M "
                f"{e.phase_turns:>7.4f} {_MODE_NAME[e.phase_mode]:>10}")
        for tg in self.core.trace_gates:
            out.append(f"adc{tg.core_index} gate: requested [{tg.requested_start_s*1e9:.4f}, "
                       f"{(tg.requested_start_s + tg.requested_dur_s)*1e9:.4f}) ns -> recorded "
                       f"batches [{tg.batch0}, {tg.batch0 + tg.batches}) "
                       f"({tg.batches} batches, {tg.batches*16/self.core.f_dac*1e6:.4f} us)")
        return "\n".join(out)


POP_LATE = 3        # batches: a queue entry is out this long after its due batch (SrlShadow II=3; the
                    # queue actually pops LEAD early, so this is conservative)
PUSH_MARGIN = 300   # batches (0.61 us): a barrier's wait_until return plus that event's stores at the
                    # 4x2's ~100 MHz CPU, before the play is in the queue
PUSH_COST = 60      # batches (0.12 us): every further event's stores (set_freq .. play) while no wait holds
                    # the kernel — the pace the kernel sustains pushing one play after another.
                    # Both measured in co-sim (sim/cosim2q_queue.py, 25 plays on one channel at period P):
                    # P = 73 (36.5 batches per play) FAILS — the kernel falls behind and pushes arrive
                    # late; P = 100 (50 per play) and P = 121 PASS. The planner's model refuses P <= 115
                    # and admits P = 121, i.e. it keeps ~15 % over what the RTL sustained.


def _push_order(sch: "Schedule", core_index: int) -> list[tuple[int, int, int, int]]:
    """Core `core_index`'s plays in the order its kernel pushes them — schedule order, an event's
    chunks back to back: [(event index, chunk number, channel, due batch)]."""
    out = []
    for idx, e in enumerate(sch.events):
        if e.channel // 3 != core_index:
            continue
        b = e.batch
        for j, (_, nb) in enumerate(sch.chunks[idx]):
            out.append((idx, j, e.channel, b))
            b += nb
    return out


def _queue_barriers(sch: "Schedule", core_index: int) -> dict[tuple[int, int], int]:
    """Where core `core_index`'s kernel must `wait_until` before a play push so that no channel's
    TimedQueue ever holds more than `queue_depth` entries: {(event index, chunk number): batch}.

    The push has no backpressure (an overfull queue silently drops entries), so before a channel's
    (k+depth)-th event the kernel waits for the k-th play to have popped — its due batch plus POP_LATE
    (the wait precedes the event's FIRST register write: set_freq / set_phase / set_amp / set_env /
    set_dur are queue entries too, tagged with the channel's current start time, so they must not
    be pushed into a full queue either). A barrier holds the whole core, and from then on the
    planner tracks a lower bound of the kernel's clock — PUSH_MARGIN after a wait that waits,
    PUSH_COST per further event — and refuses the schedule when a play would be pushed with less
    than LEAD batches to its due."""
    m = sch.core.m
    depth = m.params.queue_depth
    pushed: dict[int, list[int]] = {}
    barriers: dict[tuple[int, int], int] = {}
    clock = None                                   # lower bound of the kernel's clock, once a wait held it
    last_bar = None
    for idx, j, ch, due in _push_order(sch, core_index):
        q = pushed.setdefault(ch, [])
        if len(q) >= depth:
            bar = q[-depth] + POP_LATE
            barriers[(idx, j)] = bar               # ALWAYS emitted: the model's clock is only a lower
            if clock is None or bar > clock:       # bound, a faster kernel still needs the wait (free
                clock, last_bar = bar + PUSH_MARGIN, bar     # when the time has passed)
            else:                                  # returns at once: just this event's stores
                clock += PUSH_COST
        elif clock is not None:
            clock += PUSH_COST
        if clock is not None and due - clock < LEAD:
            raise RuntimeError(
                f"{_label(ch, m)}: the play at batch {due} (event {idx}) can only be pushed after the "
                f"queue entry from batch {last_bar - POP_LATE} pops (queue depth {depth}) and the pushes "
                f"queued behind that wait, i.e. at about batch {clock}, which leaves it {due - clock} "
                f"batches of lead (>= LEAD = {LEAD} needed; a wait costs PUSH_MARGIN = {PUSH_MARGIN}, "
                f"every further event PUSH_COST = {PUSH_COST}); space the events — no {depth + 1} plays "
                f"of one channel within {LEAD + PUSH_MARGIN + POP_LATE} batches — or split the run")
        q.append(due)
    return barriers


def plan(core: Core, reserved_base: int = 0, max_run: int | None = None) -> Schedule:
    """Place every recorded event on the hardware's grids.

    Envelope lines `reserved_base ..` hold one partial pattern per distinct (leading, trailing)
    zero pair seen on a channel; the full lines start after them. A pulse longer than the usable
    depth would wrap the free-running reader back onto a reserved line, so it is CHUNKED into
    back-to-back plays that never wrap.
    """
    m = core.m
    for k in {tg.core_index for tg in core.trace_gates}:
        n = sum(1 for tg in core.trace_gates if tg.core_index == k)
        if n > 1:
            raise RuntimeError(
                f"{n} gates on adc{k} in one run: the trace address resets on every refire, so a "
                "second window would overwrite the first — one gate per trace per run")
    for tg in core.trace_gates:
        # the gate snaps OUTWARD to whole batches, so check the snapped size, not ceil(dur)
        gb = -(-(tg.start_mu + tg.dur_mu) // 16) - tg.start_mu // 16
        if gb > m.params.rob_depth:
            raise RuntimeError(f"adc gate of {gb} batches (snapped outward) exceeds rob_depth "
                               f"{m.params.rob_depth}")
        _fill_trace_window(core, tg)
    sch = Schedule(core=core, events=sorted(core.events, key=lambda e: (e.start_mu, e.channel)))
    per_ch: dict[int, dict[tuple, int]] = {}
    _plan_phase_chain(core)

    # pass 1: edges, words, and the partial-line inventory
    for e in sch.events:
        info = m.channel(e.channel % 3)          # the core-local channel (gate / readout / demod)
        step = 16 // info.samples_per_line

        # -- the LEADING edge snaps to this channel's envelope grid (sub-batch, the
        #    board-verified partial-first-line pattern); the TRAILING edge snaps to a WHOLE batch:
        #    a trailing partial line would need a 1-batch play at the END of the chain, and co-sim
        #    showed that pattern drops the pulse valid and resets the trace address (probe
        #    2026-08-30). The end error is reported like every other snap. --
        e.start_dac = int(round(e.start_mu / step) * step)
        e.end_dac = 16 * int(round((e.start_mu + e.dur_mu) / 16))
        if e.end_dac <= e.start_dac:
            e.end_dac = (e.start_dac // 16 + 1) * 16
        e.batch, e.lead_zeros = divmod(e.start_dac, 16)
        e.trail_zeros = 0
        e.dur_batches = e.end_dac // 16 - e.batch

        # -- the carrier: phi(n) = W*n + C, C = C_const - W*t1_dac (see the module docstring) --
        W = units.freq_word(e.frequency, m.params)
        p32 = (round(e.phase_turns * 65536) % 65536) << 16
        e.freq_word = core.sets[e.set_index].freq_word
        e.phase_const = core.sets[e.set_index].phase_const
        e.amp_code = units._amp_code(e.amplitude) & 0xFFFF

        # a line is reserved only for a NON-trivial pattern; a square pulse uses the full region
        keys = per_ch.setdefault(e.channel, {})
        if e.lead_zeros:
            keys.setdefault((e.lead_zeros, 0), None)

    # pass 2: allocate the reserved lines, then the chunking (which depends on how many)
    n_reserved = max((len(v) for v in per_ch.values()), default=0)
    # TRIPLET layout: each lead pattern owns THREE lines — base+3i the partial lead line,
    # base+3i+1 and base+3i+2 full-square copies (free: envelope_images tiles the whole RAM with
    # the full square first). A lead line can then free-run across its own copies, so the first
    # play of a split pulse covers 3 batches — see _chunk_runs for why 3 is the magic number.
    for ch, keys in per_ch.items():
        for i, k in enumerate(list(keys)):
            keys[k] = reserved_base + 3 * i
    sch.env_lines, sch.first_line = per_ch, reserved_base + 3 * n_reserved
    usable = m.params.env_depth - sch.first_line
    if max_run is not None:
        usable = min(usable, max_run)          # diagnostic: force a chunk size (see _chunk_runs)
    if usable < 1:
        raise RuntimeError(f"envelope RAM exhausted: {n_reserved} reserved lines of "
                           f"{m.params.env_depth}")

    for idx, e in enumerate(sch.events):
        keys = per_ch[e.channel]
        e.env_line = keys[(e.lead_zeros, 0)] if e.lead_zeros else sch.first_line
        e.tail_line = -1

    # back-to-back timelines: the whole-batch end rounding may step past the next pulse's start
    # on the same channel — the boundary batch belongs to the FOLLOWER (its play owns the batch,
    # and its partial first line silences the leading samples); the earlier pulse is clamped and
    # its end_error reports the loss.
    for ch in {e.channel for e in sch.events}:
        evs = sorted((e for e in sch.events if e.channel == ch), key=lambda e: e.start_dac)
        for a, b in zip(evs, evs[1:]):
            if a.batch < b.batch and 0 < a.end_dac - b.batch * 16 <= 16:
                a.end_dac = b.batch * 16
                a.dur_batches = a.end_dac // 16 - a.batch
                if a.dur_batches < 1:
                    raise RuntimeError(
                        f"{_label(ch, m)}: pulse at batch {a.batch} is squeezed to nothing by the next "
                        f"pulse at batch {b.batch} — the timeline packs sub-batch pulses tighter "
                        "than the envelope grid allows")

    # chunk AFTER the clamp above — it changes dur_batches, and the runs must replay the FINAL
    # duration. The envelope reader free-runs and WRAPS at env_depth, so a run may never pass
    # the reserved block: from `env_line` it can read to the top of the RAM and no further.
    for idx, e in enumerate(sch.events):
        sch.chunks[idx] = _chunk_runs(sch, e, usable)

    # Two hardware contracts of the play parameter queue (TimedQueue.scala; the deployed impl
    # is PulseGeneratorParams' default TimedQueueImpl.SrlShadow):
    #  * II = 3 — a 2-cycle fire blank follows every pop (the shadow due register is stale for
    #    two cycles), so a play starting < 3 batches after the previous one pops a batch LATE:
    #    the duration counter passes through zero, the pulse valid line glitches low for one
    #    batch (a hole in the DAC output), and the trace recorder resets its write address (the
    #    RX wrap bug). Queued plays must start >= 3 batches apart. (RegHead would tolerate 2 —
    #    II=2 — but the conservative bound covers every impl in TimedQueueImpl.)
    #  * The play push has NO backpressure (a Flow into the generator): an overfull queue
    #    silently DROPS entries. The queue is queue_depth deep, so before its (k+depth)-th play of a
    #    channel the kernel WAITS for the k-th to have popped (a `wait_until` barrier, emitted by
    #    generate_kernel_source) — legal only while every later push still keeps its lead:
    #    _queue_barriers checks that, per core.
    for ch in {e.channel for e in sch.events}:
        plays = []
        for idx, e in enumerate(sch.events):
            if e.channel != ch:
                continue
            b = e.batch
            for _, nb in sch.chunks[idx]:
                plays.append((b, idx))
                b += nb
        plays.sort()
        for (pa, ia), (pb, ib) in zip(plays, plays[1:]):
            if pb - pa < 3:
                raise RuntimeError(
                    f"{_label(ch, m)}: queued plays at batches {pa} (event {ia}) and {pb} (event {ib}) "
                    "start closer than 3 batches apart — the parameter queue pops one entry per "
                    "3 batches (SrlShadow II=3), so the later play lands a batch late and the "
                    "channel (and the trace recording) glitches low for a batch; space the "
                    "events apart")
    for k in {e.channel // 3 for e in sch.events}:
        _queue_barriers(sch, k)

    # the read of an IQ result halts ITS core until the integral settles (window end), and that
    # core's later MMIO pushes need their LEAD margin — refuse anything on the same core scheduled
    # too close after a readout (other cores run their own kernels and are not halted)
    guard = RESULT_TAIL + LEAD
    for d in (e for e in sch.events if e.is_demod):
        d_end = d.batch + d.dur_batches
        for e in sch.events:
            if e is d or e.batch <= d.batch or e.channel // 3 != d.channel // 3:
                continue
            if e.batch < d_end + guard:
                raise RuntimeError(
                    f"event at batch {e.batch} starts within {guard} batches of the readout "
                    f"window ending at batch {d_end}: read_res halts the core there, so later "
                    f"pulses need >= READOUT_LEAD + LEAD = {guard} batches of slack (space the "
                    "schedule out, as the cal kernels' period grid does)")

    # the batch clock is 32 bits and every due test is a signed difference: keep the whole schedule
    # well inside half its range (2^31 batches = 4.37 s) — 2^30 leaves the same margin again
    horizon = max((e.batch + e.dur_batches for e in sch.events), default=0)
    if horizon >= (1 << 30):
        raise RuntimeError(f"schedule horizon of {horizon} batches ({horizon * 16 / core.f_dac:.3f} s) "
                           "exceeds the 32-bit batch clock's safe range (2^30 batches, 2.18 s)")

    # overlap check: two pulses on one channel may not overlap in time
    for ch in {e.channel for e in sch.events}:
        evs = sorted((e for e in sch.events if e.channel == ch), key=lambda e: e.start_dac)
        for a, b in zip(evs, evs[1:]):
            if b.start_dac < a.end_dac:
                raise RuntimeError(
                    f"{_label(ch, m)}: pulses overlap — one ends at DAC sample {a.end_dac}, the next starts "
                    f"at {b.start_dac}; a channel plays one pulse at a time")
    return sch


def fill_gaps(core: Core, channel: int, amplitude: float = 0.0) -> int:
    """Make `channel` fire CONTINUOUSLY by inserting amplitude-0 pulses in its gaps.

    Not an ARTIQ concept — a property of this hardware's one-shot capture: the trace records only
    while the readout channel is firing, and its write address RESETS between fires. A sequence
    with a silent gap therefore comes back as two traces overlaid at address 0 unless the gap is
    played as a zero-amplitude pulse (the DAC output is genuinely zero either way).

    Call it after building the sequence and before `plan()`. `channel` is the dds index. Returns
    the number of fillers added.
    """
    channel = _dds_flat(core.m, channel)
    evs = sorted((e for e in core.events if e.channel == channel), key=lambda e: e.start_mu)
    if not evs:
        return 0
    step = 16 // core.m.channel(channel % 3).samples_per_line
    added = 0
    for prev, nxt in zip(evs, evs[1:]):
        end_dac = int(round((prev.start_mu + prev.dur_mu) / step) * step)
        end_batch = -(-end_dac // 16)                       # ceil: the first batch after it
        start_batch = int(round(nxt.start_mu / step) * step) // 16
        if start_batch <= end_batch:
            continue                                       # already contiguous batch-wise
        start_mu, dur_mu = end_batch * 16, (start_batch - end_batch) * 16
        core.sets.append(ToneSet(
            channel=channel, set_mu=start_mu, set_s=start_mu / core.f_dac,
            frequency=prev.frequency, phase_turns=0.0, amplitude=amplitude,
            phase_mode=PHASE_MODE_TRACKING, is_filler=True))
        core.events.append(PulseEvent(
            channel=channel, set_index=len(core.sets) - 1, set_mu=start_mu,
            set_s=start_mu / core.f_dac, start_mu=start_mu, dur_mu=dur_mu,
            requested_start_s=start_mu / core.f_dac, requested_dur_s=dur_mu / core.f_dac,
            frequency=prev.frequency, phase_turns=0.0, amplitude=amplitude,
            phase_mode=PHASE_MODE_TRACKING))
        added += 1
    return added


def _plan_phase_chain(core: Core) -> None:
    """Resolve every `set()` into its phase constant, in time order per channel.

    CONTINUOUS chains through the SET events (a set that never played still moved the accumulator),
    and a capture filler is skipped so it cannot silently rewrite the phase history.
    """
    state: dict[int, tuple[int, int]] = {}                   # channel -> (W, C_const)
    for t in sorted(core.sets, key=lambda t: (t.set_mu, t.channel)):
        W = (units.demod_freq_word(t.frequency, core.m.params) if t.is_demod
             else units.freq_word(t.frequency, core.m.params)) % _M32
        p32 = (round(t.phase_turns * 65536) % 65536) << 16
        # The anchor is the UN-ROUNDED instant the user asked for: rounding it to mu would put up
        # to half a DAC sample of time into the carrier phase (1.5 deg at 82 MHz for the
        # 0.4-sample rounding of 105 us). Split it as `set_mu + frac` so the big term stays EXACT
        # integer arithmetic and only the sub-sample remainder goes through a float product.
        # demod phase advances per ADC sample (4 per batch), drive per DAC sample (16 per batch)
        rate = 4 if t.is_demod else 1                          # anchor divisor vs DAC samples
        frac = t.set_s * core.f_dac - t.set_mu                # |frac| <= 0.5 DAC samples
        anchor = (lambda w: round(w * t.set_mu / rate) + round(w * frac / rate)
                  ) if t.is_demod else (lambda w: w * t.set_mu + round(w * frac))  # noqa: E731
        if t.phase_mode == PHASE_MODE_TRACKING:
            c_const = p32
        elif t.phase_mode == PHASE_MODE_ABSOLUTE:
            c_const = p32 - anchor(W)
        else:                                                 # CONTINUOUS
            prev = state.get(t.channel)
            if prev is None:
                raise RuntimeError(
                    f"{_label(t.channel, core.m)}: PHASE_MODE_CONTINUOUS continues a phase that does not exist "
                    "yet — the first set() on a channel must be TRACKING or ABSOLUTE")
            W1, c1 = prev
            c_const = c1 + anchor(W1) - anchor(W) + p32
        t.freq_word = W
        t.phase_const = (c_const if t.is_demod else c_const + W * _TTP16) % _M32
        if not t.is_filler:                                   # a filler leaves the chain untouched
            state[t.channel] = (W, c_const % _M32)


def _chunk_runs(sch: Schedule, e: PulseEvent, usable: int) -> list[tuple[int, int]]:
    """(line, batches) runs for one event, none of which may wrap onto the reserved block.

    A pulse whose first batch is partial starts on its reserved lead line. The channel's HIGHEST
    lead line is followed only by full lines (its own triplet copies, then slots other channels
    reserved — untouched full lines in THIS channel's RAM), so it free-runs the whole pulse as
    one play — the board-verified ion-trap shape. A LOWER lead line would walk onto another
    pattern's line, so it plays a 3-batch triplet prefix (its lead line + its two full copies)
    and the rest replays from the full region 3 batches later — the closest spacing the II=3
    parameter queue (SrlShadow, the deployed impl) pops on time; a closer play pops a cycle late
    and glitches the channel: the RX wrap bug this layout exists to avoid.
    """
    depth = sch.core.m.params.env_depth
    dur_max = (1 << 16) - 1                       # the dur register is 16 bits; 65536 wraps to 0
    tail = 1 if e.tail_line >= 0 else 0
    todo, runs, line = e.dur_batches - tail, [], e.env_line
    # A channel that reserves NO partial line has a uniform envelope RAM, so the free-running
    # reader may wrap as much as it likes: one play covers any duration UP TO the dur field,
    # whatever OTHER channels reserve (Codex F4).
    if not sch.env_lines.get(e.channel):
        while todo > 0:
            n = min(todo, dur_max)
            runs.append((line, n))
            todo -= n
        return runs + ([(e.tail_line, 1)] if tail else [])
    # Triplet prefix for every lead line except the channel's highest (see the docstring).
    if line < sch.first_line and line != max(sch.env_lines[e.channel].values()):
        n0 = min(todo, 3)
        runs.append((line, n0))
        todo -= n0
        line = sch.first_line
    while todo > 0:
        room = min(depth - line, usable, dur_max) # to the top of the RAM; wrapping is not allowed
        n = min(todo, room)
        runs.append((line, n))
        todo -= n
        line = sch.first_line                     # every later run restarts in the full region
        if todo and usable <= 0:
            raise RuntimeError("no full envelope lines left to chunk into")
        if todo:
            room = usable
    if tail:
        runs.append((e.tail_line, 1))
    return runs


def envelope_images(sch: Schedule) -> dict[int, np.ndarray]:
    """Envelope RAM content per channel: reserved partial lines, then the unit envelope everywhere.

    Stored samples are the UNIT envelope — the amplitude lives in the slot's amp register — with
    `lead // step` samples zeroed at the front and `(trail + 1) // step` at the back of a reserved
    line, so a pulse can start and end inside a batch.
    """
    m = sch.core.m
    images = {}
    for ch, keys in sch.env_lines.items():           # keyed by the flat channel id
        local = ch % 3
        spl = m.channel(local).samples_per_line
        step = 16 // spl
        # PACKED lines (uint32 per stored sample), the format riscq.run.write_envelope expects
        full = Pulse(envelopes.square(spl), amp=1.0).packed_lines(m, local)
        img = np.tile(full, (m.params.env_depth, 1))
        for (lead, trail), line in keys.items():
            row = envelopes.square(spl).copy()
            row[: lead // step] = 0
            if trail:
                row[spl - (trail + 1) // step:] = 0
            img[line] = Pulse(row, amp=1.0).packed_lines(m, local)[0]
        images[ch] = img
    return images


# ── code generation: the schedule becomes a RISC-Q kernel ────────────────────────────────────────

_KERNEL_HEAD = '''"""Generated by riscq.artiqapi from a Core timeline — do not edit, regenerate."""
from riscq.lang import Array, ParamTable, kernel


@kernel
def k_sequence({params}):
    """{doc}"""
{pre}{inits}
    t1 = {origin}  # noqa: F821      the sequence origin: a runtime value
{post}'''


def generate_kernel_source(sch: Schedule, lead: int = 8192, doc: str = "", core_index: int = 0,
                           origin: str | None = None) -> str:
    """Straight-line kernel source for hardware core `core_index` — one `set_*`/`play` group per
    event of that core (on a 1-core build: every event). `origin` is the kernel expression of the
    sequence origin t1: the default `now() + lead` anchors a single core on its own clock read; a
    multi-core run passes `run_origin()` (the SoC's reset-release latch, identical on every core)
    and the kernel then also fills a `tele` array — [t1, now at entry, now just before its first
    play, now after each blocking readout] — that run() checks against the schedule. Where a channel
    would hold more than `queue_depth` plays, a `wait_until` queue barrier precedes the push (see
    _queue_barriers)."""
    events = [(idx, e) for idx, e in enumerate(sch.events) if e.channel // 3 == core_index]
    bars = _queue_barriers(sch, core_index)
    depth = sch.core.m.params.queue_depth
    chans = sorted({e.channel for _, e in events})
    nm = {c: ("demod" if c % 3 == 2 else f"ch{c % 3}") for c in chans}
    n_iq = sum(1 for _, e in events if e.is_demod)
    body: list[str] = []
    n_reads = 0
    for idx, e in events:
        n, W = nm[e.channel], _i32(e.freq_word)
        body.append(
            f"    # event {idx}: {e.frequency/1e6:.6f} MHz, phase {e.phase_turns:g} turns, "
            f"{_MODE_NAME[e.phase_mode]}; batches {e.batch}..{e.batch + e.dur_batches - 1}"
            + (f", +{e.lead_zeros} DAC samples in" if e.lead_zeros else "")
            + (f", -{e.trail_zeros} out" if e.trail_zeros else ""))
        # every event — fillers included — programs its carrier registers, so nothing ever plays
        # with an uninitialised frequency (fillers are excluded from the CONTINUOUS phase chain,
        # not from register programming; their amp is 0, so the output is silent either way).
        # The demod carrier advances per ADC sample: 4 per batch, not 16.
        rate = 4 if e.is_demod else 16
        if (idx, 0) in bars:
            # set_freq is itself a queue push (tagged with the channel's start time, which the last fire
            # auto-advanced to its end), so the barrier goes before the event's FIRST write, not the play
            body.append(f"    wait_until(t1 + {bars[(idx, 0)]})  # noqa: F821  queue barrier: the play "
                        f"pushed {depth} plays ago has popped, the queues take this event's writes")
        body.append(f"    set_freq({n}, {W})  # noqa: F821")
        body.append(f"    set_phase({n}, {n}[\"p\"], {_i32(e.phase_const)} - {W} * (t1 * {rate}))"
                    f"  # noqa: F821")
        body.append(f"    set_amp({n}, {n}[\"p\"], {_i32(e.amp_code << 16)})  # noqa: F821")
        at = e.batch
        for j, (line, nb) in enumerate(sch.chunks[idx]):
            if j and (idx, j) in bars:
                body.append(f"    wait_until(t1 + {bars[(idx, j)]})  # noqa: F821  queue barrier: the play "
                            f"pushed {depth} plays ago has popped (chunk {j})")
            body.append(f"    set_env({n}, {n}[\"p\"], {_i32(line << 16)})  # noqa: F821")
            body.append(f"    set_dur({n}, {n}[\"p\"], {_i32(nb << 16)})  # noqa: F821")
            if origin is not None and not any("play(" in b for b in body):
                body.append("    tele[2] = now()  # noqa: F821   armed: about to push the first play")
            body.append(f"    play({n}, {n}[\"p\"], t1 + {at})  # noqa: F821")
            at += nb
        if e.is_demod:
            k = 3 * e.result_slot
            body.append(f"    wait_until(t1 + {e.batch} + {READOUT_LEAD})  # noqa: F821  freshness")
            body.append(f"    out[{k}] = read_res()  # noqa: F821   HALTS until the integral settles")
            body.append(f"    out[{k + 1}] = read_real()  # noqa: F821")
            body.append(f"    out[{k + 2}] = read_imag()  # noqa: F821")
            if origin is not None:
                body.append(f"    tele[{3 + n_reads}] = now()  # noqa: F821   back from the halting read")
                n_reads += 1
    end = max((e.batch + e.dur_batches for _, e in events), default=0)
    body.append(f"    wait_until(t1 + {end + 64})  # noqa: F821")
    params = ", ".join(f"{nm[c]}: ParamTable" for c in chans)
    if n_iq:
        params += ", out: Array"
    pre = post = ""
    if origin is not None:
        params += ", tele: Array"
        pre = "    tele[1] = now()  # noqa: F821   entry: before the pulse-table init\n"
        post = "    tele[0] = t1  # noqa: F821\n"
    head = _KERNEL_HEAD.format(
        params=params, pre=pre, post=post,
        inits="\n".join(f"    init_pulse_params({nm[c]}.pulses)  # noqa: F821" for c in chans),
        doc=doc or "generated sequence", origin=origin if origin is not None else f"now() + {lead}")
    return head + "\n".join(body) + "\n"


def compile_schedule(sch: Schedule, out_dir, lead: int = 8192, doc: str = "", core_index: int = 0,
                     origin: str | None = None):
    """Generate hardware core `core_index`'s kernel, import it, and hand it to RISC-Q's own
    `compile_kernel`.

    Returns `(program, tables, envelope_images, source_path)` — the envelope images are the whole
    schedule's, keyed by flat channel id. The source is written to a real file so the kernel front
    end can read it with `inspect.getsource` — and so a human can read it too.
    """
    import importlib.util
    import sys
    from pathlib import Path

    from riscq.lang import Array as KArray, ParamTable, compile_kernel

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "generated_sequence" if core_index == 0 else f"generated_sequence_core{core_index}"
    path = out_dir / f"{stem}.py"
    path.write_text(generate_kernel_source(sch, lead=lead, doc=doc, core_index=core_index,
                                           origin=origin), encoding="utf-8")

    spec = importlib.util.spec_from_file_location(f"riscq_{stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    m = sch.core.m
    events = [e for e in sch.events if e.channel // 3 == core_index]
    tables = {}
    for ch in sorted({e.channel for e in events}):
        local = ch % 3
        spl = m.channel(local).samples_per_line
        first = next(e for e in events if e.channel == ch)
        # the table should DESCRIBE the pulse: carry the real amplitude, not a placeholder, so the
        # value `init_pulse_params` loads is already the right one
        name = "demod" if local == 2 else f"ch{local}"
        tables[name] = ParamTable(local, first.frequency,
                                  {"p": Pulse(envelopes.square(spl), amp=first.amplitude)})
    n_iq = sum(1 for e in events if e.is_demod)
    bindings = {"out": KArray(3 * n_iq)} if n_iq else {}
    if origin is not None:
        bindings["tele"] = KArray(3 + n_iq)
    return (compile_kernel(mod.k_sequence, m, tables=tables, **bindings), tables,
            envelope_images(sch), path)


# ── the host runner: build -> compile -> upload -> run -> results ────────────────────────────────

@dataclass
class CoreResult:
    """What one hardware core produced: its raw trace (when it had an adc gate) and its IQ results."""
    trace: np.ndarray | None = None       # int32 ADC samples over the gate window (None: no gate)
    t: np.ndarray | None = None           # seconds, relative to the gate start
    gate_start_mu: int = 0
    res: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))   # hardware sign bits
    real: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))  # integrals
    imag: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    tele: np.ndarray | None = None        # multi-core runs: [t1, now entry, now armed, now after reads]

    @property
    def iq(self) -> np.ndarray:
        return self.real.astype(np.float64) + 1j * self.imag.astype(np.float64)


@dataclass
class RunResult:
    """Everything one run produced, per hardware core (`cores[k]`), in physical units where possible.
    `origin` is the batch-time origin t1 a multi-core run's kernels shared; `trace`/`t`/
    `gate_start_mu`/`res`/`real`/`imag`/`iq` are core 0's (the whole result on a single-core build)."""
    schedule: Schedule
    fs: float
    cores: dict[int, CoreResult] = field(default_factory=dict)
    origin: int | None = None             # the shared t1 of a multi-core run (from the cores' telemetry)

    def _core0(self) -> CoreResult:
        return self.cores.get(0) or CoreResult()

    @property
    def trace(self): return self._core0().trace
    @property
    def t(self): return self._core0().t
    @property
    def gate_start_mu(self): return self._core0().gate_start_mu
    @property
    def res(self): return self._core0().res
    @property
    def real(self): return self._core0().real
    @property
    def imag(self): return self._core0().imag
    @property
    def iq(self): return self._core0().iq


def _sdelta(a: int, b: int) -> int:
    """a - b on the 32-bit batch clock, as the signed difference the hardware's due tests use."""
    return (int(a) - int(b) + (1 << 31)) % _M32 - (1 << 31)


def _check_telemetry(sch: Schedule, k: int, tele, origins: dict[int, int]) -> None:
    """The loud checks on one core's `tele` (see generate_kernel_source): the origin it used is the
    one every other core used, its first play was pushed >= LEAD batches before it was due, and every
    play pushed after a halting readout still had its lead."""
    tele = [int(v) & 0xFFFFFFFF for v in tele]
    t1, now_entry, now_armed = tele[:3]
    origins[k] = t1
    if len(set(origins.values())) > 1:
        raise RuntimeError(f"cores disagree on the run origin: {origins} — the run_origin latch is "
                           "not shared, the run is invalid")
    events = [e for e in sch.events if e.channel // 3 == k]
    if not events:
        return
    first = min(e.batch for e in events)
    slack = _sdelta(t1 + first, now_armed)
    if not LEAD <= slack < (1 << 31):
        raise RuntimeError(
            f"core {k}: its first play (batch {first}) was pushed only {slack} batches before it was "
            f"due (>= {LEAD} needed) — the reset-release lead does not cover this kernel's boot + "
            "setup; the run is invalid")
    if _sdelta(now_armed, now_entry) < 0:
        raise RuntimeError(f"core {k}: telemetry out of order (entry {now_entry}, armed {now_armed})")
    reads = [e for e in events if e.is_demod]
    for j, d in enumerate(reads):
        later = [e.batch for e in events if e.batch > d.batch]
        if later:
            slack = _sdelta(t1 + min(later), tele[3 + j])
            if slack < LEAD:
                raise RuntimeError(
                    f"core {k}: after the readout at batch {d.batch} the next play (batch "
                    f"{min(later)}) had only {slack} batches of lead left (>= {LEAD} needed)")


def run(drv, core: Core, work_dir, doc: str = "", max_run: int | None = None) -> RunResult:
    """Plan, compile, upload and execute the recorded timeline; fetch what it measured.

    Replaces the hand-written experiment plumbing: rq.setup, the envelope images (including the
    demod matched filter), the liveness gate on hardware, rerun with the results arrays, and the
    trace reads where adc gates exist. One kernel per hardware core. A multi-core run anchors every
    core on the SoC's shared run origin (`run_origin()`: the batch time latched at the reset release
    plus a fixed lead — the cores' own `now()` reads are up to ~2 batches apart, tens of degrees of
    relative carrier phase) and checks each core's telemetry afterwards (same origin everywhere,
    every play pushed with its lead)."""
    from riscq import run as rq
    from riscq.map import ADC_BATCH

    m = core.m
    sch = plan(core, max_run=max_run)
    env_images = envelope_images(sch)
    hw_cores = sorted({e.channel // 3 for e in sch.events} | {tg.core_index for tg in core.trace_gates})
    if not hw_cores:
        raise RuntimeError("nothing to run: the timeline has no pulses and no gates")
    shared = len(hw_cores) > 1
    if shared and not m.params.run_origin:
        raise RuntimeError(
            f"this timeline spans hardware cores {hw_cores} but the build has no shared run origin "
            "(run_origin is off): the cores' own clock reads would skew the timeline — build with "
            '"run_origin": true')
    origin = "run_origin()" if shared else None
    progs = {}
    for k in hw_cores:
        progs[k], _tables, _imgs, _src = compile_schedule(sch, work_dir, doc=doc, core_index=k,
                                                          origin=origin)
    total = max((e.batch + e.dur_batches) for e in sch.events) if sch.events else 0

    rq.setup(drv, m, progs)
    for ch, img in env_images.items():
        rq.write_envelope(drv, m, ch // 3, ch % 3, 0, img)

    if not hasattr(drv, "sim"):           # hardware only: never touch a dead dsp domain
        import time
        h0, d0 = drv.read32(m.host_ctrl + 0x104), drv.read32(m.host_ctrl + 0x100)
        time.sleep(0.01)
        h1, d1 = drv.read32(m.host_ctrl + 0x104), drv.read32(m.host_ctrl + 0x100)
        if h1 == h0 or d1 == d0:
            raise RuntimeError(f"liveness gate FAILED (hostAlive {h0:#x}->{h1:#x}, "
                               f"dspAlive {d0:#x}->{d1:#x}) — not touching the dsp domain")

    results = rq.rerun(drv, m, progs, timeout=(total + 20000) * 4 + 20_000_000)

    out = RunResult(schedule=sch, fs=ADC_BATCH * m.params.dsp_freq_hz)
    origins: dict[int, int] = {}
    for k in hw_cores:
        cr = out.cores[k] = CoreResult()
        if shared:
            cr.tele = np.asarray(results[k]["tele"], dtype=np.int64) & 0xFFFFFFFF
            _check_telemetry(sch, k, cr.tele, origins)
            out.origin = origins[k]
        if "out" in results[k]:
            raw = np.asarray(results[k]["out"], dtype=np.int64)
            cr.res, cr.real, cr.imag = (raw[0::3].astype(np.int32), raw[1::3], raw[2::3])

    for tg in core.trace_gates:
        cr = out.cores[tg.core_index]
        nbytes, chunk, parts = (m.rob_width // 8) * tg.batches, 128 * 1024, []
        for off in range(0, nbytes, chunk):
            parts.append(drv.read_block(m.robs(tg.core_index) + off, min(chunk, nbytes - off)))
        cr.trace = np.frombuffer(b"".join(parts), dtype=m.rob_dtype).astype(np.int32)
        cr.t = np.arange(cr.trace.size) / out.fs
        cr.gate_start_mu = tg.batch0 * 16
    core.last_result = out
    return out
