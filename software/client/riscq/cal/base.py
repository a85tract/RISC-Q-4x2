"""Shared calibration infrastructure (spec 06 §1, spec 08 §6 batched cut-over).

Every calibration runs its whole sweep as ONE batched run on the core (riscq.cal.kernels): the kernel
COMPUTES the swept knob on-core from a scalar sweep descriptor (a Q16 or int pair — spec 09; the input
Arrays are gone), walking it on a FIXED-period grid whose idle head is the T1 relax reset (the model
has no auto-reset, spec-M4 B0), so every readout lands at the same time-referenced demod-LO phase. The
demod carrier's discrimination phase (measured by ReadoutCalibration) is baked into the readout tables;
firing the demod IS the readout (its `dur` is the integration window). This module holds the batch
composers:
  - batches / seconds / relax_batches — the physical-units boundary (spec 13 §2: the Config and the
    calibration knobs are Hz / seconds / normalized amp; codes and batches are derived HERE),
  - gate_pulse / prep / x90_vz — the Config's own gate envelopes, qcal's two |1> preps (spec 13 §4)
    and the X90's virtual-Z frame bracket (spec 13 §7),
  - readout_tables / demod_table — the ro-drive + demod-carrier table pair (amp / env / phase / delay),
  - grid_period / batch_timeout — the host-computed fixed period + poll timeout,
  - sweep_q16 — the (x0q, dxq) Q16 sweep descriptor + the exact int x-axis the kernel realizes,
  - population / sweep_counts / rerun_counts — a params-only counts-mode sweep/rerun of ALL requested
    cores in one run, returning {q: |1> population P} (spec 13 §8 simultaneous multi-qubit),
  - acquire_shots — a per-prep raw-mode capture of all cores, returning {q: one prep state's IQ shots}.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from riscq import run as rq
from riscq.lang import ParamTable
from riscq.map import LEAD, READOUT_LEAD, READOUT_MAX_WIN_LOG2, pack16
from riscq.pulses import Pulse, envelopes, golden, units

# The PHYSICAL gap (batches) between a sequence's last drive pulse and its readout window — every
# kernel schedules its prep to end at `t_ro − SEP`. NOT a posting lead: every play is posted a full
# grid period (≥ the relax head) before its startTime, so the post→startTime contract (LEAD) never
# bounds pulse spacing (spec 16 §1.1). What SEP does is decay the excited prep for its duration —
# at SEP = LEAD = 96 that cost 1.7–1.9× of cluster SNR at the co-sim's T1 = 600 batches, the whole
# of spec 15's D2 deficit. The one place a real scheduling lead belongs after a pulse is the
# core-halting herald read, which names LEAD explicitly (`_herald_extra` / `herald_offset`).
SEP = 8                 # smallest clean prep→readout gap (spec 16 S1 sweep: 4 clips the window head)
GATE_ENV = envelopes.square(16)   # default gate envelope (4 batches at gateInterp 4)
GATE_CH = 0                       # the gate channel (the x90/x pulse table's channel)
X90, X = 0, 1                     # the kernels' compile-time `prep_gate` binding (spec 13 §4)

# ── deep-gate-train pacing (spec 14 F1) ──
# Every PulseGenerator parameter sits behind a depth-4 TimedQueue on the posted RF link, and a push
# into a FULL queue is silently DROPPED (no backpressure reaches the core), so a train may never have
# more than TRAIN_AHEAD gates scheduled ahead of the one now playing. An UNPACED train therefore
# plays its first 4 gates and drops every one after — measured in co-sim, exactly 4 of 96.
#
# Pacing alone is not enough: the core must also SUSTAIN one gate's pushes per train step, or the
# params land after their pulse has begun and the gate plays TRUNCATED. Measured on the real kernels
# at ngates=96 (co-sim gate DAC, tests/test_deep_trains): a step of 30 batches truncates ~56% of the
# gates, 32 is the first clean step, so TRAIN_STEP carries a margin over that.
#
# For a 35 ns X6Y3 X90 (18 batches) the step therefore leaves an IDLE GAP between gates — a
# documented deviation from qcal's back-to-back train. It costs wall time, not physics: the ladder's
# amplified rotation is unchanged, and the extra decoherence is a UNIFORM contrast factor across the
# sweep (every point sits on the same fixed grid), which leaves the parabola vertex — the calibrated
# amplitude — where it was. Pulses at or above TRAIN_STEP (the 70 ns X, the 400 ns CZ) are already
# slow enough and stay back-to-back.
TRAIN_AHEAD = 4         # gates scheduled ahead of the playing one (the depth-4 queue contract)
TRAIN_STEP = 36         # min batches per train gate: the core's sustained push rate (32) + margin


def train_step(dur_batches: int) -> int:
    """The grid step of an n-gate train: the pulse's own length when it is long enough for the core to
    keep up, else TRAIN_STEP (an idle gap follows each gate). `TRAIN_AHEAD * step` is also the fire's
    scheduling lead, which stays ≥ LEAD for every step this returns."""
    return max(int(dur_batches), TRAIN_STEP)


def qubits_list(qubits) -> list:
    """A calibration's `qubits` argument (spec 13 §8): a bare int is one qubit, so single-qubit
    callers keep working. Every cal compiles one program per core and issues ONE run over all of
    them (simultaneous readout is a different measurement from serial — crosstalk, converter
    summing), then fits each core on the host."""
    return list(qubits) if isinstance(qubits, (list, tuple)) else [int(qubits)]


# ── physical units → batches (spec 13 §2: the Config and the cal knobs are in Hz / seconds / amp) ──

def batches(t_s: float, m) -> int:
    """Seconds → batches (dsp cycles) for this build. THE conversion the calibrations use: every
    duration in the Config and in a calibration's constructor is seconds, and becomes batches here."""
    return units.batches(float(t_s) * 1e9, m.params)


def seconds(n_batches: float, m) -> float:
    """Batches → seconds (a fitted decay/window reported back into the Config); fractional batches
    are kept — a fitted τ is not an integer."""
    return units.ns(float(n_batches), m.params) * 1e-9


def relax_batches(cfg, m) -> int:
    """The per-point idle head before the drive, from `reset/relax` (seconds, qcal's
    reset/passive/delay — 500 us on X6Y3; the co-sim keeps a short one in its own config). It is ≫ T1,
    so the qubit relaxes back to |0> at every grid slot (the model has no auto-reset, spec-M4 B0)."""
    return batches(cfg["reset/relax"], m)


def res_sign(cfg, q: int) -> int:
    """`readout/{q}/res_sign` (±1, default +1): which |0>/|1> assignment the hardware `res` bit
    carries under the config's demod phase. +1 is ReadoutCalibration's convention (|0> on +real, so
    res=1 ⇒ |1>); −1 means the clusters land the other way round and every counts population is
    inverted — `population()` folds it, so the sweeps stay P(|1>) either way."""
    s = int(cfg.get(f"readout/{q}/res_sign", 1))
    assert s in (1, -1), f"readout/{q}/res_sign must be +1 or -1, got {s}"
    return s


def heralding(cfg) -> bool:
    """`readout/herald` (default off): whether the counts kernels post-select each shot on a
    pre-sequence herald read finding the qubit in |0> (spec 13 §8). qcal's `from_qcal` carries it;
    the co-sim configs leave it off, so the herald code folds away and the run is byte-identical."""
    return bool(cfg.get("readout/herald", False))


def demod_table(n: int, phase: float = 0.0, env=None, amp: float = 1.0) -> ParamTable:
    """The demod-carrier pulse on channel 2 — the readout demod is a plain drive channel fed to
    the carrier-triggered decoder, programmed/played with the generic init_pulse_params/set_freq/play.
    Its carrier freq is set separately via set_freq with an ADC-rate demod_freq_to_code code, so the
    table freq is unused (0). Played per shot; its `dur` (n batches) IS the readout integration window
    (n <= env_depth and the decoder's no-overflow cap 2^READOUT_MAX_WIN_LOG2). `env`/`amp` are the
    window's shape and scale (the Config's demod envelope weights the integral). `phase` is the
    discrimination-carrier phase (rad) baked into the slot — the programmable discriminator knob
    (spec 08 §2.1): it rotates the decoder IQ so `sign(sumR)` separates |0>/|1> (a compiled-in
    write_slot("phase")); ReadoutCalibration measures it."""
    assert n <= (1 << READOUT_MAX_WIN_LOG2), \
        f"demod window {n} exceeds the decoder no-overflow cap {1 << READOUT_MAX_WIN_LOG2} batches"
    return ParamTable(2, 0.0, {"sq": Pulse(envelopes.square(n) if env is None else env,
                                           amp=amp, phase=phase)})


def _channel_env(cfg, path: str, n_batches: int, channel: int, m):
    """The envelope a Config names for `channel`, built on that channel's stored-sample grid:
    `{path}/env` (a qcal env name, default square) shaped by `{path}/kwargs`."""
    ch = m.channel(channel)
    return envelopes.build(cfg.get(f"{path}/env", "square"), n_batches * ch.samples_per_line,
                           ch.samples_per_line * m.params.dsp_freq_hz,
                           **cfg.get(f"{path}/kwargs", {}))


def readout_tables(cfg, q: int, m, phase: float | None = None, win: float | None = None):
    """The (channel-1 readout drive, channel-2 demod carrier) table pair for qubit q's readout, plus
    the demod carrier code, the integration window and the demod delay (batches). All Config values
    are physical (spec 13 §3):

      readout/{q}/{freq, amp, dur, env, kwargs}  — the measurement tone: `amp` is its power (X6Y3:
          0.015–0.052), `dur` its LENGTH in seconds (not the window);
      readout/{q}/demod/{dur, delay, phase, env, kwargs} — the carrier-triggered decoder: firing the
          demod IS the readout, its `dur` is the integration window and it opens `delay` seconds after
          the drive starts (the ADC round trip, 0.34–0.75 us on X6Y3; 0 in co-sim). `phase` is the
          discrimination knob (see demod_table).

    `phase`/`win` override the config's demod phase / window (Window compiles the tables at the
    longest candidate window, then retunes the slot's `dur` field per point — spec 08 §4).
    NOTE (co-sim): the projective model emits its readout tone only while the DRIVE is on, so a co-sim
    config must size `readout/{q}/dur` to cover delay + window.
    Returns (ro_table, demod_table, demod_code, win_batches, delay_batches)."""
    ro_freq = float(cfg[f"readout/{q}/freq"])
    drive = batches(cfg[f"readout/{q}/dur"], m)
    n = batches(cfg[f"readout/{q}/demod/dur"] if win is None else win, m)
    delay = batches(cfg.get(f"readout/{q}/demod/delay", 0.0), m)
    ph = float(cfg.get(f"readout/{q}/demod/phase", 0.0)) if phase is None else float(phase)
    ro = ParamTable(1, ro_freq, {"meas": Pulse(_channel_env(cfg, f"readout/{q}", drive, 1, m),
                                               freq_hz=ro_freq, amp=float(cfg[f"readout/{q}/amp"]))})
    demod = demod_table(n, ph, _channel_env(cfg, f"readout/{q}/demod", n, 2, m),
                        float(cfg.get(f"readout/{q}/demod/amp", 1.0)))
    return ro, demod, units.demod_freq_to_code(ro_freq, m.params), n, delay


def _herald_extra(delay: int = 0) -> int:
    """The batches a heralded shot adds to the grid period (spec 13 §8): the pre-sequence herald read
    (`delay + READOUT_LEAD`, its window covered by READOUT_LEAD) PLUS a full LEAD of scheduling lead
    after it — the herald `read_res()` HALTS the core, so the following drive is posted only after
    the read returns and genuinely needs the post→startTime lead, or it drops (the qubit never
    rotates, the measurement reads |0>). This is the one post-pulse LEAD in the schedule; it is a
    posting constraint, not a physical gap, so it does NOT shrink with SEP (spec 16 §1.1)."""
    return int(delay) + READOUT_LEAD + LEAD


def herald_offset(seq_batches: int, delay: int = 0) -> int:
    """`hoff` = t_ro − t_h: place the herald read so it COMPLETES a full LEAD before the earliest
    sequence pulse (`t_ro − SEP − seq_batches`) — the read halts the core, so the drive scheduled
    right after it is posted late and needs the post→startTime lead, or it drops (spec 13 §8).
    Derived so `(t_ro − SEP − seq_batches) − (t_h + delay + READOUT_LEAD) == LEAD`."""
    return int(seq_batches) + int(delay) + READOUT_LEAD + SEP + LEAD


def grid_period(relax: int, seq_batches: int, dur: int, delay: int = 0, herald: bool = False) -> int:
    """The fixed readout-grid period (batches) for a batched sweep (spec 08 §2.2): each shot fires on
    this grid, whose idle head is the T1 relax reset, so every readout lands at the same demod-LO
    phase. `seq_batches` is the longest point's drive prelude (earliest pulse start -> t_ro), `delay`
    the demod's round-trip offset. Sized period >= relax + seq_batches + LEAD + delay + dur +
    READOUT_LEAD and rounded UP to a multiple of 8 (so the phase repeats). The internal `+ LEAD` is
    the POSTING MARGIN, not a physical gap: the kernels open on `t_ro = now() + period`, so the
    first prep's post→startTime margin is `period − seq ≥ relax + LEAD + <readout tail>`, and the
    same term is what lets the core re-post each next shot's params after `read_res` returns when
    `relax` is short (the L1/L2 probe configs run with relax ≈ 0 — at SEP = LEAD the old `+ SEP`
    covered this by accident; spec 16 S1 found it the hard way when the gap shrank). `herald`
    prepends one more readout window + its LEAD posting lead (the pre-sequence herald read, spec 13
    §8) — the relax gap is preserved (it grows by `dur`). Host-computed, fail-loud."""
    need = int(relax) + int(seq_batches) + LEAD + int(delay) + int(dur) + READOUT_LEAD
    if herald:
        need += _herald_extra(delay)
    period = -(-need // 8) * 8
    assert period % 8 == 0 and period >= need
    return period


def batch_timeout(nbatches: int) -> int:
    """poll_done timeout (host/sim cycles) for a batch of `nbatches` grid batches: 4 cycles/batch +
    boot slack (mirrors tests/test_batch)."""
    return int(nbatches) * 4 + 20_000_000


def sweep_q16(x0: int, x1: int, n: int, fold: bool = False) -> tuple[int, int, np.ndarray]:
    """(x0q, dxq) kernel params + the EXACT int x-axis the kernel realizes (host-mirrored
    integer arithmetic, not the float linspace). Asserts every point is a legal 16-bit code —
    for an AMPLITUDE or a phase ramp that assert is an overflow guard and stays on.

    `fold=True` is for a FREQUENCY ramp that runs past the DAC's half rate (a code past ±2^15 —
    a bring-up scan that starts below 4 GHz and ends above it, X6Y3's readout band): there the
    16-bit range is not a bound at all. The kernel's accumulator is a plain int32 (`-fwrapv`), so
    its wrap IS the fold the phase accumulator does in hardware, and the register keeps the code
    mod 2^16 — exactly the aliasing `units._freq_code` applies to a single tone. The axis returned
    stays UNFOLDED: it is the frequency ramp the caller asked for, and the frequency callers map it
    back to Hz as a delta from their own first point (so it never jumps a whole `fs` at the fold).
    What still fails loud is a scan wider than one full turn, which would alias onto itself."""
    # This is a 16-BIT-CODE sweep — the realized value is `q >> 16`. Amplitude / phase / time knobs
    # are 16-bit at every build width; a FREQUENCY ramp is not (a freq_width-32 build reads the whole
    # word), so frequency callers use `sweep_freq`, which mirrors the build's own width.
    dxq = 0 if n <= 1 else round(((x1 - x0) << 16) / (n - 1))
    xs = (x0 * (1 << 16) + np.arange(n, dtype=np.int64) * dxq) >> 16
    if fold:
        assert abs(x0) < (1 << 15), f"the sweep must START on a legal 16-bit code, got {x0}"
        assert int(xs.max() - xs.min()) < (1 << 16), \
            "a folded sweep may not cover more than one full turn of the band"
    else:
        assert np.all(np.abs(xs) < (1 << 15))
    return x0 << 16, dxq, xs.astype(int)


def sweep_freq(f0_hz: float, f1_hz: float, n: int, m) -> tuple[int, int, np.ndarray]:
    """A FREQUENCY ramp for the on-core accumulator at THIS build's word width: the kernel params
    (x0q, dxq) — `set_freq(ch, x0q + i*dxq)` raw, as k_vna / k_cz_pop do — and the Hz each point
    REALIZES, in the caller's own Nyquist zone (the x axis; a value written back to the config
    re-derives the same register word bit-for-bit).

    x0q is the seated word of f0 (16-bit build: code << 16; 32-bit: the SF(32) word), so its LSB
    weighs fs / 2^32 at either width and the ramp arithmetic is the same; what differs is what the
    register KEEPS — the top 16 bits (the code) or the whole word — and the axis mirrors exactly
    that. The kernel's int32 accumulator wraps like the hardware's phase accumulator, so the ramp
    may run through the band edge unfolded; wider than a full turn it would alias onto itself."""
    fw, fs = m.params.freq_width, units.sample_rate(m.params)
    assert abs(f0_hz) < fs and abs(f1_hz) < fs, \
        f"sweep endpoints {f0_hz:g}..{f1_hz:g} Hz must lie within one sample rate ({fs:g} Hz) of DC — " \
        "the range units.freq_word accepts, so every axis value converts back"
    w0 = round(f0_hz / fs * (1 << fw)) << (32 - fw)              # UNWRAPPED: keeps the Nyquist zone
    span = round((f1_hz - f0_hz) / fs * (1 << 32))
    assert abs(span) < (1 << 32), "a frequency sweep may not cover more than one full turn of the band"
    dxq = 0 if n <= 1 else round(span / (n - 1))
    q = w0 + np.arange(n, dtype=np.int64) * dxq                  # the accumulator, unwrapped
    kept = (q >> (32 - fw)) << (32 - fw)                         # what the register keeps at this width
    assert np.abs(kept).max() < (1 << 32), \
        "a point of the realized ramp rounds to the sample rate itself, which no register word holds"
    assert int(kept.max() - kept.min()) < (1 << 32), \
        "the realized ramp (the rounded step times n-1) covers a full turn: its ends alias onto each other"
    # the kernel params are int32: the accumulator wraps mod 2^32 exactly like the hardware's, so the
    # wrapped increment realizes the same points (dxq = 2^31 for a half-band sweep in two points)
    return int(units._wrap_signed(w0, 32)), int(units._wrap_signed(dxq, 32)), kept * fs / (1 << 32)


def gate_sigma(m, pulse: Pulse, carrier_hz: float, amp_code: int) -> float:
    """Σ over the pulse of the per-batch drive estimate amp_est = √(2·mean(sample²)) on the
    bit-exact DAC golden — exactly what TwoLevelModel integrates, so θ = rabi_rad_per_amp·gate_sigma
    is the qubit's rotation angle. Linear in amp_code (used to convert a fitted Rabi rate)."""
    lines = pulse.packed_lines(m, 0)   # gate channel
    w = golden.pulse_window(lines, int(amp_code), units._freq_code(carrier_hz, m.params),
                            0, 0, len(lines))
    return float(sum(math.sqrt(2 * np.mean(row.astype(float) ** 2)) for row in w))


def socmap(drv):
    """Derive the build's SocMap from the driver — the co-sim exposes its SocParams JSON on
    `drv.sim`, the board server on `drv.board` (RemoteDriver deliberately has no `.sim`, spec 10 §6)."""
    from riscq.map import SocMap, SocParams
    src = drv.sim if hasattr(drv, "sim") else drv.board
    return SocMap(SocParams.from_json(src.get_params()))


@dataclass
class Result:
    """A calibration outcome (spec 06 §1; multi-qubit per spec 13 §8). `data` and `fit` are PER-QUBIT
    dicts keyed by the qubit label (`data[q]["x"]`, `fit[q]`) — a cal runs every requested qubit
    simultaneously in one run; a single-qubit cal is just the one-key case. `proposal` is a single
    merged `{config path → value}` dict (the paths already carry `q`, so there is no collision).
    `oks` is the per-qubit verdict (`ok` = all of them); `.apply()` writes the passing qubits'
    proposals; `.plot()` draws one qubit's sweep (or all)."""

    ok: bool
    data: dict
    fit: dict
    proposal: dict = field(default_factory=dict)
    cfg: object = None
    label: str = ""
    oks: dict = field(default_factory=dict)   # per-qubit ok; empty = one all-qubit verdict (legacy)

    def apply(self):
        """Write the proposal into the Config. With per-qubit `oks`, a failed qubit's paths are
        skipped and the rest still apply (qcal writes per qubit too — one drifted resonator must not
        veto the other seven); it refuses outright only when NO qubit passed."""
        good = {str(q) for q, v in self.oks.items() if v}
        if not self.ok and not good:
            raise RuntimeError(f"{self.label}: fit failed, refusing to update config")
        for path, value in self.proposal.items():
            if self.ok or path.split("/")[1] in good:
                self.cfg[path] = value
        return self.cfg

    def plot(self, ax=None, q=None):
        import matplotlib.pyplot as plt          # lazy: headless CI must import cal without it
        qs = [q] if q is not None else list(self.data)
        if ax is None:
            _, axes = plt.subplots(len(qs), squeeze=False)
            axes = list(axes.ravel())
        else:
            axes = [ax] * len(qs)
        for a, qi in zip(axes, qs):
            x, y = self.data[qi].get("x"), self.data[qi].get("y")
            if x is not None and y is not None:
                a.plot(x, y, "o-")
            a.set_title(f"{self.label} q{qi} ({'ok' if self.ok else 'FAILED'})")
        return axes[0] if len(qs) == 1 else axes


# ── gate pulses and the |1> prep, from the Config tree (spec 13 §4) ──

def qubit_freq(cfg, q: int) -> float:
    return float(cfg[f"qubit/{q}/freq"])


def gate_pulse(cfg, q: int, m, name: str = "x90") -> Pulse:
    """The Config's own gate pulse `qubit/{q}/{name}` ('x90' or 'x'): its envelope (`env` + `kwargs`,
    built on the gate channel's stored-sample grid and `dur` seconds long — the X6Y3 FAST_DRAG), its
    normalized `amp` and its own axis `phase`. A config that names no envelope (the co-sim's) gets the
    default square gate."""
    path = f"qubit/{q}/{name}"
    env = GATE_ENV if f"{path}/env" not in cfg \
        else _channel_env(cfg, path, batches(cfg[f"{path}/dur"], m), GATE_CH, m)
    return Pulse(env, freq_hz=qubit_freq(cfg, q), amp=float(cfg[f"{path}/amp"]),
                 phase=float(cfg.get(f"{path}/phase", 0.0)))


def prep(cfg, q: int, m, gate: str = "X90") -> tuple[ParamTable, int, int]:
    """qcal's two |1> preps (readout.py:129-132) → (gate table, the kernels' compile-time `prep_gate`
    binding, the prep's length in batches):

      'X90' → TWO X90 plays (one `play` + one bare `fire`; B0's startTime auto-advance makes the train
              contiguous, the same trick as k_rabi's n-gate train);
      'X'   → ONE play of the config's own X pulse (`qubit/{q}/x/*`) — on X6Y3 a double-LENGTH,
              same-amplitude FAST_DRAG, which is why the old "π = 2× the X90 amp" synthesis is gone.

    The table carries both slots when the config defines an X; the kernel's `prep_gate` fold picks one
    and the dead branch is eliminated before slot resolution, so an X90 prep still compiles on a config
    that has no X."""
    assert gate in ("X90", "X"), f"prep gate must be 'X90' or 'X', got {gate!r}"
    pulses = {"x90": gate_pulse(cfg, q, m)}
    if f"qubit/{q}/x/amp" in cfg:
        pulses["x"] = gate_pulse(cfg, q, m, "x")
    table = ParamTable(GATE_CH, qubit_freq(cfg, q), pulses)
    if gate == "X":
        assert "x" in pulses, f"prep gate 'X' needs qubit/{q}/x/* in the config"
        return table, X, pulses["x"].dur_batches(m, GATE_CH)
    return table, X90, 2 * pulses["x90"].dur_batches(m, GATE_CH)


def x90_vz(cfg, q: int) -> dict:
    """The X90's virtual-Z frame bracket (spec 13 §7) as the two SEATED phase words every X90-playing
    kernel binds: `compile_kernel(..., **x90_vz(cfg, q))`.

    qcal's X90 is virtualz(vz0) · FAST_DRAG · virtualz(vz1) — the pair `qubit/{q}/x90/vz` = [before,
    after] (rad) that Phase calibrates — so a play is `set_phase_offset(frame + vz0); play; frame +=
    vz0 + vz1`. The kernels need `vz0` and the frame step `vzsum` = vz0 + vz1; both are seated words
    (spec 12), summed in the code domain so they wrap mod one turn like every other phase. A config
    with no pair (every co-sim one) gets [0, 0] and the bracket is a no-op."""
    v0, v1 = cfg.get(f"qubit/{q}/x90/vz", [0.0, 0.0])
    c0, c1 = units._phase_code(float(v0)), units._phase_code(float(v1))
    return {"vz0": pack16(c0), "vzsum": pack16(c0 + c1)}


def ef_vz(cfg, q: int, name: str = "x90") -> dict:
    """The EF gate's own virtual-Z frame bracket — `x90_vz`'s EF twin (spec 14 §3 finding 6), bound as
    `evz0`/`evzsum` so an EF kernel can carry BOTH brackets at once (the GE prep's vz0/vzsum and this
    one) without the names colliding: `compile_kernel(..., **x90_vz(cfg, q), **ef_vz(cfg, q))`.

    qcal's EF X90 is the same virtualz(vz0) · FAST_DRAG · virtualz(vz1) triplet as the GE one — the
    pair `qubit/{q}/EF/{name}/vz` that `EFPhase` calibrates — so every EF gate in its EF
    Amplitude/Frequency/Phase circuits plays it, and the config of record carries a non-trivial pair
    on all 8 qubits (q2 −0.163 rad, q6 −0.133). The EF X is a bare FAST_DRAG with no pair, so
    `name='x'` finds no key and folds to a no-op — as does every co-sim config."""
    v0, v1 = cfg.get(f"qubit/{q}/EF/{name}/vz", [0.0, 0.0])
    c0, c1 = units._phase_code(float(v0)), units._phase_code(float(v1))
    return {"evz0": pack16(c0), "evzsum": pack16(c0 + c1)}


# ── EF-subspace gate table (spec two-qubit/01 §4.1) ──

def ef_pulse(cfg, q: int, m, name: str = "x90") -> Pulse:
    """The Config's own EF gate pulse `qubit/{q}/EF/{name}` ('x90' or 'x' — the X6Y3 EF X the
    sandwich CZ shelves with, spec 04 §1) — its envelope (`env` + `kwargs`, default square) and
    normalized `amp`/`phase`, BASEBAND (freq_hz=None): the EF carrier is programmed at runtime by
    set_freq, not baked into the pulse, so it can share the gate channel with the GE prep."""
    path = f"qubit/{q}/EF/{name}"
    env = GATE_ENV if f"{path}/env" not in cfg \
        else _channel_env(cfg, path, batches(cfg[f"{path}/dur"], m), GATE_CH, m)
    return Pulse(env, amp=float(cfg[f"{path}/amp"]), phase=float(cfg.get(f"{path}/phase", 0.0)))


def ef_table(cfg, q: int, m, name: str = "x90") -> tuple[ParamTable, int, int]:
    """The gate table + SEATED carrier words for the EF calibrations → (table, ge_freq, ef_freq). The
    table carries the GE prep X90 (slot "x90") and the EF gate `name` (slot "ef" — the X90 by
    default, the config's EF X for the π-amplitude cal), BOTH baseband so the kernel drives each
    segment at its own carrier: `ge_freq` = the config's GE carrier, `ef_freq` its EF carrier
    (`qubit/{q}/EF/freq`). ONE NCO per channel, retuned between segments (spec 01 §4.1)."""
    ge = gate_pulse(cfg, q, m)                                   # the config GE X90
    ge = Pulse(ge.env, amp=ge.amp, phase=ge.phase)              # strip freq_hz → baseband
    table = ParamTable(GATE_CH, qubit_freq(cfg, q), {"x90": ge, "ef": ef_pulse(cfg, q, m, name)})
    return (table, units.freq_to_code(qubit_freq(cfg, q), m.params),
            units.freq_to_code(float(cfg[f"qubit/{q}/EF/freq"]), m.params))


# ── batched sweeps: one run is the whole sweep (spec 08 §2, §6; spec 09 computed knobs) ──

def population(counts, shots: int, sign: int = 1) -> np.ndarray:
    """The |1> population from the hardware-classified counts: P = counts/shots under the +1 res-sign
    convention, 1 − counts/shots under −1 (spec 13 §3 — the sign is CONSUMED, not just proposed)."""
    p = np.asarray(counts, dtype=float) / shots
    return p if sign > 0 else 1.0 - p


def population_heralded(out, sign: int = 1) -> np.ndarray:
    """The |1> population from a HERALDED counts run (spec 13 §8): `out` is the interleaved
    (count, kept) pairs the kernel writes per point — count = classified |1>s, kept = shots that
    passed the pre-sequence herald (qubit found in |0>). P = count/kept (self-normalised over the
    post-selected shots), res-sign folded. A point that rejected every shot (kept == 0, never on a
    clean |0> qubit) reports 0.5 rather than dividing by zero."""
    o = np.asarray(out, dtype=float).reshape(-1, 2)
    count, kept = o[:, 0], o[:, 1]
    p = np.divide(count, kept, out=np.full(count.shape, 0.5), where=kept > 0)
    return p if sign > 0 else 1.0 - p


def _counts_to_pop(out, q, shots, signs, herald) -> np.ndarray:
    """Decode one core's `out` array into a |1> population: heralded runs write interleaved
    (count, kept) pairs (P = count/kept, spec 13 §8), plain runs write one count per point (P =
    count/shots). Res-sign folded either way."""
    return (population_heralded(out[q]["out"], signs[q]) if herald
            else population(out[q]["out"], shots, signs[q]))


def sweep_counts(drv, m, progs, params, shots: int, timeout: int, signs, herald: bool = False) -> dict:
    """Batched counts-mode sweep across ALL requested cores (spec 13 §8): ONE `rq.run` of every core's
    `progs[q]` — the swept knob is COMPUTED on-core from the scalar `params[q]` (a Q16 or int pair,
    e.g. {"a0q": ..., "daq": ..., "prep": 1}), so there are no input Arrays. Returns `{q: P}` — each
    core's self-normalised populations in [0, 1] (no |0> reference / projection), folded through that
    qubit's res-sign `signs[q]`. `herald` selects the interleaved (count, kept) decode (spec 13 §8)."""
    out = rq.run(drv, m, progs, params=params, results=["out"], timeout=timeout)
    return {q: _counts_to_pop(out, q, shots, signs, herald) for q in progs}


def rerun_counts(drv, m, progs, params, shots: int, timeout: int, signs, herald: bool = False) -> dict:
    """Like `sweep_counts` but a `rq.rerun` of already-`setup` cores (no reload) — the reruns a cal
    issues per detuning / per prep state / per window over the one resident image (spec 08 §4).
    Returns `{q: P}`."""
    out = rq.rerun(drv, m, progs, params=params, results=["out"], timeout=timeout)
    return {q: _counts_to_pop(out, q, shots, signs, herald) for q in progs}


def _levels_pop(out_arr, npts: int, shots: int, classifier, level: int) -> np.ndarray:
    """P(`level`) per point from a RAW capture's IQ (out sized 2·npts·shots, the point-major cursor the
    RAW kernels write): reshape to (npts, shots, 2), classify every shot into a level with the 3-level
    ClassifierN, and take the fraction that landed in `level` (spec two-qubit/01 §4.1 — the {|1>, |2>}
    readout the hardware res sign cannot do, so it is host-side over the RAW shots)."""
    iq = np.asarray(out_arr, dtype=float).reshape(npts, shots, 2)
    return np.array([float(np.mean(classifier.classify(iq[i]) == level)) for i in range(npts)])


def sweep_levels(drv, m, progs, params, npts: int, shots: int, timeout: int, classifiers,
                 level: int = 2) -> dict:
    """Batched RAW-mode sweep across ALL cores → `{q: P(level)}` via each core's pre-trained 3-level
    ClassifierN (spec 01 §4.1): ONE `rq.run` of every core's `progs[q]`, the swept knob COMPUTED on-core
    from the scalar `params[q]`. `classifiers[q]` is core q's ClassifierN, `level` the level to count
    (2 → P(|2>), the target of an EF drive)."""
    out = rq.run(drv, m, progs, params=params, results=["out"], timeout=timeout)
    return {q: _levels_pop(out[q]["out"], npts, shots, classifiers[q], level) for q in progs}


def rerun_levels(drv, m, progs, params, npts: int, shots: int, timeout: int, classifiers,
                 level: int = 2) -> dict:
    """Like `sweep_levels` but a `rq.rerun` of already-`setup` cores (the per-detuning reruns EFFrequency
    issues over one resident image, mirroring `rerun_counts`). Returns `{q: P(level)}`."""
    out = rq.rerun(drv, m, progs, params=params, results=["out"], timeout=timeout)
    return {q: _levels_pop(out[q]["out"], npts, shots, classifiers[q], level) for q in progs}


def acquire_shots(drv, m, progs, prep: int, shots: int, timeout: int) -> dict:
    """Per-prep raw-IQ capture across ALL cores (spec 09, 13 §8): one RAW-mode RERUN of the
    already-`setup` `progs` with the `prep` scalar written for this run. prep=0 → |0>, prep=1 → |1>
    (two reruns of the resident image, no reload). Returns `{q: (shots, 2)}` real/imag — `shots` is the
    number of IQ pairs each program writes (Separation passes npts·shots and reshapes per point)."""
    out = rq.rerun(drv, m, progs, params={q: {"prep": int(prep)} for q in progs}, results=["out"],
                   timeout=timeout)
    return {q: out[q]["out"].reshape(shots, 2).astype(float) for q in progs}
