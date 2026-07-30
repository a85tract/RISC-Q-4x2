"""Two-qubit CZ calibration (specs/two-qubit/01). Q0's config-arithmetic seed and the shared
conventions the whole family uses:

  - `pair_key` — qcal's `two_qubit` tuple-repr key ("(0, 1)"), so a config round-trips with qcal;
  - `coupler_core` — the channel-role lookup: which homogeneous-array core plays a coupler's CZ drive
    (a coupler is an ordinary extra core, spec 01 §1 — held in the config, not hard-coded);
  - `calc_cz_frequency` — the parametric-CZ drive-frequency seed from the 1Q spectrum (no hardware);
  - `joint_populations` — the two-qubit readout: per-core per-shot bits zipped by shot index.

The JAZZ / CZ resonance / conditionality / local-phase experiment classes join this module in Q3–Q5.
"""

from __future__ import annotations

import copy
import math
import re

import numpy as np

from riscq import run as rq
from riscq.cal import fits, kernels
from riscq.cal.config import _amp_phase
from riscq.cal.base import (GATE_CH, Result, batch_timeout, batches, ef_pulse, gate_pulse,
                            grid_period, population, qubit_freq, readout_tables, relax_batches,
                            res_sign, seconds, socmap, sweep_q16, x90_vz)
from riscq.lang import Array, ParamTable, compile_kernel
from riscq.map import LEAD, pack16
from riscq.pulses import Pulse, envelopes, units


def pair_key(pair) -> str:
    """The `two_qubit` config key for a qubit pair — qcal's tuple repr, space included:
    (0, 1) -> "(0, 1)" (== str((0, 1)), so a config stays interchangeable with qcal, spec 01 §2)."""
    i, j = pair
    return f"({int(i)}, {int(j)})"


def coupler_core(cfg, pair) -> int:
    """The homogeneous-array core index that plays coupler (i, j)'s CZ drive. A coupler is an ordinary
    extra core whose gate channel is the CZ drive (spec 01 §1); the qubit<->core map is identity (qubit
    q on core q) and each coupler's core is carried in the config as `two_qubit/(i, j)/core`, not
    hard-coded — sim-2q1c maps coupler (0, 1) to core 2."""
    return int(cfg[f"two_qubit/{pair_key(pair)}/core"])


def calc_cz_frequency(cfg, pairs, state: str = "02", form: str = "parametric") -> None:
    """Seed each pair's CZ drive frequency from the 1Q spectrum — pure config arithmetic, no hardware.
    The CZ activates the |11> <-> |{02, 20}> avoided crossing, and `form` picks the gate form's seed
    arithmetic (spec 04 §4.4) from the two two-excitation levels

        f_11 = f_GE(i) + f_GE(j)                                   # the |11> energy
        f_02 = f_GE(j) + f_EF(j)   /   f_20 = f_GE(i) + f_EF(i)    # the intermediate |02> / |20>

      'parametric' — the coupler-flux tone at the level DETUNING: CZ/freq = |f_state − f_11| (qcal
                     `calculate_parametric_cz_frequency`, cz.py:37-77);
      'drive'      — the two-qubit-drive form's IN-BAND tone: CZ/freq = (f_11 + f_state)/4, half the
                     two-photon midpoint (qcal's commented-out midpoint form — reproduces X6Y3's
                     plain pairs' calibrated freqs to within ~100 MHz, spec 04 §1).

    GE freqs are `qubit/{q}/freq`, EF freqs `qubit/{q}/EF/freq` (the EF prerequisite, Q1) — both in Hz.
    Writes `two_qubit/(i, j)/CZ/freq`. `state` picks the intermediate level ('02' the target's EF, the
    default; '20' the control's — walkthrough stage 4)."""
    assert state in ("02", "20"), f"intermediate state must be '02' or '20', got {state!r}"
    assert form in ("parametric", "drive"), f"form must be 'parametric' or 'drive', got {form!r}"
    for pair in pairs:
        i, j = int(pair[0]), int(pair[1])
        f_ge_i, f_ge_j = float(cfg[f"qubit/{i}/freq"]), float(cfg[f"qubit/{j}/freq"])
        f_11 = f_ge_i + f_ge_j
        f_02 = f_ge_j + float(cfg[f"qubit/{j}/EF/freq"])
        f_20 = f_ge_i + float(cfg[f"qubit/{i}/EF/freq"])
        f_int = f_02 if state == "02" else f_20
        cfg[f"two_qubit/{pair_key(pair)}/CZ/freq"] = \
            (f_11 + f_int) / 4 if form == "drive" else abs(f_int - f_11)


def joint_populations(bits_by_core: dict, order) -> np.ndarray:
    """Joint two-qubit populations [P(00), P(01), P(10), P(11)] from per-core per-shot bits, zipped by
    SHOT INDEX (spec 01 §5). Both cores record their shots in the same deterministic order — a fixed
    grid walked by one program per core in one run — so shot k on `order[0]` (control) and shot k on
    `order[1]` (target) are the same repetition. `bits_by_core[c]` is a 1-D array of that core's 0/1
    classified shots; the two lengths MUST match — a mismatch means the per-core streams desynced (the
    alignment contract, not a silent truncation). The result is indexed 2*b_control + b_target."""
    ctrl, tgt = order
    bc = np.asarray(bits_by_core[ctrl], dtype=int).ravel()
    bt = np.asarray(bits_by_core[tgt], dtype=int).ravel()
    if bc.shape != bt.shape:
        raise ValueError(f"shot-count mismatch: control {bc.shape} vs target {bt.shape} — streams desynced")
    counts = np.bincount(2 * bc + bt, minlength=4)
    return counts / counts.sum()


def _signed_fft_freq(t: np.ndarray, z: np.ndarray) -> float:
    """The signed dominant FFT frequency of a complex signal z on the uniform grid t (qcal's
    `est_freq_fft`, zz.py:539-576): the two conditional Ramseys can run at opposite-sign fringe
    frequencies, and a real cosine fit only gives |f| — the complex quadrature I − jQ resolves the
    SIGN. Returns cycles per unit of t (Hz when t is seconds)."""
    n = len(t)
    dt = (t[-1] - t[0]) / (n - 1) if n > 1 else 1.0
    spec = np.abs(np.fft.fft(z - np.mean(z)))
    freqs = np.fft.fftfreq(n, d=dt)
    return float(freqs[int(np.argmax(spec))])


def _fit_complex_freq(t: np.ndarray, z: np.ndarray):
    """Fit the complex quadrature z = I − jQ to a decaying phasor A·e^{i(2πf·t+φ)}·e^{−t/τ} and return
    (signed f, ok). Using BOTH quadratures at once — and seeding f from the complex FFT — pins the
    fringe's sign AND magnitude in one fit, which is far steadier on the short, noisy, low-contrast
    co-sim fringes than an unsigned real-cosine fit patched with a separate FFT sign (that pair let one
    branch lock onto a spurious harmonic). curve_fit runs on the stacked [Re, Im]."""
    from scipy.optimize import curve_fit
    t = np.asarray(t, float)
    z = np.asarray(z, complex)
    span = (t[-1] - t[0]) or 1.0
    seed = _signed_fft_freq(t, z)

    def model(_t, ar, ai, f, tau):
        e = (ar + 1j * ai) * np.exp(1j * 2 * np.pi * f * _t) * np.exp(-_t / tau)
        return np.concatenate([e.real, e.imag])

    p0 = [z[0].real or 1e-3, z[0].imag or 1e-3, seed, span]
    try:
        popt, pcov = curve_fit(model, t, np.concatenate([z.real, z.imag]), p0=p0, maxfev=20000)
    except (RuntimeError, ValueError, TypeError):
        return seed, False
    ok = bool(np.all(np.isfinite(popt)) and np.all(np.isfinite(np.diag(pcov))))
    return float(popt[2]), ok


class JAZZ:
    """Residual-ZZ Ramsey (Joint Amplification of ZZ, spec two-qubit/01 §4.3; qcal zz.py:43-288). The
    first end-to-end two-qubit measurement — X90s + idles only, no coupler or EF drive — so it works at
    the zero-bias coupling point of part 1 (and is later the cost function for the part-3 park servo).

    A Hahn-echo Ramsey on the target with the control in |0> vs |1>: `X90 · idle · [pi on both] · idle ·
    Rz(2*pi*detuning*t) · close` (the midpoint pi on both qubits is BIRD — it refocuses each qubit's own
    detuning but preserves the control-conditional ZZ). The target fringe frequency therefore differs
    between the two control states by exactly the ZZ, so **ZZ11 = f(control=1) − f(control=0)** (Hz).
    Both cores run one role of `k_jazz` on the shared grid; the four sequences (control |0>/|1> × X90/Y90
    close) are runtime reruns of the one resident image (`prep`, `quad` scalars).

    Per control state the X90 close gives the in-phase fringe I(t) and the Y90 close the quadrature
    Q(t); the complex I − jQ is fit to a decaying phasor (`_fit_complex_freq`) for the SIGNED fringe
    frequency in one step (steadier than an unsigned cosine fit patched with an FFT sign). Writes
    `two_qubit/(i, j)/ZZ11`.

    Co-sim note (spec 01 §4.3): the model's ONLY qubit reset is relaxation, which also decoheres this
    long multi-pulse Ramsey, so co-sim fringes are low-contrast and the ZZ MAGNITUDE is SNR-limited
    there — the model ZZ physics (the split tracks ζ's sign, vanishes at ζ=0) is verified host-side
    (test_models.test_twoqubit_jazz_zz_physics); the co-sim gate checks the end-to-end run + the
    ζ=0 null. Full-accuracy co-sim recovery needs a long (slow) grid and is deferred."""

    def __init__(self, cfg, pair, detune=2e6, points=20, t0=40e-9, dt=40e-9, shots=120):
        self.cfg = cfg
        self.pair = (int(pair[0]), int(pair[1]))
        self.detune = float(detune)
        self.points, self.t0, self.dt = int(points), float(t0), float(dt)
        self.shots = int(shots)
        self.data, self.fit = {}, {}

    def _signed_freq(self, t, I, Q):
        """The signed fringe frequency (Hz) from the complex quadrature z = I − jQ (both quadratures
        fit jointly for sign + magnitude, qcal's I − jQ convention)."""
        z = (np.asarray(I, float) - np.mean(I)) - 1j * (np.asarray(Q, float) - np.mean(Q))
        f, ok = _fit_complex_freq(t, z)
        return f, ok

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg = self.cfg
        ctrl, tgt = self.pair
        w0, dw = batches(self.t0, m), batches(self.dt, m)       # half-wait grid; full echo delay = 2w
        waits = [w0 + i * dw for i in range(self.points)]
        t_s = np.array([seconds(2 * w, m) for w in waits])      # fit x-axis: the full delay in seconds
        dc = units._freq_code(self.detune, m.params)            # applied virtual-detuning code
        hpi = pack16(units._phase_code(math.pi / 2))            # pi/2 seated: the Y90 close
        # the virtual detuning phi = 2*pi*detuning*t: over the full delay t = 2w batches, the seated
        # phase is 16*dc*(2w) = 32*dc*w — the k_ramsey `16*dc*wait` idiom at the doubled delay.
        p0, dp = pack16(32 * dc * w0), pack16(32 * dc * dw)

        # ONE grid for both cores: the control's prelude (prep + echo) is the longer of the two.
        progs, signs, dur, ddly = {}, {}, 0, 0
        tables, ds = {}, {}
        for role, q in ((kernels.CONTROL, ctrl), (kernels.TARGET, tgt)):
            gate = ParamTable(GATE_CH, qubit_freq(cfg, q), {"x90": gate_pulse(cfg, q, m)})
            ro, demod, code, dur, ddly = readout_tables(cfg, q, m)
            tables[q] = (role, gate, ro, demod, code)
            ds[q] = gate.pulses["x90"].dur_batches(m, GATE_CH)
        seq = 5 * max(ds.values()) + 2 * waits[-1]              # control: prep 2d + echo 2d + close d
        period = grid_period(relax_batches(cfg, m), seq, dur, ddly)
        timeout = batch_timeout(self.points * self.shots * period)
        for q, (role, gate, ro, demod, code) in tables.items():
            # leave this role's runtime scalar UNBOUND (a rewritable symbol) and bind the other role's
            # (a dead reference here, folded away): control reruns `prep`, target reruns `quad`.
            fixed = {"quad": 0} if role == kernels.CONTROL else {"prep": 0}
            progs[q] = compile_kernel(kernels.k_jazz, m, tables=dict(gate=gate, ro=ro, demod=demod),
                                      out=Array(self.points), npts=self.points, shots=self.shots,
                                      period=period, code=code, ddly=ddly, role=role, hpi=hpi,
                                      w0=w0, dw=dw, p0=p0, dp=dp, **fixed, **x90_vz(cfg, q))
            signs[q] = res_sign(cfg, q)
        rq.setup(drv, m, progs)

        # the four sequences: control |0>/|1> × X90/Y90 close, reruns of the one resident image. The
        # runtime scalar is per role — the compile-dead branch drops its param, so the control program
        # keeps only `prep` and the target only `quad`.
        P = {}
        for c in (0, 1):
            for quad in (0, 1):
                par = {ctrl: {"prep": c}, tgt: {"quad": quad}}
                out = rq.rerun(drv, m, progs, params=par, results=["out"], timeout=timeout)
                P[(c, quad)] = population(out[tgt]["out"], self.shots, signs[tgt])
        f0, ok0 = self._signed_freq(t_s, P[(0, 0)], P[(0, 1)])
        f1, ok1 = self._signed_freq(t_s, P[(1, 0)], P[(1, 1)])
        zz = f1 - f0

        self.data = {self.pair: {"t": t_s, "I0": P[(0, 0)], "Q0": P[(0, 1)],
                                 "I1": P[(1, 0)], "Q1": P[(1, 1)], "f0": f0, "f1": f1}}
        self.fit = {self.pair: (f0, f1)}
        ok = ok0 and ok1
        proposal = {f"two_qubit/{pair_key(self.pair)}/ZZ11": zz} if ok else {}
        return Result(ok, self.data, self.fit, proposal, cfg, f"JAZZ {self.pair}")


# ── CZ pulse table + config accessors (spec two-qubit/01 §2, §4.4-4.6; layout-aware per 04 §4.2) ──
#
# Two gate forms share one pulse-list schema: the coupler-drive form (one physical pulse on a
# dedicated coupler core, spec 01 §2) and qcal's two-qubit-drive form (two consecutive drives on the
# pair's own gate channels; X6Y3, spec 04 §1). String entries are pulse REFERENCES (the (5,6)/(6,7)
# EF-X shelving sandwich) and shift every position, so nothing is found by index: drives by the
# qcal `find_pulse_index` walk, virtual-Z entries by their `channel` key.

def cz_coupler_form(cfg, pair) -> bool:
    """Gate-form detection (spec 04 §4.1): a pair with a dedicated coupler core recorded at
    `two_qubit/(i, j)/core` is the coupler-drive form; without it, the two-qubit-drive form."""
    return f"two_qubit/{pair_key(pair)}/core" in cfg


def _channel_qubit(channel) -> int | None:
    """The core index a pulse-entry channel names — 'Q<i>.qdrv' (qcal) or bare 'Q<i>' (the synthetic
    co-sim configs) → i (the qubit↔core map is identity, spec 01 §1); None for anything else (a
    coupler channel like 'C0_1')."""
    mm = re.fullmatch(r"Q(\d+)(?:\..*)?", str(channel))
    return int(mm.group(1)) if mm else None


def _cz_pulses(cfg, pair) -> list:
    """A pair's raw CZ pulse list."""
    return cfg[f"two_qubit/{pair_key(pair)}/CZ/pulse"]


def _cz_drive_indices(pulses) -> list[int]:
    """The indices of the PHYSICAL drive entries of a CZ pulse list — qcal's `find_pulse_index` walk
    (qcal/calibration/utils.py:11-28) over the whole list: skip string references and virtualz
    entries; what remains is the drive — ONE entry in the coupler form, TWO consecutive ones
    (control, then target with the calibrated relative phase) in the two-qubit-drive form."""
    return [i for i, p in enumerate(pulses)
            if not isinstance(p, str) and p.get("env") != "virtualz"]


def _cz_vz_entry(pulses, q: int) -> dict | None:
    """Qubit q's virtual-Z entry of a CZ pulse list, matched by its `channel` key (04 §4.2) — the
    pair's own ZI/IZ corrections and the spectator entries alike. None when q has no entry."""
    for p in pulses:
        if not isinstance(p, str) and p.get("env") == "virtualz" \
                and _channel_qubit(p.get("channel")) == q:
            return p
    return None


def _cz_entry(cfg, pair, drive: int = 0) -> dict:
    """A pair's `drive`-th physical CZ drive entry: 0 = the coupler drive (coupler form) / the
    CONTROL drive (two-qubit-drive form), 1 = the TARGET drive (04 §1)."""
    pulses = _cz_pulses(cfg, pair)
    return pulses[_cz_drive_indices(pulses)[drive]]


def _cz_amp(cfg, pair) -> float:
    """The CZ drive's normalized amplitude (equal on both lines in the two-qubit-drive form)."""
    return float(_cz_entry(cfg, pair).get("kwargs", {}).get("amp", 1.0))


def _cz_dur_batches(cfg, pair, m) -> int:
    """The CZ drive's length in batches (the drive entry's `time`, seconds)."""
    return batches(float(_cz_entry(cfg, pair)["time"]), m)


def _local_phase(cfg, pair, q: int) -> float:
    """QUBIT q's virtual-Z phase (rad) in a pair's CZ pulse list (the control's = ZI, the target's
    = IZ, a neighbour's = its spectator correction; spec 01 §3, §4.6) — matched by channel, 0.0
    when the entry is absent."""
    p = _cz_vz_entry(cfg.get(f"two_qubit/{pair_key(pair)}/CZ/pulse") or [], q)
    return 0.0 if p is None else float(p.get("kwargs", {}).get("phase", 0.0))


def _local_phase_code(cfg, pair, q: int) -> int:
    """`_local_phase` as the seated virtual-Z word the kernels fold per CZ."""
    return pack16(units._phase_code(_local_phase(cfg, pair, q)))


def _cz_pulse_set(cfg, pair, key: str, value) -> list:
    """A deep copy of a pair's CZ pulse list with the physical drive's `time` (dur) or `amp` updated
    on EVERY drive entry — one in the coupler form, both lines in the two-qubit-drive form (qcal
    calibrates them jointly, 04 §4.3). The proposal payload, since the entries sit inside a
    slash-path list leaf, so the whole list is written back."""
    pulses = copy.deepcopy(_cz_pulses(cfg, pair))
    for i in _cz_drive_indices(pulses):
        if key == "time":
            pulses[i]["time"] = float(value)
        else:
            pulses[i].setdefault("kwargs", {})["amp"] = float(value)
    return pulses


def _cz_pulse(cfg, pair, m, dur_batches: int, drive: int = 0, amp=None) -> Pulse:
    """The BASEBAND (freq_hz=None — the kernel programs the carrier at runtime) CZ Pulse of a pair's
    `drive`-th physical line: a `dur_batches`-long envelope at the entry's amp/phase. `amp` overrides
    the config amp (the AMP-ladder centre). The envelope's shape kwargs are the entry's `kwargs` minus
    amp/phase (X6Y3 carries `ramp_fraction` there, spec 04 §3), merged over the legacy
    `{env_func, ...}` dict form."""
    entry = _cz_entry(cfg, pair, drive)
    a0, phase, kw = _amp_phase(entry.get("kwargs"))
    a = a0 if amp is None else float(amp)
    envspec = entry.get("env", "square")
    ch = m.channel(GATE_CH)
    n = int(dur_batches) * ch.samples_per_line
    if isinstance(envspec, dict):                              # the {env_func, ...kwargs} dict form
        name = envspec.get("env_func", "square")
        ekw = {**{k: v for k, v in envspec.items() if k != "env_func"}, **kw}
    else:                                                      # a qcal env NAME: shape kwargs in `kwargs`
        name, ekw = str(envspec), kw
    env = envelopes.build(name, n, ch.samples_per_line * m.params.dsp_freq_hz, **ekw)
    return Pulse(env, amp=a, phase=phase)


def _cz_rel_phase_set(cfg, pair, phase: float) -> list:
    """A fresh CZ pulse list with the TARGET drive line's tone phase set — qcal RelativePhase's
    `CZ/pulse/{idx+1}/kwargs/phase` param (cz.py:1553-1560); the proposal payload (a list leaf,
    written back whole). Two-qubit-drive form only (the second physical drive entry)."""
    pulses = copy.deepcopy(_cz_pulses(cfg, pair))
    idx = _cz_drive_indices(pulses)
    if len(idx) < 2:
        raise ValueError(f"pair {tuple(pair)}: no target drive line (coupler form?) — "
                         f"the relative phase is a two-qubit-drive knob")
    pulses[idx[1]].setdefault("kwargs", {})["phase"] = float(phase)
    return pulses


def cz_table(cfg, pair, m, dur_batches: int, amp=None) -> ParamTable:
    """The COUPLER-form CZ pulse table (channel GATE_CH, one slot 'cz', spec 01 §1): the physical
    drive entry's baseband pulse (`_cz_pulse`), the table carrier seeded at the config `CZ/freq`.
    `dur_batches` is the ENVELOPE length (czmax for a DUR sweep — the swept dur field truncates its
    flat top)."""
    f_cz = float(cfg[f"two_qubit/{pair_key(pair)}/CZ/freq"])
    return ParamTable(GATE_CH, f_cz, {"cz": _cz_pulse(cfg, pair, m, dur_batches, 0, amp)})


_EF_X_REF = re.compile(r"single_qubit/(\d+)/EF/X/pulse")


def cz_sandwich(cfg, pair) -> int | None:
    """The EF-SHELVED qubit of an EF-sandwich pair, or None for a plain pulse list (spec 04 §1 /
    X4). X6Y3's (5,6)/(6,7) bracket their two drive tones with STRING-REFERENCE pre/post-pulses —
    `single_qubit/6/EF/X/pulse`, a qcal config-path pulse reuse — that play that qubit's EF X around
    the tone: |1>→|2> before (shelve), |2>→|1> after (un-shelve), so the CZ activates in the shelved
    manifold. Exactly that layout is supported: TWO identical references, one before the first
    drive and one after the last, naming the EF X of a PAIR MEMBER whose `qubit/{q}/EF/{freq, x}`
    the config carries. Anything else is a loud error, never a silent mis-play."""
    pulses = _cz_pulses(cfg, pair)
    refs = [(i, p) for i, p in enumerate(pulses) if isinstance(p, str)]
    if not refs:
        return None

    def fail(why):
        raise ValueError(f"pair {tuple(pair)}: unsupported CZ string-reference layout — {why} "
                         f"(X4 supports X6Y3's EF-X sandwich: one identical "
                         f"'single_qubit/<q>/EF/X/pulse' reference before and one after the drives)")

    if len(refs) != 2 or refs[0][1] != refs[1][1]:
        fail(f"expected 2 identical references, got {[p for _, p in refs]}")
    mm = _EF_X_REF.fullmatch(refs[0][1])
    if not mm:
        fail(f"reference {refs[0][1]!r} is not an EF X pulse path")
    q = int(mm.group(1))
    if q not in (int(pair[0]), int(pair[1])):
        fail(f"referenced qubit {q} is not a member of the pair")
    drives = _cz_drive_indices(pulses)
    if not (refs[0][0] < drives[0] and refs[1][0] > drives[-1]):
        fail("the references must bracket the drive tones (pre + post)")
    for key in (f"qubit/{q}/EF/freq", f"qubit/{q}/EF/x/amp"):
        if key not in cfg:
            fail(f"missing {key} (the shelved qubit's EF calibration)")
    return q


def _sandwich_binds(cfg, pair, m):
    """The EF-sandwich compile bindings of a drive-form CZ kernel (spec 04 §1 / X4) →
    (shelf | None, {core: {'sw', 'fef'}}, tail). `tail` = LEAD + the EF X length in batches — the
    extra grid segment between the cz tone(s) and the close/readout region (the un-shelving EF X
    plus its retune gap) — bound on BOTH cores so they place the tone at the same absolute slot
    (the mirrored pre-shelve segment lengthens the prelude by the same amount: a grid period
    accounts 2·tail). `sw`=1 with the shelf's seated EF carrier `fef` binds only on the SHELF core,
    the only one that plays the EF X. A plain (or coupler-form) pair binds all zeros."""
    shelf = cz_sandwich(cfg, pair)
    binds = {int(q): {"sw": 0, "fef": 0} for q in pair}
    if shelf is None:
        return None, binds, 0
    efd = ef_pulse(cfg, shelf, m, "x").dur_batches(m, GATE_CH)
    binds[shelf] = {"sw": 1, "fef": units.freq_to_code(float(cfg[f"qubit/{shelf}/EF/freq"]),
                                                       m.params)}
    return shelf, binds, LEAD + efd


def cz_drive_table(cfg, pair, q: int, drive: int, m, dur_batches: int, amp=None) -> ParamTable:
    """The TWO-QUBIT-DRIVE form's gate table for qubit core q (spec 04 §4.1): its GE 'x90' slot (the
    prep/close pulses, at the table's GE carrier) plus a baseband 'cz' slot from the pair's `drive`-th
    line (0 = the control line, 1 = the target line — whose slot phase IS the calibrated relative
    phase). The kernel retunes the ONE gate NCO f_GE → f_CZ around the cz slot (the EF mechanism,
    spec 01 §4.1). An EF-sandwich pair (spec 04 §1 / X4) resolves its string references through the
    config (`cz_sandwich`) into a third, baseband 'ef' slot holding the SHELF qubit's own EF X — on
    BOTH cores: the shelf core plays it (the `sw` kernel fold); the partner carries it as a
    never-fired pad so the two gate tables stay the same shape, because unequal slot counts shift a
    core's `now() + period` grid by tens of batches (the X3 finding) and would break the two-tone
    lock-step."""
    shelf = cz_sandwich(cfg, pair)
    pulses = {"x90": gate_pulse(cfg, q, m),
              "cz": _cz_pulse(cfg, pair, m, dur_batches, drive, amp)}
    if shelf is not None:
        pulses["ef"] = ef_pulse(cfg, shelf, m, "x")
    return ParamTable(GATE_CH, qubit_freq(cfg, q), pulses)


def _cz_freq_word(cfg, pair, m) -> int:
    """The config `CZ/freq` as the SEATED carrier word the drive-form kernels retune to (`fcz`)."""
    return units.freq_to_code(float(cfg[f"two_qubit/{pair_key(pair)}/CZ/freq"]), m.params)


# ── CZ resonance & return: sweep the coupler pulse (spec 01 §4.4) ────────────────────────────────

class CZSweep:
    """CZ resonance / return sweep (spec 01 §4.4; qcal `ParamSweep`, sweep.py:177-223). Prep |11> on
    both qubits, sweep ONE field of the coupler pulse, and read the CONTROL core's |1> population — a
    clean 2-level signal because |02> (the parametric partner of |11>) has the control in |0>, so the
    control P(1) DIPS as population leaves |11> and RETURNS as the round trip closes (the target's |2>
    is not 2-level separable — 3-level P(02) is the §4.5 enhancement, off-core here). Three knobs run
    the same `k_cz_pop` (spec 01 §4.4 table):

      'freq' — sweep the coupler carrier ±`span` Hz around `CZ/freq`; the P(1) dip locates f_CZ, fit by
               a parabola (qcal's argmax-mean replaced by a proper fit, §4.4 gotcha). Writes `CZ/freq`.
      'dur'  — sweep the slot's dur field (`set_dur`) under a fixed max-length envelope (truncating the
               flat top); the first full |11>→|02>→|11> round trip is one cosine period. Writes the
               coupler pulse `time`.
      'amp'  — sweep the coupler amp; the full 2π return is one cosine period. Writes the pulse `amp`.

    `lo`/`hi` override the swept endpoints (freq: Hz; dur: seconds; amp: normalized) — else defaults are
    derived from the config (freq: ±span; dur: 0→2×; amp: 0→2×). A single batched run. The gate form is
    read off the config layout (`cz_coupler_form`, spec 04 §4.1): coupler form = three cores on the
    shared grid, the COUPLER carrying the sweep (and never firing its readout, spec 01 §1); two-qubit-
    drive form = the two qubit cores play their own CZ lines with the sweep bound LOCKSTEP on both
    (`k_cz_pop` DRIVE_FORM — the freq knob retunes both NCOs, dur/amp hit both cz slots; qcal
    calibrates the lines jointly)."""

    def __init__(self, cfg, pair, knob="freq", span=10e6, lo=None, hi=None, points=21, shots=120):
        assert knob in ("freq", "dur", "amp"), f"knob must be freq/dur/amp, got {knob!r}"
        self.cfg, self.pair = cfg, (int(pair[0]), int(pair[1]))
        self.knob, self.span = knob, float(span)
        self.lo, self.hi = lo, hi
        self.points, self.shots = int(points), int(shots)
        self.data, self.fit = {}, {}

    def _sweep(self, m):
        """(knob code, kernel knob, czmax, x0, dx, xs-int, x-axis physical) for this run."""
        cfg, pair, pk = self.cfg, self.pair, pair_key(self.pair)
        czd = _cz_dur_batches(cfg, pair, m)
        if self.knob == "freq":
            f_cz = float(cfg[f"two_qubit/{pk}/CZ/freq"])
            lo = units._freq_code(f_cz - self.span if self.lo is None else self.lo, m.params)
            hi = units._freq_code(f_cz + self.span if self.hi is None else self.hi, m.params)
            x0, dx, xs = sweep_q16(lo, hi, self.points)
            xax = np.array([units.code_to_freq(int(x), m.params) for x in xs])
            return kernels.FREQ, czd, int(x0), int(dx), xs, xax
        if self.knob == "amp":
            a0 = _cz_amp(cfg, pair)
            lo = units._amp_code(0.02 if self.lo is None else self.lo)
            hi = units._amp_code(min(0.99, 2 * a0) if self.hi is None else self.hi)
            x0, dx, xs = sweep_q16(lo, hi, self.points)
            xax = np.array([int(x) / units.AMP_SCALE for x in xs])
            return kernels.AMP, czd, int(x0), int(dx), xs, xax
        lo = batches(1.0 / m.params.dsp_freq_hz if self.lo is None else self.lo, m)   # ~0 dur
        hi = 2 * czd if self.hi is None else batches(self.hi, m)
        dd = 0 if self.points <= 1 else round((hi - lo) / (self.points - 1))
        xs = np.array([lo + i * dd for i in range(self.points)])
        xax = np.array([seconds(int(x), m) for x in xs])
        return kernels.DUR, int(hi), int(lo), int(dd), xs, xax               # czmax = hi (the envelope)

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg, pair, pk = self.cfg, self.pair, pair_key(self.pair)
        ctrl, tgt = pair
        coupler = cz_coupler_form(cfg, pair)
        shelf, sw_binds, tail = _sandwich_binds(cfg, pair, m)     # zeros on plain/coupler pairs
        kknob, czmax, x0, dx, xs, xax = self._sweep(m)
        ro_c, demod_c, code_c, dur_c, ddly_c = readout_tables(cfg, ctrl, m)
        xd = gate_pulse(cfg, ctrl, m).dur_batches(m, GATE_CH)
        # drive form: the GE prep ends a full LEAD before the retuned CZ segment (phasor regen);
        # an EF sandwich adds a (LEAD + EF X) segment on each side of the tone (2·tail)
        period = grid_period(relax_batches(cfg, m),
                             2 * xd + czmax + (0 if coupler else LEAD) + 2 * tail, dur_c, ddly_c)
        timeout = batch_timeout(self.points * self.shots * period)
        progs, signs = {}, {}
        if coupler:
            for q in (ctrl, tgt):                                 # both qubits prep |11> and read out
                gate = ParamTable(GATE_CH, qubit_freq(cfg, q), {"x90": gate_pulse(cfg, q, m)})
                ro, demod, code, dur, ddly = readout_tables(cfg, q, m)
                progs[q] = compile_kernel(kernels.k_cz_pop, m,
                                          tables=dict(gate=gate, ro=ro, demod=demod),
                                          out=Array(self.points), npts=self.points, shots=self.shots,
                                          period=period, code=code, ddly=ddly, role=kernels.CONTROL,
                                          knob=kknob, form=kernels.COUPLER_FORM, xd=xd, czmax=czmax,
                                          fcz=0, fef=0, sw=0, tail=0, x0=0, dx=0, **x90_vz(cfg, q))
                signs[q] = res_sign(cfg, q)
            cztable = cz_table(cfg, pair, m, czmax)               # coupler: envelope built at czmax
            progs[coupler_core(cfg, pair)] = compile_kernel(
                kernels.k_cz_pop, m, tables=dict(gate=cztable, ro=ro_c, demod=demod_c),
                out=Array(self.points), npts=self.points, shots=self.shots, period=period,
                code=code_c, ddly=ddly_c, role=kernels.COUPLER, knob=kknob,
                form=kernels.COUPLER_FORM, xd=xd, czmax=czmax, fcz=0, fef=0, sw=0, tail=0,
                x0=x0, dx=dx, **x90_vz(cfg, ctrl))
        else:                                                     # drive form: 2 cores, LOCKSTEP sweep
            fcz = _cz_freq_word(cfg, pair, m)
            for q, drive in ((ctrl, 0), (tgt, 1)):
                gate = cz_drive_table(cfg, pair, q, drive, m, czmax)
                ro, demod, code, dur, ddly = readout_tables(cfg, q, m)
                progs[q] = compile_kernel(kernels.k_cz_pop, m,
                                          tables=dict(gate=gate, ro=ro, demod=demod),
                                          out=Array(self.points), npts=self.points, shots=self.shots,
                                          period=period, code=code, ddly=ddly, role=kernels.CONTROL,
                                          knob=kknob, form=kernels.DRIVE_FORM, xd=xd, czmax=czmax,
                                          fcz=fcz, tail=tail, x0=int(x0), dx=int(dx),
                                          **sw_binds[q], **x90_vz(cfg, q))
                signs[q] = res_sign(cfg, q)
        out = rq.run(drv, m, progs, results=["out"], timeout=timeout)
        P1 = population(out[ctrl]["out"], self.shots, signs[ctrl])   # the control's dip / return
        self.data = {pair: {"x": xax, "y": P1}}

        ok, prop = False, {}
        if self.knob == "freq":
            fit = fits.fit_parabola(xax, P1)
            in_rng = fit.ok and xax.min() <= fit.value <= xax.max()
            if fit.ok and fit.params["a"] > 0 and in_rng and (P1.max() - P1.min()) > 0.15:
                prop, ok = {f"two_qubit/{pk}/CZ/freq": float(fit.value)}, True
        else:                                                     # dur / amp: the full-return period
            fit = fits.fit_cosine(xax, P1)
            full = 1.0 / fit.value if (fit.ok and fit.value > 0) else math.nan
            if fit.ok and fit.value > 0 and xax.min() <= full <= xax.max():
                key = "time" if self.knob == "dur" else "amp"
                prop = {f"two_qubit/{pk}/CZ/pulse": _cz_pulse_set(cfg, pair, key, full)}
                ok = True
        self.fit = {pair: fit}
        return Result(ok, self.data, self.fit, prop, cfg, f"CZSweep {pair} {self.knob}")


# ── CZ conditionality: the tomographic R metric (spec 01 §4.5) ───────────────────────────────────

def _cz_cond_period(cfg, m, pair, ngates: int) -> int:
    """The `k_cz_cond` batch grid for an `ngates` CZ train. One run sizes its own; an RPE ladder
    passes the DEEPEST rung's `period` into every `_cz_cond_progs` compile so all depths share ONE
    grid — per-rung periods give the shallow rungs a shorter relax head, a depth-dependent
    state-prep error the ladder cannot absorb (spec 14 F5 finding 2)."""
    coupler = cz_coupler_form(cfg, pair)
    _, _, tail = _sandwich_binds(cfg, pair, m)                    # zeros on plain/coupler pairs
    czd = _cz_dur_batches(cfg, pair, m)
    ctrl = int(pair[0])
    _, _, _, dur_c, ddly_c = readout_tables(cfg, ctrl, m)
    xd = gate_pulse(cfg, ctrl, m).dur_batches(m, GATE_CH)
    # drive form: a LEAD gap on each side of the retuned cz train (prep → train, train → close);
    # an EF sandwich adds a (LEAD + EF X) segment on each side of the train (2·tail)
    return grid_period(relax_batches(cfg, m),
                       3 * xd + ngates * czd + (0 if coupler else 2 * LEAD) + 2 * tail,
                       dur_c, ddly_c)


def _cz_cond_progs(cfg, m, pair, knob, x0, dx, points, ngates, shots, amp=None, ramsey=None,
                   period=None):
    """Compile the `k_cz_cond` conditionality-tomography programs for a pair, form-dispatched (spec
    04 §4.1): coupler form = 3 cores, the COUPLER carrying the swept knob (the qubit cores bind a dead
    0 sweep); two-qubit-drive form = the 2 qubit cores playing their own CZ lines, the sweep bound
    LOCKSTEP on both. `amp` overrides the CZ drive amp (both lines — the AMP-ladder centre). `ramsey`
    picks WHICH qubit plays the Ramsey (the TARGET role: Y90 prep + swept close; default the pair's
    target) — CZRPE's (3, 1) rung Ramseys the physical CONTROL with the target prepped |1>, the roles
    swapped but each line still playing its own CZ tone (spec 14 F5 finding 5). Each core binds its
    OWN local virtual-Z word into both `zi`/`iz` (its role reads exactly one). `period` overrides the
    self-sized batch grid (`_cz_cond_period` — the RPE ladder's shared-grid contract). Returns
    (progs, tables, signs, timeout) — `tables` so a host-paced caller (RelativePhase,
    CZAmpFreqSweep) can locate a cz slot for `rq.write_slot`; it holds the gate table of EVERY
    compiled core, so `"cz" in table.pulses` picks out the physical CZ lines in either form."""
    ctrl, tgt = int(pair[0]), int(pair[1])
    ramsey = tgt if ramsey is None else int(ramsey)
    assert ramsey in (ctrl, tgt), f"ramsey qubit {ramsey} is not in pair {(ctrl, tgt)}"
    coupler = cz_coupler_form(cfg, pair)
    shelf, sw_binds, tail = _sandwich_binds(cfg, pair, m)         # zeros on plain/coupler pairs
    czd = _cz_dur_batches(cfg, pair, m)
    hpi = pack16(units._phase_code(math.pi / 2))
    kknob = kernels.FREQ if knob == "freq" else kernels.AMP
    form = kernels.COUPLER_FORM if coupler else kernels.DRIVE_FORM
    fcz = 0 if coupler else _cz_freq_word(cfg, pair, m)
    ro_c, demod_c, code_c, dur_c, ddly_c = readout_tables(cfg, ctrl, m)
    xd = gate_pulse(cfg, ctrl, m).dur_batches(m, GATE_CH)
    if period is None:
        period = _cz_cond_period(cfg, m, pair, ngates)
    timeout = batch_timeout(points * shots * period)
    progs, tables, signs = {}, {}, {}
    for q, drive in ((ctrl, 0), (tgt, 1)):
        role = kernels.TARGET if q == ramsey else kernels.CONTROL
        gate = (ParamTable(GATE_CH, qubit_freq(cfg, q), {"x90": gate_pulse(cfg, q, m)}) if coupler
                else cz_drive_table(cfg, pair, q, drive, m, czd, amp=amp))
        ro, demod, code, dur, ddly = readout_tables(cfg, q, m)
        own = _local_phase_code(cfg, pair, q)
        fixed = {"quad": 0} if role == kernels.CONTROL else {"prep": 0}   # bind the dead scalar (as JAZZ)
        progs[q] = compile_kernel(kernels.k_cz_cond, m, tables=dict(gate=gate, ro=ro, demod=demod),
                                  out=Array(points), npts=points, shots=shots, period=period, code=code,
                                  ddly=ddly, role=role, knob=kknob, form=form, ngates=ngates, hpi=hpi,
                                  zi=own, iz=own, xd=xd, czd=czd, fcz=fcz, tail=tail,
                                  x0=0 if coupler else int(x0), dx=0 if coupler else int(dx),
                                  **sw_binds[q], **fixed, **x90_vz(cfg, q))
        tables[q], signs[q] = gate, res_sign(cfg, q)
    if coupler:
        cztable = cz_table(cfg, pair, m, czd, amp=amp)
        progs[coupler_core(cfg, pair)] = compile_kernel(
            kernels.k_cz_cond, m, tables=dict(gate=cztable, ro=ro_c, demod=demod_c),
            out=Array(points), npts=points, shots=shots, period=period, code=code_c, ddly=ddly_c,
            role=kernels.COUPLER, knob=kknob, form=form, ngates=ngates, hpi=hpi, zi=0, iz=0,
            xd=xd, czd=czd, fcz=fcz, fef=0, sw=0, tail=0, x0=int(x0), dx=int(dx), prep=0, quad=0,
            **x90_vz(cfg, ctrl))
        tables[coupler_core(cfg, pair)] = cztable
    return progs, tables, signs, timeout


def _cond_R(drv, m, progs, pair, shots, signs, timeout):
    """The four tomography reruns over already-`setup` `_cz_cond_progs` — {control |0>/|1>} × {close
    Y90/X90}, reruns of one resident image (`prep`, `quad` scalars) — and

        R = √((P0_C1X − P0_C0X)² + (P0_C1Y − P0_C0Y)²)

    per point on the target, 1 at a conditional-π CZ (spec 01 §4.5). Returns (R, the four target P(0)
    branches)."""
    ctrl, tgt = int(pair[0]), int(pair[1])
    P0 = {}
    for c in (0, 1):                                             # control |0>/|1>
        for quad in (0, 1):                                     # close Y90 (X-seq) / X90 (Y-seq)
            par = {ctrl: {"prep": c}, tgt: {"quad": quad}}
            out = rq.rerun(drv, m, progs, params=par, results=["out"], timeout=timeout)
            P0[(c, quad)] = 1.0 - population(out[tgt]["out"], shots, signs[tgt])   # target P(0)
    R = np.sqrt((P0[(1, 0)] - P0[(0, 0)]) ** 2 + (P0[(1, 1)] - P0[(0, 1)]) ** 2)
    return R, P0


def _cz_cond_R(cfg, drv, m, pair, knob, x0, dx, points, ngates, shots, amp=None):
    """Compile + setup + measure: the whole-sweep conditionality R (`_cz_cond_progs` + `_cond_R`)."""
    progs, tables, signs, timeout = _cz_cond_progs(cfg, m, pair, knob, x0, dx, points, ngates, shots,
                                                   amp=amp)
    rq.setup(drv, m, progs)
    return _cond_R(drv, m, progs, pair, shots, signs, timeout)


class CZFrequency:
    """CZ conditionality vs the CZ carrier — qcal `cz.Frequency` (spec 01 §4.5, cz.py:1082-1089).
    Sweep the CZ drive freq ±`span` Hz around `CZ/freq`, measure the conditionality R at each, and write
    the argmax (the freq at which the CZ is most conditional). `ngates` CZ repetitions amplify a residual
    phase error. A parabola refines the peak; the run fails (no write) if R never rises above a genuine
    conditional response (0.5). Form-dispatched through `_cz_cond_progs` (spec 04 §4.1): the coupler
    form sweeps the coupler NCO, the two-qubit-drive form retunes BOTH qubit NCOs in lock-step."""

    def __init__(self, cfg, pair, span=6e6, points=15, ngates=1, shots=120):
        self.cfg, self.pair = cfg, (int(pair[0]), int(pair[1]))
        self.span, self.points = float(span), int(points)
        self.ngates, self.shots = int(ngates), int(shots)
        self.data, self.fit = {}, {}

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg, pair, pk = self.cfg, self.pair, pair_key(self.pair)
        f_cz = float(cfg[f"two_qubit/{pk}/CZ/freq"])
        lo = units._freq_code(f_cz - self.span, m.params)
        hi = units._freq_code(f_cz + self.span, m.params)
        x0, dx, xs = sweep_q16(lo, hi, self.points)
        xax = np.array([units.code_to_freq(int(x), m.params) for x in xs])
        R, P0 = _cz_cond_R(cfg, drv, m, pair, "freq", x0, dx, self.points, self.ngates, self.shots)
        self.data = {pair: {"x": xax, "R": R, "P0": P0}}
        i = int(np.argmax(R))
        fit = fits.fit_parabola(xax, R)
        f_star = (float(fit.value) if fit.ok and fit.params["a"] < 0
                  and xax.min() <= fit.value <= xax.max() else float(xax[i]))
        self.fit = {pair: fit}
        ok = float(R[i]) > 0.5
        prop = {f"two_qubit/{pk}/CZ/freq": f_star} if ok else {}
        return Result(ok, self.data, self.fit, prop, cfg, f"CZFrequency {pair}")


class CZAmpFreqSweep:
    """The 2D CZ (amp × freq) SEED landscape — qcal `cz.AmpFreqSweep` (spec 14 §2 row 4.2,
    cz.py:247-420). The reference runs this BEFORE the 1D chain: with the pair uncalibrated, a 1D
    freq sweep taken at the wrong amp (or a 1D amp ladder at the wrong carrier) can miss the
    activation entirely, so the whole (amp, f_CZ) plane is mapped once and `CZFrequency` /
    `CZAmplitude` start from its argmax.

    Every point is exactly `CZFrequency`'s measurement — the four conditionality-tomography
    sequences ({control |0>/|1>} × {Y90/X90 close}) giving R, 1 at a conditional-π CZ (spec 01
    §4.5) — laid out as a HOST amp loop × the ON-CORE freq sweep:

      - the freq axis is `CZFrequency`'s: ±`span` Hz around `CZ/freq`, bound as `k_cz_cond`'s swept
        knob (both NCOs in lock-step in the drive form, the coupler's in the coupler form);
      - the amp axis is the HOST knob — `rq.write_slot("amp")` on the cz slot of EVERY core that
        plays a physical CZ line, i.e. BOTH lines in lock-step in the two-qubit-drive form (qcal
        sweeps `.../{idx}/kwargs/amp` and `.../{idx+1}/kwargs/amp` over the same array; X6Y3 holds
        the lines equal, 04 §4.3) — on the ONE resident image, no recompile (the
        `RelativePhase`/`Window` host pacing, spec 08 §4). A pulse's envelope is a normalized shape
        and its amp a slot field, so a slot write IS a recompile at that amp.

    Analysis: the plain 2D argmax of R, no fit. R = 1 (its ceiling) wherever the {|11>, |02>}
    rotation closes a full 2π — that is what stamps the conditional π — so the landscape is the
    familiar CHEVRON: the 2π contour runs through the resonant config-amp point and opens outwards,
    a detuned cell reaching the same total angle only at a higher amp (the generalised Rabi rate
    √(Ω² + Δ²)). The apex of that chevron is the resonance, and it is the argmax whenever the amp
    axis brackets it — an off-resonance arm reaches R = 1 only where a grid cell happens to land on
    the contour, so it undershoots the apex, by a margin that depends on where the grid cuts the
    arms. qcal reads the same landscape off a 31×31 plot and picks the apex by eye; we take the
    argmax and write it, leaving the refinement to the 1D chain (`CZFrequency` parabola,
    `CZAmplitude` ladder) that follows. Writes BOTH `CZ/freq` and the drive `amp` (`_cz_pulse_set`
    — every physical line), and refuses (no write) below a genuine conditional response
    (max R ≤ 0.5), `CZFrequency`'s acceptance.

    `data[pair]["R"]` is indexed [amp, freq] (the run order: one host amp step per on-core sweep;
    qcal's `_R` is the transpose); `data[pair]["P0"][k]` keeps the amp row's four tomography
    branches (`CZFrequency`'s convention)."""

    def __init__(self, cfg, pair, amps=None, span=6e6, points=15, ngates=1, shots=120):
        self.cfg, self.pair = cfg, (int(pair[0]), int(pair[1]))
        self.amps = None if amps is None else np.asarray(amps, dtype=float)
        self.span, self.points = float(span), int(points)
        self.ngates, self.shots = int(ngates), int(shots)
        self.data, self.fit = {}, {}

    def _amp_axis(self, cfg) -> np.ndarray:
        """The normalized-amp axis: the caller's (qcal passes `amplitudes` explicitly) or 7 points
        over 0.5x–1.5x the config amp — the window that brackets the 2π round trip whenever the
        config amp is within a factor ~2 of it."""
        if self.amps is not None:
            return self.amps
        a0 = _cz_amp(cfg, self.pair)
        return np.clip(np.linspace(0.5 * a0, 1.5 * a0, 7), 0.002, 0.999)

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg, pair, pk = self.cfg, self.pair, pair_key(self.pair)
        amps = self._amp_axis(cfg)
        f_cz = float(cfg[f"two_qubit/{pk}/CZ/freq"])
        lo = units._freq_code(f_cz - self.span, m.params)
        hi = units._freq_code(f_cz + self.span, m.params)
        x0, dx, xs = sweep_q16(lo, hi, self.points)
        fax = np.array([units.code_to_freq(int(x), m.params) for x in xs])
        progs, tables, signs, timeout = _cz_cond_progs(cfg, m, pair, "freq", x0, dx, self.points,
                                                       self.ngates, self.shots, amp=float(amps[0]))
        rq.setup(drv, m, progs)
        lines = {q: t.slot_of("cz") for q, t in tables.items() if "cz" in t.pulses}
        R, branches = np.zeros((len(amps), self.points)), []
        for k, a in enumerate(amps):
            for q, slot in lines.items():          # LOCK-STEP: every physical line at the same amp
                rq.write_slot(drv, m, q, progs[q], "gate", slot, "amp",
                              units._amp_code(float(a)))
            R[k], P0 = _cond_R(drv, m, progs, pair, self.shots, signs, timeout)
            branches.append(P0)
        ka, kf = np.unravel_index(int(np.argmax(R)), R.shape)
        amp_star, f_star, r_max = float(amps[ka]), float(fax[kf]), float(R[ka, kf])
        self.data = {pair: {"amps": amps, "freqs": fax, "R": R, "P0": branches}}
        self.fit = {pair: {"amp": amp_star, "freq": f_star, "R": r_max}}
        ok = r_max > 0.5
        prop = {f"two_qubit/{pk}/CZ/freq": f_star,
                f"two_qubit/{pk}/CZ/pulse": _cz_pulse_set(cfg, pair, "amp", amp_star)} if ok else {}
        return Result(ok, self.data, self.fit, prop, cfg, f"CZAmpFreqSweep {pair}")


class CZAmplitude:
    """CZ conditionality vs the CZ drive amp, error-amplification ladder — qcal `cz.Amplitude` (spec 01
    §4.5, cz.py:745-778). For each `n_gates` in the ladder (1, 3, 5, 7, 9 — odd keeps the net gate a CZ),
    sweep the CZ amp in a window that NARROWS with n (relative half-width `window`/n around the running
    estimate) and take the parabola vertex of R (its maximum) as the refined amp — n CZs amplify a
    residual amplitude error into a sharper R peak. Writes the drive `amp` — EVERY physical line
    (`_cz_pulse_set`): the coupler pulse, or both lines of the two-qubit-drive form swept in lock-step
    (qcal calibrates them jointly, spec 04 §4.3)."""

    def __init__(self, cfg, pair, n_gates=(1, 3, 5, 7, 9), window=0.3, points=11, shots=120):
        self.cfg, self.pair = cfg, (int(pair[0]), int(pair[1]))
        self.n_gates = tuple(int(n) for n in n_gates)
        self.window, self.points, self.shots = float(window), int(points), int(shots)
        self.data, self.fit = {}, {}

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg, pair, pk = self.cfg, self.pair, pair_key(self.pair)
        amp = _cz_amp(cfg, pair)
        passes, moved = [], False
        for n in self.n_gates:
            wn = self.window / n
            lo = units._amp_code(max(0.002, amp * (1 - wn)))
            hi = units._amp_code(min(0.999, amp * (1 + wn)))
            x0, dx, xs = sweep_q16(lo, hi, self.points)
            xax = np.array([int(x) / units.AMP_SCALE for x in xs])
            R, P0 = _cz_cond_R(cfg, drv, m, pair, "amp", x0, dx, self.points, n, self.shots,
                               amp=amp)
            fit = fits.fit_parabola(xax, R)
            if fit.ok and fit.params["a"] < 0 and xax.min() <= fit.value <= xax.max():
                amp, moved = float(np.clip(fit.value, 0.002, 0.999)), True
            passes.append({"n": n, "x": xax, "R": R, "fit": fit, "amp": amp})
        self.data = {pair: {"passes": passes}}
        self.fit = {pair: passes[-1]["fit"]}
        prop = {f"two_qubit/{pk}/CZ/pulse": _cz_pulse_set(cfg, pair, "amp", amp)} if moved else {}
        return Result(moved, self.data, self.fit, prop, cfg, f"CZAmplitude {pair}")


class RelativePhase:
    """CZ conditionality vs the TARGET line's tone phase — qcal `cz.RelativePhase` (spec 04 §4.3,
    cz.py:1482-1673). The two-qubit-drive CZ is two simultaneous tones on the pair's own gate
    channels; R peaks where the target line's phase (relative to the control's) combines them into
    the most conditional gate. TWO-QUBIT-DRIVE FORM ONLY — a coupler pair has one physical tone, no
    relative phase to turn (loud assert).

    HOST-PACED like Window (spec 08 §4): the `k_cz_cond` tomography programs are compiled ONCE at the
    config phase with a DEAD on-core sweep (npts=1, the FREQ word pinned at `CZ/freq`), and each
    point is a `rq.write_slot("phase")` on the TARGET core's cz slot — the next rerun's
    init_pulse_params programs it, no recompile — followed by the four `_cond_R` tomography reruns.
    The sweep is `points` phases spanning `span` rad (default one full turn) centred on the config
    value; the refine is exactly CZFrequency's (parabola vertex a<0 in-range, argmax fallback; qcal
    fits the same parabola, cz.py:1650-1673) and the run fails (no write) below a genuine conditional
    response (max R ≤ 0.5). Writes the target drive entry's `kwargs/phase` (`_cz_rel_phase_set`)."""

    def __init__(self, cfg, pair, span=2 * math.pi, points=15, ngates=1, shots=120):
        self.cfg, self.pair = cfg, (int(pair[0]), int(pair[1]))
        self.span, self.points = float(span), int(points)
        self.ngates, self.shots = int(ngates), int(shots)
        self.data, self.fit = {}, {}

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg, pair, pk = self.cfg, self.pair, pair_key(self.pair)
        assert not cz_coupler_form(cfg, pair), \
            f"pair {pair}: RelativePhase needs the two-qubit-drive form (a coupler pair has ONE tone)"
        ctrl, tgt = pair
        p0 = float(_cz_entry(cfg, pair, drive=1).get("kwargs", {}).get("phase", 0.0))
        phis = p0 + np.linspace(-self.span / 2, self.span / 2, self.points)
        progs, tables, signs, timeout = _cz_cond_progs(cfg, m, pair, "freq",
                                                       _cz_freq_word(cfg, pair, m), 0,
                                                       1, self.ngates, self.shots)
        rq.setup(drv, m, progs)
        slot = tables[tgt].slot_of("cz")
        R, branches = np.zeros(self.points), []
        for k, phi in enumerate(phis):
            rq.write_slot(drv, m, tgt, progs[tgt], "gate", slot, "phase",
                          units._phase_code(float(phi)))
            r, P0 = _cond_R(drv, m, progs, pair, self.shots, signs, timeout)
            R[k] = float(r[0])
            branches.append(P0)
        self.data = {pair: {"x": phis, "R": R, "P0": branches}}
        i = int(np.argmax(R))
        fit = fits.fit_parabola(phis, R)
        phi_star = (float(fit.value) if fit.ok and fit.params["a"] < 0
                    and phis.min() <= fit.value <= phis.max() else float(phis[i]))
        self.fit = {pair: fit}
        ok = float(R[i]) > 0.5
        prop = {f"two_qubit/{pk}/CZ/pulse": _cz_rel_phase_set(cfg, pair, phi_star)} if ok else {}
        return Result(ok, self.data, self.fit, prop, cfg, f"RelativePhase {pair}")


# ── CZ local single-qubit phases (spec 01 §4.6) ──────────────────────────────────────────────────

def _fringe_peak(phi, P):
    """The φ at which a Ramsey P fringe peaks — the first-harmonic phase of P over the full-turn sweep
    φ (`atan2` of Σ (P−P̄)·e^{iφ}), which is robust to the cosine sign that a curve fit leaves ambiguous.
    Returns (φ_peak in (−π, π], the fringe contrast)."""
    P = np.asarray(P, float)
    z = complex(np.sum((P - P.mean()) * np.exp(1j * np.asarray(phi, float))))
    return math.atan2(z.imag, z.real), float(P.max() - P.min())


def _mean_offset(a: float, b: float) -> float:
    """The midpoint of a and b along the SHORTER arc, wrap-aware, in (−π, π]. Arithmetic — NOT a
    circular mean. Well-conditioned only when the two offsets are CLOSE: antipodal inputs (~π apart)
    sit exactly on its wrap boundary, where noise flips which side the midpoint lands — π away. That
    degeneracy is why the spectator-|1> branch's conditional π is removed (`_branch_correction`)
    BEFORE the midpoint, never averaged across."""
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return (a + d / 2 + math.pi) % (2 * math.pi) - math.pi


def _branch_correction(off0: float, off1: float) -> float:
    """The written local-phase correction from the two spectator-branch fringe peaks (`off0` =
    spectator |0>, `off1` = spectator |1>) — qcal's combination (cz.py:2013-2051): remove the
    conditional π from the |1> branch, then the shorter-arc midpoint.

    Derivation on OUR sequence `Y90 · cz · Rz(φ) · Y90`, reading P(1) (qcal closes Y⁻90 and reads
    P(0) — the same fringe): the prep puts the ACTIVE qubit on +X and the CZ leaves it at azimuth
    ψ_s per spectator state s — ψ₀ = θ_local + θ_ZZ, ψ₁ = ψ₀ + π + δ (the conditional π plus the
    residual conditionality error δ = −2·θ_ZZ − π, zero at an exact conditional π).

    A frame word SUBTRACTS from the accrued angle — the kernels apply it as the close's
    `set_phase_offset`, so P(1) = (1 + cos(ψ_s − φ))/2 and each branch peaks at φ = +ψ_s (the
    convention is pinned on RTL by the CZRPE zero-amp gate, where a planted config vz comes back
    NEGATED; spec 14 §3 finding 9). So off0 = ψ₀ is ITSELF the correction — writing it makes the
    kernels' effective local phase θ_raw − ψ₀, i.e. exactly −θ_ZZ — and off1 = ψ₀ + π + δ carries
    the conditional π. Removing it (qcal's NEGATIVE-amplitude fit prior on the |1> branch: its
    reported zero is the raw fringe's trough = peak ∓ π) leaves ψ₀ + δ, and the shorter-arc midpoint
    with off0 is ψ₀ + δ/2 = θ_local − π/2, which lands the effective local phase on **+π/2 exactly,
    whatever θ_ZZ is** — the residual conditional error split evenly between the branches, and
    continuous in δ of either sign. The RAW midpoint instead sits at ψ₀ + (π + δ)/2 ≡ local ± π/2
    with a noise-unstable sign (the raw peaks are ~π apart — `_mean_offset`'s wrap boundary),
    calibrating an exp(iπ/4·ZZ)-like composite instead of diag(1,1,1,−1) (spec 04 §3, fixed in X1).
    The ∓π shift's own sign is immaterial mod 2π."""
    return _mean_offset(off0, off1 - math.pi)


def _cz_local_set(cfg, pair, zi: float, iz: float) -> list:
    """A fresh CZ pulse list with the control's ZI and the target's IZ virtual-Z phases set — the
    entries matched by CHANNEL (04 §4.2), never by position (the LocalPhases proposal: both live in
    the list leaf, so the whole list is written back)."""
    pulses = copy.deepcopy(_cz_pulses(cfg, pair))
    for q, phase in ((int(pair[0]), zi), (int(pair[1]), iz)):
        p = _cz_vz_entry(pulses, q)
        if p is None:
            raise ValueError(f"pair {tuple(pair)}: no virtualz entry for qubit {q} in the CZ pulse list")
        p.setdefault("kwargs", {})["phase"] = float(phase)
    return pulses


def _phi_sweep(points: int):
    """A full-turn virtual-Z sweep for the Ramsey-peak cals (LocalPhases / SpectatorPhase):
    `points` phases from −π, endpoint-EXCLUSIVE — +π IS −π, `units._phase_code` wraps both to the
    SAME code, so an inclusive −π→+π span silently collapses to a zero step (the X3-caught bug: the
    class-level sweep never swept; the analysis tests used their own endpoint-exclusive axes).
    Returns (plain-code start, plain-code step, the radian axis `_fringe_peak` runs on)."""
    c0 = units._phase_code(-math.pi)
    dc = 0 if points <= 1 else round((1 << 16) / points)         # 2π = 65536 plain-code units
    phi_ax = (c0 + dc * np.arange(points)) * math.pi / (1 << 15)
    return c0, dc, phi_ax


def _cz_spectator_set(cfg, pair, q: int, phase: float) -> list:
    """A fresh CZ pulse list with SPECTATOR qubit q's virtual-Z phase set — the channel-matched
    entry (04 §4.2) of the PAIR's list (spectator corrections live with the gate that causes them).
    The SpectatorPhase proposal payload (a list leaf, written back whole, like `_cz_local_set`)."""
    pulses = copy.deepcopy(_cz_pulses(cfg, pair))
    p = _cz_vz_entry(pulses, q)
    if p is None:
        raise ValueError(f"pair {tuple(pair)}: no virtualz entry for spectator qubit {q} "
                         f"in the CZ pulse list")
    p.setdefault("kwargs", {})["phase"] = float(phase)
    return pulses


class SpectatorPhase:
    """CZ spectator phase — qcal `SpectatorPhase` (spec 04 §4.5; cz.py:2246-2647). The pair's CZ
    also kicks the frame of a NEIGHBOUR that is not in the gate (static ZZ to the pair during the
    window + Stark from the CZ tones); this measures that kick with a Ramsey ON THE SPECTATOR
    bracketing one CZ fire — qcal's circuits: C0 = Y90(s) · CZ(pair) · Y⁻90(s), C1 the same with an
    X (X90·X90) prepping the CONDITIONAL qubit first (cz.py:2400-2434) — and writes the spectator's
    channel-matched virtual-Z entry of the pair's own `CZ/pulse` list.

    Kernel mapping (3 cores, all `k_cz_local`): the SPECTATOR qubit compiles the COUPLER_FORM
    ACTIVE branch — a plain `Y90 · window · Rz(φ) · Y90` Ramsey with NO retune and NO cz slot,
    exactly what a bystander plays — with its bracketed window bound to `czd + 2·LEAD`, which puts
    the pair's tone (a LEAD short of t_close, the DRIVE_FORM phasor-regen slot) nominally
    MID-WINDOW with a LEAD margin each side. The margin is load-bearing: each core's grid is its
    own `now() + period`, and the spectator's SHORTER gate table (no cz slot) makes its
    init preamble faster, so its grid runs tens of batches EARLY relative to the pair (the co-sim
    measures ~58 on sim-2q1c — the first unequal-table multi-core kernel; equal tables skew ≤2).
    The symmetric bracket absorbs any sub-LEAD offset by construction, and the extra idle inside it
    only adds the spectator's static idle phase, which belongs in the correction anyway. The two
    PAIR cores compile the DRIVE_FORM SPECTATOR role — prep |0>/|1> (the runtime `sp` scalar, live
    only on the `conditional` core; the other binds sp=0 dead), retune f_GE → f_CZ, fire their OWN
    cz line in lock-step, no readout. Only the spectator reads out. TWO-QUBIT-DRIVE FORM ONLY (loud
    assert) — the coupler-form CONTROL+COUPLER role wiring is not built (no coupler config carries
    spectator entries; X6Y3, the chip in scope, has no couplers).

    Analysis: per conditional branch c the fringe P(1) peaks at φ = +ψ_c — the φ the kernels then
    SUBTRACT, so it is the absolute vz correction that cancels the accrued kick (exactly
    LocalPhases' convention/X1 derivation, `_branch_correction`); the
    write-back is qcal's combination — the PLAIN MEAN of the two branch corrections (np.mean,
    cz.py:2512-2530), with NO conditional-π removal, because the spectator is OUTSIDE the gate: its
    |1>-branch fringe shifts only by the small spectator conditionality δ, never by π (qcal fits
    both branches with the same cosine prior). Documented deviations from qcal: (1) the mean is
    wrap-aware (`_mean_offset`, the shorter-arc midpoint — identical to np.mean except across the
    ±π boundary, where np.mean is wrong; well-conditioned here since the branches sit CLOSE, far
    from the midpoint's antipodal degeneracy); (2) qcal sweeps the config vz param itself,
    re-writing the gate each point — ours applies the identical phase as the on-core Rz(φ) between
    the CZ window and the close (a Z commutes across the idle), one resident batched program; (3)
    qcal reads P(0) under a Y⁻90 close with per-branch cosine fits — ours reads P(1) under the Y90
    close with the `_fringe_peak` first harmonic, the same extremum (the LocalPhases machinery)."""

    def __init__(self, cfg, pair, spectator, conditional=None, points=15, shots=120):
        self.cfg, self.pair = cfg, (int(pair[0]), int(pair[1]))
        self.spectator = int(spectator)
        self.conditional = int(conditional) if conditional is not None else self.pair[0]
        assert self.spectator not in self.pair, \
            f"spectator {self.spectator} is a member of pair {self.pair} — the pair's own phases " \
            f"are LocalPhases' job"
        assert self.conditional in self.pair, \
            f"conditional qubit {self.conditional} must be a member of pair {self.pair}"
        assert not cz_coupler_form(cfg, self.pair), \
            f"pair {self.pair}: SpectatorPhase supports the two-qubit-drive form only — the " \
            f"coupler-form (CONTROL+COUPLER) roles are not wired (no coupler config carries " \
            f"spectator entries)"
        self.points, self.shots = int(points), int(shots)
        self.data, self.fit = {}, {}

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg, pair, pk = self.cfg, self.pair, pair_key(self.pair)
        spect, cond = self.spectator, self.conditional
        czd = _cz_dur_batches(cfg, pair, m)
        fcz = _cz_freq_word(cfg, pair, m)
        hpi = pack16(units._phase_code(math.pi / 2))
        c0, dc, phi_ax = _phi_sweep(self.points)                 # full turn, endpoint-exclusive
        p0, dp = pack16(int(c0)), pack16(int(dc))

        shelf, sw_binds, tail = _sandwich_binds(cfg, pair, m)    # zeros on plain pairs
        ro_s, demod_s, code_s, dur_s, ddly_s = readout_tables(cfg, spect, m)
        xd = gate_pulse(cfg, spect, m).dur_batches(m, GATE_CH)
        # the same grid as LocalPhases' drive form: prep + tone + close with a LEAD gap each side
        # (an EF sandwich adds a (LEAD + EF X) segment on each side of the tone, 2·tail)
        period = grid_period(relax_batches(cfg, m), 3 * xd + czd + 2 * LEAD + 2 * tail,
                             dur_s, ddly_s)
        timeout = batch_timeout(self.points * self.shots * period)
        common = dict(npts=self.points, shots=self.shots, period=period, ddly=ddly_s, hpi=hpi,
                      xd=xd, p0=p0, dp=dp)
        progs = {}
        # the spectator: the COUPLER_FORM ACTIVE Ramsey, window czd + 2·LEAD (the docstring's
        # ±LEAD bracket margins; a sandwich widens the pair's tone block by 2·tail, so the window
        # widens with it and keeps the same LEAD margins); fcz/sw/fef/tail dead (it never retunes),
        # sp dead (it never preps).
        progs[spect] = compile_kernel(
            kernels.k_cz_local, m,
            tables=dict(gate=ParamTable(GATE_CH, qubit_freq(cfg, spect),
                                        {"x90": gate_pulse(cfg, spect, m)}),
                        ro=ro_s, demod=demod_s),
            out=Array(self.points), code=code_s, role=kernels.ACTIVE, form=kernels.COUPLER_FORM,
            czd=czd + 2 * LEAD + 2 * tail, fcz=0, fef=0, sw=0, tail=0, sp=0,
            **common, **x90_vz(cfg, spect))
        drive_of = {pair[0]: 0, pair[1]: 1}
        for q in pair:            # the pair: DRIVE_FORM SPECTATOR roles, each firing its OWN line
            gate = cz_drive_table(cfg, pair, q, drive_of[q], m, czd)
            ro, demod, code, dur, ddly = readout_tables(cfg, q, m)
            fixed = {} if q == cond else {"sp": 0}   # only the conditional core preps |0>/|1>
            progs[q] = compile_kernel(kernels.k_cz_local, m,
                                      tables=dict(gate=gate, ro=ro, demod=demod),
                                      out=Array(self.points), code=code, role=kernels.SPECTATOR,
                                      form=kernels.DRIVE_FORM, czd=czd, fcz=fcz, tail=tail,
                                      **sw_binds[q], **fixed, **common, **x90_vz(cfg, q))
        rq.setup(drv, m, progs)
        sign, offs, branches = res_sign(cfg, spect), [], {}
        for sp in (0, 1):
            out = rq.rerun(drv, m, progs, params={cond: {"sp": sp}}, results=["out"],
                           timeout=timeout)
            P = population(out[spect]["out"], self.shots, sign)
            off, contrast = _fringe_peak(phi_ax, P)
            branches[sp] = {"P": P, "offset": off, "contrast": contrast}
            offs.append(off if contrast > 0.15 else math.nan)
        ok = not any(math.isnan(o) for o in offs)
        corr = _mean_offset(offs[0], offs[1]) if ok else math.nan   # qcal's mean, wrap-aware — NO π
        self.data = {pair: {"phi": phi_ax, "spectator": spect, "branches": branches, "corr": corr}}
        prop = {f"two_qubit/{pk}/CZ/pulse": _cz_spectator_set(cfg, pair, spect, corr)} if ok else {}
        return Result(ok, self.data, self.fit, prop, cfg, f"SpectatorPhase {pair} Q{spect}")


class LocalPhases:
    """CZ local single-qubit phases — qcal `cz.LocalPhases` (spec 01 §4.6, cz.py:1848-1946). A CZ leaves
    each qubit with a single-qubit Z the config must undo; this measures it with a Ramsey around ONE CZ
    on the ACTIVE qubit (`Y90 · cz · Rz(φ) · Y90`) while the partner SPECTATES in |0> or |1>. The φ that
    peaks the ACTIVE fringe IS the phase that qubit accrued for the spectator state — a frame word
    subtracts, so the peak is the correction; the correction written is
    the |0>-branch peak midpointed with the |1>-branch peak AFTER its conditional π is removed
    (`_branch_correction` — the qcal parity fix of spec 04 §3/X1; the raw midpoint carried ±π/2 of the
    conditional phase with a noise-unstable sign). Run with the ACTIVE role on the control (→ ZI) and
    the target (→ IZ), written into each qubit's channel-matched virtual-Z entry (`_cz_local_set`) — the
    same entries the CZ kernel macro folds in (spec 01 §3). Fix order: conditional phase (§4.5) → local
    phases → re-check (§4.8); local phases do NOT change the conditionality R (a single-qubit Z shifts
    both control branches equally), so they are the last coherent-error pass, not part of the R chain.
    Form-dispatched (spec 04 §4.1): on a two-qubit-drive pair both cores fire their own CZ line around
    the ACTIVE Ramsey — the retune round-trip's constant frame slip lands in exactly the vz this
    calibration writes (spec 04 §1)."""

    def __init__(self, cfg, pair, points=15, shots=120):
        self.cfg, self.pair = cfg, (int(pair[0]), int(pair[1]))
        self.points, self.shots = int(points), int(shots)
        self.data, self.fit = {}, {}

    def _offset(self, drv, m, active, spect, ccore, czd, hpi, p0, dp, phi_ax):
        """One qubit's local phase: compile `k_cz_local` form-dispatched (spec 04 §4.1 — coupler form:
        3 cores, the COUPLER drives; drive form: the 2 qubit cores each firing their OWN CZ line,
        `ccore` is None), setup, and rerun the spectator in |0>/|1> — the two peak offsets combined by
        `_branch_correction` (the |1> branch's conditional π removed before the midpoint)."""
        cfg = self.cfg
        coupler = ccore is not None
        form = kernels.COUPLER_FORM if coupler else kernels.DRIVE_FORM
        fcz = 0 if coupler else _cz_freq_word(cfg, self.pair, m)
        shelf, sw_binds, tail = _sandwich_binds(cfg, self.pair, m)   # zeros on plain/coupler pairs
        ro_a, demod_a, code_a, dur_a, ddly_a = readout_tables(cfg, active, m)
        xd = gate_pulse(cfg, active, m).dur_batches(m, GATE_CH)
        # drive form: a LEAD gap on each side of the retuned CZ segment (prep → tone, tone → close);
        # an EF sandwich adds a (LEAD + EF X) segment on each side of the tone (2·tail)
        period = grid_period(relax_batches(cfg, m),
                             3 * xd + czd + (0 if coupler else 2 * LEAD) + 2 * tail, dur_a, ddly_a)
        timeout = batch_timeout(self.points * self.shots * period)
        if coupler:
            gate_a = ParamTable(GATE_CH, qubit_freq(cfg, active), {"x90": gate_pulse(cfg, active, m)})
            gate_s = ParamTable(GATE_CH, qubit_freq(cfg, spect), {"x90": gate_pulse(cfg, spect, m)})
        else:                                    # each core's cz slot is ITS OWN line of the pair
            drive_of = {int(self.pair[0]): 0, int(self.pair[1]): 1}
            gate_a = cz_drive_table(cfg, self.pair, active, drive_of[active], m, czd)
            gate_s = cz_drive_table(cfg, self.pair, spect, drive_of[spect], m, czd)
        ro_s, demod_s, code_s, dur_s, ddly_s = readout_tables(cfg, spect, m)
        common = dict(npts=self.points, shots=self.shots, period=period, ddly=ddly_a, form=form,
                      hpi=hpi, xd=xd, czd=czd, fcz=fcz, tail=tail, p0=p0, dp=dp)
        progs = {}
        progs[active] = compile_kernel(kernels.k_cz_local, m,
                                       tables=dict(gate=gate_a, ro=ro_a, demod=demod_a),
                                       out=Array(self.points), code=code_a, role=kernels.ACTIVE, sp=0,
                                       **sw_binds[active], **common, **x90_vz(cfg, active))
        progs[spect] = compile_kernel(kernels.k_cz_local, m,               # sp unbound → rerun scalar
                                      tables=dict(gate=gate_s, ro=ro_s, demod=demod_s),
                                      out=Array(self.points), code=code_s, role=kernels.SPECTATOR,
                                      **sw_binds[spect], **common, **x90_vz(cfg, spect))
        if coupler:
            progs[ccore] = compile_kernel(kernels.k_cz_local, m,
                                          tables=dict(gate=cz_table(cfg, self.pair, m, czd), ro=ro_a,
                                                      demod=demod_a),
                                          out=Array(self.points), code=code_a, role=kernels.COUPLER,
                                          sp=0, fef=0, sw=0, **common, **x90_vz(cfg, active))
        rq.setup(drv, m, progs)
        sign, offs, data = res_sign(cfg, active), [], {}
        for sp in (0, 1):
            out = rq.rerun(drv, m, progs, params={spect: {"sp": sp}}, results=["out"], timeout=timeout)
            P = population(out[active]["out"], self.shots, sign)
            off, contrast = _fringe_peak(phi_ax, P)
            data[sp] = {"P": P, "offset": off, "contrast": contrast}
            offs.append(off if contrast > 0.15 else math.nan)
        if any(math.isnan(o) for o in offs):
            return math.nan, data
        return _branch_correction(offs[0], offs[1]), data

    def run(self, drv) -> Result:
        m = socmap(drv)
        cfg, pair, pk = self.cfg, self.pair, pair_key(self.pair)
        ctrl, tgt = pair
        ccore = coupler_core(cfg, pair) if cz_coupler_form(cfg, pair) else None
        czd = _cz_dur_batches(cfg, pair, m)
        hpi = pack16(units._phase_code(math.pi / 2))
        c0, dc, phi_ax = _phi_sweep(self.points)     # full turn, endpoint-exclusive (the X3 fix —
        p0, dp = pack16(int(c0)), pack16(int(dc))    # ±π share one code, c0→c1 collapsed to dc = 0)
        zi, zdata = self._offset(drv, m, ctrl, tgt, ccore, czd, hpi, p0, dp, phi_ax)   # control → ZI
        iz, idata = self._offset(drv, m, tgt, ctrl, ccore, czd, hpi, p0, dp, phi_ax)   # target  → IZ
        self.data = {pair: {"phi": phi_ax, "ZI": zdata, "IZ": idata, "zi": zi, "iz": iz}}
        ok = not (math.isnan(zi) or math.isnan(iz))
        prop = {f"two_qubit/{pk}/CZ/pulse": _cz_local_set(cfg, pair, zi, iz)} if ok else {}
        return Result(ok, self.data, self.fit, prop, cfg, f"LocalPhases {pair}")
