"""QuantumModel — the ADC seam (spec 05 §3). The cocotb bench calls a model once per batch with
the current DAC samples; the model returns the ADC samples that close the physics loop. Models run
IN the sim process and are selected at runtime over Pyro5 (a JSON-serializable spec → build_model),
because the co-sim fixture is session-scoped — one sim process serves the whole test run, so the
model must be reconfigurable without restarting it.

The qutip dependency is confined to TwoLevelModel (imported inside its constructor); ZeroModel and
LoopbackModel are pure numpy.
"""

from __future__ import annotations

import cmath
import math
from collections import deque
from typing import Protocol

import numpy as np

from riscq.map import ADC_BATCH, BATCH_SIZE
from riscq.pulses import units

_I16_MIN, _I16_MAX = -(1 << 15), (1 << 15) - 1


def _clip16(x) -> np.ndarray:
    """Round to nearest and clip to SInt16 (the converter code range)."""
    return np.clip(np.rint(x), _I16_MIN, _I16_MAX).astype(np.int64)


class QuantumModel(Protocol):
    def dac_ids(self) -> list[int]:
        """Which physical DACs this model reads (the bench samples only these each batch)."""
        ...

    def adc_batch(self, t: int, dac: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        """t: batch index. dac[i]: this batch's BATCH_SIZE int16 samples of DAC i.
        Returns {adc_id: ADC_BATCH int16 samples} for the ADCs this model drives. Causal."""
        ...


class ZeroModel:
    """ADC silence — the default (DAC-only tests). Drives nothing; the bench holds every ADC at 0."""

    def dac_ids(self) -> list[int]:
        return []

    def adc_batch(self, t, dac):
        return {}


class LoopbackModel:
    """A DAC → ADC path (spec 05 §3): ADC lane j = gain · DAC[src] sample 4·j (decimate the 16
    DAC samples/batch down to the ADC's 4/batch), delayed by `delay` batches.

    The 4× decimation is exactly why the demod frequency code is 4× the DAC's for the same physical
    tone: a DAC carrier at code F has per-DAC-sample phase advance π·F/2^15, so sampling every 4th
    DAC sample gives per-ADC-sample advance π·(4F)/2^15 — the tone appears on the ADC at demod code
    4F. The M3 golden loops the gate DAC (0) back into the core's ADC (0) and finds that peak."""

    def __init__(self, gain: float = 1.0, delay: int = 0, src: int = 0, dst: int = 0):
        self.gain = float(gain)
        self.delay = int(delay)
        self.src = int(src)
        self.dst = int(dst)
        self._buf: deque = deque()

    def dac_ids(self) -> list[int]:
        return [self.src]

    def adc_batch(self, t, dac):
        lanes = _clip16(self.gain * dac[self.src][0:BATCH_SIZE:ADC_BATCH].astype(float))
        self._buf.append(lanes)
        if len(self._buf) <= self.delay:
            return {self.dst: np.zeros(ADC_BATCH, dtype=np.int64)}
        return {self.dst: self._buf.popleft()}


class ToneModel:
    """A batch-time-locked CW ADC tone at a PHYSICAL frequency (mirrors PulseTableSocSim.adcWord),
    generated from the physical ADC sample rate (ADC_BATCH samples/batch, one batch per dsp cycle):

        adc[j] = amp · cos(2π · f_hz · (ADC_BATCH·t + j) / (ADC_BATCH · dsp_freq_hz) + phase)

    Because it is a function of the batch time `t` (not of the exact sim cycle), it stays
    phase-coherent with the free-running demod LO regardless of co-sim sampling granularity — the
    robust readout-datapath source (a loopback that re-decimates the DAC every cycle is not, so it
    is only a coarse echo). The demod code that matches this tone is the measurement that PINS the
    demod freq↔code convention: for f_hz it lands at units.demod_freq_to_code(f_hz) = 4·freq_to_code
    (the DAC's code for the same f), because the ADC has 4 samples/batch vs the DAC's 16."""

    def __init__(self, m, adc: int = 0, freq_hz: float = 0.0, amp: float = 20000.0,
                 phase: float = 0.0):
        self.adc = int(adc)
        self.amp = float(amp)
        self.phase = float(phase)
        self._w = 2.0 * math.pi * float(freq_hz) / (ADC_BATCH * m.params.dsp_freq_hz)

    def dac_ids(self) -> list[int]:
        return []

    def adc_batch(self, t, dac):
        k = np.arange(ADC_BATCH)
        return {self.adc: _clip16(self.amp * np.cos(self._w * (ADC_BATCH * t + k) + self.phase))}


class TwoLevelModel:
    """qutip-backed driven two-level qubit (spec 05 §3) — the calibration-grade ADC model.

    Drive: reads the core's gate DAC each batch and rotates the qubit by `rabi_rad_per_amp · amp_est`
    (the MAGNITUDE, from amp_est = sqrt(2·mean(sample²)), a phase-blind RMS of the batch's RF drive
    — so a mis-scaled amplitude under/over-rotates, which M4's Amplitude calibration recovers) about
    the xy-plane AXIS recovered by demodulating that same DAC against the qubit frequency `f_ge`
    (see `_drive_axis`). A resonant drive (f_drive == f_ge) has a fixed axis; a detuned carrier — or
    a virtual-Z `set_phase` — ramps the axis, so the Ramsey fringe (two X90s around a wait) and any
    phase error fall straight out of the drive, with NO explicit free-precession term. M3's resonant
    x-rotations are the f_drive == f_ge, axis-≈const case (magnitude unchanged), so it stays exact.

    Readout: emits a CW tone on the core's ADC whose complex amplitude tracks <σz> (soft
    measurement — projective collapse is OMITTED by default; a state-tracking tone + optional
    shot noise is sufficient for every M4 acceptance). Ground (<σz>=+1) and excited (<σz>=−1) tones
    are π out of phase, so the demod's integrated-real SIGN discriminates the state (the `res` bit).

    Projective readout (`collapse=True`, spec 08 §2.4 / B1): needed to validate counts mode, where
    `sign(real)` of a *sampled* shot must be binomial, not a smeared step of ⟨σz⟩. At each readout
    window the model samples s ∈ {0,1} with p₁ = (1−⟨σz⟩)/2, emits the definite-state tone (+amp for
    s=0, −amp for s=1) for the whole window, and collapses the Bloch vector to the sampled pole — so
    repeated reads of a re-prepared superposition give *bimodal* IQ clusters with binomial statistics.
    The window is detected from the READOUT DRIVE (channel 1 → the core's readout DAC): every
    measurement plays it (spec §2.1), and its rising edge is the window opening. Soft mode (default)
    still ignores ch1, so every existing test is unchanged. No qutip solve — the model already tracks
    the Bloch vector, so collapse is a pole assignment.

    Dispersive readout (`chi != 0`, spec 13 Q2): the flat tone above is a caricature — its |0>/|1>
    tones are exactly π out of phase at EVERY frequency, so cluster separation is strictly ∝ |z| and
    the |0>-magnitude peak and the max-separation frequency coincide (which would make Separation's
    acceptance gate vacuous). With `chi` set, the readout is instead a driven RESONATOR whose centre
    shifts by ±chi with the qubit state (`_dispersive`), so the two states' IQ responses differ in
    magnitude AND phase, and max |S21| at |0> (at f_r + chi) is NOT max separation (at f_r). Default
    `chi = 0` ⇒ the flat-tone path, bit-identical to before.

    AC-Stark drive phase (`stark_rad_per_sigma != 0`, spec 13 Q3): the drive above rotates ONLY about
    its xy-axis, so a gate carries no phase error and the X90's virtual-Z pair is correct at 0 — i.e.
    the Phase calibration has nothing to find. With `stark_rad_per_sigma` set, every driven batch also
    rotates the Bloch vector about +z by `stark_rad_per_sigma · amp_est` (a detuning proportional to
    the drive, which is what an ac-Stark shift is), interleaved with the xy rotation. A pulse of drive
    integral σ (= base.gate_sigma, the same Σ amp_est that sets θ = rabi_rad_per_amp·σ) therefore
    accrues ε = stark_rad_per_sigma·σ radians of Z — the phase the calibration's virtual-Z corrects.
    Default 0 ⇒ no z-rotation, bit-identical to before.

    Ground-truth attributes (M4's Amplitude/Frequency/T1/T2 calibrations recover these): `f_ge`,
    `rabi_rad_per_amp`, `t1`, `t2` (t1/t2 in BATCHES). qutip prepares the initial state and defines
    the Pauli operators; the per-batch evolution is the exact analytic Bloch-vector map (a full
    qutip solve per batch would make the co-sim far too slow). Optional gaussian readout noise
    (`noise_scale`, seeded by `noise_seed`) gives realistic IQ scatter for the readout calibrations."""

    _DRIVE_FLOOR = 100.0   # amp_est below this = idle batch (DAC ≈ 0), no rotation

    def __init__(self, m, core: int = 0, rabi_rad_per_amp: float = 0.0, readout_code: int = 2048,
                 readout_amp: float = 20000.0, readout_phase: float = 0.0, f_ge: float = 0.0,
                 t1: float | None = None, t2: float | None = None, init_excited: bool = False,
                 noise_scale: float = 0.0, noise_seed: int = 0, collapse: bool = False,
                 f_r: float = 0.0, kappa: float = 0.0, chi: float = 0.0,
                 stark_rad_per_sigma: float = 0.0):
        import qutip   # confined to TwoLevelModel (spec 05 §3)
        self._q = qutip

        self.params = m.params
        self.gate_dac = m.gate_dac(core)
        self.ro_dac = m.ro_dac(core)   # readout-drive DAC: the projective window trigger (collapse mode)
        self.adc = m.adc_of(core)
        self.rabi_rad_per_amp = float(rabi_rad_per_amp)
        self.stark_rad_per_sigma = float(stark_rad_per_sigma)   # the drive's Z rotation (spec 13 Q3)
        self.readout_code = int(readout_code)
        self.readout_amp = float(readout_amp)
        self.readout_phase = float(readout_phase)
        self.f_ge = float(f_ge)
        self._f_ge_code = units._freq_code(self.f_ge, m.params)   # PLAIN reference code (16-bit phase math)
        self.t1 = t1
        self.t2 = t2

        psi = qutip.basis(2, 1) if init_excited else qutip.basis(2, 0)
        rho = qutip.ket2dm(psi)
        # Bloch vector (bx, by, bz); bz = <σz> (ground = +1). qutip seeds it, numpy evolves it.
        self._b = np.array([qutip.expect(qutip.sigmax(), rho),
                            qutip.expect(qutip.sigmay(), rho),
                            qutip.expect(qutip.sigmaz(), rho)], dtype=float)
        self._t1_decay = math.exp(-1.0 / t1) if t1 else 1.0
        self._t2_decay = math.exp(-1.0 / t2) if t2 else 1.0
        self._noise_scale = float(noise_scale)
        self._rng = np.random.default_rng(noise_seed)
        # projective readout (collapse mode): a decoupled RNG for the shot sampling (so collapse
        # statistics don't depend on whether readout noise is drawing), a rising-edge detector on the
        # readout drive, and the per-window latched tone amplitude (±readout_amp for the sampled state).
        self.collapse = bool(collapse)
        self._crng = np.random.default_rng(noise_seed + 0xC0BE)
        self._ro_active = False
        self._shot_amp = 0.0
        # dispersive readout (chi != 0): a driven resonator at f_r of linewidth kappa, pulled ±chi by
        # the qubit state. chi = 0 (default) keeps the flat π-out-of-phase tone above, untouched.
        self.f_r, self.kappa, self.chi = float(f_r), float(kappa), float(chi)
        assert not self.chi or self.kappa > 0, "a dispersive readout (chi != 0) needs kappa > 0"

    def dac_ids(self) -> list[int]:
        return [self.gate_dac, self.ro_dac] if (self.collapse or self.chi) else [self.gate_dac]

    def sigma_z(self) -> float:
        return float(self._b[2])

    def state(self):
        """The current density matrix as a qutip Qobj: ρ = (I + b·σ)/2."""
        q, (bx, by, bz) = self._q, self._b
        return 0.5 * (q.qeye(2) + bx * q.sigmax() + by * q.sigmay() + bz * q.sigmaz())

    def adc_batch(self, t, dac):
        samples = dac[self.gate_dac].astype(float)
        amp_est = math.sqrt(2.0 * float(np.mean(samples * samples)))
        if amp_est > self._DRIVE_FLOOR:
            if self.rabi_rad_per_amp:
                self._rotate(self.rabi_rad_per_amp * amp_est, self._drive_axis(t, samples))
            if self.stark_rad_per_sigma:      # the ac-Stark Z, accrued WITH the drive (spec 13 Q3)
                self._rotate_z(self.stark_rad_per_sigma * amp_est)
        if self.t1 or self.t2:
            self._b[0] *= self._t2_decay
            self._b[1] *= self._t2_decay
            self._b[2] = 1.0 + (self._b[2] - 1.0) * self._t1_decay

        amp = self._projective_amp(dac) if self.collapse else self.readout_amp * self._b[2]
        # (soft: |0>→+amp, |1>→−amp, a continuous step of ⟨σz⟩; projective: a latched ±amp per shot)
        if self.chi:                                  # dispersive: the resonator answers the DRIVE
            lanes = self._dispersive(dac[self.ro_dac], amp / self.readout_amp)
        else:
            k = np.arange(ADC_BATCH)
            ang = math.pi * self.readout_code * (ADC_BATCH * t + k) / (1 << 15) + self.readout_phase
            lanes = amp * np.cos(ang)
        if self._noise_scale:
            lanes = lanes + self._rng.normal(0.0, self._noise_scale, ADC_BATCH)
        return {self.adc: _clip16(lanes)}

    def _dispersive(self, ro, sz: float) -> np.ndarray:
        """The dispersive readout response (spec 13 Q2): this batch's ADC lanes.

        The resonator is a Lorentzian S(f) = 1 / (1 + 2i(f − f_r ∓ chi)/kappa) whose centre is pulled
        to f_r + chi by |0> and f_r − chi by |1>; the qubit's state enters as the mixture
        S = (1−p₁)·S|0> + p₁·S|1> (p₁ = (1 − ⟨σz⟩)/2, so a collapsed shot is one pure branch). Off
        resonance the two states' responses differ in BOTH magnitude and phase — the point of the whole
        thing: argmax |S(|0>)| sits at f_r + chi, argmax |S(|0>) − S(|1>)| at f_r.

        The response is built FROM THE DRIVE ITSELF: decimate the readout-drive DAC to the ADC's lanes,
        form its analytic signal (the quadrature from the carrier ω, which `_carrier_code` recovers),
        scale by |S| and rotate by arg S. So the emitted tone carries the drive's own amplitude (the
        lever qcal's Fidelity sweeps — the flat tone's amplitude is a model constant) and, crucially,
        the drive's own PHASE. Re-synthesizing it from absolute time instead would ring: the readout
        carrier's phase is referenced to the pulse start, not to t=0, so at any code but the one the
        grid period was rounded for (grid_period's `% 8`) the demod would see a different phase every
        shot — a smeared IQ cluster that fakes exactly the frequency dependence Separation measures.
        Silent when undriven."""
        x = ro.astype(float)
        if math.sqrt(2.0 * float(np.mean(x * x))) <= self._DRIVE_FLOOR:
            return np.zeros(ADC_BATCH)                            # not driven ⇒ no response
        code = self._carrier_code(x)                              # the drive's own DAC-rate code
        p1 = 0.5 * (1.0 - sz)
        f = units.code_to_freq(code, self.params)
        s = (1.0 - p1) * self._lorentzian(f, +1.0) + p1 * self._lorentzian(f, -1.0)
        i_lanes, q_lanes = self._analytic(x, code)                # the drive, as the ADC samples it
        psi = self.readout_phase + cmath.phase(s)                 # Re{(I + iQ) · |S|·e^{i·arg S}}
        return ((self.readout_amp / units.AMP_SCALE) * abs(s)
                * (i_lanes * math.cos(psi) - q_lanes * math.sin(psi)))

    @staticmethod
    def _analytic(x: np.ndarray, code: int):
        """(I, Q) of the drive at the ADC's sample instants: fit x[k] = p·cos(ωk) + q·sin(ωk) over the
        batch (ω = π·code/2^15), which pins the tone's amplitude AND phase exactly, then read both
        quadratures off the fit at the decimated lanes (ADC lane j = DAC sample 4j, see LoopbackModel).
        A 2-parameter fit over the whole batch, rather than a sample-to-sample quadrature, so the
        pulse's first batch has no edge artifact and the model keeps no history."""
        w = math.pi * code / (1 << 15)
        k = np.arange(BATCH_SIZE)
        (p, q), *_ = np.linalg.lstsq(np.stack([np.cos(w * k), np.sin(w * k)], axis=1), x, rcond=None)
        j = np.arange(0, BATCH_SIZE, BATCH_SIZE // ADC_BATCH)     # the ADC's DAC-sample instants
        cj, sj = np.cos(w * j), np.sin(w * j)
        return p * cj + q * sj, p * sj - q * cj      # A·cos(ωj + θ), A·sin(ωj + θ)

    def _lorentzian(self, f: float, state: float) -> complex:
        """S(f) for a resonator pulled by `state` (+1 = |0> → f_r + chi, −1 = |1> → f_r − chi)."""
        return 1.0 / (1.0 + 2j * (f - self.f_r - state * self.chi) / self.kappa)

    @staticmethod
    def _carrier_code(x: np.ndarray) -> int:
        """The carrier code of a clean tone, from ONE batch of its DAC samples: for
        x[k] = A·cos(ωk + θ) the identity x[k+1] + x[k−1] = 2·cos(ω)·x[k] holds sample by sample, so a
        least-squares cos ω over the batch inverts to ω — amplitude- and phase-blind, and exact (to
        well under 1 LSB of code) for the square-envelope tone the readout drive always is. The code is
        the per-sample phase advance in pi units, SF(16): ω/π · 2^15 (riscq.pulses.units)."""
        num = float(np.sum(x[1:-1] * (x[2:] + x[:-2])))
        den = 2.0 * float(np.sum(x[1:-1] * x[1:-1]))
        w = math.acos(min(1.0, max(-1.0, num / den))) if den else 0.0
        return round(w / math.pi * (1 << 15))

    def _projective_amp(self, dac) -> float:
        """Projective readout amplitude: on the readout drive's rising edge (window opening) sample
        s ~ Bernoulli((1−<σz>)/2), collapse the Bloch vector to that pole, and latch the definite-state
        tone amplitude (+amp for s=0/|0>, −amp for s=1/|1>). Held across the window so the decoder
        integrates a definite state, and until the next window so the reads in between are self-consistent."""
        ro = dac[self.ro_dac].astype(float)
        ro_on = math.sqrt(2.0 * float(np.mean(ro * ro))) > self._DRIVE_FLOOR
        if ro_on and not self._ro_active:                          # rising edge = window opening
            s = 1 if self._crng.random() < (1.0 - self._b[2]) / 2.0 else 0
            self._b[:] = (0.0, 0.0, 1.0 - 2.0 * s)                 # collapse to the sampled pole
            self._shot_amp = self.readout_amp * (1.0 - 2.0 * s)    # +amp (s=0) / −amp (s=1)
        elif not ro_on:
            self._shot_amp = 0.0                                   # no drive ⇒ silent (decoder ignores it)
        self._ro_active = ro_on
        return self._shot_amp

    def _drive_axis(self, t: int, samples: np.ndarray) -> float:
        """The xy-plane rotation axis (rad): arg of the gate DAC demodulated against f_ge over this
        batch's 16 samples at ABSOLUTE sample index s = 16·t + k. A resonant drive (f_drive == f_ge)
        yields a fixed axis; a detuned one ramps at (f_drive − f_ge), which is exactly the Ramsey /
        virtual-Z physics. The f_ge·s phase is reduced mod 2^16 as integers (the hardware's 16-bit
        phase wrap) before scaling, so it stays exact for large batch times."""
        k = np.arange(BATCH_SIZE)
        ph = (self._f_ge_code * (BATCH_SIZE * t + k)) % (1 << 16)
        b = complex(np.sum(samples * np.exp(-1j * math.pi * ph / (1 << 15))))
        return math.atan2(b.imag, b.real)

    def _rotate_z(self, zeta: float) -> None:
        """Rotate the Bloch vector by ζ about +z — the same right-hand sense as `_rotate`'s xy-axis
        rotations, so a POSITIVE `stark_rad_per_sigma` advances the qubit's phase in the same
        direction as a positive channel phaseOffset advances the drive axis (which is why the Phase
        calibration recovers a POSITIVE virtual-Z for a positive Stark term: the frame has to chase
        the phase the qubit accrued)."""
        c, s = math.cos(zeta), math.sin(zeta)
        bx, by = self._b[0], self._b[1]
        self._b[0] = bx * c - by * s
        self._b[1] = bx * s + by * c

    def _rotate(self, theta: float, phi: float) -> None:
        """Rotate the Bloch vector by θ about the xy-plane axis (cos φ, sin φ, 0) (Rodrigues);
        φ = 0 reduces to M3's rotation about +x."""
        c, s = math.cos(theta), math.sin(theta)
        cp, sp = math.cos(phi), math.sin(phi)
        bx, by, bz = self._b
        n_dot_b = bx * cp + by * sp
        self._b[0] = bx * c + sp * bz * s + cp * n_dot_b * (1.0 - c)
        self._b[1] = by * c - cp * bz * s + sp * n_dot_b * (1.0 - c)
        self._b[2] = bz * c + (cp * by - sp * bx) * s


class MultiModel:
    """Several independent QuantumModels driven together (spec 13 §8): each sub-model reads its OWN
    core's gate DAC and drives its OWN core's readout tone. On this build several cores SHARE a readout
    DAC/ADC (`ro_dac`/`adc_of` map cores 0–6 → DAC 14 / ADC 0), so simultaneous readout is frequency
    multiplexed — the sub-models emit at different readout codes and their ADC lanes SUM on the shared
    converter (the physical converter summing), each core's demod integrating out its own tone.
    `dac_ids()` unions the sub-models'; `adc_batch()` sums per ADC (re-clipping to the converter range)."""

    def __init__(self, submodels):
        self._models = list(submodels)

    def dac_ids(self) -> list[int]:
        ids: list[int] = []
        for md in self._models:
            for did in md.dac_ids():
                if did not in ids:
                    ids.append(did)
        return ids

    def adc_batch(self, t, dac):
        out: dict[int, np.ndarray] = {}
        for md in self._models:
            for aid, lanes in md.adc_batch(t, dac).items():
                out[aid] = lanes if aid not in out \
                    else _clip16(out[aid].astype(float) + lanes.astype(float))
        return out


def build_model(spec: dict, m) -> QuantumModel:
    """Construct the model named by a JSON-serializable spec (`{"kind": ...}`), in the sim
    process — this is what makes the session-scoped fixture reconfigurable over Pyro5."""
    kind = spec.get("kind", "zero")
    if kind == "zero":
        return ZeroModel()
    if kind == "multi":
        return MultiModel([build_model(s, m) for s in spec["models"]])
    if kind == "loopback":
        return LoopbackModel(gain=spec.get("gain", 1.0), delay=spec.get("delay", 0),
                             src=spec.get("src", 0), dst=spec.get("dst", 0))
    if kind == "tone":
        return ToneModel(m, adc=spec.get("adc", 0), freq_hz=spec.get("freq_hz", 0.0),
                         amp=spec.get("amp", 20000.0), phase=spec.get("phase", 0.0))
    if kind == "twolevel":
        return TwoLevelModel(
            m, core=spec.get("core", 0), rabi_rad_per_amp=spec.get("rabi_rad_per_amp", 0.0),
            readout_code=spec.get("readout_code", 2048), readout_amp=spec.get("readout_amp", 20000.0),
            readout_phase=spec.get("readout_phase", 0.0), f_ge=spec.get("f_ge", 0.0),
            t1=spec.get("t1"), t2=spec.get("t2"), init_excited=spec.get("init_excited", False),
            noise_scale=spec.get("noise_scale", 0.0), noise_seed=spec.get("noise_seed", 0),
            collapse=spec.get("collapse", False), f_r=spec.get("f_r", 0.0),
            kappa=spec.get("kappa", 0.0), chi=spec.get("chi", 0.0),
            stark_rad_per_sigma=spec.get("stark_rad_per_sigma", 0.0))
    raise ValueError(f"unknown QuantumModel kind {kind!r}")
