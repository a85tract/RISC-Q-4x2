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

    def ground_truth(self) -> dict:
        """The model's exact state, as plain JSON-serialisable data.

        This is a TEST OBSERVATION, not part of the ADC seam: the bench exposes it over
        `drv.sim.model_state()` so a co-sim test can assert what the played signal did to the
        qubit without re-measuring it through shot statistics
        (specs/software-test-refactor/01 §4.3). It has no hardware counterpart —
        `RemoteDriver` has no `.sim` — so nothing under `riscq/` outside `riscq/sim/` may call it.

        Models with no quantum state return `{}`.
        """
        return {}

    def fast_forward(self, n: int) -> None:
        """Advance `n` IDLE batches without generating ADC samples (specs/software/15 §3.3).

        Only a host-side driver that knows the DAC is silent may call this — the co-sim never does,
        it simulates every batch for real. It exists for the spec-15 virtual QubiC board, whose
        passive-reset gaps (tens of thousands of batches per shot) would otherwise cost one Python
        `adc_batch` call each. Models with no time evolution ignore it.
        """
        return None


class ZeroModel:
    """ADC silence — the default (DAC-only tests). Drives nothing; the bench holds every ADC at 0."""

    def dac_ids(self) -> list[int]:
        return []

    def adc_batch(self, t, dac):
        return {}

    def ground_truth(self) -> dict:
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
                 stark_rad_per_sigma: float = 0.0, decay_in_window: bool = False):
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
        # T1 DURING the window (spec 15 §3.3's t1-tail scenario). Off by default: the latched tone
        # above is constant across the window, so |0> and |1> come back as two clean clouds. With it
        # on, a |1> shot may jump to |0> partway through and integrate part of each tone — the heavy
        # tail between the clusters that a real T1 puts there, and the one regime where an
        # unsupervised GMM and a linear discriminant genuinely disagree.
        self.decay_in_window = bool(decay_in_window)
        self._decay_at = -1        # batches into the window at which the latched |1> jumps to |0>
        self._win_k = 0
        # dispersive readout (chi != 0): a driven resonator at f_r of linewidth kappa, pulled ±chi by
        # the qubit state. chi = 0 (default) keeps the flat π-out-of-phase tone above, untouched.
        self.f_r, self.kappa, self.chi = float(f_r), float(kappa), float(chi)
        assert not self.chi or self.kappa > 0, "a dispersive readout (chi != 0) needs kappa > 0"

    def dac_ids(self) -> list[int]:
        return [self.gate_dac, self.ro_dac] if (self.collapse or self.chi) else [self.gate_dac]

    def sigma_z(self) -> float:
        return float(self._b[2])

    def ground_truth(self) -> dict:
        """`bloch` = (⟨σx⟩, ⟨σy⟩, ⟨σz⟩). The readout can only ever see ⟨σz⟩ (the tone amplitude
        tracks it); the xy components — what a virtual-Z calibration actually moves — are visible
        here and nowhere else."""
        return {"bloch": [float(v) for v in self._b]}

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
        self._relax()

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

    def _relax(self) -> None:
        """One batch of T1/T2 relaxation: an AFFINE map on the Bloch vector — xy scaled by the T2
        decay, z pulled toward the ground pole (+1) by the T1 one. Linear, so `fast_forward` can
        apply n of them in closed form."""
        if self.t1 or self.t2:
            self._b[0] *= self._t2_decay
            self._b[1] *= self._t2_decay
            self._b[2] = 1.0 + (self._b[2] - 1.0) * self._t1_decay

    def fast_forward(self, n: int) -> None:
        """n idle batches of relaxation in closed form (spec 15 §3.3): the n-th power of `_relax`'s
        affine map. Exact in exact arithmetic — `x**n` and n multiplications differ only by float
        rounding — and the ADC is not generated, so the caller must know the DAC is silent."""
        if n <= 0 or not (self.t1 or self.t2):
            return
        self._b[0] *= self._t2_decay ** n
        self._b[1] *= self._t2_decay ** n
        self._b[2] = 1.0 + (self._b[2] - 1.0) * self._t1_decay ** n

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
            # how many batches this |1> survives: geometric with the per-batch decay probability
            # 1 − exp(−1/T1), i.e. the discrete T1 the rest of the model already uses
            self._decay_at = int(self._crng.geometric(1.0 - self._t1_decay)) \
                if (s and self.decay_in_window and self.t1) else -1
            self._win_k = 0
        elif not ro_on:
            self._shot_amp = 0.0                                   # no drive ⇒ silent (decoder ignores it)
            self._decay_at = -1
        if ro_on:
            self._win_k += 1
            if 0 <= self._decay_at <= self._win_k:                 # the jump, mid-window
                self._shot_amp = self.readout_amp
                self._b[:] = (0.0, 0.0, 1.0)
                self._decay_at = -1
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


class ThreeLevelModel:
    """A driven qutrit (|0>, |1>, |2>) for the EF calibration and 3-level readout (spec two-qubit/01
    §4.1, §5). Two transitions share the core's gate DAC, ONE carrier at a time (the kernel re-programs
    the channel freq around the EF segment): a GE-resonant drive rotates {|0>, |1>}, an EF-resonant one
    rotates {|1>, |2>}. Each batch the model demodulates the gate DAC against BOTH f_ge and f_ef and
    drives whichever transition the carrier matches — by the same amp_est·rate angle and demod-recovered
    axis TwoLevelModel uses — so a mis-scaled EF amplitude under/over-rotates (EF Amplitude recovers it)
    and a detuned EF carrier ramps the axis (the EF Ramsey fringe, EF Frequency).

    Readout: each level emits the readout tone at a distinct phase (`level_phases`), so |0>/|1>/|2>
    land as three separated IQ clusters a ClassifierN tells apart. In `collapse` mode the readout
    drive's rising edge samples a definite level from |psi|^2, collapses to it, and latches that level's
    tone for the window (bi/trimodal shot statistics); soft mode emits the population-weighted phasor.

    Pure-state numpy evolution, no decoherence: the light co-sim gates recover a planted EF freq / amp
    from an undamped fringe / Rabi, which is all Q1 needs (full-physics runs are manual, notebook-style)."""

    _DRIVE_FLOOR = 100.0                                    # amp_est below this = idle batch (as TwoLevelModel)
    _DEFAULT_PHASES = (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)   # 3 tones 120° apart → 3 clusters

    def __init__(self, m, core: int = 0, f_ge: float = 0.0, f_ef: float = 0.0,
                 rabi_ge_rad_per_amp: float = 0.0, rabi_ef_rad_per_amp: float = 0.0,
                 readout_code: int = 2048, readout_amp: float = 20000.0, readout_phase: float = 0.0,
                 level_phases=None, init_level: int = 0, collapse: bool = False,
                 t1: float | None = None, noise_scale: float = 0.0, noise_seed: int = 0):
        self.params = m.params
        self.gate_dac = m.gate_dac(core)
        self.ro_dac = m.ro_dac(core)
        self.adc = m.adc_of(core)
        self.rabi = {(0, 1): float(rabi_ge_rad_per_amp), (1, 2): float(rabi_ef_rad_per_amp)}
        self._code = {(0, 1): units._freq_code(float(f_ge), m.params),      # plain reference codes
                      (1, 2): units._freq_code(float(f_ef), m.params)}
        self.readout_code = int(readout_code)
        self.readout_amp = float(readout_amp)
        self.readout_phase = float(readout_phase)
        self.level_phases = tuple(self._DEFAULT_PHASES if level_phases is None else level_phases)
        self._psi = np.zeros(3, dtype=complex)
        self._psi[int(init_level)] = 1.0
        # amplitude damping toward |0> per IDLE batch (t1 in BATCHES): the batched cals fire on a fixed
        # grid whose idle head is ≫ t1, so each shot starts from |0> — WITHOUT it the model (which has no
        # auto-reset) carries the previous shot's collapsed level into the next prep and the sweep scrambles
        # (spec two-qubit/01 §6, the counterpart of TwoLevelModel's t1/t2). Pure-state: the excited
        # amplitudes shrink and the lost norm flows to |0>, so the idle head relaxes |1>/|2> away.
        self.t1 = t1
        self._t1_decay = math.exp(-1.0 / t1) if t1 else 1.0
        self.collapse = bool(collapse)
        self._noise_scale = float(noise_scale)
        self._rng = np.random.default_rng(noise_seed)
        self._crng = np.random.default_rng(noise_seed + 0xC0BE)
        self._ro_active = False
        self._shot_level = None            # latched sampled level (collapse mode); None ⇒ silent

    def dac_ids(self) -> list[int]:
        return [self.gate_dac, self.ro_dac] if self.collapse else [self.gate_dac]

    def populations(self) -> np.ndarray:
        return np.abs(self._psi) ** 2

    def ground_truth(self) -> dict:
        """`populations` = (P0, P1, P2). |2⟩ is invisible to the hardware `res` bit (one
        threshold, two outcomes), so leakage assertions have to come from here."""
        return {"populations": [float(p) for p in self.populations()]}

    def _demod(self, t: int, samples: np.ndarray, code: int) -> complex:
        """The gate DAC demodulated against `code` over this batch (mirrors TwoLevelModel._drive_axis):
        magnitude tells which carrier is on, arg is the drive axis. Phase reduced mod 2^16 as ints."""
        k = np.arange(BATCH_SIZE)
        ph = (code * (BATCH_SIZE * t + k)) % (1 << 16)
        return complex(np.sum(samples * np.exp(-1j * math.pi * ph / (1 << 15))))

    def _rotate(self, pair, theta: float, phi: float) -> None:
        """Rotate the {a, b} 2-level subspace of psi by Bloch angle `theta` about xy-axis `phi`
        (U = exp(-i θ/2 (cosφ σx + sinφ σy)) embedded in C^3), leaving the third level untouched."""
        a, b = pair
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        em, ep = cmath.exp(-1j * phi), cmath.exp(1j * phi)
        pa, pb = self._psi[a], self._psi[b]
        self._psi[a] = c * pa - 1j * em * s * pb
        self._psi[b] = -1j * ep * s * pa + c * pb

    def adc_batch(self, t, dac):
        samples = dac[self.gate_dac].astype(float)
        amp_est = math.sqrt(2.0 * float(np.mean(samples * samples)))
        if amp_est > self._DRIVE_FLOOR:
            b01, b12 = (self._demod(t, samples, self._code[(0, 1)]),
                        self._demod(t, samples, self._code[(1, 2)]))
            pair, b = ((0, 1), b01) if abs(b01) >= abs(b12) else ((1, 2), b12)
            if self.rabi[pair]:
                self._rotate(pair, self.rabi[pair] * amp_est, math.atan2(b.imag, b.real))
        elif self.t1:
            self._relax()          # idle batch: amplitude-damp |1>/|2> toward |0> (the grid reset)

        phasor = self._projective_phasor(dac) if self.collapse else \
            sum(self.populations()[k] * cmath.exp(1j * self.level_phases[k]) for k in range(3))
        k = np.arange(ADC_BATCH)
        ang = math.pi * self.readout_code * (ADC_BATCH * t + k) / (1 << 15) + self.readout_phase
        lanes = self.readout_amp * (phasor.real * np.cos(ang) - phasor.imag * np.sin(ang))
        if self._noise_scale:
            lanes = lanes + self._rng.normal(0.0, self._noise_scale, ADC_BATCH)
        return {self.adc: _clip16(lanes)}

    def _relax(self) -> None:
        """Amplitude-damp toward |0> one idle batch: shrink the |1>/|2> amplitudes by the t1 factor and
        pour the lost norm back into |0> (phase-preserving, so the state stays pure). Over the grid's
        idle head (≫ t1) |1>/|2> vanish and the qubit resets to |0> — the reset the batched sweep needs."""
        self._psi[1] *= self._t1_decay
        self._psi[2] *= self._t1_decay
        lost = 1.0 - float(np.vdot(self._psi, self._psi).real)          # norm bled off |1>/|2>
        if lost > 0.0:
            a0 = self._psi[0]
            mag0 = math.sqrt(max(0.0, abs(a0) ** 2 + lost))
            self._psi[0] = mag0 * (a0 / abs(a0)) if abs(a0) > 1e-12 else mag0

    def _projective_phasor(self, dac) -> complex:
        """On the readout drive's rising edge sample a definite level from |psi|^2, collapse to it, and
        latch that level's tone phasor for the window (silent when undriven)."""
        ro = dac[self.ro_dac].astype(float)
        ro_on = math.sqrt(2.0 * float(np.mean(ro * ro))) > self._DRIVE_FLOOR
        if ro_on and not self._ro_active:
            p = self.populations()
            lvl = int(self._crng.choice(3, p=p / p.sum()))
            self._psi[:] = 0.0
            self._psi[lvl] = 1.0
            self._shot_level = lvl
        elif not ro_on:
            self._shot_level = None
        self._ro_active = ro_on
        return 0j if self._shot_level is None else cmath.exp(1j * self.level_phases[self._shot_level])


class TwoQubitModel:
    """Two qutrits + a flux coupler for the CZ calibration (spec two-qubit/01 §6). ONE model holds the
    JOINT 3x3 state psi[a, b] (a = control level, b = target level) — the CZ entangles the pair, so it
    cannot be two independent per-qubit models. It reads three DACs (both gate channels + the coupler
    drive) and drives both qubits' readout tones, frequency-multiplexed onto the shared ADC.

    Every rotation is the SAME demodulate-then-rotate mechanism the one-qubit models use (a resonant
    carrier gives a fixed axis, a detuned one ramps it — the Ramsey/off-resonant-Rabi physics with no
    explicit precession term), lifted to the joint state:

      - single-qubit GE/EF drive: demod that qubit's gate DAC against its f_ge/f_ef, rotate its {0,1}
        or {1,2} subspace for every level of the partner (a product-preserving embedded rotation),
        exactly as ThreeLevelModel does for one qutrit.
      - parametric CZ (`coupler` set — the default): demod the coupler DAC against f_CZ =
        |f_EF(target) - f_GE(control)| (the |11>-|02> detuning, spec 01 §4.2) and rotate the
        {|11>, |02>} PSEUDO-QUBIT by rabi_cz*amp_est about the recovered axis. On resonance (coupler
        carrier == f_CZ) the axis is fixed, so the subspace Rabi-flops |11>->|02>->|11> and a full
        2*pi round trip is -I on {|11>, |02>} — i.e. |11> picks up the conditional pi phase that IS
        the CZ, while |00>/|01>/|10> are untouched. A detuned coupler carrier ramps the axis, which
        is exactly the off-resonant Rabi Ω²/(Ω²+Δ²) (rotating about a ramping equatorial axis at
        ramp-rate Δ == a static (Ω/2)σx-(Δ/2)σz), so the transfer peaks at f_CZ — the resonance the
        CZ Frequency cal finds.
      - two-qubit-drive CZ (`coupler=None`; a `build_model` spec WITHOUT a "coupler" key — the X6Y3
        form, spec 04 §4.6): no coupler exists, the CZ is two simultaneous in-band tones on the
        pair's OWN gate channels at f_CZ = (f_11 + f_02)/4 = (f_ge[0] + 2·f_ge[1] + f_ef[1])/4 (the
        drive-form seed arithmetic, spec 04 §1 — computed from the model's own spectrum, like the
        coupler path's detuning). Each gate DAC gains a THIRD demod, against f_CZ, and the per-batch
        argmax over {GE, EF, CZ} decides which drive that line is (the same comparative mechanism
        that already separates GE from EF); a batch whose carrier is the CZ tone contributes the
        phasor A_i·e^{iφ_i} (A_i = the line's RMS amplitude — detuning-blind, as the coupler path's
        amp_est; φ_i = its f_CZ-demod arg). The EFFECTIVE drive is the COHERENT SUM of the two lines,
        E = A_c·e^{iφ_c} + A_t·e^{iφ_t}, and the {|11>, |02>} pseudo-qubit rotates by rabi_cz·|E|
        about axis arg E — the SAME `_rotate(2, ...)` mechanics as the coupler path, so the detuning
        response (both NCOs retune LOCKSTEP, both demod args ramp together → the Ω²/(Ω²+Δ²) transfer
        peaking at f_CZ) and the conditional-π round trip are identical. What the closed form ADDS is
        the relative-phase dependence: equal lines give |E| = 2A·cos(Δφ/2) — maximal when the two
        absolute-time-referenced tones align (Δφ = 0), extinguished at Δφ = π — the real optimum the
        RelativePhase calibration finds. One line alone still activates at half strength.
      - residual ZZ (`zz_rad_per_batch`): the static ζ|11><11| term, a phase e^{-iζ} on |11> EVERY
        batch. In a target Ramsey it shifts the fringe frequency by ζ only when the control is |1>, so
        JAZZ recovers ζ = f(control=1) - f(control=0) (spec 01 §4.3). Default 0 = the part-1 zero-bias
        point (DC hardware, part 3, wires a nonzero ζ in later).

    Readout: each qubit emits the readout tone at a phase set by its level (`level_phases`, 3 tones
    120° apart -> 3 IQ clusters a ClassifierN tells apart); the two tones ride distinct readout codes
    and SUM on the shared ADC (frequency-multiplexed, each core's demod integrating out its own). Soft
    mode emits the population-weighted phasor of each qubit's MARGINAL; `collapse` mode samples one
    JOINT (a, b) from |psi|² on the readout-drive rising edge, collapses the pair to it, and latches
    each qubit's level for the window — so the two cores' per-shot bits are drawn from the correlated
    pair state and zip into the joint P(00..11) (spec 01 §5).

    Pure-state numpy evolution; optional `t1` amplitude-damps the pair toward |00> on IDLE batches so
    the batched grid's idle head resets each shot (the counterpart of ThreeLevelModel's reset). The
    light co-sim gates recover planted f_CZ / ζ / conditional phase; full physics is manual (§6)."""

    _DRIVE_FLOOR = 100.0
    _DEFAULT_PHASES = (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)

    def __init__(self, m, control: int = 0, target: int = 1, coupler: int | None = 2,
                 f_ge=(0.0, 0.0), f_ef=(0.0, 0.0), rabi_ge=(0.0, 0.0), rabi_ef=(0.0, 0.0),
                 rabi_cz_rad_per_amp: float = 0.0, zz_rad_per_batch: float = 0.0,
                 readout_code=(2048, 2048), readout_amp=(20000.0, 20000.0), readout_phase=(0.0, 0.0),
                 level_phases=None, init=(0, 0), collapse: bool = False, t1: float | None = None,
                 noise_scale: float = 0.0, noise_seed: int = 0):
        self.params = m.params
        self.gate = [m.gate_dac(control), m.gate_dac(target)]
        self.coupler_dac = None if coupler is None else m.gate_dac(coupler)
        self.ro_dac = m.ro_dac(control)                # the shared readout-drive DAC (window trigger)
        self.adc = [m.adc_of(control), m.adc_of(target)]
        self.rabi = [{(0, 1): float(rabi_ge[i]), (1, 2): float(rabi_ef[i])} for i in (0, 1)]
        self._code = [{(0, 1): units._freq_code(float(f_ge[i]), m.params),   # plain reference codes
                       (1, 2): units._freq_code(float(f_ef[i]), m.params)} for i in (0, 1)]
        if coupler is None:                            # drive form: the in-band (f_11 + f_02)/4 tone
            f_cz = (float(f_ge[0]) + 2.0 * float(f_ge[1]) + float(f_ef[1])) / 4.0
        else:                                          # |11>-|02> parametric resonance (spec 01 §4.2)
            f_cz = abs(float(f_ef[1]) - float(f_ge[0]))
        self._cz_code = units._freq_code(f_cz, m.params)
        self.rabi_cz = float(rabi_cz_rad_per_amp)
        self.zz = float(zz_rad_per_batch)
        self.readout_code = [int(readout_code[0]), int(readout_code[1])]
        self.readout_amp = [float(readout_amp[0]), float(readout_amp[1])]
        self.readout_phase = [float(readout_phase[0]), float(readout_phase[1])]
        self.level_phases = tuple(self._DEFAULT_PHASES if level_phases is None else level_phases)
        self._psi = np.zeros((3, 3), dtype=complex)
        self._psi[int(init[0]), int(init[1])] = 1.0
        self.t1 = t1
        self._t1_decay = math.exp(-1.0 / t1) if t1 else 1.0
        self.collapse = bool(collapse)
        self._noise_scale = float(noise_scale)
        self._rng = np.random.default_rng(noise_seed)
        self._crng = np.random.default_rng(noise_seed + 0xC0BE)
        self._ro_active = False
        self._shot = None                              # latched joint shot (a, b); None ⇒ silent

    def dac_ids(self) -> list[int]:
        ids = [self.gate[0], self.gate[1]]
        if self.coupler_dac is not None:
            ids.append(self.coupler_dac)
        if self.collapse and self.ro_dac not in ids:
            ids.append(self.ro_dac)
        return ids

    def populations(self) -> np.ndarray:
        return np.abs(self._psi) ** 2                  # [a, b] joint pops

    def marginals(self):
        p = self.populations()
        return p.sum(axis=1), p.sum(axis=0)            # (control, target) single-qubit populations

    def ground_truth(self) -> dict:
        """`populations` = the full 3×3 joint grid, `marginals` = (control, target). The joint
        state is what a CZ acts on and what per-core readout can only see marginals of."""
        ctrl, tgt = self.marginals()
        return {"populations": self.populations().tolist(),
                "marginals": [ctrl.tolist(), tgt.tolist()]}

    def _demod(self, t: int, samples: np.ndarray, code: int) -> complex:
        """The DAC demodulated against `code` over this batch (as ThreeLevelModel._demod): magnitude
        tells whether the carrier is on, arg is the drive axis. Phase reduced mod 2^16 as ints."""
        k = np.arange(BATCH_SIZE)
        ph = (code * (BATCH_SIZE * t + k)) % (1 << 16)
        return complex(np.sum(samples * np.exp(-1j * math.pi * ph / (1 << 15))))

    def _rotate(self, who: int, pair, theta: float, phi: float) -> None:
        """Rotate a 2-level subspace by Bloch angle `theta` about xy-axis `phi`
        (U = exp(-i θ/2 (cosφ σx + sinφ σy))). who=0/1 rotates the control/target qutrit's `pair`
        subspace across every partner level (a product-preserving embedded rotation); who=2 rotates the
        {|11>, |02>} pseudo-qubit (the parametric coupling)."""
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        em, ep = cmath.exp(-1j * phi), cmath.exp(1j * phi)
        if who == 2:                                   # the {|11>, |02>} coupling
            pa, pb = self._psi[1, 1], self._psi[0, 2]
            self._psi[1, 1] = c * pa - 1j * em * s * pb
            self._psi[0, 2] = -1j * ep * s * pa + c * pb
            return
        a, b = pair
        if who == 0:                                   # control subspace: rows a, b
            pa, pb = self._psi[a, :].copy(), self._psi[b, :].copy()
            self._psi[a, :] = c * pa - 1j * em * s * pb
            self._psi[b, :] = -1j * ep * s * pa + c * pb
        else:                                          # target subspace: columns a, b
            pa, pb = self._psi[:, a].copy(), self._psi[:, b].copy()
            self._psi[:, a] = c * pa - 1j * em * s * pb
            self._psi[:, b] = -1j * ep * s * pa + c * pb

    def _drive_qubit(self, idx: int, samples: np.ndarray, t: int):
        """Drive qubit `idx` from its gate DAC this batch: demod against its f_ge and f_ef, rotate
        whichever transition the carrier matches (as ThreeLevelModel). In the drive form (no coupler)
        the same channel also carries the pair's CZ line, so a THIRD demod against f_CZ joins the
        argmax: a CZ-carrier batch rotates nothing here and instead returns its line phasor
        A·e^{iφ} (A = the RMS amplitude, φ = the f_CZ-demod arg) for `_drive_cz_lines` to combine.
        Returns (driven, cz_phasor-or-None)."""
        samples = samples.astype(float)
        amp_est = math.sqrt(2.0 * float(np.mean(samples * samples)))
        if amp_est <= self._DRIVE_FLOOR:
            return False, None
        b01 = self._demod(t, samples, self._code[idx][(0, 1)])
        b12 = self._demod(t, samples, self._code[idx][(1, 2)])
        if self.coupler_dac is None:                   # drive form: the CZ tone rides this channel too
            bcz = self._demod(t, samples, self._cz_code)
            if abs(bcz) > abs(b01) and abs(bcz) > abs(b12):
                return True, amp_est * cmath.exp(1j * math.atan2(bcz.imag, bcz.real))
        pair, b = ((0, 1), b01) if abs(b01) >= abs(b12) else ((1, 2), b12)
        if self.rabi[idx][pair]:
            self._rotate(idx, pair, self.rabi[idx][pair] * amp_est, math.atan2(b.imag, b.real))
        return True, None

    def _drive_coupler(self, samples: np.ndarray, t: int) -> bool:
        """Drive the {|11>, |02>} coupling from the coupler DAC: demod against f_CZ, rotate the
        pseudo-qubit by rabi_cz*amp_est about the recovered axis. Returns whether it was driven."""
        samples = samples.astype(float)
        amp_est = math.sqrt(2.0 * float(np.mean(samples * samples)))
        if amp_est <= self._DRIVE_FLOOR:
            return False
        if self.rabi_cz:
            b = self._demod(t, samples, self._cz_code)
            self._rotate(2, None, self.rabi_cz * amp_est, math.atan2(b.imag, b.real))
        return True

    def _drive_cz_lines(self, e0, e1) -> None:
        """Drive-form CZ activation (spec 04 §4.6): combine the two gate lines' phasors COHERENTLY —
        E = A_c·e^{iφ_c} + A_t·e^{iφ_t} — and rotate the {|11>, |02>} pseudo-qubit by rabi_cz·|E|
        about axis arg E (the exact `_drive_coupler` mechanics with |E| in place of the one line's
        amp_est). Anti-phase lines cancel (|E| under the drive floor ⇒ no rotation) — the Δφ = π
        null of the 2A·cos(Δφ/2) closed form."""
        if not self.rabi_cz:
            return
        e = (e0 if e0 is not None else 0j) + (e1 if e1 is not None else 0j)
        if abs(e) <= self._DRIVE_FLOOR:
            return
        self._rotate(2, None, self.rabi_cz * abs(e), math.atan2(e.imag, e.real))

    def adc_batch(self, t, dac):
        d0, e0 = self._drive_qubit(0, dac[self.gate[0]], t)
        d1, e1 = self._drive_qubit(1, dac[self.gate[1]], t)
        if self.coupler_dac is not None:               # coupler form: the dedicated CZ-drive channel
            dc = self._drive_coupler(dac[self.coupler_dac], t)
        else:                                          # drive form: the two gate lines combine
            dc = False
            self._drive_cz_lines(e0, e1)
        if self.zz:                                    # the static ζ|11><11| term (spec 01 §4.3)
            self._psi[1, 1] *= cmath.exp(-1j * self.zz)
        if not (d0 or d1 or dc) and self.t1:
            self._relax()                              # idle batch: reset the pair toward |00>
        if self.collapse:
            self._update_shot(dac)
        return self._emit(t)

    def _emit(self, t: int) -> dict:
        """Both qubits' readout tones, frequency-multiplexed and summed on the shared ADC."""
        out: dict[int, np.ndarray] = {}
        k = np.arange(ADC_BATCH)
        for idx in (0, 1):
            phasor = self._phasor(idx)
            ang = math.pi * self.readout_code[idx] * (ADC_BATCH * t + k) / (1 << 15) \
                + self.readout_phase[idx]
            lanes = self.readout_amp[idx] * (phasor.real * np.cos(ang) - phasor.imag * np.sin(ang))
            a = self.adc[idx]
            out[a] = lanes if a not in out else out[a] + lanes
        if self._noise_scale:
            for a in out:
                out[a] = out[a] + self._rng.normal(0.0, self._noise_scale, ADC_BATCH)
        return {a: _clip16(v) for a, v in out.items()}

    def _phasor(self, idx: int) -> complex:
        """Qubit `idx`'s readout phasor: its collapsed level's tone (collapse mode) or the
        population-weighted sum over its marginal (soft mode)."""
        if self.collapse:
            return 0j if self._shot is None else cmath.exp(1j * self.level_phases[self._shot[idx]])
        pops = self.marginals()[idx]
        return sum(pops[L] * cmath.exp(1j * self.level_phases[L]) for L in range(3))

    def _update_shot(self, dac) -> None:
        """On the readout drive's rising edge sample one JOINT (a, b) from |psi|², collapse the pair to
        it, and latch it for the window (both qubits read the same correlated shot); silent when undriven."""
        ro = dac[self.ro_dac].astype(float)
        on = math.sqrt(2.0 * float(np.mean(ro * ro))) > self._DRIVE_FLOOR
        if on and not self._ro_active:
            p = self.populations().ravel()
            i = int(self._crng.choice(9, p=p / p.sum()))
            self._psi[:] = 0.0
            self._psi[i // 3, i % 3] = 1.0
            self._shot = (i // 3, i % 3)
        elif not on:
            self._shot = None
        self._ro_active = on

    def _relax(self) -> None:
        """Amplitude-damp the pair toward |00> one idle batch: shrink every amplitude but |00>'s by the
        t1 factor and pour the lost norm back into |00> (phase-preserving). Over the idle head (≫ t1)
        the pair resets to |00> — the reset the batched sweep needs (ThreeLevelModel._relax, lifted)."""
        a00 = self._psi[0, 0]
        self._psi *= self._t1_decay
        self._psi[0, 0] = a00
        lost = 1.0 - float(np.vdot(self._psi, self._psi).real)
        if lost > 0.0:
            mag = math.sqrt(max(0.0, abs(a00) ** 2 + lost))
            self._psi[0, 0] = mag * (a00 / abs(a00)) if abs(a00) > 1e-12 else mag


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

    def ground_truth(self) -> dict:
        """`models` = each sub-model's ground truth, in the order they were built — i.e. the order
        of the `models` list in the spec passed to `set_model`."""
        return {"models": [md.ground_truth() for md in self._models]}

    def fast_forward(self, n: int) -> None:
        for md in self._models:
            if hasattr(md, "fast_forward"):
                md.fast_forward(n)


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
            stark_rad_per_sigma=spec.get("stark_rad_per_sigma", 0.0),
            decay_in_window=spec.get("decay_in_window", False))
    if kind == "threelevel":
        return ThreeLevelModel(
            m, core=spec.get("core", 0), f_ge=spec.get("f_ge", 0.0), f_ef=spec.get("f_ef", 0.0),
            rabi_ge_rad_per_amp=spec.get("rabi_ge_rad_per_amp", 0.0),
            rabi_ef_rad_per_amp=spec.get("rabi_ef_rad_per_amp", 0.0),
            readout_code=spec.get("readout_code", 2048), readout_amp=spec.get("readout_amp", 20000.0),
            readout_phase=spec.get("readout_phase", 0.0), level_phases=spec.get("level_phases"),
            init_level=spec.get("init_level", 0), collapse=spec.get("collapse", False),
            t1=spec.get("t1"), noise_scale=spec.get("noise_scale", 0.0),
            noise_seed=spec.get("noise_seed", 0))
    if kind == "twoqubit":
        return TwoQubitModel(   # no "coupler" key ⇒ the two-qubit-drive form (spec 04 §4.6)
            m, control=spec.get("control", 0), target=spec.get("target", 1),
            coupler=spec.get("coupler"), f_ge=spec.get("f_ge", (0.0, 0.0)),
            f_ef=spec.get("f_ef", (0.0, 0.0)), rabi_ge=spec.get("rabi_ge", (0.0, 0.0)),
            rabi_ef=spec.get("rabi_ef", (0.0, 0.0)),
            rabi_cz_rad_per_amp=spec.get("rabi_cz_rad_per_amp", 0.0),
            zz_rad_per_batch=spec.get("zz_rad_per_batch", 0.0),
            readout_code=spec.get("readout_code", (2048, 2048)),
            readout_amp=spec.get("readout_amp", (20000.0, 20000.0)),
            readout_phase=spec.get("readout_phase", (0.0, 0.0)),
            level_phases=spec.get("level_phases"), init=spec.get("init", (0, 0)),
            collapse=spec.get("collapse", False), t1=spec.get("t1"),
            noise_scale=spec.get("noise_scale", 0.0), noise_seed=spec.get("noise_seed", 0))
    raise ValueError(f"unknown QuantumModel kind {kind!r}")
