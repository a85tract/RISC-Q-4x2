"""Autocalibration sequences (spec 06 §4). Ordinary host python — the ordering, retries, and
conditional re-runs need no DSL because they run on the host between on-core calibrations."""

from __future__ import annotations

from riscq.cal.qubit import Amplitude, Frequency, Phase
from riscq.cal.readout import Fidelity, ReadoutCalibration, ReadoutFidelity, Separation


def calibration_x6y3(cfg, qubits, drv, apply=True, verbose=False):
    """The Calibration_X6Y3 flow: ReadoutCalibration → Separation → Fidelity → ReadoutFidelity →
    Frequency → Amplitude (coarse) → Amplitude (fine, relative) → Phase, over `qubits` (a bare int is
    one qubit) SIMULTANEOUSLY — every step calibrates all cores in one run (spec 13 §8). Each cell is
    `r = Cal(...).run(drv); r.apply()`. The fine (n_gates=4) amplitude sweep is `relative_amp`, i.e.
    0.7–1.3× whatever the coarse step just wrote (the notebook's own fine pass), so it needs no host
    arithmetic at all. Returns the list of Results. `Window` (the demod-window sweep) is OURS, not
    qcal's, so it is deliberately not in this chain (spec 13 §5).

    The point/shot counts are the lighter end of each cal's range: a chained autocal only needs each
    step to IMPROVE its estimate (the next step refines it), not the sub-1 % single-shot precision the
    standalone cals target — so the counts sweeps run ~64–120 shots (Rabi RATE precision, spec 08 B2,
    is the standalone cals' concern, not the chain's) and the VNA a few points. Phase runs BOTH qcal
    sequences (two runs), so it is sized at half the shots. Every step stays a real cal; the sequence
    just sizes them for turnaround."""
    results = []

    def step(cal):
        r = cal.run(drv)
        if verbose:
            print(f"  {r.label}: ok={r.ok} proposal={r.proposal}")
        if apply and r.ok:
            r.apply()
        results.append(r)
        return r

    step(ReadoutCalibration(cfg, qubits))
    step(Separation(cfg, qubits, points=7, shots=16))    # two prep reruns (spec 13 §5): 2 x the shots
    step(Fidelity(cfg, qubits, points=5, shots=16))      # the readout-AMP sweep (qcal's knob)
    step(ReadoutFidelity(cfg, qubits, shots=24))
    step(Frequency(cfg, qubits, points=11, shots=64))
    step(Amplitude(cfg, qubits, n_gates=1, points=15, shots=120))   # coarse → sets qubit/q/x90/amp
    step(Amplitude(cfg, qubits, n_gates=4, amp_span=(0.7, 1.3), relative_amp=True,
                   shots=120))   # fine → refines it (qcal's relative pass, 4 X90s amplify the error)
    step(Phase(cfg, qubits, points=7, shots=64, relative_phase=True))   # the notebook's per-qubit pass
    return results
