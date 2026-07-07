"""Autocalibration sequences (spec 06 §4). Ordinary host python — the ordering, retries, and
conditional re-runs need no DSL because they run on the host between on-core calibrations."""

from __future__ import annotations

from riscq.cal.qubit import Amplitude, Frequency, Phase
from riscq.cal.readout import Fidelity, ReadoutCalibration, ReadoutFidelity, Separation
from riscq.pulses import units


def calibration_x6y3(cfg, qubit, drv, apply=True, verbose=False):
    """The Calibration_X6Y3 flow: ReadoutCalibration → Separation → Fidelity → ReadoutFidelity →
    Frequency → Amplitude (coarse) → Amplitude (fine, relative) → Phase. Each cell is
    `r = Cal(...).run(drv); r.apply()`. The fine (n_gates=4) amplitude sweep is centred on the
    coarse result, so it is built after the coarse step applies. Returns the list of Results.

    The point/shot counts are the lighter end of each cal's range: a chained autocal only needs each
    step to IMPROVE its estimate (the next step refines it), not the sub-1 % single-shot precision the
    standalone cals target — so the counts sweeps run ~64–120 shots (Rabi RATE precision, spec 08 B2,
    is the standalone cals' concern, not the chain's) and the VNA a few points. Every step stays a
    real cal; the sequence just sizes them for turnaround."""
    results = []

    def step(cal):
        r = cal.run(drv)
        if verbose:
            print(f"  {r.label}: ok={r.ok} proposal={r.proposal}")
        if apply and r.ok:
            r.apply()
        results.append(r)
        return r

    step(ReadoutCalibration(cfg, qubit))
    step(Separation(cfg, qubit, points=9, shots=8))
    step(Fidelity(cfg, qubit))
    step(ReadoutFidelity(cfg, qubit))
    step(Frequency(cfg, qubit, points=11, shots=64))
    step(Amplitude(cfg, qubit, n_gates=1, points=15, shots=120))   # coarse → sets qubit/q/x90/amp
    cur = int(round(float(cfg[f"qubit/{qubit}/x90/amp"]) * units.AMP_SCALE))
    step(Amplitude(cfg, qubit, n_gates=4, amp_span=(max(400, int(0.8 * cur)), int(1.2 * cur)),
                   shots=120))   # fine → refines it (relative, span centred on the coarse result)
    step(Phase(cfg, qubit, points=11, shots=120))
    return results
