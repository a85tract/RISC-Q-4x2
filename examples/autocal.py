"""single_qubit_autocalibrate_v2 (spec 06 §4) — a plain host-python autocalibration script.

The control flow between calibrations is ordinary python: it needs no kernel DSL because it runs on
the host. What we keep from qcal's ``single_qubit_autocalibrate_v2``:

  * an ``expts_to_run`` gating dict — skip or run each experiment;
  * incremental frequency updates — each Ramsey nudges the *current* drive frequency toward f_ge;
  * the automatic **Ramsey-after-Amplitude re-run** rule — changing the drive amplitude shifts the
    qubit's AC-Stark-shifted frequency, so the frequency calibration is re-run whenever the amplitude
    moved (only if Ramsey is enabled).

The order is: spectroscopy (readout resonator) -> Ramsey (drive frequency) -> Rabi (drive amplitude)
-> T1. Each step runs one on-core calibration and, when its fit succeeds, applies the proposal into
the shared Config. Returns the dict of Results keyed by step name.
"""

from __future__ import annotations

from riscq.cal import Amplitude, Frequency, Separation, T1

DEFAULT_EXPTS = {"spectroscopy": True, "ramsey": True, "rabi": True, "t1": True}
AMP_RERUN_THRESHOLD = 1e-3   # an amplitude move larger than this re-runs Ramsey (AC-Stark)


def single_qubit_autocalibrate(cfg, qubit, drv, expts_to_run=None, apply=True, verbose=False,
                               spec_kw=None, ramsey_kw=None, rabi_kw=None, t1_kw=None):
    """Run the gated spectroscopy -> Ramsey -> Rabi -> T1 sequence on one qubit.

    ``expts_to_run`` overrides the default gating dict (all on). ``*_kw`` pass through knobs to the
    matching calibration class (e.g. ``ramsey_kw={"points": 7}`` for a fast co-sim run).
    """
    expts = dict(DEFAULT_EXPTS, **(expts_to_run or {}))
    results = {}

    def step(name, cal):
        r = cal.run(drv)
        if verbose:
            print(f"  {name}: ok={r.ok} proposal={r.proposal}")
        if apply and r.ok:
            r.apply()
        results[name] = r
        return r

    if expts["spectroscopy"]:
        step("spectroscopy", Separation(cfg, qubit, **(spec_kw or {})))

    if expts["ramsey"]:
        step("ramsey", Frequency(cfg, qubit, **(ramsey_kw or {})))

    if expts["rabi"]:
        amp_before = float(cfg[f"qubit/{qubit}/x90/amp"])
        step("rabi", Amplitude(cfg, qubit, **(rabi_kw or {})))
        amp_after = float(cfg[f"qubit/{qubit}/x90/amp"])
        # AC-Stark: the drive amplitude changed, so the drive-power-dependent qubit frequency moved.
        # Re-run Ramsey to re-centre the drive (only if the frequency calibration is enabled).
        if expts["ramsey"] and abs(amp_after - amp_before) > AMP_RERUN_THRESHOLD:
            if verbose:
                print(f"  amplitude moved {amp_before:.4f} -> {amp_after:.4f}; re-running Ramsey "
                      "(AC-Stark)")
            step("ramsey_restark", Frequency(cfg, qubit, **(ramsey_kw or {})))

    if expts["t1"]:
        step("t1", T1(cfg, qubit, **(t1_kw or {})))

    return results
