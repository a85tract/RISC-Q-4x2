"""Host unit tests for the readout-completeness pieces of spec 14 F2: the confusion correction
(`rcorr`, qcal's `rcorr_cmat`) and the 3-level `ReadoutFidelity`'s guards.

`rcorr` is the one piece of readout analysis with an exact answer — push a known population vector
through a known confusion matrix and the correction has to return it bit-for-bit (up to float error),
so these are equalities, not tolerances.
"""

import math

import numpy as np
import pytest

from riscq.cal import Config, ReadoutFidelity
from riscq.cal.base import GATE_ENV, batch_timeout, acquire_shots, gate_sigma
from riscq.cal.readout import ClassifierN, _rawiq_prog, rcorr
from riscq.pulses import Pulse, units
from riscq import run as rq

F_GE, F_EF = 50e6, 40e6          # the planted qutrit's two carriers
GE_AMP, EF_AMP = 0.5, 0.5


def test_rcorr_inverts_a_planted_confusion():
    """The defining property: p_meas = p_true @ cmat (row = PREPARED, col = MEASURED), so correcting
    p_meas returns p_true exactly. A 3-level matrix with realistic leakage between every pair."""
    cmat = np.array([[0.97, 0.02, 0.01],       # |0> read as 0/1/2
                     [0.04, 0.93, 0.03],
                     [0.02, 0.08, 0.90]])
    assert np.allclose(cmat.sum(1), 1.0)
    p_true = np.array([0.20, 0.50, 0.30])
    p_meas = p_true @ cmat
    assert not np.allclose(p_meas, p_true)      # the confusion really moved the populations
    assert np.allclose(rcorr(p_meas, cmat), p_true)


def test_rcorr_is_a_no_op_on_a_perfect_readout():
    p = np.array([0.1, 0.6, 0.3])
    assert np.allclose(rcorr(p, np.eye(3)), p)


def test_rcorr_corrects_a_stack_of_populations():
    """A whole sweep at once: one row per point (a Leakage scan's P(|0>,|1>,|2>) per phase)."""
    cmat = np.array([[0.95, 0.05], [0.10, 0.90]])
    p_true = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.25, 0.75]])
    got = rcorr(p_true @ cmat, cmat)
    assert got.shape == p_true.shape and np.allclose(got, p_true)


def test_rcorr_does_not_clip_an_unphysical_correction():
    """A population the confusion cannot explain corrects OUT of [0, 1] — reported, not squashed, so a
    miscalibrated matrix is visible rather than silently plausible."""
    cmat = np.array([[0.9, 0.1], [0.1, 0.9]])
    out = rcorr(np.array([1.0, 0.0]), cmat)     # cleaner than the matrix allows
    assert out[0] > 1.0 and out[1] < 0.0


def test_rcorr_rejects_a_mismatched_matrix():
    with pytest.raises(AssertionError, match="does not match"):
        rcorr(np.array([0.5, 0.5]), np.eye(3))


def test_three_level_fidelity_requires_a_pretrained_classifier():
    """Confusing the very shots a classifier was fitted to measures the fit, not the readout — so the
    3-level mode takes the classifier from outside (the EFAmplitude/EFPhase convention) and refuses to
    run without one."""
    cfg = Config()
    with pytest.raises(AssertionError, match="pre-trained"):
        ReadoutFidelity(cfg, 0, n_levels=3)
    with pytest.raises(AssertionError, match="n_levels"):
        ReadoutFidelity(cfg, 0, n_levels=4)
    clf = ClassifierN([np.zeros((4, 2)), np.ones((4, 2)), 2 * np.ones((4, 2))])
    assert ReadoutFidelity(cfg, 0, n_levels=3, classifier=clf).classifiers == {0: clf}
    assert ReadoutFidelity(cfg, 0).classifiers == {}          # the 2-level path needs none


# ── the 3-level confusion, on a planted qutrit (spec 14 F2) ──

def _s(batches, m):
    return float(batches) / m.params.dsp_freq_hz


def _cfg3(m, q=0):
    """A qutrit Config: GE + EF gates and a 3-level readout tone. The EF X is a real π in {|1>, |2>},
    which is what the |2> prep plays after the GE π."""
    c = Config()
    c[f"qubit/{q}/freq"] = F_GE
    c[f"qubit/{q}/x90/amp"] = GE_AMP
    c[f"qubit/{q}/T1"] = _s(120, m)
    c[f"qubit/{q}/EF/freq"] = F_EF
    c[f"qubit/{q}/EF/x90/amp"] = EF_AMP
    c[f"qubit/{q}/EF/x/amp"] = EF_AMP
    c[f"readout/{q}/freq"] = float(units.demod_code_to_freq(2048, m.params))
    c[f"readout/{q}/amp"] = 0.5
    c[f"readout/{q}/dur"] = _s(56, m)
    c[f"readout/{q}/demod/dur"] = _s(40, m)
    c["reset/relax"] = _s(8000, m)      # >= 5x the population T1 below, so each shot starts |0>
    return c


def _plant(level):
    """The model for a REFERENCE cloud: level `level` planted and held. No T1 — the planted level has
    to survive the grid's relax head to be read out, which is exactly why the reference clouds are
    captured on a decay-free model (the EF cals' `_train_3level` does the same)."""
    return {"kind": "threelevel", "core": 0, "readout_code": 2048, "readout_amp": 18000.0,
            "init_level": level, "collapse": True, "noise_scale": 400.0, "noise_seed": 7 + level}


def _qutrit(m, **kw):
    """The planted qutrit: a GE Rabi rate making the two-X90 prep a π, an EF rate making the EF X a π
    in {|1>, |2>}, and a T1 that lets each shot's relax head reset the level between shots."""
    ge = gate_sigma(m, Pulse(GATE_ENV, freq_hz=F_GE, amp=GE_AMP), F_GE, units._amp_code(GE_AMP))
    ef = gate_sigma(m, Pulse(GATE_ENV, amp=EF_AMP), F_EF, units._amp_code(EF_AMP))
    return {"kind": "threelevel", "core": 0, "f_ge": F_GE, "f_ef": F_EF,
            "rabi_ge_rad_per_amp": math.pi / (2 * ge), "rabi_ef_rad_per_amp": math.pi / ef,
            "readout_code": 2048, "readout_amp": 18000.0, "init_level": 0, "collapse": True,
            # amplitude damping: the POPULATION time constant is t1/2 batches. The |2> prep is the
            # longest sequence here (GE pi + LEAD + EF pi + SEP ~ 200 batches), so t1 has to leave it
            # standing — at t1 = 600 it decayed to 0.375 and the row misclassified as |0>.
            "t1": 3000, "noise_scale": 400.0, "noise_seed": 5, **kw}


@pytest.mark.cosim
def test_three_level_fidelity_measures_a_real_confusion(cosim):
    """(F2 gate) The 3×3 matrix from REAL preps: |0> (idle), |1> (GE π) and |2> (GE π + EF π, the new
    `_ef_prep_prog`), each classified by a classifier trained on planted reference clouds — so the
    matrix measures the readout, not the fit. Every prepared level must dominate its own row, and the
    matrix must be a usable `rcorr` input (it inverts a planted population back)."""
    drv, m = cosim
    q = 0
    cfg = _cfg3(m, q)
    shots = 32
    prog, period = _rawiq_prog(m, cfg, q, "X90", shots)
    timeout = batch_timeout(shots * period)
    clouds = []
    for level in range(3):                       # reference clouds: PLANT each level, read it out
        drv.sim.set_model(_plant(level))
        rq.setup(drv, m, {q: prog})
        clouds.append(acquire_shots(drv, m, {q: prog}, 0, shots, timeout)[q])
    clf = ClassifierN(clouds)
    assert clf.separation > 1.0, f"3-level clusters not separated ({clf.separation:.2f})"

    drv.sim.set_model(_qutrit(m))                # now the real qutrit: the preps must DRIVE there
    r = ReadoutFidelity(cfg, q, shots=shots, n_levels=3, classifier=clf).run(drv)
    conf = r.data[q]["confusion"]
    print(f"\n[cmat3] fidelity={r.data[q]['fidelity']:.3f}\n{np.round(conf, 3)}")
    assert conf.shape == (3, 3) and np.allclose(conf.sum(1), 1.0)
    # Each prepared level dominates its own row. The |2> row is the weak one (~0.5) and that is the
    # co-sim's |2> prep, not this program: `test_ef_amplitude_recovers_the_ef_rabi` tops out at
    # P(|2>) ≈ 0.56 on the same model. An imperfect matrix is exactly what `rcorr` exists to undo.
    for level in range(3):
        assert conf[level].argmax() == level, \
            f"prepared |{level}> classified as |{conf[level].argmax()}>:\n{np.round(conf, 3)}"
    assert conf[0, 0] > 0.9 and conf[1, 1] > 0.8 and conf[2, 2] > 0.4
    assert r.proposal[f"readout/{q}/cmat"] == conf.tolist()     # YAML-safe, in the Config
    p_true = np.array([0.2, 0.5, 0.3])
    assert np.allclose(rcorr(p_true @ conf, conf), p_true)      # the measured matrix is invertible
