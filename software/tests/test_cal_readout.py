"""The readout-completeness pieces of spec 14 F2: the confusion correction (`rcorr`, qcal's
`rcorr_cmat`), the 3-level `ReadoutFidelity`'s guards, and — in co-sim — that the three real preps
actually reach levels 0/1/2.

`rcorr` is the one piece of readout analysis with an exact answer — push a known population vector
through a known confusion matrix and the correction has to return it bit-for-bit (up to float error),
so these are equalities, not tolerances. That is the whole 3×3 confusion ARITHMETIC, host-pure, and
it is why the co-sim half no longer measures a matrix (specs/software-test-refactor/02 §3.6): the
old gate spent `relax = 8000` × 32 shots × 3 preps — the second-worst test in the suite — to
rediscover, through shot noise, that a diagonal-dominant matrix inverts.

What is left for the simulator is the one claim the host cannot make: that the three PRODUCTION
prep programs — idle, the GE π, and `_ef_prep_prog`'s GE π + EF π — really drive the qutrit to
levels 0, 1 and 2. That is an **L2 state probe** (01 §4): |2⟩ is invisible to the hardware `res`
bit (one threshold, two outcomes), so it is read off `drv.sim.model_state()["populations"]`, one
noiseless shot per prep against the analytic target.
"""

import math

import numpy as np
import pytest

from riscq.cal import Classifier, Config, ReadoutFidelity
from riscq.cal.base import GATE_ENV, gate_sigma
from riscq.cal.readout import ClassifierN, _ef_prep_prog, _rawiq_prog, rcorr, res_fidelity
from riscq.pulses import Pulse, units
from tests.probe import Probe

# The planted qutrit's two carriers, an anharmonicity of exactly 4096 codes (100 MHz on this build)
# apart. `ThreeLevelModel` picks which transition a batch drives by demodulating the gate DAC against
# BOTH carriers over that batch's 16 samples and taking the larger — and 16 samples is a very short
# window: at the old 50/40 MHz pair (Δ = 410 codes) the two demods came back within 1.6 % of each
# other and the winner flipped batch to batch, so the "EF π" also drove GE and the |2> prep topped
# out at 0.49. A separation of 4096 codes puts exactly one full turn of the difference frequency in
# a batch, so each carrier's demod is EXACTLY zero against the other transition and both preps are
# the textbook gates their planted rates say they are.
F_GE, F_EF = 150e6, 50e6
GE_AMP, EF_AMP = 0.5, 0.5


def _clusters(s0: complex, s1: complex, n: int = 400, sigma: float = 0.05, seed: int = 3):
    """Two IQ clouds around planted complex responses — the shape a readout actually produces."""
    rng = np.random.default_rng(seed)
    return [np.column_stack([s.real + rng.normal(0, sigma, n), s.imag + rng.normal(0, sigma, n)])
            for s in (s0, s1)]


def _proposed_phase(iq0, iq1) -> float:
    """ReadoutCalibration's rule: rotate the |0>->|1> cluster axis onto +real."""
    return -math.atan2(*(iq0.mean(0) - iq1.mean(0))[::-1])


def test_res_fidelity_is_perfect_when_the_two_responses_are_antipodal():
    """A flat readout tone puts |1> exactly pi out of phase from |0>, so the rotated midpoint lands
    on the imaginary axis and the hardware's hard-zero threshold sits between the clusters."""
    iq0, iq1 = _clusters(0.6 + 0.8j, -0.6 - 0.8j)
    assert res_fidelity(iq0, iq1, _proposed_phase(iq0, iq1)) > 0.99


def test_res_fidelity_is_perfect_for_a_dispersive_readout_probed_at_f_r():
    """At f_r the two Lorentzian responses are complex CONJUGATES, so their difference is purely
    imaginary and their sum purely real — rotating the axis onto +real again centres the midpoint."""
    s0 = 1.0 / (1.0 + 0.6j)
    iq0, iq1 = _clusters(s0, s0.conjugate(), sigma=0.02)
    assert res_fidelity(iq0, iq1, _proposed_phase(iq0, iq1)) > 0.99


def test_res_fidelity_collapses_off_resonance_even_though_the_clusters_are_well_separated():
    """spec 15 §9.7's finding: probed OFF f_r the responses are neither antipodal nor conjugate, so
    a ROTATION cannot move the midpoint off the real axis. Both clusters end up on the +real side of
    the hardware's hard-zero threshold and `res` discriminates at chance — while the host
    classifier, which puts its boundary where the data is, is untouched. That is exactly why
    ReadoutCalibration gates on this and not on `separation` alone."""
    chi, kappa, det = 1.2e6, 4.0e6, 1.5e6
    s0 = 1.0 / (1.0 + 2j * (det - chi) / kappa)
    s1 = 1.0 / (1.0 + 2j * (det + chi) / kappa)
    iq0, iq1 = _clusters(s0, s1, sigma=0.02)
    clf = Classifier(iq0, iq1)
    assert clf.separation > 2.0, "the clusters are well separated to a host classifier"
    assert res_fidelity(iq0, iq1, _proposed_phase(iq0, iq1)) < 0.6, \
        "the hard-zero res bit should be at chance here"


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


# ── L2: the three REAL preps reach levels 0 / 1 / 2 (spec 14 F2) ──

def _s(batches, m):
    return float(batches) / m.params.dsp_freq_hz


def _cfg3(m, q=0):
    """A qutrit Config: GE + EF gates and a 3-level readout tone. The EF X is a real π in {|1>, |2>},
    which is what the |2> prep plays after the GE π.

    `reset/relax` is at its floor: an L2 probe re-issues `set_model`, which rebuilds the qutrit in
    |0> in ZERO simulated cycles, so the 8000-batch idle head the old shot-statistics version needed
    to reset between shots buys nothing here (01 §4.2)."""
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
    c["reset/relax"] = _s(8, m)
    return c


def _qutrit(m):
    """The planted qutrit: a GE Rabi rate making the two-X90 prep an exact π and an EF rate making the
    EF X an exact π in {|1>, |2>} — so each prep is a textbook gate and the expected populations are
    the ideal ones. 01 §4.6: no decay, no noise, no collapse (the probe reads the state, not shots)."""
    ge = gate_sigma(m, Pulse(GATE_ENV, freq_hz=F_GE, amp=GE_AMP), F_GE, units._amp_code(GE_AMP))
    ef = gate_sigma(m, Pulse(GATE_ENV, amp=EF_AMP), F_EF, units._amp_code(EF_AMP))
    return {"kind": "threelevel", "core": 0, "f_ge": F_GE, "f_ef": F_EF,
            "rabi_ge_rad_per_amp": math.pi / (2 * ge), "rabi_ef_rad_per_amp": math.pi / ef,
            "readout_code": 2048, "readout_amp": 18000.0, "init_level": 0}


# One test per PROGRAM, because each program is its own image and an image load is ~7 k simulated
# batches — the two together overrun the 20 k per-test cap (specs/software-test-refactor/02 §1).
#
# Both replace the halves of a 3 × 32-shot, `relax = 8000` confusion measurement whose |2> row only
# ever reached 0.5. The preps are now pinned to 0.02 instead of to "dominates its row", and the 3×3
# matrix ARITHMETIC — the row-stochastic shape and the `rcorr` inverse — is the host-pure half above.

@pytest.mark.cosim
def test_ge_preps_reach_levels_0_and_1(cosim):
    """L2 (spec 14 F2) — the |0> and |1> rows of the 3-level confusion come from ONE `_rawiq_prog`
    image whose `prep` runtime scalar picks idle or the GE π (two X90 plays). Both must land on the
    level they name, and the |1> prep must not leak into |2>.

    The target is analytic and exact: the GE rate is planted so the two-X90 prep is a π in
    {|0>, |1>}, so the prep is a textbook gate and the populations are 1 at the intended level and 0
    elsewhere. Nothing is fitted and nothing is sampled — |2> is invisible to the `res` bit, so the
    populations come off `model_state()`."""
    _, m = cosim
    q = 0
    prog, _ = _rawiq_prog(m, _cfg3(m, q), q, "X90", 1)
    p, spec = Probe(cosim, {q: prog}), _qutrit(m)
    for prep, want in ((0, [1.0, 0.0, 0.0]), (1, [0.0, 1.0, 0.0])):
        pops = p.state(spec, {q: {"prep": prep}})["populations"]
        print(f"\n[prep {prep}] populations={np.round(pops, 4).tolist()} want={want}")
        assert pops == pytest.approx(want, abs=0.02), \
            f"the prep={prep} program left the qutrit at {np.round(pops, 4).tolist()}, not {want}"


@pytest.mark.cosim
def test_ef_prep_reaches_level_2(cosim):
    """L2 (spec 14 F2) — the |2> row's prep: `_ef_prep_prog`, a GE π followed by an EF π at the
    config's EF X amplitude, on its own image with the carrier retuned mid-shot. It is the program
    `ReadoutFidelity._run_3level` runs, unchanged.

    Both rates are planted exact (GE: the two-X90 prep is a π in {|0>, |1>}; EF: the EF X is a π in
    {|1>, |2>}), so the analytic target is a clean |2> — which also makes it the sharpest available
    statement about the mid-shot GE→EF retune: any slip in WHEN the new carrier takes effect leaves
    population behind in |1>."""
    _, m = cosim
    q = 0
    prog, par, _ = _ef_prep_prog(m, _cfg3(m, q), q, 1)
    pops = Probe(cosim, {q: prog}).state(_qutrit(m), {q: par})["populations"]
    print(f"\n[prep |2>] populations={np.round(pops, 4).tolist()} want=[0.0, 0.0, 1.0]")
    assert pops == pytest.approx([0.0, 0.0, 1.0], abs=0.02), \
        f"the GE π + EF π prep left the qutrit at {np.round(pops, 4).tolist()}, not |2>"
