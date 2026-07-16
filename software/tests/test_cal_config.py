"""Host unit tests for riscq.cal.Config: slash-path get/set/missing, YAML round-trip, deep copy, and
the qcal adapter (spec 13 Q0) against the real X6Y3 config.yaml — every qubit's gate/readout parameter
round-trips, the frequency codes are in range for the build's SocParams, `hardware/*` matches it, and
save_qcal re-emits a tree that still loads (with everything it does not calibrate untouched)."""

import math
from pathlib import Path

import pytest
import yaml

from riscq.cal import Config
from riscq.map import SocParams
from riscq.pulses import envelopes, units

SW_ROOT = Path(__file__).resolve().parents[1]
QCAL_YAML = SW_ROOT.parent / "build" / "qcal-x6y3-config" / "config.yaml"   # the artefact of record
PARAMS = SocParams.load(SW_ROOT / "configs" / "zcu216-14q.json")            # the X6Y3-class build


def test_get_set_slash_paths():
    cfg = Config()
    cfg["readout/0/freq"] = 5_000_000
    cfg["readout/0/amp"] = 0.3
    cfg["qubit/2/amp"] = 0.9
    assert cfg["readout/0/freq"] == 5_000_000
    assert cfg["readout/0/amp"] == 0.3
    assert cfg["qubit/2/amp"] == 0.9
    # intermediate nodes were created as nested dicts
    assert cfg["readout/0"] == {"freq": 5_000_000, "amp": 0.3}


def test_missing_key_raises_and_contains_get():
    cfg = Config({"readout": {"0": {"freq": 1}}})
    assert "readout/0/freq" in cfg
    assert "readout/1/freq" not in cfg
    assert cfg.get("readout/1/freq", 42) == 42
    with pytest.raises(KeyError):
        cfg["readout/1/freq"]
    with pytest.raises(KeyError):
        cfg[""]


def test_overwrite_scalar_with_subtree():
    cfg = Config()
    cfg["a/b"] = 1
    cfg["a/b/c"] = 2                     # b was a scalar; setting deeper replaces it with a dict
    assert cfg["a/b/c"] == 2


def test_yaml_round_trip(tmp_path):
    cfg = Config()
    cfg["readout/0/freq"] = 5_000_000
    cfg["qubit/0/x90/amp"] = 0.42
    cfg["qubit/0/x90/phase"] = 0.0
    path = tmp_path / "cfg.yaml"
    cfg.save(path)
    back = Config.load(path)
    assert back.to_dict() == cfg.to_dict()
    assert back["qubit/0/x90/amp"] == 0.42


def test_copy_is_deep():
    cfg = Config({"a": {"b": 1}})
    dup = cfg.copy()
    dup["a/b"] = 99
    assert cfg["a/b"] == 1               # original untouched
    assert dup["a/b"] == 99


# ── the qcal adapter, on the real X6Y3 config (spec 13 Q0) ──

@pytest.fixture(scope="module")
def qcal_tree():
    if not QCAL_YAML.exists():
        pytest.skip(f"the X6Y3 reference config is not present ({QCAL_YAML})")
    with open(QCAL_YAML) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def qcal_cfg(qcal_tree):
    return Config.from_qcal(QCAL_YAML)


def test_from_qcal_round_trips_every_qubit(qcal_cfg, qcal_tree):
    """Every gate/readout parameter of all 8 qubits maps into our paths — the X90 is the FAST_DRAG
    slot (its own axis `phase`) bracketed by the virtual-Z pair (`vz`), the X a REAL pulse (double
    length, same amp — not 2x the X90 amplitude), and the demod phase is degrees in qcal, radians
    here."""
    qubits = list(qcal_tree["single_qubit"])
    assert len(qubits) == 8
    for q in qubits:
        ge = qcal_tree["single_qubit"][q]["GE"]
        assert qcal_cfg[f"qubit/{q}/freq"] == ge["freq"]
        assert qcal_cfg[f"qubit/{q}/T1"] == ge["T1"]
        assert qcal_cfg[f"qubit/{q}/T2"] == ge["T2*"]

        x90, x = [p for p in ge["X90"]["pulse"] if p["env"] != "virtualz"][0], ge["X"]["pulse"][0]
        vz = [p for p in ge["X90"]["pulse"] if p["env"] == "virtualz"]
        assert qcal_cfg[f"qubit/{q}/x90/env"] == x90["env"] == "FAST_DRAG"
        assert qcal_cfg[f"qubit/{q}/x90/dur"] == x90["time"] == 35e-9
        assert qcal_cfg[f"qubit/{q}/x90/amp"] == x90["kwargs"]["amp"]
        assert qcal_cfg[f"qubit/{q}/x90/phase"] == x90["kwargs"]["phase"]        # the pulse's own axis
        assert qcal_cfg[f"qubit/{q}/x90/vz"] == [p["kwargs"]["phase"] for p in vz]   # the frame pair
        assert qcal_cfg[f"qubit/{q}/x90/kwargs"] == {k: v for k, v in x90["kwargs"].items()
                                                     if k not in ("amp", "phase")}
        assert qcal_cfg[f"qubit/{q}/x/dur"] == x["time"] == 70e-9                # DOUBLE LENGTH
        assert qcal_cfg[f"qubit/{q}/x/amp"] == x["kwargs"]["amp"]
        assert qcal_cfg[f"qubit/{q}/x/amp"] < 2 * qcal_cfg[f"qubit/{q}/x90/amp"]  # NOT double amplitude
        assert qcal_cfg[f"qubit/{q}/x/phase"] == x["kwargs"]["phase"] == 0.0
        assert f"qubit/{q}/x/vz" not in qcal_cfg                                 # a bare pulse, no frame

        ro = qcal_tree["readout"][q]
        assert qcal_cfg[f"readout/{q}/freq"] == ro["freq"]
        assert qcal_cfg[f"readout/{q}/amp"] == ro["amp"] and 0.01 < ro["amp"] < 0.06
        assert qcal_cfg[f"readout/{q}/dur"] == ro["time"]
        assert qcal_cfg[f"readout/{q}/env"] == ro["env"] == "cosine_square"
        assert qcal_cfg[f"readout/{q}/kwargs"] == ro["kwargs"]
        assert qcal_cfg[f"readout/{q}/demod/dur"] == ro["demod"]["time"]
        assert qcal_cfg[f"readout/{q}/demod/delay"] == ro["demod"]["delay"]
        assert qcal_cfg[f"readout/{q}/demod/phase"] == pytest.approx(
            math.radians(ro["demod"]["phase"]))                                  # degrees → radians
        assert qcal_cfg[f"readout/{q}/demod/env"] == ro["demod"]["env"]
        assert qcal_cfg[f"readout/{q}/demod/amp"] == ro["demod"]["kwargs"]["amp"]

    assert qcal_cfg["readout/herald"] is True
    assert qcal_cfg["reset/relax"] == qcal_tree["reset"]["passive"]["delay"] == 500e-6

    # the two phase knobs are NOT interchangeable in the real artefact: q5's FAST_DRAG carries its own
    # axis phase on top of the frame pair, and q6's virtual-Z pair is unequal (a per-qubit fine pass)
    assert qcal_cfg["qubit/5/x90/phase"] != 0.0
    assert qcal_cfg["qubit/6/x90/vz"][0] != qcal_cfg["qubit/6/x90/vz"][1]


def test_from_qcal_codes_in_range(qcal_cfg, qcal_tree):
    """Every X6Y3 tone lands on a legal 16-bit code for this build: the 5.4–5.5 GHz drives and the
    6.5–6.8 GHz readouts alias below the 8 GS/s DAC's Nyquist, and each demod code is 4x its drive
    code folded — the pulses/units contract, exercised on the real frequencies."""
    for q in qcal_tree["single_qubit"]:
        for path in (f"qubit/{q}/freq", f"readout/{q}/freq"):
            code = units._freq_code(qcal_cfg[path], PARAMS)
            assert -(1 << 15) <= code < (1 << 15)
        demod = units._demod_code(qcal_cfg[f"readout/{q}/freq"], PARAMS)
        assert -(1 << 15) <= demod < (1 << 15)
        # the durations are non-degenerate on this build's 2 ns batch
        assert units.batches(qcal_cfg[f"qubit/{q}/x90/dur"] * 1e9, PARAMS) > 0
        assert units.batches(qcal_cfg[f"readout/{q}/demod/dur"] * 1e9, PARAMS) > 0


def test_from_qcal_hardware_matches_socparams(qcal_cfg):
    """hardware/* is CHECKED against the build, never imported: DAC 8 GS/s, ADC 2 GS/s, qdrv/rdrv/rdlo
    interpolation 4/16/4."""
    qcal_cfg.check_hardware(PARAMS)                    # the real X6Y3 tree matches zcu216-14q
    bad = qcal_cfg.copy()
    bad._qcal["hardware"]["sample_rate"]["DAC"] = 4e9
    with pytest.raises(ValueError, match="DAC sample rate"):
        bad.check_hardware(PARAMS)
    bad = qcal_cfg.copy()
    bad._qcal["hardware"]["interpolation_ratio"]["rdlo"] = 2
    with pytest.raises(ValueError, match="rdlo interpolation"):
        bad.check_hardware(PARAMS)


def test_save_qcal_writes_back_only_the_calibrated_fields(qcal_cfg, qcal_tree, tmp_path):
    """save_qcal re-emits a tree that still loads: the calibrated fields carry the new values (the
    X90's amp + its virtual-Z PAIR, the X's amp, GE freq/T1/T2*, the readout freq/amp, the demod
    window + its phase back in DEGREES) and everything else — two-qubit gates, EF, reset pulses —
    round-trips untouched."""
    cal = qcal_cfg.copy()
    cal["qubit/3/freq"] = 5.4321e9
    cal["qubit/3/x90/amp"] = 0.123
    cal["qubit/3/x90/phase"] = 0.05
    cal["qubit/3/x90/vz"] = [0.25, 0.25]
    cal["qubit/3/x/amp"] = 0.234
    cal["qubit/3/T1"] = 6.6e-5
    cal["readout/3/freq"] = 6.7e9
    cal["readout/3/amp"] = 0.0345
    cal["readout/3/demod/dur"] = 8e-7
    cal["readout/3/demod/phase"] = math.radians(-30.0)

    path = tmp_path / "config.yaml"
    cal.save_qcal(path)
    with open(path) as f:
        out = yaml.safe_load(f)

    ge = out["single_qubit"][3]["GE"]
    assert ge["freq"] == 5.4321e9 and ge["T1"] == 6.6e-5
    assert [p["kwargs"]["phase"] for p in ge["X90"]["pulse"] if p["env"] == "virtualz"] == [0.25, 0.25]
    drive = [p for p in ge["X90"]["pulse"] if p["env"] != "virtualz"]
    assert [p["kwargs"]["amp"] for p in drive] == [0.123]
    assert [p["kwargs"]["phase"] for p in drive] == [0.05]
    assert ge["X"]["pulse"][0]["kwargs"]["amp"] == 0.234
    assert out["readout"][3]["freq"] == 6.7e9 and out["readout"][3]["amp"] == 0.0345
    assert out["readout"][3]["demod"]["time"] == 8e-7
    assert out["readout"][3]["demod"]["phase"] == pytest.approx(-30.0)          # radians → degrees

    # untouched subtrees round-trip bit-for-bit
    assert out["two_qubit"] == qcal_tree["two_qubit"]
    assert out["reset"] == qcal_tree["reset"]
    assert out["hardware"] == qcal_tree["hardware"]
    assert out["single_qubit"][3]["EF"] == qcal_tree["single_qubit"][3]["EF"]
    assert out["single_qubit"][0] == qcal_tree["single_qubit"][0]               # q0 was not calibrated

    # and the emitted tree loads straight back into a Config with the same values
    back = Config.from_qcal(path)
    assert back["qubit/3/x90/amp"] == 0.123
    assert back["readout/3/demod/phase"] == pytest.approx(math.radians(-30.0))
    assert back.to_dict() == cal.to_dict()


def test_qcal_envelopes_build(qcal_cfg):
    """The X6Y3 envelopes are buildable on this build's channel grids: the FAST_DRAG gate (35 ns of
    2 GS/s gate samples) and the cosine_square readout tone, both peaking at FULL."""
    x90 = envelopes.build("FAST_DRAG", 70, 2e9, **qcal_cfg["qubit/0/x90/kwargs"])
    assert x90.shape == (70,) and abs(x90).max() == pytest.approx(envelopes.FULL)
    assert abs(x90.imag).max() > 0                       # the DRAG quadrature is present (alpha=1)
    ro = envelopes.build("cosine_square", 250, 5e8, **qcal_cfg["readout/0/kwargs"])
    assert ro.shape == (250,) and abs(ro).max() == pytest.approx(envelopes.FULL)
    assert abs(ro[0]) < 1e-9 and abs(ro[125]) == pytest.approx(envelopes.FULL)   # ramps up to a flat top
