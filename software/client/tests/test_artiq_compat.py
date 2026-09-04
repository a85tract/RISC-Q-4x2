"""riscq.artiq_compat: the character-identical ARTIQ experiment shape.

Host-pure: recording only — the driver half of run_experiment is exercised with monkeypatched
seams (Codex F4/F9: lifecycle, analyze-only-on-success, cleanup on failure)."""
from pathlib import Path

import pytest

from riscq import artiq_compat as C
from riscq import artiqapi as A
from riscq.artiq_compat import (EnvExperiment, kernel, parallel, sequential, delay,
                                MHz, us, PHASE_MODE_ABSOLUTE)
from riscq.map import SocMap, SocParams

CONFIGS = Path(__file__).resolve().parents[3] / "gateware" / "configs"


def _map(name="rfsoc4x2-1q-fine"):
    path = CONFIGS / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{name}.json not in this checkout")
    return SocMap(SocParams.from_json(path.read_text()))


DB = {
    "core":     {"type": "board", "host": "192.168.3.1", "bundle": "rfsoc4x2-1q-fine"},
    "gate_dds": {"type": "dds", "channel": 0},
    "ro_dds":   {"type": "dds", "channel": 1},
    "adc":      {"type": "adc"},
    "dm":       {"type": "demod"},
    "readout":  "ro_dds",                                     # alias
}


class IonTrap(EnvExperiment):
    """The verified ion-trap sequence, in the exact ARTIQ shape."""

    def build(self):
        self.setattr_device("core")
        self.setattr_device("gate_dds")
        self.setattr_device("ro_dds")
        self.setattr_device("adc")

    @kernel
    def run(self):
        self.core.reset()
        with parallel:
            with sequential:
                with parallel:
                    with sequential:
                        self.gate_dds.set(83.765*MHz, phase=0.0, amplitude=0.4)
                        self.gate_dds.sw.pulse(100*us)
                    with sequential:
                        self.ro_dds.set(80.235*MHz, phase=0.5, amplitude=0.4)
                        self.ro_dds.sw.pulse(100*us)
                delay(5*us)
                self.ro_dds.set(82.0*MHz, phase=0.25, amplitude=0.4,
                                phase_mode=PHASE_MODE_ABSOLUTE)
                self.ro_dds.sw.pulse(20*us)
            with sequential:
                self.adc.gate(125*us)


def _explicit_ion_trap(m):
    core = A.Core(m)
    gate, ro, adc = A.DDSChannel(core, 0, "gate_dds"), A.DDSChannel(core, 1, "ro_dds"), \
        A.ADCChannel(core)
    with A.parallel(core):
        with A.branch(core):
            with A.parallel(core):
                with A.branch(core):
                    gate.set(83.765*MHz, phase=0.0, amplitude=0.4); gate.sw.pulse(100*us)
                with A.branch(core):
                    ro.set(80.235*MHz, phase=0.5, amplitude=0.4); ro.sw.pulse(100*us)
            A.delay(core, 5*us)
            ro.set(82.0*MHz, phase=0.25, amplitude=0.4, phase_mode=A.PHASE_MODE_ABSOLUTE)
            ro.sw.pulse(20*us)
        with A.branch(core):
            adc.gate(125*us)
    return core


def test_compat_records_the_same_timeline_as_the_explicit_api():
    """Same sets, same events, same trace gates, same cursor — the compat layer is sugar only."""
    m = _map()
    core = A.Core(m)
    C.record_experiment(IonTrap, DB, core)
    ref = _explicit_ion_trap(m)
    assert core.sets == ref.sets
    assert core.events == ref.events
    assert core.trace_gates == ref.trace_gates
    assert core.now_mu() == ref.now_mu()


def test_bare_statements_inside_parallel_raise():
    """ARTIQ's compiler parallelizes bare statements; this layer cannot, so it must refuse them
    on EVERY mutation path (set / pulse / delay / at_mu / gate), not silently serialize."""
    m = _map()

    def make(body):
        class Bad(EnvExperiment):
            def build(self):
                self.setattr_device("core")
                self.setattr_device("ro_dds")
                self.setattr_device("adc")

            @kernel
            def run(self):
                with parallel:
                    body(self)
        return Bad

    for name, body in [
        ("set",   lambda s: s.ro_dds.set(82*MHz)),
        ("pulse", lambda s: s.ro_dds.sw.pulse(1*us)),
        ("delay", lambda s: delay(1*us)),
        ("at_mu", lambda s: C.at_mu(64)),
        ("gate",  lambda s: s.adc.gate(1*us)),
    ]:
        with pytest.raises(RuntimeError, match="wrap each parallel arm"):
            C.record_experiment(make(body), DB, A.Core(m))


def test_nested_sequential_does_not_rewind():
    """`sequential` is a parallel ARM only directly under `parallel`; nested deeper it is plain
    grouping and must not rewind the cursor to the parallel start (Codex F6)."""
    m = _map()

    class Nested(EnvExperiment):
        def build(self):
            self.setattr_device("core")
            self.setattr_device("ro_dds")

        @kernel
        def run(self):
            with parallel:
                with sequential:
                    self.ro_dds.set(82*MHz, amplitude=0.4)
                    self.ro_dds.sw.pulse(10*us)
                    with sequential:                    # plain grouping, NOT a new arm
                        self.ro_dds.set(82*MHz, amplitude=0.4)
                        self.ro_dds.sw.pulse(10*us)

    core = A.Core(m)
    C.record_experiment(Nested, DB, core)
    e0, e1 = core.events
    assert e1.start_mu == e0.start_mu + core.seconds_to_mu(10*us), \
        "the nested sequential pulse must FOLLOW the first, not restart the arm"


def test_context_unbinds_after_an_exception():
    m = _map()

    class Boom(EnvExperiment):
        @kernel
        def run(self):
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        C.record_experiment(Boom, DB, A.Core(m))
    with pytest.raises(RuntimeError, match="no experiment is running"):
        delay(1*us)


def test_run_must_be_marked_kernel():
    m = _map()

    class NoMark(EnvExperiment):
        def run(self):
            pass

    with pytest.raises(RuntimeError, match="@kernel"):
        C.record_experiment(NoMark, DB, A.Core(m))


def test_device_db_is_validated_eagerly():
    m = _map()
    cases = [
        ({}, "needs a 'core' entry"),
        ({"core": {"type": "dds", "channel": 0}}, "must be type 'board' or 'cosim'"),
        ({"core": {"type": "usb"}}, "unknown type"),
        ({"core": DB["core"], "x": {"type": "laser"}}, "unknown type"),
        ({"core": DB["core"], "x": {"type": "dds"}}, "missing \\['channel'\\]"),
        ({"core": DB["core"], "x": "y"}, "unknown device"),
        ({"core": DB["core"], "x": "y", "y": "x"}, "alias cycle"),
        ({"core": DB["core"], "x": {"type": "board", "host": "h", "bundle": "b"}},
         "belongs only in 'core'"),
        ({"core": "hw", "hw": {"type": "board", "host": "h", "bundle": "b"}},
         "not an alias"),
    ]
    for db, msg in cases:
        with pytest.raises(ValueError, match=msg):
            C._validate_db(db)
    assert C._validate_db(DB) is DB                           # pure check, raw db returned

    class UsesBadChannel(EnvExperiment):
        @kernel
        def run(self):
            pass

    bad = dict(DB); bad["x"] = {"type": "dds", "channel": 7}
    with pytest.raises(ValueError, match="has 2 dds channels"):
        C.record_experiment(UsesBadChannel, bad, A.Core(m))


def test_alias_is_the_same_device_object():
    """ARTIQ DeviceManager semantics: alias and target are interchangeable, tone state
    included — not two wrappers over two channels (Codex F3)."""
    m = _map()

    class UsesAlias(EnvExperiment):
        def build(self):
            self.setattr_device("core")
            self.setattr_device("ro_dds")
            self.setattr_device("readout")               # alias of ro_dds

        @kernel
        def run(self):
            self.readout.set(80.235*MHz, phase=0.5, amplitude=0.4)   # set via the ALIAS
            self.ro_dds.sw.pulse(10*us)                              # pulse via the TARGET

    core = A.Core(m)
    exp = C.record_experiment(UsesAlias, DB, core)
    assert exp.readout is exp.ro_dds
    (e,) = core.events
    assert e.frequency == 80.235*MHz and e.phase_turns == 0.5


def test_reset_is_refused_inside_open_frames():
    """core.reset() under parallel/sequential would wipe the frame stack mid-flight and bypass
    the bare-parallel guard (Codex F2) — refuse it."""
    m = _map()

    class BadReset(EnvExperiment):
        def build(self):
            self.setattr_device("core")

        @kernel
        def run(self):
            with parallel:
                with sequential:
                    self.core.reset()

    with pytest.raises(RuntimeError, match="top level of run"):
        C.record_experiment(BadReset, DB, A.Core(m))


def test_open_driver_releases_on_partial_failure(monkeypatch):
    """The driver must not leak when load/get_params/set_model raises AFTER acquisition
    (Codex F1) — exercised on the real _open_driver with stubbed backends."""
    released = []

    class FakeProxy:
        def _pyroRelease(self):
            released.append("board")

    class FakeBoard:
        def load(self, bundle):
            raise RuntimeError("no such bundle")

    class FakeRemote:
        def __init__(self, host):
            self._proxy, self.board = FakeProxy(), FakeBoard()

    import riscq.driver.remote as remote_mod
    monkeypatch.setattr(remote_mod, "RemoteDriver", FakeRemote)
    with pytest.raises(RuntimeError, match="no such bundle"):
        C._open_driver({"type": "board", "host": "h", "bundle": "b"})
    assert released == ["board"]

    class FakeSim:
        def get_params(self):
            raise RuntimeError("handshake died")

    class FakeSrvDrv:
        sim = FakeSim()

    import riscq_sim.cosim as server_mod
    monkeypatch.setattr(server_mod, "start", lambda cfg, bld: FakeSrvDrv())
    monkeypatch.setattr(server_mod, "stop", lambda drv: released.append("cosim"))
    with pytest.raises(RuntimeError, match="handshake died"):
        C._open_driver({"type": "cosim", "config": "c", "build": "b"})
    assert released == ["board", "cosim"]


def test_multicore_device_db_channels():
    """On a 2-core build `adc`/`demod` take `channel` = the hardware core (default 0) and dds
    channels run 0..3; the same keys are refused on a 1-core build at device construction."""
    m2 = _map("rfsoc4x2-2q-fine")
    db = {
        "core":  {"type": "board", "host": "192.168.3.1", "bundle": "rfsoc4x2-2q-fine"},
        "dds_a": {"type": "dds", "channel": 1},
        "dds_b": {"type": "dds", "channel": 3},
        "adc_a": {"type": "adc"},
        "adc_b": {"type": "adc", "channel": 1},
        "dm_b":  {"type": "demod", "channel": 1},
    }

    class Both(EnvExperiment):
        def build(self):
            for n in ("core", "dds_a", "dds_b", "adc_a", "adc_b", "dm_b"):
                self.setattr_device(n)

        @kernel
        def run(self):
            with parallel:
                with sequential:
                    self.dds_a.set(82*MHz, amplitude=0.4); self.dds_a.sw.pulse(2*us)
                with sequential:
                    self.dds_b.set(82*MHz, amplitude=0.4); self.dds_b.sw.pulse(2*us)
                with sequential:
                    self.adc_a.gate(3*us)
                with sequential:
                    self.adc_b.gate(3*us)

    core = A.Core(m2)
    exp = C.record_experiment(Both, db, core)
    assert exp.adc_b._adc.core_index == 1 and exp.dm_b._dm.core_index == 1
    assert exp.adc_a._adc.core_index == 0
    assert {e.channel for e in core.events} == {1, 4}            # flat ids: core 0 / core 1 readout
    assert sorted(tg.core_index for tg in core.trace_gates) == [0, 1]

    one = dict(DB); one["adc_b"] = {"type": "adc", "channel": 1}
    with pytest.raises(ValueError, match="traces 0..0"):
        C.record_experiment(Both, one, A.Core(_map()))


def test_alias_to_core_shares_the_core_device():
    """An alias pointing AT `core` resolves to the same CoreDevice (the reverse — `core`
    itself being an alias — is refused at validation, tested above)."""
    m = _map()

    class C2(EnvExperiment):
        def build(self):
            self.setattr_device("core")
            self.setattr_device("core2")

        @kernel
        def run(self):
            pass

    db = dict(DB); db["core2"] = "core"
    exp = C.record_experiment(C2, db, A.Core(m))
    assert exp.core2 is exp.core


def test_run_experiment_lifecycle(monkeypatch):
    """Success: hardware runs, THEN analyze; failure in the hardware run: analyze is skipped;
    the driver closer runs in both cases (Codex F4)."""
    m = _map()
    params = (CONFIGS / "rfsoc4x2-1q-fine.json").read_text()
    closed = []
    monkeypatch.setattr(C, "_open_driver",
                        lambda entry: ("DRV", params, lambda: closed.append(True)))

    ran, analyzed = [], []

    class Exp(EnvExperiment):
        def build(self):
            self.setattr_device("core")
            self.setattr_device("ro_dds")

        @kernel
        def run(self):
            self.ro_dds.set(82*MHz, amplitude=0.4)
            self.ro_dds.sw.pulse(10*us)

        def analyze(self):
            analyzed.append(self.last_result)

    monkeypatch.setattr(A, "run", lambda drv, core, wd, doc="": ran.append(drv) or "RESULT")
    exp = C.run_experiment(Exp, DB, workdir="unused")
    assert ran == ["DRV"] and analyzed == ["RESULT"] and exp.last_result == "RESULT"
    assert closed == [True]

    def fail(drv, core, wd, doc=""):
        raise RuntimeError("hw died")
    monkeypatch.setattr(A, "run", fail)
    analyzed.clear(); closed.clear()
    with pytest.raises(RuntimeError, match="hw died"):
        C.run_experiment(Exp, DB, workdir="unused")
    assert analyzed == [] and closed == [True]


def test_core_reset_clears_the_same_core():
    """Devices keep their Core reference across reset(): reset clears, never rebinds."""
    m = _map()

    class Twice(EnvExperiment):
        def build(self):
            self.setattr_device("core")
            self.setattr_device("ro_dds")

        @kernel
        def run(self):
            self.ro_dds.set(82*MHz, amplitude=0.4)
            self.ro_dds.sw.pulse(10*us)
            self.core.reset()                                # drop it all
            self.ro_dds.set(82*MHz, amplitude=0.4)
            self.ro_dds.sw.pulse(20*us)

    core = A.Core(m)
    C.record_experiment(Twice, DB, core)
    assert len(core.events) == 1 and core.events[0].dur_mu == core.seconds_to_mu(20*us)
