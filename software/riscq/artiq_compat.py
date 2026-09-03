"""Character-identical ARTIQ experiment shape over riscq.artiqapi.

The user writes exactly what they would write for ARTIQ — an `EnvExperiment` subclass with
`build`/`prepare`/`run`/`analyze`, `setattr_device`, a bare `with parallel:` /
`with sequential:`, bare `delay(...)` — and `run_experiment(Exp, device_db)` plays the role of
`artiq_run`: build, record the kernel, execute it on the hardware (or co-sim), then `analyze`.

    from riscq.artiq_compat import *

    class IonTrap(EnvExperiment):
        def build(self):
            self.setattr_device("core")
            self.setattr_device("ro_dds")
            self.setattr_device("adc")

        @kernel
        def run(self):
            self.core.reset()
            with parallel:
                with sequential:
                    self.ro_dds.set(82*MHz, phase=0.25, amplitude=0.4)
                    self.ro_dds.sw.pulse(20*us)
                with sequential:
                    self.adc.gate(30*us)

        def analyze(self):
            self.trace = self.adc.fetch_trace()

    device_db = {
        "core":   {"type": "board", "host": "192.168.3.1", "bundle": "rfsoc4x2-1q-fine"},
        "ro_dds": {"type": "dds", "channel": 1},
        "adc":    {"type": "adc"},
    }
    exp = run_experiment(IonTrap, device_db)

WHAT IS AND IS NOT ARTIQ HERE (the honest contract):
  * an "ARTIQ-syntax restricted subset", not ARTIQ's compiler. `@kernel` marks the method (and
    `run_experiment` REQUIRES the mark on `run`) but the body is recorded by executing it as
    ordinary Python; the RISC-V kernel is generated from the recorded schedule. Consequences:
    no in-kernel branching on hardware state, and the hardware runs AFTER `run()` returns —
    mixed host/kernel orchestration inside `run` is not supported (hence the required mark:
    the whole method is one kernel).
  * ARTIQ's `with parallel:` parallelizes every immediate STATEMENT; a runtime layer cannot
    see statements, so every parallel arm must be wrapped in `with sequential:` (ARTIQ's own
    recommended style). A timeline operation directly inside a bare `parallel` raises
    instead of silently serializing.
  * the device db is a RISC-Q device DB (types `board`/`cosim`/`dds`/`adc`/`demod`, plus
    string aliases), validated eagerly — not ARTIQ's `type=local/module/class` schema.
"""

from __future__ import annotations

import contextvars
from pathlib import Path

from riscq import artiqapi as _A
from riscq.artiqapi import (  # noqa: F401  (re-exported: the ARTIQ names)
    GHz, Hz, MHz, kHz, ms, ns, s, us,
    PHASE_MODE_ABSOLUTE, PHASE_MODE_CONTINUOUS, PHASE_MODE_TRACKING,
)
from riscq.map import SocMap, SocParams

__all__ = [
    "EnvExperiment", "kernel", "run_experiment", "parallel", "sequential",
    "delay", "delay_mu", "now_mu", "at_mu",
    "s", "ms", "us", "ns", "Hz", "kHz", "MHz", "GHz",
    "PHASE_MODE_CONTINUOUS", "PHASE_MODE_ABSOLUTE", "PHASE_MODE_TRACKING",
]

_current: contextvars.ContextVar["_A.Core"] = contextvars.ContextVar("riscq_current_core")


def _core() -> "_A.Core":
    try:
        return _current.get()
    except LookupError:
        raise RuntimeError(
            "no experiment is running — bare delay/parallel/... work only inside an "
            "EnvExperiment executed by run_experiment()") from None


def _stack(core) -> list:
    if not hasattr(core, "_compat_stack"):
        core._compat_stack = []
    return core._compat_stack


def _guard(core) -> None:
    """A timeline operation directly inside a bare `with parallel:` (no `with sequential:` arm
    open) would silently SERIALIZE where ARTIQ would parallelize — refuse it loudly."""
    st = _stack(core)
    if st and st[-1][0] == "P":
        raise RuntimeError(
            "timeline operation directly inside `with parallel:` — this layer cannot "
            "statement-split like ARTIQ's compiler, so wrap each parallel arm in "
            "`with sequential:`")


# ── bare timeline verbs ──────────────────────────────────────────────────────────────────────────

def delay(dt: float) -> None:
    c = _core(); _guard(c); c.delay(dt)


def delay_mu(dt: int) -> None:
    c = _core(); _guard(c); c.delay_mu(dt)


def now_mu() -> int:
    return _core().now_mu()


def at_mu(t: int) -> None:
    c = _core(); _guard(c); c.at_mu(t)


# ── bare `with parallel:` / `with sequential:` ───────────────────────────────────────────────────

class _Parallel:
    def __enter__(self):
        c = _core()
        cm = _A.parallel(c)
        cm.__enter__()
        _stack(c).append(("P", cm))
        return self

    def __exit__(self, *exc):
        c = _core()
        kind, cm = _stack(c).pop()
        assert kind == "P"
        return cm.__exit__(*exc)


class _Sequential:
    """One parallel arm when directly under `parallel`; plain grouping anywhere else (a nested
    `sequential` must NOT rewind to the parallel start)."""

    def __enter__(self):
        c = _core()
        st = _stack(c)
        cm = _A.branch(c) if (st and st[-1][0] == "P") else _A.sequential(c)
        cm.__enter__()
        st.append(("S", cm))
        return self

    def __exit__(self, *exc):
        c = _core()
        kind, cm = _stack(c).pop()
        assert kind == "S"
        return cm.__exit__(*exc)


parallel = _Parallel()
sequential = _Sequential()


# ── guarded devices (interception happens here, so artiqapi stays untouched) ─────────────────────

class _GuardedSwitch:
    def __init__(self, dds):
        self._dds = dds

    def pulse(self, duration: float) -> None:
        _guard(self._dds.core); self._dds.sw.pulse(duration)

    def pulse_mu(self, duration_mu: int) -> None:
        _guard(self._dds.core); self._dds.sw.pulse_mu(duration_mu)

    def on(self):
        raise NotImplementedError("use pulse()/pulse_mu(); a free-running switch has no "
                                  "end time to schedule")
    off = on


class DDSDevice:
    """`artiq.coredevice.ad9910.AD9910` shape, guarded for the bare-parallel rule."""

    def __init__(self, dds: "_A.DDSChannel"):
        self._dds = dds
        self.sw = _GuardedSwitch(dds)

    def set(self, frequency, phase=0.0, amplitude=1.0, phase_mode=None) -> None:
        _guard(self._dds.core)
        kw = {} if phase_mode is None else {"phase_mode": phase_mode}
        self._dds.set(frequency, phase=phase, amplitude=amplitude, **kw)

    def set_phase_mode(self, phase_mode: int) -> None:
        self._dds.set_phase_mode(phase_mode)

    def pulse(self, duration: float) -> None:      # convenience alias, same as sw.pulse
        self.sw.pulse(duration)


class ADCDevice:
    def __init__(self, adc: "_A.ADCChannel"):
        self._adc = adc

    def gate(self, duration: float):
        _guard(self._adc.core); return self._adc.gate(duration)

    def gate_mu(self, duration_mu: int):
        _guard(self._adc.core); return self._adc.gate_mu(duration_mu)

    def fetch_trace(self):
        return self._adc.fetch_trace()


class DemodDevice:
    def __init__(self, dm: "_A.DemodChannel"):
        self._dm = dm

    def set(self, frequency, phase=0.0) -> None:
        _guard(self._dm.core); self._dm.set(frequency, phase=phase)

    def set_phase_mode(self, phase_mode: int) -> None:
        self._dm.set_phase_mode(phase_mode)

    def gate(self, duration: float):
        _guard(self._dm.core); return self._dm.gate(duration)

    def gate_mu(self, duration_mu: int):
        _guard(self._dm.core); return self._dm.gate_mu(duration_mu)

    def fetch_iq(self):
        return self._dm.fetch_iq()


class CoreDevice:
    """`self.core`: reset + the mu helpers. `reset()` clears the SAME underlying timeline the
    devices hold references to (it never rebinds the context)."""

    def __init__(self, core: "_A.Core"):
        self._core = core

    def reset(self) -> None:
        if _stack(self._core):
            raise RuntimeError(
                "core.reset() inside `with parallel:`/`with sequential:` — reset drops the whole "
                "recorded timeline, so call it only at the top level of run()")
        self._core.clear()

    def seconds_to_mu(self, t: float) -> int:
        return self._core.seconds_to_mu(t)

    def mu_to_seconds(self, mu: int) -> float:
        return self._core.mu_to_seconds(mu)


# ── the experiment class and its runner ──────────────────────────────────────────────────────────

def kernel(fn):
    """ARTIQ's `@kernel`, as a MARK: the body is recorded as ordinary Python and the whole
    method becomes one hardware kernel. run_experiment refuses an unmarked `run`."""
    fn.__riscq_kernel__ = True
    return fn


class EnvExperiment:
    """build()/prepare()/run()/analyze(), `setattr_device` — ARTIQ's experiment shape."""

    def build(self):
        pass

    def prepare(self):
        pass

    def run(self):
        raise NotImplementedError("an experiment must define run()")

    def analyze(self):
        pass

    def setattr_device(self, name: str) -> None:
        setattr(self, name, self.__riscq_devices__[name])


_DEV_KEYS = {"board": {"host", "bundle"}, "cosim": {"config", "build"},
             "dds": {"channel"}, "adc": set(), "demod": set()}


def _validate_db(db: dict) -> dict:
    """Eager validation (alias chains followed for CHECKING only; the raw db is returned so
    _build_devices can preserve alias identity — alias and target share one device)."""
    if "core" not in db:
        raise ValueError("device_db needs a 'core' entry ({'type': 'board'|'cosim', ...})")
    if isinstance(db["core"], str):
        raise ValueError("device_db['core'] must be the driver entry itself, not an alias")
    out = {}
    for name in db:
        entry, seen = db[name], [name]
        while isinstance(entry, str):                       # alias chain
            if entry in seen:
                raise ValueError(f"device_db alias cycle: {' -> '.join(seen + [entry])}")
            seen.append(entry)
            if entry not in db:
                raise ValueError(f"device_db alias {name!r} -> unknown device {entry!r}")
            entry = db[entry]
        if not isinstance(entry, dict) or "type" not in entry:
            raise ValueError(f"device_db[{name!r}] must be a dict with a 'type' (or an alias)")
        typ = entry["type"]
        if typ not in _DEV_KEYS:
            raise ValueError(f"device_db[{name!r}]: unknown type {typ!r} "
                             f"(have {sorted(_DEV_KEYS)})")
        missing = _DEV_KEYS[typ] - set(entry)
        if missing:
            raise ValueError(f"device_db[{name!r}] ({typ}) is missing {sorted(missing)}")
        owner = seen[-1]                                # the alias chain's final TARGET name
        if name == "core" and typ not in ("board", "cosim"):
            raise ValueError(f"device_db['core'] must be type 'board' or 'cosim', not {typ!r}")
        if owner != "core" and typ in ("board", "cosim"):
            raise ValueError(f"device_db[{owner!r}]: type {typ!r} belongs only in 'core'")
        out[name] = entry
    return db


def _open_driver(entry: dict):
    """Returns (drv, params_text, closer). Once the driver is ACQUIRED, every failure path
    releases it — including failures in load/set_model/get_params below."""
    if entry["type"] == "board":
        from riscq.driver import remote
        drv = remote.RemoteDriver(entry["host"])
        def closer():
            try:
                drv._proxy._pyroRelease()
            except Exception:
                pass
        try:
            drv.board.load(entry["bundle"])
            params = drv.board.get_params()
        except BaseException:
            closer()
            raise
        return drv, params, closer
    from riscq.sim import server
    drv = server.start(entry["config"], entry["build"])
    def closer():
        server.stop(drv)
    try:
        if entry.get("model"):
            drv.sim.set_model(entry["model"])
        params = drv.sim.get_params()
    except BaseException:
        closer()
        raise
    return drv, params, closer


def _build_devices(db: dict, core: "_A.Core") -> dict:
    """One device object per TARGET entry; an alias gets the SAME object (ARTIQ's
    DeviceManager semantics — alias and target are interchangeable, state included)."""
    def target(name):
        n = name
        while isinstance(db[n], str):
            n = db[n]
        return n

    devs: dict = {}
    for name in db:
        tn = target(name)
        if tn not in devs:
            entry = db[tn]
            if tn == "core":
                devs[tn] = CoreDevice(core)
            elif entry["type"] == "dds":
                devs[tn] = DDSDevice(_A.DDSChannel(core, int(entry["channel"]), tn))
            elif entry["type"] == "adc":
                devs[tn] = ADCDevice(_A.ADCChannel(core))
            else:
                devs[tn] = DemodDevice(_A.DemodChannel(core))
        devs[name] = devs[tn]
    return devs


def record_experiment(exp_cls, device_db: dict, core: "_A.Core"):
    """build + prepare + run on an EXISTING core (no hardware): the recording half of
    run_experiment, separated so host-pure tests need no driver."""
    db = _validate_db(device_db)
    if not getattr(exp_cls.run, "__riscq_kernel__", False):
        raise RuntimeError(
            f"{exp_cls.__name__}.run must be decorated with @kernel — the whole method is "
            "recorded and executed as ONE hardware kernel (host code interleaved with "
            "hardware execution inside run() is not supported by this layer)")
    exp = exp_cls()
    exp.__riscq_devices__ = _build_devices(db, core)
    token = _current.set(core)
    try:
        exp.build()
        exp.prepare()
        exp.run()
    finally:
        _current.reset(token)
    return exp


def run_experiment(exp_cls, device_db: dict, workdir: str | Path = "artiq_compat_work",
                   doc: str = ""):
    """The `artiq_run` role: record the experiment, execute it, then analyze. Returns the
    experiment instance (RunResult at .last_result); analyze() runs only after a successful
    hardware run."""
    db = _validate_db(device_db)
    drv, params_text, closer = _open_driver(db["core"])
    try:
        core = _A.Core(SocMap(SocParams.from_json(params_text)))
        exp = record_experiment(exp_cls, device_db, core)
        exp.last_result = _A.run(drv, core, workdir, doc=doc or exp_cls.__name__)
        exp.analyze()
        return exp
    finally:
        closer()
