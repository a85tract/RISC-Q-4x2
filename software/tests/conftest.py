from pathlib import Path

import pytest

SW_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = SW_ROOT / "configs"


def pytest_addoption(parser):
    parser.addoption("--cosim", action="store_true", default=False,
                     help="run the verilator co-simulation tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "cosim: verilator co-simulation test (needs --cosim)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--cosim"):
        return
    skip = pytest.mark.skip(reason="co-sim test: pass --cosim to run")
    for item in items:
        if "cosim" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def cosim(request):
    """A running verilator co-sim of the sim-2q build: (CosimDriver, SocMap)."""
    if not request.config.getoption("--cosim"):
        pytest.skip("needs --cosim")
    from riscq.map import SocMap, SocParams
    from riscq.sim import server

    drv = server.start(CONFIGS / "sim-2q.json", SW_ROOT / "build" / "sim-2q")
    m = SocMap(SocParams.from_json(drv.sim.get_params()))
    yield drv, m
    server.stop(drv)


@pytest.fixture(scope="session")
def cosim_2q1c(request):
    """A running verilator co-sim of the sim-2q1c (3-core: 2 qubits + 1 coupler) build, with the
    explicit dac_map/adc_map (specs/two-qubit/01 §1): (CosimDriver, SocMap)."""
    if not request.config.getoption("--cosim"):
        pytest.skip("needs --cosim")
    from riscq.map import SocMap, SocParams
    from riscq.sim import server

    drv = server.start(CONFIGS / "sim-2q1c.json", SW_ROOT / "build" / "sim-2q1c")
    m = SocMap(SocParams.from_json(drv.sim.get_params()))
    yield drv, m
    server.stop(drv)
