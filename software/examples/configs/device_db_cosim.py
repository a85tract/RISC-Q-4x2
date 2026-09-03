"""Device db for co-simulation (no hardware): the bit-accurate RTL model with a DAC0 -> ADC0 loopback.

    from configs.device_db_cosim import device_db
    exp = run_experiment(MyExperiment, device_db)

Paths are relative to the repository root (run from there, or make them absolute). Needs the
`full` docker image (Verilator, mill, cocotb) — see sim/README.md. The first run of a config
generates the RTL and verilates it (a few minutes); later runs start in seconds.
"""
device_db = {
    "core": {
        "type": "cosim",
        "config": "gateware/configs/rfsoc4x2-1q-fine.json",
        "build": "sim/build/rfsoc4x2-1q-fine",
        # what the "cable" does: DAC0 into ADC0, 0.9 gain, 5-sample delay (single-DAC bundle:
        # both drive channels are on DAC0, so the ADC sees their sum, as on the bench)
        "model": {"kind": "loopback", "src": 0, "dst": 0, "gain": 0.9, "delay": 5},
    },
    "gate_dds": {"type": "dds", "channel": 0},
    "ro_dds":   {"type": "dds", "channel": 1},
    "adc":      {"type": "adc"},
    "demod":    {"type": "demod"},
}
