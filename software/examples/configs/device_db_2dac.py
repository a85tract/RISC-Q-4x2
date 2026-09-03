"""Device db for the two-DAC bundle: gate drive on DAC0, readout drive on DAC1, ADC0 readout.

    from configs.device_db_2dac import device_db          # board
    from configs.device_db_2dac import device_db_cosim    # co-simulation of the same bundle

Bench wiring for the receive-side checks: the readout tone is on DAC1, so loop DAC1 -> ADC0.
"""
_devices = {
    "gate_dds": {"type": "dds", "channel": 0},     # -> DAC0
    "ro_dds":   {"type": "dds", "channel": 1},     # -> DAC1
    "adc":      {"type": "adc"},                   # ADC0
    "demod":    {"type": "demod"},
}

device_db = {
    "core": {"type": "board", "host": "192.168.3.1", "bundle": "rfsoc4x2-2dac-fine"},
    **_devices,
}

device_db_cosim = {
    "core": {
        "type": "cosim",
        "config": "gateware/configs/rfsoc4x2-2dac-fine.json",
        "build": "sim/build/rfsoc4x2-2dac-fine",
        "model": {"kind": "loopback", "src": 1, "dst": 0, "gain": 0.9, "delay": 5},   # DAC1 -> ADC0
    },
    **_devices,
}
