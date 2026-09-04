"""Device db for the two-core bundle: dds 0/1 on DAC_A with the trace on ADC_A, dds 2/3 on DAC_B
with the trace on ADC_B — one timeline across both.

    from configs.device_db_2q import device_db          # board (loop DAC_A -> ADC_A, DAC_B -> ADC_B)
    from configs.device_db_2q import device_db_cosim    # co-simulation of the same bundle

`adc`/`demod` take "channel" = the hardware core (0 = the DAC_A/ADC_A side, 1 = DAC_B/ADC_B).
"""
_devices = {
    "dds_a0": {"type": "dds", "channel": 0},      # core 0 gate drive     -> DAC_A
    "dds_a1": {"type": "dds", "channel": 1},      # core 0 readout drive  -> DAC_A (recorded by adc_a)
    "dds_b0": {"type": "dds", "channel": 2},      # core 1 gate drive     -> DAC_B
    "dds_b1": {"type": "dds", "channel": 3},      # core 1 readout drive  -> DAC_B (recorded by adc_b)
    "adc_a":  {"type": "adc", "channel": 0},      # raw trace of ADC_A
    "adc_b":  {"type": "adc", "channel": 1},      # raw trace of ADC_B
    "demod_a": {"type": "demod", "channel": 0},   # IQ readout of core 0
    "demod_b": {"type": "demod", "channel": 1},   # IQ readout of core 1
}

device_db = {
    "core": {"type": "board", "host": "192.168.3.1", "bundle": "rfsoc4x2-2q-fine"},
    **_devices,
}

device_db_cosim = {
    "core": {
        "type": "cosim",
        "config": "gateware/configs/rfsoc4x2-2q-fine.json",
        "build": "sim/build/rfsoc4x2-2q-fine",
        # the two "cables": core 0 is SoC DAC1/ADC1 (connectors DAC_A/ADC_A), core 1 is DAC0/ADC0
        "model": {"kind": "multi", "models": [
            {"kind": "loopback", "src": 1, "dst": 1, "gain": 0.9, "delay": 5},
            {"kind": "loopback", "src": 0, "dst": 0, "gain": 0.9, "delay": 5}]},
    },
    **_devices,
}
