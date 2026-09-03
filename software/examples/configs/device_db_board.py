"""Device db for the RFSoC 4x2 on the bench: single-DAC bundle (gate + readout summed on DAC0).

    from configs.device_db_board import device_db
    exp = run_experiment(MyExperiment, device_db)

Edit `host` to your board's IP (board_setup.sh / start_server.sh print it).
"""
device_db = {
    "core":     {"type": "board", "host": "192.168.3.1", "bundle": "rfsoc4x2-1q-fine"},
    "gate_dds": {"type": "dds", "channel": 0},     # gate drive   (DAC0)
    "ro_dds":   {"type": "dds", "channel": 1},     # readout drive (DAC0 too, on this bundle)
    "adc":      {"type": "adc"},                   # raw ADC0 trace
    "demod":    {"type": "demod"},                 # hardware IQ readout on ADC0
}
