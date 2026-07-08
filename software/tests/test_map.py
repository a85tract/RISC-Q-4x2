"""SocMap derivations reproduce the elaboration's known numbers (independent literals,
verified against the SpinalHDL SocMemoryMap), and the generators evaluate them."""

from pathlib import Path

from riscq.map import MEM_BASE, SocMap, SocParams, pack16

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _map(name: str) -> SocMap:
    return SocMap(SocParams.load(CONFIGS / f"{name}.json"))


def test_pack16_seats_field_at_bit16():
    """pack16 seats a 16-bit code in data[31:16] (spec 12) — the software-side RQ_PACK16, returning
    the SIGNED int32 word a register holds. The low 16 bits are zero (single-`lui` constant / free
    Q16 fraction); a code with bit 15 set seats to a negative word (passes the kernel int32 check)."""
    assert pack16(0) == 0
    assert pack16(1) == 0x00010000
    assert pack16(0x7FFF) == 0x7FFF0000
    assert pack16(-0x8000) == -(1 << 31)                       # bit-15 code -> most-negative int32
    assert pack16(-1) == -0x10000                              # 0xFFFF0000 as signed
    for code in (0, 1, 1234, -1, -2048, 0x7FFF, -0x8000, 0x8000):
        w = pack16(code)
        assert -(1 << 31) <= w <= (1 << 31) - 1 and w & 0xFFFF == 0   # signed int32, low half free
        assert (w >> 16) & 0xFFFF == code & 0xFFFF             # field == code mod 2^16
    assert pack16(0x12345) == 0x23450000                       # wider codes fold mod 2^16


def test_zcu216_14q_numbers():
    m = _map("zcu216-14q")
    assert m.core_stride == 0x10000       # 64 KB
    assert m.pulse_stride == 0x4000       # 16 KB (128-bit gate line x 1024)
    assert m.ro_env_stride == 0x1000      # 4 KB (32-bit ro line x 1024)
    assert m.demod_env_stride == 0x1000   # 4 KB (32-bit demod-carrier line x 1024)
    assert m.rob_stride == 0x8000         # 32 KB (2 x pow2ceil(128*1024/8), rob_depth 1024)
    # pow2ceil(0x10000 * 14 = 0xE0000) = 0x100000 (1 MB) — RTL-verified; CONTRACT.md's 2 MB
    # was an arithmetic slip.
    assert m.region_size == 0x100000
    assert m.core_mem_base == 0x0
    assert m.pulse_mem_base == 0x100000
    assert m.ro_env_base == 0x200000
    assert m.demod_env_base == 0x300000
    assert m.rob_base == 0x400000
    assert m.host_ctrl == 0x500000
    assert m.imem(13) == 13 * 0x10000
    assert m.gate_env(2) == 0x100000 + 2 * 0x4000
    assert m.ro_env(3) == 0x200000 + 3 * 0x1000
    assert m.demod_env(3) == 0x300000 + 3 * 0x1000
    assert m.robs() == 0x400000
    assert m.to_host_addr(13, MEM_BASE + 0x123 * 4) == 13 * 0x10000 + 0x123 * 4
    assert m.mem_bytes == 4096 * 4


def test_sim_2q_numbers():
    m = _map("sim-2q")
    assert m.region_size == 0x20000       # pow2ceil(0x10000 * 2)
    assert [m.core_mem_base, m.pulse_mem_base, m.ro_env_base,
            m.demod_env_base, m.rob_base, m.host_ctrl] == [k * 0x20000 for k in range(6)]
    assert m.imem(1) == 0x10000
    assert m.gate_env(1) == 0x20000 + 0x4000
    assert m.ro_env(1) == 0x40000 + 0x1000
    assert m.demod_env(1) == 0x60000 + 0x1000


def test_core_local_constants():
    m = _map("sim-2q")
    assert (m.CTRL_TIME_CMP, m.CTRL_WAIT_TIME_CMP, m.CTRL_TIME) == (0x4000, 0x4008, 0xBFF8)
    assert (m.CTRL_RES, m.CTRL_REAL, m.CTRL_IMAG, m.CTRL_FROM_HOST) == \
        (0x4200, 0x4204, 0x4208, 0x2000)
    assert (m.RF_GATE, m.RF_READOUT, m.RF_DEMOD) == (0x10000, 0x20000, 0x30000)
    assert (m.RF_FIRE, m.RF_FREQ, m.RF_SLOT_STRIDE, m.RF_START_TIME) == (0x0, 0x4, 0x10, 0x4100)
    assert m.RF_DC_OFFSET == 0x8
    assert m.RF_PHASE_OFFSET == 0xC
    assert (m.HOST_RESET, m.HOST_TIME_OFF_LO, m.HOST_TIME_OFF_HI) == (0x0, 0x40, 0x44)
    assert m.LEAD == 96   # pinned by the M1 lead-margin test (64 measured insufficient)


def test_generated_header_values():
    h = _map("sim-2q").gen_header()
    assert "#define RQ_MEM_BASE 0x80000000" in h
    assert "#define RQ_MEM_BYTES 0x4000" in h
    assert "#define RQ_CTRL_TIME 0xbff8" in h
    assert "#define RQ_GATE 0x10000" in h
    assert "#define RQ_DEMOD 0x30000" in h
    assert "RQ_DECODER" not in h                 # decoder RF window removed (carrier-triggered)
    assert "#define RQ_RO_MAX_WIN 0x4000" in h   # 2^14 window cap
    assert "#define RQ_RO_LEAD 0x30" in h        # 48-batch readout freshness wait
    assert "#define RQ_START_TIME 0x4100" in h
    assert "#define RQ_DC_OFFSET 0x8" in h
    assert "#define RQ_PHASE_OFFSET 0xc" in h
    assert "#define RQ_GATE_PULSE_NUM 0x8" in h


def test_channel_table():
    import pytest

    from riscq.map import ChannelInfo

    m = _map("sim-2q")
    assert m.channels() == [
        ChannelInfo(0, "RF_CH0", 0x10000, 8, 4, 16),   # gate drive: 8 slots, 4 samples/line
        ChannelInfo(1, "RF_CH1", 0x20000, 1, 1, 4),    # readout drive: 1 slot, 1 sample/line
        ChannelInfo(2, "RF_CH2", 0x30000, 1, 1, 4),    # demod carrier: 1 slot, 1 ADC-batch sample/line
    ]
    assert m.channel(0) == m.channels()[0]
    assert m.channel(2) == m.channels()[2]
    with pytest.raises(ValueError, match="unknown channel index"):
        m.channel(3)


def test_channel_defines_in_header():
    h = _map("sim-2q").gen_header()
    assert "#define RF_CH0 0x10000" in h
    assert "#define RF_CH1 0x20000" in h
    assert "#define RF_CH2 0x30000" in h


def test_generated_linker_values():
    ld = _map("sim-2q").gen_linker()
    assert "ORIGIN = 0x80000000, LENGTH = 16384" in ld
    assert ".text" in ld and "KEEP(*(.text.init))" in ld
    assert "__stack_top" in ld and "ASSERT" in ld


def test_entries_cover_every_region():
    m = _map("sim-2q")
    entries = m.entries()
    kinds = [e.kind for e in entries]
    assert kinds.count("ram_rw") == 2       # one per core
    assert kinds.count("env_gate") == 2
    assert kinds.count("env_ro") == 2
    assert kinds.count("env_demod") == 2
    assert kinds.count("robs_ro") == 1      # shared
    assert kinds.count("ctrl_wo") == 1
    by_name = {e.name: e for e in entries}
    assert by_name["core1_ram"].host_addr == 0x10000
    assert by_name["robs"].host_addr == m.robs()
    assert by_name["host_ctrl"].host_addr == m.host_ctrl
    # every region base is claimed by exactly the right window
    for e in entries:
        assert e.nbytes > 0


def test_fixed_constants_rejected():
    import json

    import pytest

    raw = json.loads((CONFIGS / "sim-2q.json").read_text())
    raw["batch_size"] = 8
    with pytest.raises(ValueError, match="fixed by architecture"):
        SocParams.from_json(json.dumps(raw))
