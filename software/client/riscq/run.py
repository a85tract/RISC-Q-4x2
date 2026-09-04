"""The run layer: pure functions over (Driver, SocMap) implementing the boot/run protocol.
Backend-agnostic — the same code drives co-sim and hardware."""

from __future__ import annotations

import time as _time

import numpy as np
import serpent

from riscq.build import Image, Program
from riscq.map import SocMap, pack16

STATUS_DONE_MASK = 0xFFFF0000
STATUS_DONE = 0xD04E0000
STATUS_RUNNING = 1
MAGIC = 0x52515121


def reset(drv, m: SocMap, on: bool) -> None:
    """Assert (on=True) / release the shared core reset. Powers up asserted."""
    drv.write32(m.host_ctrl + m.HOST_RESET, 1 if on else 0)


def set_time_offset(drv, m: SocMap, t: int) -> None:
    drv.write32(m.host_ctrl + m.HOST_TIME_OFF_LO, t & 0xFFFFFFFF)
    drv.write32(m.host_ctrl + m.HOST_TIME_OFF_HI, (t >> 32) & 0xFFFFFFFF)


def load_program(drv, m: SocMap, core: int, image: Image) -> None:
    """Block-write the flat image into the core's RAM window (load address 0x80000000)."""
    drv.write_block(m.to_host_addr(core, image.entry), image.data)


def write_var(drv, m: SocMap, core: int, program: Program, name: str, value: int) -> None:
    """Write a named int32 global — before or during a run (the RAM host port is live)."""
    drv.write32(m.to_host_addr(core, program.var_addr(name)), int(value) & 0xFFFFFFFF)


def read_var(drv, m: SocMap, core: int, program: Program, name: str) -> int:
    """Read a named global; returns the raw unsigned 32-bit value."""
    return drv.read32(m.to_host_addr(core, program.var_addr(name)))


def read_array(drv, m: SocMap, core: int, program: Program, name: str) -> np.ndarray:
    """Fetch a named array global as int32 (element count from the ELF symbol size)."""
    addr, size = program.var_addr(name), program.var_size(name)
    if size < 4 or size % 4:
        raise ValueError(f"symbol {name!r} has size {size}, not an int32 array")
    return np.frombuffer(drv.read_block(m.to_host_addr(core, addr), size), dtype="<i4").copy()


def write_array(drv, m: SocMap, core: int, program: Program, name: str, values) -> None:
    """Fill a named int32 array global — a host-preloaded input Array (`slots`/`times`) lives in
    .data (RQ_PARAM) so this pre-run write survives boot (spec 02 §3.1)."""
    addr, size = program.var_addr(name), program.var_size(name)
    buf = np.asarray(values, dtype="<i4").tobytes()
    if len(buf) > size:
        raise ValueError(f"{len(buf)} B into array {name!r} of {size} B")
    drv.write_block(m.to_host_addr(core, addr), buf)


# rq_slot = { int32 phase, amp, env, dur } — the pulse-table entry layout (fw/riscq.h)
_SLOT_BYTES = 16
_SLOT_FIELD_OFF = {"phase": 0, "amp": 4, "env": 8, "dur": 12}


def load_tables(drv, m: SocMap, core: int, program: Program) -> None:
    """Fill each live ParamTable's .data `struct rq_slot[]` by symbol+offset with its compiled
    design-time codes — the kernel's init_pulse_params programs the hardware slots from it. .data
    (RQ_PARAM) survives start.S's .bss zeroing, like a param global (spec 02 §3.2). Every field is
    seated in data[31:16] (pack16, spec 12) so init_pulse_params' set_* write it raw."""
    for name, slot_codes in program.tables.items():
        buf = b"".join((pack16(c) & 0xFFFFFFFF).to_bytes(4, "little")
                       for slot in slot_codes for c in slot)
        drv.write_block(m.to_host_addr(core, program.var_addr(name)), buf)


def write_slot(drv, m: SocMap, core: int, program: Program, table: str, slot: int,
               field: str, value: int) -> None:
    """Retune one field of one ParamTable slot in .data by name (`gate[0].amp`) — no recompile,
    the next init_pulse_params picks it up (spec 02 §3.2). `value` is a plain code; it is seated in
    data[31:16] (pack16, spec 12) to match the compiled table fields."""
    addr = program.var_addr(table) + _SLOT_BYTES * slot + _SLOT_FIELD_OFF[field]
    drv.write32(m.to_host_addr(core, addr), pack16(value) & 0xFFFFFFFF)


def write_params(drv, m: SocMap, core: int, program: Program,
                 values: dict[str, int] | None = None) -> None:
    """Write the program's param globals: any compile-time auto value first, then the caller's
    `values` — an explicit value overrides an auto one (host-side retuning without recompiling).
    Kernel params are all user (auto=None) now; the pulse codes live in the .data tables
    (load_tables) instead of auto-param globals."""
    values = values or {}
    for name, auto in program.params.items():
        if auto is not None and name not in values:
            write_var(drv, m, core, program, name, auto)
    for name, value in values.items():
        write_var(drv, m, core, program, name, value)


def load_envelopes(drv, m: SocMap, core: int, program: Program) -> None:
    """Upload the program's envelope-RAM images (compile_kernel's allocator output)."""
    for channel, image in program.envelopes.items():
        for line0, lines in image:
            write_envelope(drv, m, core, channel, line0, lines)


def park_core(drv, m: SocMap, core: int) -> None:
    """Write a `j .` self-loop (0x6f) at the core's reset vector. Reset release boots ALL cores;
    a core whose RAM holds stale/garbage words would run wild (bus decode errors), so every core
    must hold either a program or this park word before the release."""
    drv.write32(m.imem(core), 0x6F)


def poll_done(drv, m: SocMap, core: int, program: Program, timeout: int = 2_000_000) -> int:
    """Poll __rq_status until DONE (0xD04E....); returns the full status word.
    `timeout` is in sim/host-clock cycles when the driver has sim extras, else read iterations.
    Loud TimeoutError with the last status on expiry."""
    addr = m.to_host_addr(core, program.var_addr("__rq_status"))
    sim = getattr(drv, "sim", None)
    chunk = 20_000
    spent = 0
    status = drv.read32(addr)
    while (status & STATUS_DONE_MASK) != STATUS_DONE:
        if spent >= timeout:
            raise TimeoutError(f"core {core} not DONE after {timeout} cycles "
                               f"(__rq_status = {status:#010x})")
        if sim is not None:
            status = sim.poll_word(addr, not_equal=status, timeout_cycles=min(chunk, timeout - spent))
        else:
            _time.sleep(0.001)
            status = drv.read32(addr)
        spent += chunk if sim is not None else 1
    return status


def check_magic(drv, m: SocMap, core: int, program: Program) -> None:
    magic = read_var(drv, m, core, program, "__rq_magic")
    if magic != MAGIC:
        raise RuntimeError(f"core {core}: __rq_magic = {magic:#010x} != {MAGIC:#010x} "
                           f"— image not loaded / wrong map")


# ── remote batching: serpent-safe Program reduction for the server-side runner (spec 08 §5) ──
# Only JSON-ish primitives + `bytes` cross the Pyro5/serpent wire, and `bytes` arrive on the far
# side as a {'data': <base64>, 'encoding': 'base64'} dict — so a Program (bytes image + np.uint32
# envelope arrays) is reduced to bytes + int-lists here and rebuilt with `_prog_from_wire`.

def _wire_bytes(x) -> bytes:
    """Unwrap a bytes field that may have crossed serpent as a base64 dict."""
    return serpent.tobytes(x) if isinstance(x, dict) else bytes(x)


def _params_json(m: SocMap) -> str:
    return m.params.to_json()


def _prog_to_wire(prog: Program) -> dict:
    """Reduce a Program to serpent-safe primitives (bytes + dicts + int lists). Envelope lines
    (np.uint32 (n, wpl)) become bytes + shape; everything else is ints/strings."""
    img = prog.image
    return {
        "data": img.data,
        "entry": int(img.entry),
        "symbols": {name: [int(addr), int(size)] for name, (addr, size) in img.symbols.items()},
        "params": {name: (None if v is None else int(v)) for name, v in prog.params.items()},
        "arrays": {name: int(n) for name, n in prog.arrays.items()},
        "tables": {name: [[int(c) for c in slot] for slot in slots]
                   for name, slots in prog.tables.items()},
        "envelopes": {int(chan): [_env_to_wire(line0, lines) for line0, lines in image]
                      for chan, image in prog.envelopes.items()},
    }


def _env_to_wire(line0, lines) -> list:
    arr = np.ascontiguousarray(lines, dtype="<u4")
    return [int(line0), arr.tobytes(), int(arr.shape[0]), int(arr.shape[1])]


def _prog_from_wire(wire: dict) -> Program:
    """Inverse of `_prog_to_wire`: rebuild a real Image + Program, unwrapping any serpent
    base64-dict byte fields (the image `data` and each envelope's packed lines)."""
    symbols = {name: (int(pair[0]), int(pair[1])) for name, pair in wire["symbols"].items()}
    image = Image(data=_wire_bytes(wire["data"]), symbols=symbols, entry=int(wire["entry"]))
    params = {name: (None if v is None else int(v)) for name, v in wire["params"].items()}
    arrays = {name: int(n) for name, n in wire["arrays"].items()}
    tables = {name: [tuple(int(c) for c in slot) for slot in slots]
              for name, slots in wire["tables"].items()}
    envelopes = {int(chan): [_env_from_wire(entry) for entry in entries]
                 for chan, entries in wire["envelopes"].items()}
    return Program(image, params=params, arrays=arrays, envelopes=envelopes, tables=tables)


def _env_from_wire(entry) -> tuple:
    line0, data, n_lines, wpl = entry
    arr = np.frombuffer(_wire_bytes(data), dtype="<u4").reshape(int(n_lines), int(wpl))
    return int(line0), arr


def setup(drv, m: SocMap, progs: dict[int, Program]) -> None:
    """Once per session: hold reset, load each core's image + envelopes + tables, and park the
    unassigned cores. Leaves reset ASSERTED so the first `rerun` writes into a quiescent, loaded
    core — the image then stays put across reruns (spec 03 §2, 08 §4). With a `drv.remote` extras
    object present, the whole load runs server-side in one RPC (spec 08 §5)."""
    remote = getattr(drv, "remote", None)
    if remote is not None:
        remote.setup(_params_json(m), {core: _prog_to_wire(prog) for core, prog in progs.items()})
        return
    reset(drv, m, on=True)
    for core, prog in progs.items():
        load_program(drv, m, core, prog.image)
        load_envelopes(drv, m, core, prog)
        load_tables(drv, m, core, prog)
    for core in range(m.params.qubit_num):   # un-programmed cores boot too: park them
        if core not in progs:
            park_core(drv, m, core)


def rerun(drv, m: SocMap, progs: dict[int, Program],
          params: dict[int, dict[str, int]] | None = None,
          arrays: dict[int, dict[str, object]] | None = None,
          results: list[str] | None = None,
          timeout: int = 2_000_000) -> dict[int, dict[str, np.ndarray]]:
    """Re-run an already-`setup` batch without any reload: check magic -> clear status -> write
    params + host input arrays -> one reset release for all cores -> poll -> read results ->
    re-assert reset. Reuses the loaded image/envelopes/tables, so a whole sweep costs O(1) driver
    ops (spec 08 §4). Reset is held on entry (`setup` or the previous `rerun` left it asserted);
    `__rq_status` is cleared while it is held so a stale DONE from the previous run can't race the
    poll (start.S raises RUNNING only after zeroing .bss). With a `drv.remote` extras object present,
    the whole batch (params + arrays in, poll, results out) runs server-side in one RPC (spec 08 §5)."""
    remote = getattr(drv, "remote", None)
    if remote is not None:
        raw = remote.rerun(list(progs), params or {}, arrays or {}, results, timeout)
        return {int(core): {name: np.frombuffer(buf, dtype="<i4").copy() for name, buf in d.items()}
                for core, d in raw.items()}
    params = params or {}
    arrays = arrays or {}
    for core, prog in progs.items():
        check_magic(drv, m, core, prog)                    # guard: image still loaded
        write_var(drv, m, core, prog, "__rq_status", 0)    # clear stale DONE, reset held
        write_params(drv, m, core, prog, params.get(core, {}))
        for name, values in arrays.get(core, {}).items():
            write_array(drv, m, core, prog, name, values)
    reset(drv, m, on=False)
    for core, prog in progs.items():
        poll_done(drv, m, core, prog, timeout=timeout)
    out = {core: {name: read_array(drv, m, core, prog, name)
                  for name in (list(prog.arrays) if results is None else results)}
           for core, prog in progs.items()}
    reset(drv, m, on=True)
    return out


def run(drv, m: SocMap, progs: dict[int, Program],
        params: dict[int, dict[str, int]] | None = None,
        arrays: dict[int, dict[str, object]] | None = None,
        results: list[str] | None = None,
        timeout: int = 2_000_000) -> dict[int, dict[str, np.ndarray]]:
    """The one-shot convenience: `setup` (load) then a single `rerun` (spec 08 §4)."""
    setup(drv, m, progs)
    return rerun(drv, m, progs, params, arrays, results, timeout)


def _env_window(m: SocMap, core: int, channel: int) -> tuple[int, int, int]:
    """(host base, bytes per line, words per line) of one core's envelope RAM for channel `channel`
    (the logical RF channel index: 0 gate / 1 ro / 2 demod)."""
    ch = m.channel(channel)
    return m.env_base(channel, core), ch.line_bytes, ch.samples_per_line


def write_envelope(drv, m: SocMap, core: int, channel: int, line0: int, lines) -> None:
    """Upload packed envelope lines ((n_lines, words_per_line) uint32, from riscq.pulses) at
    RAM line `line0`, for the logical RF channel index `channel`. A gate line is 4 words at host
    line*16 + {0,4,8,12} — contiguous, so this is one block write; readout is 1 word per line at
    line*4. The envelope banks are host WRITE-ONLY (BramWriteFiber — the generator reads them
    internally), so there is no read_envelope; the write→DAC path is verified bit-exact in
    tests/test_pulse.test_dac_window_bit_exact_*."""
    lines = np.ascontiguousarray(lines, dtype="<u4")
    base, line_bytes, wpl = _env_window(m, core, channel)
    if lines.ndim != 2 or lines.shape[1] != wpl:
        raise ValueError(f"channel {channel} lines must be (n, {wpl}) words, got shape {lines.shape}")
    if line0 < 0 or line0 + lines.shape[0] > m.params.env_depth:
        raise ValueError(f"lines [{line0}, {line0 + lines.shape[0]}) outside envelope RAM "
                         f"depth {m.params.env_depth}")
    drv.write_block(base + line0 * line_bytes, lines.tobytes())


def read_robs(drv, m: SocMap, nbytes: int | None = None, core: int = 0) -> np.ndarray:
    """Diagnostic fetch of a raw-ADC trace BRAM as its lanes (int16 on a rob_per_core build — `core`'s
    own trace — else the shared int32-lane trace)."""
    nbytes = m.rob_bytes if nbytes is None else nbytes
    return np.frombuffer(drv.read_block(m.robs(core), nbytes), dtype=m.rob_dtype).copy()
