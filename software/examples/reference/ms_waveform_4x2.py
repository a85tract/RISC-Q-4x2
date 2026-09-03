"""Play the ion-trap two-pulse waveform (Tests/waveform_generator.py) on the RISC-Q refactor
PulseTableSoc and capture it in the robs trace — co-sim (verilator + LoopbackModel) or board.

Sequence (all on DAC_A via dac_map [[0,0]]):
  pulse 1 (100 us): tone A (82+1.765 MHz, phase 0)   on the GATE channel (ch0)
                  + tone B (82-1.765 MHz, phase -180) on the READOUT channel (ch1)
  gap 5 us, then pulse 2 (20 us): 82 MHz, phase +90, on the READOUT channel (ch1)

Frequency words: default = the 16-bit code grid (120 kHz steps: 698 / 669 / 683 -> 83.76 / 80.28 /
81.96 MHz, 5 / 45 / 40 kHz off the generator); `--exact` (freq_width 32 builds, M7b) = the
generator's TRUE tones via units.freq_word (NCO realisation errors +0.61 / 0.00 / -0.61 Hz).

Phases are referenced to each pulse's ENVELOPE start and compensated IN-KERNEL: the carrier is
absolute-time (phase16 = ((W * 16 * tau) mod 2^fw) >> (fw - 16)) and the envelope batch played at
time t carries the carrier of tau = t - TIME_TO_PULSE (golden.py: 36 batches = 73.2 ns), so the
register written for a pulse starting at batch t is `target - W * ((t - 36) * 16)` (int32 wrap).
Earlier versions of this script (and the dev line) referenced the phases to the absolute-time
frame instead — 36 batches after the physical edge, a frequency-dependent offset of -49 / +44 /
-2 deg for the three tones, and a "D ~ 260 ns" that was really ~187 ns + 73 ns.

ONE program, ONE continuous capture (rob_depth 65536 >= 61440 batches): the readout channel's
single slot fires tone B -> an amp-0 gap filler -> pulse 2 back-to-back (readoutPulse.valid stays
high, rbAddr never resets), with in-kernel slot retunes between the plays. The npz (schema 2:
FULL words, freq_width, mode, source, time_to_pulse, TARGET phases) feeds
Tests/experiment/compare_ms_waveform.py.

Co-sim (in the toolchain container):
  cd /work/RISC-Q/software && PYTHONPATH=. /opt/venv312/bin/python \
      /work/exp/ms_waveform_4x2.py --cosim --out /work/exp/ms_capture_cosim.npz
Board (through the deployed riscq package + PynqDriver): --board (M4 bring-up first).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from riscq import run as rq
from riscq.lang import ParamTable, compile_kernel, kernel
from riscq.map import ADC_BATCH, BATCH_SIZE, SocMap, SocParams
from riscq.pulses import Pulse, envelopes, units
from riscq.pulses.golden import TIME_TO_PULSE      # carrier time reference vs envelope batch (36)

# ── the quantised sequence (16-bit words; batch = 16 DAC samples at dspClk 491.52 MHz) ──
F_DAC = 7.86432e9
CODE_A, CODE_B, CODE_V = 698, 669, 683          # 83.76 / 80.28 / 81.96 MHz
DUR1, GAP, DUR2 = 49152, 2458, 9830             # batches: 100 us / ~5 us / 20 us
PH_A, PH_B = 0, 32768                           # 0 deg, -180 deg at the aligned t1
PH_V = 16384                                    # +90 deg at pulse 2's OWN start (target)
AMP = 0.4                                       # per tone; two tones sum on DAC_A
# --exact (M7b, freq_width 32): the generator's TRUE tones instead of the 16-bit codes above.
# compare_ms_waveform.py re-derives these from Tests/waveform_generator.py's own defaults.
F_V_HZ, F_OMEGA_HZ = 82.0e6, 1.765e6
# --edge reserves ONE envelope line for the partial (leading-zero) batch. The reader wraps at
# env_depth and tone B is DUR1 = 3 x env_depth batches long, so a single 49152-batch play would
# traverse EVERY line three times and punch three holes into pulse 1 (Codex F1). Tone B is therefore
# played as CHUNKS of <= env_depth - 1 batches that all start at EDGE_LINE + 1 and never wrap onto
# it. The chunk count/length below divide DUR1 exactly and stay under env_depth.
EDGE_LINE = 0             # the reserved line (holds the partial batch); tone B starts at EDGE_LINE+1
B_CHUNKS = 4              # 4 x 12288 = 49152 = DUR1, each chunk < env_depth so it never wraps
GAP_IDEAL_S = 5.0e-6      # the generator's gap; 5 us = 2457.6 batches is NOT on the 2.0345 ns grid, so the
                          # hardware starts pulse 2 (GAP - 2457.6) = +0.4 batch = +814 ps late. --frame
                          # absolute puts that time offset into pulse 2's CARRIER phase (the envelope edge
                          # cannot move; the carrier phase can, in 0.0055 deg steps), so the carrier matches
                          # waveform.npz on the absolute time axis; --frame own keeps +90 deg at the edge.


@kernel
def k_ms_waveform(gate: ParamTable, ro: ParamTable, dur1: int, gap: int, dur2: int,
                  cqa: int, cq: int, cq2: int, gq: int, d2q: int, paq: int, pbq: int, p2q: int,
                  a0q: int, aq: int, ttp: int, ez: int, elq: int, b1q: int, nch: int, chunk: int,
                  chq: int):
    """The FULL sequence in ONE program with ONE CONTINUOUS capture: ch1 fires three back-to-back
    windows — tone B (dur1) -> an amp=0 gap filler (gap batches; the DAC contribution is zero but
    `readoutPulse.valid` stays high, so rbAddr never resets) -> pulse 2 (dur2, carrier v, P2).
    Back-to-back fires are the upstream B0 auto-advance pattern (X90*X90 trains); the single ch1
    slot is retuned in-kernel between the plays (posted-link order keeps each fire's view).
    Setter args are the seated register words (riscq.h contract: data[31:16]).
    PHASES ARE COMPENSATED IN-KERNEL, REFERENCED TO THE ENVELOPE START: the carrier is
    absolute-time, phase16(n) = ((W * n) mod 2^fw) >> (fw - 16) with n = 16 * tau, and the
    envelope batch played at time t carries the carrier of tau = t - TIME_TO_PULSE (golden.py:
    36 batches = 73.2 ns; measured identically in co-sim and on the board). So the register for a
    pulse whose ENVELOPE starts at batch t is `target_seated - W * ((t - ttp) * 16)` (int32 wrap
    via -fwrapv; NOT a shift, whose overflow stays undefined; residual <= 1 LSB16 = 0.0055 deg).
    Historic runs (and the dev line's "D ~ 260 ns") referenced the phases to the absolute-time
    frame instead, i.e. 36 batches AFTER the envelope edge - a frequency-dependent phase offset
    (f * 73.2 ns mod 1 turn: -48.6 / +44.4 / -2.1 deg for the three tones) that this fixes."""
    init_pulse_params(gate.pulses)  # noqa: F821
    init_pulse_params(ro.pulses)  # noqa: F821
    set_freq(gate, cqa)  # noqa: F821
    set_freq(ro, cq)  # noqa: F821
    t1 = now() + 8192  # noqa: F821
    t1 = (t1 & -4096) + 4096              # align (legacy words: F*16*t1 == 0 mod 2^16); no shift UB
    set_phase(gate, gate["a"], paq - cqa * ((t1 - ttp) * 16))  # noqa: F821
    set_phase(ro, ro["b"], pbq - cq * ((t1 - ttp) * 16))  # noqa: F821
    play(gate, gate["a"], t1)  # noqa: F821
    # window 1: tone B as `nch` back-to-back chunks of `chunk` batches, all reading from envelope
    # line b1q so the reserved partial line is never traversed (the carrier is absolute-time, so
    # chunking is seamless: the phase register is one constant offset for the whole train).
    set_env(ro, ro["b"], b1q)  # noqa: F821
    set_dur(ro, ro["b"], chq)  # noqa: F821
    for i in range(nch):  # noqa: F821
        play(ro, ro["b"], t1 + i * chunk)  # noqa: F821
    set_dur(ro, ro["b"], gq)  # noqa: F821
    set_amp(ro, ro["b"], a0q)  # noqa: F821
    play(ro, ro["b"], t1 + dur1)  # noqa: F821   window 2: amp-0 gap filler (true silence on DAC)
    set_freq(ro, cq2)  # noqa: F821              carrier is absolute-time: switch timing is free
    set_env(ro, ro["b"], elq)  # noqa: F821      pulse 2's envelope line (--edge: the partial one)
    set_dur(ro, ro["b"], d2q)  # noqa: F821
    set_amp(ro, ro["b"], aq)  # noqa: F821
    set_phase(ro, ro["b"], p2q - cq2 * ((t1 + dur1 + gap - ttp) * 16 + ez))  # noqa: F821
    play(ro, ro["b"], t1 + dur1 + gap)  # noqa: F821   window 3: pulse 2
    wait_until(t1 + dur1 + gap + dur2 + 64)  # noqa: F821


def _i32(v):
    """Seated words as int32 (kernel params are int32; 32768 << 16 is -2^31, not 2^31)."""
    return (int(v) + (1 << 31)) % (1 << 32) - (1 << 31)


def _const_pulse(m, ch):
    # compile-time 1-line placeholder; the real constant envelope is tiled over the whole env RAM
    # after setup() (the free-running reader wraps harmlessly) and dur/phase go in via write_slot.
    return Pulse(envelopes.square(m.channel(ch).samples_per_line), amp=AMP)


def _liveness_gate(drv, m):
    """Both liveness counters must advance before any dsp-domain access (three PS wedges on this
    bench; hostCtrl +0x100 dspAlive / +0x104 hostAlive are host-clock, safe to read always)."""
    import time
    h0, d0 = drv.read32(m.host_ctrl + 0x104), drv.read32(m.host_ctrl + 0x100)
    time.sleep(0.01)
    h1, d1 = drv.read32(m.host_ctrl + 0x104), drv.read32(m.host_ctrl + 0x100)
    if h1 == h0 or d1 == d0:
        raise RuntimeError(f"liveness gate FAILED (hostAlive {h0:#x}->{h1:#x}, "
                           f"dspAlive {d0:#x}->{d1:#x}) — not touching the dsp domain")


def _fill_constant_env(drv, m, core, channel):
    line = _const_pulse(m, channel).packed_lines(m, channel)   # (1, words)
    rq.write_envelope(drv, m, core, channel, 0, np.tile(line, (m.params.env_depth, 1)))


def _run(drv, m, out_path, exact=False, frame="own", edge=False):
    fq = 16 * m.params.dsp_freq_hz / 65536      # Hz per code
    if exact:
        if m.params.freq_width != 32:
            raise SystemExit(f"--exact needs a freq_width 32 build (this one: {m.params.freq_width})")
        f_a, f_b, f_v = F_V_HZ + F_OMEGA_HZ, F_V_HZ - F_OMEGA_HZ, F_V_HZ
    else:
        f_a, f_b, f_v = CODE_A * fq, CODE_B * fq, CODE_V * fq
    w_a, w_b, w_v = (units.freq_word(f, m.params) for f in (f_a, f_b, f_v))
    gate = ParamTable(0, f_a, {"a": _const_pulse(m, 0)})
    ro = ParamTable(1, f_b, {"b": _const_pulse(m, 1)})
    assert gate.freq_code(m) == w_a and ro.freq_code(m) == w_b, "table word != kernel word"
    for f, w in ((f_a, w_a), (f_b, w_b), (f_v, w_v)):
        print(f"  tone {f/1e6:.6f} MHz -> word {w:#011x} = {units.word_to_freq(w, m.params)/1e6:.6f} MHz",
              flush=True)
    # ── where pulse 2's envelope actually opens, in DAC samples after pulse 1's scheduled start ──
    # The generator wants gap_ideal_dac = 39321.6 DAC samples: neither a batch nor a DAC sample.
    # Default: open on the next batch boundary (2458 batches = +6.4 DAC samples late).
    # --edge: open one batch EARLY and zero the leading `edge_zeros` DAC samples of that envelope
    # line, so the first non-zero sample lands on the DAC-sample grid (+0.4 sample = +51 ps).
    spl = m.channel(1).samples_per_line
    step = 16 // spl                                          # DAC samples per stored envelope sample
    gap_ideal_dac = GAP_IDEAL_S * m.params.dsp_freq_hz * 16    # 39321.6
    gap_b, dur2_b, edge_zeros, edge_line = GAP, DUR2, 0, 0
    b1_line = 0                                               # tone B's envelope line
    if edge:
        into = gap_ideal_dac - (GAP - 1) * 16                 # 9.6 DAC samples into batch GAP-1
        edge_zeros = int(round(into / step) * step)           # the reachable start on THIS grid
        if edge_zeros % step or not 0 < edge_zeros < 16:
            raise SystemExit(f"--edge: {into:.1f} DAC samples is not reachable with "
                             f"{spl} samples/line (step {step})")
        gap_b, dur2_b = GAP - 1, DUR2 + 1                     # start a batch early, same end batch
        edge_line, b1_line = EDGE_LINE, EDGE_LINE + 1
    nch, b_chunk = (B_CHUNKS, DUR1 // B_CHUNKS) if edge else (1, DUR1)
    if edge and (b_chunk >= m.params.env_depth or b_chunk * nch != DUR1):
        raise SystemExit(f"--edge: {nch} x {b_chunk} != {DUR1} or the chunk wraps env_depth "
                         f"{m.params.env_depth}")
    start_dac = gap_b * 16 + edge_zeros
    late_dac = start_dac - gap_ideal_dac                       # +6.4 (default) or +0.4 (--edge)
    if frame == "absolute":
        # carrier phase at the ACTUAL first non-zero sample = target + f*late (W is phase per sample)
        ph_v_seated = (PH_V << 16) + round(w_v * late_dac)
        adv = (round(w_v * late_dac) >> 16) * 360 / 65536
    else:
        ph_v_seated = PH_V << 16                               # +90 deg at pulse 2's own start
        adv = 0.0
    f_dac = 16 * m.params.dsp_freq_hz
    print(f"  pulse 2 opens at DAC sample {start_dac} after pulse 1 (generator wants "
          f"{gap_ideal_dac:.1f}): {late_dac/f_dac*1e12:+.1f} ps late"
          + (f"; --edge: line {edge_line} zeroes its first {edge_zeros} DAC samples "
             f"({edge_zeros//step} of {spl} stored), tone B = {nch} x {b_chunk} batches from line "
             f"{b1_line}, filler {gap_b}, pulse 2 {dur2_b}" if edge else
             "; envelope opens on the batch boundary")
          + (f"; frame=absolute: carrier target advanced {adv:.2f} deg" if frame == "absolute" else ""),
          flush=True)
    total = DUR1 + gap_b + dur2_b                              # unchanged: 49152 + 2458 + 9830
    if m.params.rob_depth < total:
        raise SystemExit(f"rob_depth {m.params.rob_depth} < sequence {total} batches — rebuild "
                         "with rob_depth 65536 for the one-shot continuous capture")
    prog = compile_kernel(k_ms_waveform, m, tables=dict(gate=gate, ro=ro),
                          dur1=DUR1, gap=gap_b, dur2=dur2_b,
                          cqa=w_a, cq=w_b, cq2=w_v,
                          gq=gap_b << 16, d2q=dur2_b << 16,
                          paq=_i32(PH_A << 16), pbq=_i32(PH_B << 16), p2q=_i32(ph_v_seated),
                          a0q=0, aq=(units._amp_code(AMP) & 0xFFFF) << 16,
                          ttp=TIME_TO_PULSE, ez=edge_zeros, elq=edge_line << 16,
                          b1q=b1_line << 16, nch=nch, chunk=b_chunk, chq=_i32(b_chunk << 16))
    rq.setup(drv, m, {0: prog})
    _fill_constant_env(drv, m, 0, 0)
    _fill_constant_env(drv, m, 0, 1)
    if edge:
        # ONE partial line at EDGE_LINE: the leading `edge_zeros` DAC samples silent, the rest the
        # SAME unit envelope the tiling wrote (the amplitude is the slot's amp register, not here).
        env = envelopes.square(spl).copy(); env[: edge_zeros // step] = 0
        rq.write_envelope(drv, m, 0, 1, edge_line, Pulse(env, amp=1.0).packed_lines(m, 1))
        print(f"  edge line {edge_line}: {[f'{abs(v):.2f}' for v in env]} (unit envelope, "
              f"first {edge_zeros} DAC samples zero)", flush=True)
    if not hasattr(drv, "sim"):
        _liveness_gate(drv, m)      # hardware only: never touch a dead dsp domain (wedge guard)

    timeout = (total + 20000) * 4 + 20_000_000
    rq.write_slot(drv, m, 0, prog, "gate", 0, "dur", DUR1)
    rq.write_slot(drv, m, 0, prog, "ro", 0, "dur", DUR1)     # phases are set in-kernel (see kernel)
    rq.rerun(drv, m, {0: prog}, timeout=timeout)
    # chunked trace readout: one giant read_block starves the co-sim bench's RPC service window
    # (and chunking costs nothing on hardware).
    nbytes = 4 * ADC_BATCH * total
    chunk = 128 * 1024
    parts = []
    for off in range(0, nbytes, chunk):
        parts.append(drv.read_block(m.robs() + off, min(chunk, nbytes - off)))
        print(f"  trace read {off // 1024 + len(parts[-1]) // 1024} / {nbytes // 1024} KiB",
              flush=True)
    trace = np.frombuffer(b"".join(parts), dtype="<i4").astype(np.int32)
    print(f"one-shot trace: {total} batches, max|s| = {np.abs(trace).max()}", flush=True)

    np.savez(out_path, trace=trace,
             rob_depth=m.params.rob_depth, adc_batch=ADC_BATCH, batch_size=BATCH_SIZE,
             fs_hz=ADC_BATCH * m.params.dsp_freq_hz, dsp_hz=m.params.dsp_freq_hz, fdac_hz=F_DAC,
             f_words=np.array([w_a, w_b, w_v], dtype=np.int64),   # FULL seated words
             schema=2, mode=("exact" if exact else "quantised"),   # schema 2: FULL words, TARGET phases
             source=("cosim" if hasattr(drv, "sim") else "board"), time_to_pulse=TIME_TO_PULSE,
             frame=frame, gap_ideal_batches=GAP_IDEAL_S * m.params.dsp_freq_hz,
             p2_start_dac=DUR1 * 16 + start_dac, edge_zeros=edge_zeros, samples_per_line=spl,
             b_chunks=nch, b_chunk_batches=b_chunk,
             freq_width=m.params.freq_width,
             ph_words=np.array([PH_A, PH_B, PH_V]),               # TARGET phases at each pulse start
             dur1=DUR1, gap=gap_b, dur2=dur2_b, amp=AMP)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosim", action="store_true")
    ap.add_argument("--board", action="store_true")
    ap.add_argument("--remote", default=None, metavar="HOST",
                    help="run via the riscq board server at HOST (upload bundle + load + run)")
    ap.add_argument("--xsa", default=None, help="--remote: XSA to upload")
    ap.add_argument("--board-json", default=None, help="--remote: board.json to upload")
    ap.add_argument("--config", default=None, help="SocParams JSON (default: rfsoc4x2-1q)")
    ap.add_argument("--build", default=None, help="co-sim build dir")
    ap.add_argument("--delay", type=int, default=5, help="co-sim loopback delay (batches)")
    ap.add_argument("--out", default="ms_capture.npz")
    ap.add_argument("--exact", action="store_true",
                    help="play the generator's TRUE tones (needs freq_width 32) instead of the 16-bit codes")
    ap.add_argument("--edge", action="store_true",
                    help="put pulse 2's envelope edge on the DAC-sample grid: open one batch early "
                         "with the leading part of that envelope line zeroed (needs an envelope "
                         "with >= 2 samples per line, i.e. readout_interp <= 8)")
    ap.add_argument("--frame", choices=("own", "absolute"), default=None,
                    help="pulse-2 phase reference: its own envelope start, or the generator's absolute time "
                         "axis (compensates the batch-grid rounding of the 5 us gap in the carrier phase); "
                         "default: absolute with --exact, own otherwise")
    ap.add_argument("--bundle", default="rfsoc4x2-1q", help="--remote: board bundle name to upload/load")
    a = ap.parse_args()

    here = Path(__file__).resolve().parent
    frame = a.frame or ("absolute" if a.exact else "own")
    if a.cosim:
        from riscq_sim import cosim as server
        cfg = a.config or "/work/RISC-Q/software/configs/rfsoc4x2-1q.json"
        bld = a.build or "/work/RISC-Q/software/build/rfsoc4x2-1q"
        drv = server.start(cfg, bld)
        try:
            m = SocMap(SocParams.from_json(drv.sim.get_params()))
            drv.sim.set_model({"kind": "loopback", "src": 0, "dst": 0,
                               "gain": 0.9, "delay": a.delay})
            _run(drv, m, a.out, exact=a.exact, frame=frame, edge=a.edge)
        finally:
            server.stop(drv)
    elif a.remote:
        import json
        import time
        from riscq.driver.remote import RemoteDriver, upload_bundle
        cfg = a.config or "/work/RISC-Q/software/configs/rfsoc4x2-1q.json"
        drv = RemoteDriver(a.remote)
        board = json.loads(Path(a.board_json).read_text()) if a.board_json else None
        if a.xsa:
            print(f"uploading bundle {a.bundle} ...", flush=True)
            upload_bundle(drv, a.bundle, a.xsa, cfg, board=board)
        drv.board.load(a.bundle)
        m = SocMap(SocParams.from_json(drv.board.get_params()))
        print(f"loaded: {drv.board.info()}  params: env_depth={m.params.env_depth} rob_depth={m.params.rob_depth} freq_width={m.params.freq_width}", flush=True)
        # liveness gate BEFORE any dsp-domain access (three wedges on this bench; the server's
        # setup path is unguarded): both counters must advance or we stop here.
        hc = m.host_ctrl                     # driver addresses are AXI-window-relative
        h0, d0 = drv.read32(hc + 0x104), drv.read32(hc + 0x100)
        time.sleep(0.01)
        h1, d1 = drv.read32(hc + 0x104), drv.read32(hc + 0x100)
        print(f"liveness: hostAlive {h0:#x}->{h1:#x}  dspAlive {d0:#x}->{d1:#x}", flush=True)
        if h1 == h0 or d1 == d0:
            raise SystemExit("liveness gate FAILED — not touching the dsp domain")
        _run(drv, m, a.out, exact=a.exact, frame=frame, edge=a.edge)
    elif a.board:
        from riscq.board.pynq_driver import PynqDriver
        import json
        cfg = a.config or str(here / "rfsoc4x2-1q.json")
        board = json.loads((here / "board-rfsoc4x2.json").read_text())
        drv = PynqDriver(str(here / "PulseTableSoc.xsa"), cfg, board=board)
        m = SocMap(SocParams.from_json(Path(cfg).read_text()))
        _run(drv, m, a.out, exact=a.exact, frame=frame, edge=a.edge)
    else:
        ap.error("pick --cosim or --board")


if __name__ == "__main__":
    sys.exit(main())
