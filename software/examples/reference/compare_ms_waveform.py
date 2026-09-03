"""Verify the ONE-SHOT robs capture of the two-pulse ion-trap waveform against its intent —
the full pulse1 + gap + pulse2 sequence in one continuous trace (rob_depth 65536 >= 61440
batches; capture starts at the pulse-1 fire = trace sample 0, the analog path shifts the
signal by one delay D within the window).

    python compare_ms_waveform.py ms_capture.npz

Checks (dev-line bar: D ~ 260 ns, residuals < 0.5 deg, corr 1.0000; acceptance per plan M6.17):
  1. envelope boundaries: 100 us two-tone, ~5 us TRUE-SILENCE gap, 20 us single tone
  2. no clipping, RMS sane, all three quantised tones present, no unexpected peaks
  3. amplitude ratios ~ 1
  4. ONE analog delay D jointly explaining the three tone phases (residuals < 1 deg)
  5. sample-by-sample correlation vs the D-rebuilt reference (> 0.99 per pulse)
  6. DIRECT comparison with Tests/waveform_generator.py's own waveform, rebuilt on the capture
     grid by the generator's code and delayed by D: correlation and NRMSE per pulse. This is the
     "is it the SAME pulse as Tests/waveform.npz" check; it gates the verdict only when the words
     are the generator's exact tones (a 16-bit-quantised run is 5/45/40 kHz off and must FAIL it).
"""
import sys

import numpy as np


def envelope(x, fs, win_us=0.5):
    n = int(win_us * 1e-6 * fs)
    return np.sqrt(np.convolve(x.astype(float) ** 2, np.ones(n) / n, mode="same"))


def peak_freq(x, fs, fmin=60e6, fmax=100e6, n=2):
    X = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    f = np.fft.rfftfreq(x.size, 1 / fs)
    X = X * ((f > fmin) & (f < fmax))
    idx = []
    for _ in range(n):
        i = int(np.argmax(X)); idx.append(i)
        X[max(0, i - 40): i + 40] = 0
    return sorted(f[i] for i in idx)


def _generator_intent():
    """The intended physics, read from `Tests/waveform_generator.py`'s OWN defaults (its function
    signature), not from the capture's metadata — so a wrong stimulus cannot confirm itself."""
    import ast
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "waveform_generator.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "generate_two_pulse_waveform")
    args = [a.arg for a in fn.args.args]
    defaults = dict(zip(args[-len(fn.args.defaults):],
                        [ast.literal_eval(d) for d in fn.args.defaults]))
    return {"v": defaults["v_MHz"] * 1e6, "omega": defaults["omega_MHz"] * 1e6,
            "dur1_s": defaults["first_duration_us"] * 1e-6,
            "dur2_s": defaults["second_duration_us"] * 1e-6,
            "gap_s": defaults["inter_pulse_gap_us"] * 1e-6,
            "phase2_deg": defaults["second_phase_deg"]}


def main(path, force_frame=None):
    d = np.load(path)
    fs = float(d["fs_hz"]); lanes = int(d["adc_batch"])
    fdac = float(d["fdac_hz"])
    # npz schema: legacy files (no "schema") hold bare 16-bit codes and the PROGRAMMED PH_V;
    # schema 2 holds the FULL seated words (data[31:16] at fw 16), `freq_width`, `mode`
    # ("exact" | "quantised") and TARGET phases (0 / -180 / +90 at each pulse's own start).
    schema = int(d["schema"]) if "schema" in d.files else 1
    fw = int(d["freq_width"]) if schema >= 2 else 16
    mode = str(d["mode"]) if schema >= 2 else "quantised"
    source = str(d["source"]) if "source" in d.files else "board"
    ttp = int(d["time_to_pulse"]) if "time_to_pulse" in d.files else None
    frame = str(d["frame"]) if "frame" in d.files else "own"
    edge_zeros = int(d["edge_zeros"]) if "edge_zeros" in d.files else 0
    if force_frame is not None and force_frame != frame:
        print(f"*** NEGATIVE CONTROL: capture was made in the {frame!r} frame, scoring it against the "
              f"{force_frame!r} reference -- it MUST fail the pulse-2 carrier gate ***")
        frame = force_frame
    print(f"capture: source={source}, schema={schema}, freq_width={fw}, mode={mode}, "
          f"time_to_pulse={ttp} (kernel phase reference: envelope batch t carries the carrier of t - ttp)")
    if schema >= 2:
        # Sign-off artifacts must have been produced with the envelope-referenced kernel (golden.py
        # TIME_TO_PULSE = 36); older schema-2 files predate it and are only valid as negative controls.
        assert ttp == 36, f"time_to_pulse={ttp}: not the envelope-referenced kernel (expected 36)"
    def plain(w):
        w = int(w)
        if schema < 2:
            return w
        return (w >> 16) & 0xFFFF if fw == 16 else w & 0xFFFFFFFF
    words = [plain(w) for w in d["f_words"]]
    fA, fB, fV = (w * fdac / (1 << fw) for w in words)
    phA0 = int(d["ph_words"][0]) / 65536 * 2 * np.pi
    phB0 = int(d["ph_words"][1]) / 65536 * 2 * np.pi - 2 * np.pi   # -180 stored as +32768
    phV_nom = np.pi / 2          # +90 deg at pulse 2's own envelope start ...
    dur1, gap, dur2 = int(d["dur1"]), int(d["gap"]), int(d["dur2"])
    x = d["trace"].astype(np.int64)
    L1, Lg, L2 = dur1 * lanes, gap * lanes, dur2 * lanes
    # The HARDWARE's pulse-2 start, in (possibly FRACTIONAL) ADC samples: with --edge the envelope
    # opens one batch early with its leading `edge_zeros` DAC samples silent, and ADC_BATCH = 4
    # samples per batch, so 10 DAC samples = 2.5 ADC samples. Legacy captures opened on the batch
    # boundary. `n2_i` is the integer form used for slicing/searching (Codex F2).
    n2 = float(d["p2_start_dac"]) * lanes / 16.0 if "p2_start_dac" in d.files else float(L1 + Lg)
    n2_i = int(round(n2))
    # Derived measurements must follow the ACTUAL edge, not the batch bookkeeping (Codex F3):
    Lg = n2 - L1                                     # the true silent gap, in ADC samples
    L2 = x.size - n2                                 # the true non-zero pulse-2 length
    # ... unless the capture was made with --frame absolute: then the carrier is referenced to the
    # generator's IDEAL start (dur1 + gap_ideal), which the batch grid cannot hit -- the hardware
    # starts (gap - gap_ideal) batches late and the kernel advanced the carrier phase by f * that,
    # so the carrier matches waveform.npz on the absolute time axis while the envelope edge does not.
    # The ideal gap comes from the GENERATOR's own defaults (intent), not from the stimulus metadata:
    # a wrong GAP_IDEAL_S in the stimulus would otherwise move BOTH the compensation and the reference
    # and still pass (Codex F1). The capture's own value is only cross-checked against it.
    _int0 = _generator_intent(); _dsp0 = float(d["dsp_hz"])
    gap_ideal = _int0["gap_s"] * _dsp0                                      # batches, 2457.6
    if "gap_ideal_batches" in d.files:
        assert abs(float(d["gap_ideal_batches"]) - gap_ideal) < 1e-6, (
            f"capture says gap_ideal {float(d['gap_ideal_batches'])} batches, the generator's "
            f"{_int0['gap_s']*1e6:.3f} us is {gap_ideal} batches")
    n2_ref = n2 if frame != "absolute" else (dur1 + gap_ideal) * lanes      # fractional sample index
    late_s = (n2 - n2_ref) / fs
    if frame == "absolute":
        phV_nom += 2 * np.pi * fV * late_s          # expected carrier phase AT THE HARDWARE start
    if edge_zeros:
        print(f"--edge capture: envelope opened one batch early with its first {edge_zeros} DAC "
              f"samples zeroed ({int(d['samples_per_line'])} stored samples/line); tone B was played "
              f"as {int(d['b_chunks'])} x {int(d['b_chunk_batches'])} batches so the reserved line "
              f"is never traversed")
    print(f"pulse-2 phase frame: {frame}; hardware start {n2/fs*1e6:.4f} us, reference start "
          f"{n2_ref/fs*1e6:.4f} us (hardware late by {late_s*1e12:+.0f} ps = {360*fV*late_s:+.2f} deg "
          f"at {fV/1e6:.3f} MHz)")
    end_hw = dur1 + gap + dur2                                   # batches
    end_ideal = dur1 + gap_ideal + _int0["dur2_s"] * _dsp0
    print(f"pulse-2 end: hardware {end_hw} batches vs ideal {end_ideal:.1f} -> "
          f"{(end_hw - end_ideal)/_dsp0*1e12:+.0f} ps (the start and duration roundings cancel)")
    print(f"one-shot trace: {x.size} samples ({x.size/fs*1e6:.1f} us) @ {fs/1e6:.2f} MS/s; "
          f"tones A {fA/1e6:.4f} B {fB/1e6:.4f} V {fV/1e6:.4f} MHz")

    # ── INTENT CHECK (Codex round-3 finding 6): the metadata below was written by the stimulus
    # script, so verifying the capture against it alone could self-confirm a wrong intent. Check
    # it INDEPENDENTLY against Tests/waveform_generator.py's own defaults — the physics the
    # experiment is supposed to reproduce — and against the exact expected trace length. ──
    intent = _generator_intent()
    dsp = float(d["dsp_hz"])
    exact = True
    for name, want_hz, word in (("v+omega", intent["v"] + intent["omega"], words[0]),
                                ("v-omega", intent["v"] - intent["omega"], words[1]),
                                ("v",       intent["v"],                   words[2])):
        if mode == "exact":                       # the wide word nearest the intent
            want_word = round(want_hz * (1 << fw) / (16 * dsp))
        else:                                     # the 16-bit code grid, seated in this width
            want_word = round(want_hz * 65536 / (16 * dsp)) << (fw - 16)
        assert word == want_word, (f"{name}: capture metadata says word {word}, the generator's "
                                   f"{want_hz/1e6:.4f} MHz -> {want_word} ({mode}, {fw} bits)")
        got_hz = word * fdac / (1 << fw)
        exact &= abs(got_hz - want_hz) < 2.0                           # within 2 Hz of the intent
        print(f"  {name:8s} requested {want_hz/1e6:.6f} MHz, realised {got_hz/1e6:.6f} MHz "
              f"(realised - requested {got_hz - want_hz:+.1f} Hz)")
    assert (mode == "exact") == exact, f"npz says mode={mode!r} but the words are {'exact' if exact else 'quantised'}"
    for name, want_s, batches in (("pulse 1", intent["dur1_s"], dur1),
                                  ("gap",     intent["gap_s"],  gap),
                                  ("pulse 2", intent["dur2_s"], dur2)):
        got_s = batches * 16 / (16 * dsp)
        assert abs(got_s - want_s) < 0.5e-6, f"{name}: {got_s*1e6:.2f} us != {want_s*1e6:.2f} us"
    assert int(d["ph_words"][0]) == 0, "tone A must be programmed at 0 deg"
    assert int(d["ph_words"][1]) == 32768, "tone B must be programmed at -180 deg"
    assert x.size == (dur1 + gap + dur2) * lanes, \
        f"trace is {x.size} samples, the sequence is {(dur1 + gap + dur2) * lanes}"
    print(f"intent check vs waveform_generator.py defaults: tones "
          f"{(intent['v'] + intent['omega'])/1e6:.3f}/{(intent['v'] - intent['omega'])/1e6:.3f}/"
          f"{intent['v']/1e6:.3f} MHz, {intent['dur1_s']*1e6:.0f}+{intent['gap_s']*1e6:.0f}+"
          f"{intent['dur2_s']*1e6:.0f} us — OK (schema {schema}, {fw}-bit words, "
          f"{'EXACT tones' if exact else 'quantised to the 16-bit code grid'})")

    ok = True
    m = int(1e-6 * fs)                               # 1 us guard at segment edges

    # ── 1. boundaries from the envelope: two pulses, the right lengths, a true-silence gap ──
    env = envelope(x.astype(float), fs)
    thr = 0.3 * np.percentile(env, 99)
    on = env > thr
    dch = np.diff(on.astype(int))
    rises = list(np.flatnonzero(dch == 1)); falls = list(np.flatnonzero(dch == -1))
    if on[0]: rises = [0] + rises
    if on[-1]: falls = falls + [on.size - 1]
    segs = [(a, b) for a, b in zip(rises, falls) if (b - a) / fs > 2e-6]
    print("detected pulses:", [f"{a/fs*1e6:.2f}..{b/fs*1e6:.2f} us ({(b-a)/fs*1e6:.2f} us)"
                               for a, b in segs])
    if len(segs) != 2:
        print("MS_COMPARE: FAIL (expected exactly 2 envelope segments)"); return 1
    (a1, b1), (a2, b2) = segs
    ok &= abs((b1 - a1) / fs - dur1 * lanes / fs) < 0.5e-6
    ok &= abs((b2 - a2) / fs - dur2 * lanes / fs) < 0.5e-6
    ok &= abs((a2 - b1) / fs - gap * lanes / fs) < 0.5e-6
    print(f"boundaries within 0.5 us: {ok} (gap measured {(a2-b1)/fs*1e6:.3f} us, "
          f"nominal {gap*lanes/fs*1e6:.3f} us)")
    gap_rms = float(np.sqrt(np.mean(x[b1 + m: a2 - m].astype(float) ** 2))) if a2 - b1 > 2 * m else 0.0
    p1_rms = float(np.sqrt(np.mean(x[a1 + m: b1 - m].astype(float) ** 2)))
    print(f"gap rms {gap_rms:.0f} vs pulse-1 rms {p1_rms:.0f} (true silence: < 5 %)")
    ok &= gap_rms < 0.05 * p1_rms

    # ── 2. clipping / spectra / stray peaks ──
    seg1 = x[a1 + m: b1 - m].astype(float)
    seg2 = x[a2 + m: b2 - m].astype(float)
    ok &= np.abs(x).max() < 32700 and p1_rms > 500
    pk1 = peak_freq(seg1, fs, n=2); pk2 = peak_freq(seg2, fs, n=1)
    print(f"pulse-1 peaks {[round(f/1e6,4) for f in pk1]} MHz (expect {fB/1e6:.4f}, {fA/1e6:.4f}); "
          f"pulse-2 peak {pk2[0]/1e6:.4f} MHz (expect {fV/1e6:.4f})")
    ok &= abs(pk1[0] - fB) < 3 * fs / seg1.size and abs(pk1[1] - fA) < 3 * fs / seg1.size
    ok &= abs(pk2[0] - fV) < 3 * fs / seg2.size

    def no_stray(xseg, expected):
        X = np.abs(np.fft.rfft(xseg * np.hanning(xseg.size)))
        f = np.fft.rfftfreq(xseg.size, 1 / fs)
        inband = np.zeros_like(f, dtype=bool)
        for fe in expected:
            inband |= np.abs(f - fe) < 1e6
        top = max(X[np.abs(f - fe) < 1e6].max() for fe in expected)
        stray = X[(~inband) & (f > 1e6)].max()
        return stray < 0.1 * top, stray / top

    s1ok, r1 = no_stray(seg1, [fA, fB]); s2ok, r2 = no_stray(seg2, [fV])
    print(f"stray peaks: pulse 1 {r1:.3f} of top tone, pulse 2 {r2:.3f} (limit 0.1)")
    ok &= s1ok and s2ok

    # ── 3+4. amplitudes and the joint single-delay fit (phases referenced to the NOMINAL
    #        starts: pulse 1 at sample 0, pulse 2 at sample n2; the analog delay D shifts all) ──
    def tone_fit(xseg, n_abs0, f):
        n = n_abs0 + np.arange(xseg.size)
        z = 2 * np.mean(xseg * np.exp(-2j * np.pi * f * n / fs))
        return abs(z), np.angle(z)

    A_a, ph_a = tone_fit(seg1, a1 + m, fA)
    A_b, ph_b = tone_fit(seg1, a1 + m, fB)
    A_v, ph_v = tone_fit(seg2, a2 + m - n2, fV)      # pulse-2 phase in its own start frame
    print(f"amplitudes A {A_a:.0f} B {A_b:.0f} V {A_v:.0f}  "
          f"(B/A {A_b/A_a:.3f}, V/A {A_v/A_a:.3f}; expected ~1)")
    ok &= abs(A_b / A_a - 1) < 0.1 and abs(A_v / A_a - 1) < 0.15

    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    Dg = np.arange(-0.2e-6, 3.0e-6, 0.02e-9)
    rA = wrap(ph_a - (phA0 - 2 * np.pi * fA * Dg))
    rB = wrap(ph_b - (phB0 - 2 * np.pi * fB * Dg))
    rV = wrap(ph_v - (phV_nom - 2 * np.pi * fV * Dg))
    k = int(np.argmin(rA ** 2 + rB ** 2 + rV ** 2))
    D = Dg[k]; res = np.degrees([rA[k], rB[k], rV[k]])
    print(f"joint delay D = {D*1e9:.2f} ns; residuals A {res[0]:+.2f} B {res[1]:+.2f} "
          f"V {res[2]:+.2f} deg")
    ok &= float(np.max(np.abs(res))) < 1.0

    # ── 4b. the ENVELOPE start vs the carrier time reference: D from the tone phases must equal
    #        the delay of the envelope edge (pulse 2 starts sharply: a single tone at +90 deg).
    #        The hardware plays the carrier of tau = t - TIME_TO_PULSE (36 batches, 73.2 ns) on the
    #        envelope batch t; the kernel compensates for it, so a mismatch here (73 ns) means the
    #        phases are referenced to the wrong instant. The hardware reference can only be off by
    #        WHOLE batches (2.03 ns = 60 deg at 82 MHz), so agreement within 1.0 ns certifies the
    #        time reference even with the ~0.85 ns late bias of the analog edge. The threshold onset
    #        of a pulse that starts at a zero crossing (+90 deg) is biased LATE by up to one ADC
    #        sample (0.51 ns) plus the 0.1-threshold rise (~0.19 ns): ~0.70 ns = ~21 deg at 82 MHz,
    #        so the phases at the detected onset are reported with that caveat, not gated.
    amp2 = float(np.abs(seg2).max())
    look = x[max(0, n2_i - 400): n2_i + 4000].astype(float)
    onset = max(0, n2_i - 400) + int(np.flatnonzero(np.abs(look) > 0.1 * amp2)[0])
    D_env = (onset - n2) / fs
    print(f"envelope onset (pulse 2, threshold 0.1) at {D_env*1e9:.2f} ns vs tone-phase D {D*1e9:.2f} ns "
          f"-> difference {(D - D_env)*1e9:+.2f} ns (a 36-batch convention error would show as +73.2)")
    ok &= abs(D - D_env) < 1.0e-9          # a one-batch error (2.03 ns) minus the <= 0.85 ns edge bias still fails
    edge = onset - n2                     # the same pipeline+analog delay applies to pulse 1
    def phase_at_onset(xseg, first_abs, f, start_abs):
        n = first_abs - start_abs + np.arange(xseg.size)          # samples since the ONSET sample
        return np.degrees(np.angle(2 * np.mean(xseg * np.exp(-2j * np.pi * f * n / fs))))
    on_ph = [phase_at_onset(seg1, a1 + m, fA, edge), phase_at_onset(seg1, a1 + m, fB, edge),
             phase_at_onset(seg2, a2 + m, fV, n2 + edge)]
    tgt = [0.0, -180.0, float(np.degrees(phV_nom))]
    err = [((p - t + 180) % 360) - 180 for p, t in zip(on_ph, tgt)]
    print(f"phases AT THE ENVELOPE ONSET: A {on_ph[0]:+.2f} B {on_ph[1]:+.2f} V {on_ph[2]:+.2f} deg "
          f"(targets 0 / -180 / +90; errors {err[0]:+.2f} / {err[1]:+.2f} / {err[2]:+.2f} deg; "
          f"informational: the detected onset is late by <= ~0.70 ns = ~21 deg (1 sample + 0.1 threshold)"
          f"{'' if source == 'cosim' else ', plus the analog edge on the board'})")
    # the envelope edges must sit at nominal + D (coarse, +-0.4 us)
    ok &= abs(a1 / fs - D) < 0.4e-6 and abs((a2 - n2) / fs - D) < 0.4e-6

    # ── 5. correlation vs the D-rebuilt reference ──
    def corr_ref(xseg, n_abs0, tones):
        n = (n_abs0 + np.arange(xseg.size)) / fs
        ref = sum(np.cos(2 * np.pi * f * (n - D) + ph) for f, ph in tones)
        return float(np.dot(xseg, ref) / (np.linalg.norm(xseg) * np.linalg.norm(ref) + 1e-12))

    c1 = corr_ref(seg1, a1 + m, [(fA, phA0), (fB, phB0)])
    c2 = corr_ref(seg2, a2 + m - n2, [(fV, phV_nom)])
    print(f"correlation vs reference: pulse 1 {c1:.4f}, pulse 2 {c2:.4f}")
    ok &= c1 > 0.99 and c2 > 0.99

    # ── 6. DIRECT comparison with the generator's own waveform (Tests/waveform_generator.py) ──
    g_ok = direct_generator_check(x.astype(float), fs, D, intent, (a1, b1), (a2, b2),
                                  {"n2": n2, "n2_ref": n2_ref, "gap_samples": Lg, "dur2_samples": L2,
                                   "frame": frame, "fV": fV})
    if exact:
        ok &= g_ok
    else:
        print("  (quantised run: the direct-generator check is informational, expected to fail)")

    print("MS_COMPARE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def direct_generator_check(x, fs, D, intent, seg1, seg2, m):
    """Rebuild the intended waveform ON THE CAPTURE GRID with the generator's own `_make_pulse`
    (the code that wrote Tests/waveform.npz — asserted bit-exact at its 1000 MS/s), delay it by
    the fitted analog delay D (fractional-sample FFT phase ramp on a zero-padded copy), fit ONE
    amplitude scale, and score each pulse (1 us edge guards) by correlation and NRMSE."""
    import importlib.util
    import pathlib
    import types
    try:                                   # the generator imports pyplot at module level; only
        import matplotlib.pyplot  # noqa: F401   its plotting path needs it, which we never call
    except ImportError:
        sys.modules["matplotlib"] = types.ModuleType("matplotlib")
        sys.modules["matplotlib.pyplot"] = types.ModuleType("matplotlib.pyplot")
    src = pathlib.Path(__file__).resolve().parents[1] / "waveform_generator.py"
    spec = importlib.util.spec_from_file_location("waveform_generator", src)
    g = importlib.util.module_from_spec(spec)
    sys.modules["waveform_generator"] = g      # dataclasses need the module registered
    spec.loader.exec_module(g)

    def pulses(rate_mhz):
        first = g._make_pulse((g.PulseSpec((intent["v"] + intent["omega"]) / 1e6, 1.0, 0.0),
                               g.PulseSpec((intent["v"] - intent["omega"]) / 1e6, 1.0, -180.0)),
                              intent["dur1_s"] * 1e6, rate_mhz, "real")
        second = g._make_pulse((g.PulseSpec(intent["v"] / 1e6, 1.0, intent["phase2_deg"]),),
                               intent["dur2_s"] * 1e6, rate_mhz, "real")
        return first, second

    def build(rate_mhz):
        first, second = pulses(rate_mhz)
        gap = np.zeros(g._sample_count(intent["gap_s"] * 1e6, rate_mhz))
        return np.concatenate((first, gap, second))

    npz = src.parent / "waveform.npz"
    if npz.exists():
        w = np.load(npz)["waveform"]
        ref1000 = build(1000.0)
        dmax = float(np.abs(ref1000 - w).max()) if ref1000.size == w.size else np.inf
        assert dmax < 1e-9, f"generator helpers no longer reproduce Tests/waveform.npz (max diff {dmax})"
        print(f"generator self-check: _make_pulse reproduces Tests/waveform.npz (max |diff| {dmax:.1e}; "
              "0.0 on the machine that wrote it, ULP-level elsewhere)")
    # Each pulse in ITS OWN start frame (generator t=0 at the pulse start), placed at the
    # hardware's nominal starts (pulse 1: sample 0; pulse 2: sample n2) and delayed by ONE shared
    # D. The sequence-level sample counts differ by the generator's rounding (gap +0.4 sample,
    # pulse 2 -0.4 sample at this rate) — reported, not hidden; no sequence-wide sample equality
    # is claimed.
    n = x.size
    first, second = pulses(fs / 1e6)
    gen_gap = g._sample_count(intent["gap_s"] * 1e6, fs / 1e6)
    hw_gap = n2_hw = None
    (a1, b1), (a2, b2) = seg1, seg2
    n2_hw = m["n2"]; hw_gap = m["gap_samples"]; hw_dur2 = m["dur2_samples"]
    print(f"grid quantisation at {fs/1e6:.2f} MS/s: gap generator {gen_gap} vs hardware {hw_gap} samples "
          f"({(hw_gap - gen_gap)/fs*1e12:+.0f} ps), pulse 2 generator {second.size} vs hardware "
          f"{hw_dur2} samples ({(hw_dur2 - second.size)/fs*1e12:+.0f} ps)")

    def delayed(sig, start, D):
        # start may be FRACTIONAL (the generator's ideal pulse-2 start is not on the ADC grid): the
        # integer part places the pulse, the fraction rides in the FFT phase ramp together with D
        s0 = int(np.floor(start)); D_eff = D + (start - s0) / fs
        pad = int(2 ** np.ceil(np.log2(sig.size + 8192)))
        f = np.fft.rfftfreq(pad, 1 / fs)
        y = np.fft.irfft(np.fft.rfft(sig, pad) * np.exp(-2j * np.pi * f * D_eff), pad)
        out = np.zeros(n)
        k = min(y.size, n - s0)
        out[s0: s0 + k] = y[:k]
        return out

    n2_ref = m["n2_ref"]
    print(f"generator pulse 2 placed at its "
          f"{'IDEAL start (absolute frame)' if m['frame'] == 'absolute' else 'hardware start (own frame)'}, "
          f"sample {n2_ref:.1f}; the hardware envelope opens at sample {m['n2']} "
          f"({(m['n2'] - n2_ref)/fs*1e12:+.0f} ps)")
    ref_d = delayed(first, 0, D) + delayed(second, n2_ref, D)
    g1 = int(1e-6 * fs)
    idx = np.r_[a1 + g1: b1 - g1, a2 + g1: b2 - g1]
    scale = float(np.dot(x[idx], ref_d[idx]) / np.dot(ref_d[idx], ref_d[idx]))
    out = scale > 0                      # ONE positive amplitude scale over both guarded pulses
    for name, sl in (("pulse 1", slice(a1 + g1, b1 - g1)), ("pulse 2", slice(a2 + g1, b2 - g1))):
        xs, rs = x[sl], scale * ref_d[sl]
        corr = float(np.dot(xs, rs) / (np.linalg.norm(xs) * np.linalg.norm(rs) + 1e-12))
        nrmse = float(np.sqrt(np.mean((xs - rs) ** 2)) / np.sqrt(np.mean(xs ** 2)))
        print(f"vs Tests/waveform_generator.py ({name}, D = {D*1e9:.2f} ns, scale {scale:.1f}): "
              f"correlation {corr:.4f}, NRMSE {nrmse*100:.2f} %")
        out &= corr > 0.999 and nrmse < 0.05
    # pulse-2 CARRIER phase against the generator on the REFERENCE time axis. In the absolute frame
    # this is the check that the batch-grid rounding of the 5 us gap was compensated in the carrier;
    # in the own frame it is trivially ~0 because the reference sits at the hardware's own start.
    fV = m["fV"]; lo, hi = a2 + g1, b2 - g1; nn = np.arange(lo, hi)
    dphi = np.degrees(np.angle(np.mean(x[lo:hi] * np.exp(-2j * np.pi * fV * nn / fs)))
                      - np.angle(np.mean(ref_d[lo:hi] * np.exp(-2j * np.pi * fV * nn / fs))))
    dphi = (dphi + 180) % 360 - 180                # principal-angle subtraction can straddle +-180
    print(f"pulse-2 carrier phase, capture minus generator on the reference axis ({m['frame']} frame): "
          f"{dphi:+.2f} deg (gate < 1 deg)")
    out &= abs(dphi) < 1.0
    print("SAME_AS_GENERATOR:", "yes" if out else "NO")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    forced = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--as-frame=")), None)
    sys.exit(main(args[0] if args else "ms_capture.npz", forced))
