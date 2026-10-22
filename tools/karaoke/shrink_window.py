#!/usr/bin/env python3
"""Make the Spleeter vocals model's inference window dynamic (or pin it shorter).

The sherpa-onnx export bakes a fixed 512-frame time axis into the graph I/O
"x"/"y" [2, num_splits, 512, 1024] and into stale declared value_info shapes.
The graph itself is time-agnostic (Conv/ConvTranspose with stride 1 on the
frame axis; the upsampling crops are RELATIVE slices), so rewriting the I/O
dims to a symbolic axis and dropping the stale value_info yields a model that
accepts any window length with the same weights.

Why: SpleeterStream re-infers the whole window every pass but masks only the
newest `advance` frames, so per-pass compute scales with the window length.
A 128-frame window is ~4x cheaper per pass than 512 at equal advance, letting
the adaptive batch run smaller and the playback delay shrink with it.

Modes:
  default         — dynamic frame axis, saved as vocals.dyn.onnx (any window
                    at run() time; the app picks per device).
  --pin N         — fixed frame axis N, saved as vocals.wN.onnx.

Verification (must pass before anything is written), two gates:
  1. BIT-EXACT at 512 — the converted model fed the full 512-row window must
     reproduce the original model's masks EXACTLY (same ops, same values); any
     nonzero error means the rewrite corrupted the graph.
  2. STATISTICAL at --window — the masked-region rows (the trailing `advance`
     rows, frames with window-advance frames of left context) must agree in
     distribution: p99.9 mask error < 2e-3 and energy-weighted RMS < 5e-4
     (mask = (m-v)^2/(v^2+(m-v)^2+eps), m being the input row itself; a few
     isolated loud-bin outliers at fp16 noise level are inaudible, hence
     percentile + energy gates rather than max).

Empirical boundary (measured on the sherpa fp16 export): windows >= 256
reproduce the 512 reference (384+ bit-exact, 256 at fp16 noise); windows
<= 128 diverge completely (p99 mask error ~1.0 — the down/up path degenerates),
so the tool refuses --window < 256.

Usage:
  shrink_window.py --models-dir DIR [--pin 256] [--window 256] [--advance 64]
"""

import argparse
import hashlib
import os
import sys
import urllib.request

import numpy as np

WINDOW = 512
MODEL_BINS = 1024

MODELS = {
    "vocals.onnx": (
        "https://huggingface.co/csukuangfj/sherpa-onnx-spleeter-2stems-fp16/resolve/main/vocals.fp16.onnx",
        "24cef84aedcd1fe87c0b743ef3370ad34dc1fabf6c9014d6128a75a538c7b668",
        19681017,
    ),
}


def sha256_file(path):
    md = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md.update(chunk)
    return md.hexdigest()


def ensure_models(models_dir):
    os.makedirs(models_dir, exist_ok=True)
    paths = {}
    for name, (url, want_sha, want_bytes) in MODELS.items():
        path = os.path.join(models_dir, name)
        if os.path.isfile(path):
            got_sha, got_bytes = sha256_file(path), os.path.getsize(path)
            if got_sha == want_sha and got_bytes == want_bytes:
                paths[name] = path
                continue
            print(f"{name}: present but mismatched, re-downloading")
        tmp = path + ".part"
        print(f"{name}: downloading {url}")
        urllib.request.urlretrieve(url, tmp)
        got_sha, got_bytes = sha256_file(tmp), os.path.getsize(tmp)
        if got_sha != want_sha or got_bytes != want_bytes:
            os.remove(tmp)
            sys.exit(f"{name}: download mismatch (pins NOT edited)")
        os.replace(tmp, path)
        paths[name] = path
    return paths


def rewrite_window(src_path, pin=None):
    """Return a ModelProto with the frame axis dynamic, or fixed at [pin] frames."""
    import onnx

    m = onnx.load(src_path)
    g = m.graph

    def set_frame_dim(value_info, name):
        dims = value_info.type.tensor_type.shape.dim
        if len(dims) != 4:
            sys.exit(f"{name} is not 4-D: {[d.dim_value or d.dim_param for d in dims]}")
        if dims[2].dim_value not in (WINDOW, 0) and not dims[2].dim_param:
            sys.exit(f"{name} frame axis is {dims[2].dim_value}, expected {WINDOW}")
        if pin is None:
            dims[2].dim_param = "frames"
        else:
            dims[2].dim_param = ""
            dims[2].dim_value = pin

    for i in g.input:
        set_frame_dim(i, f"input {i.name}")
    for o in g.output:
        set_frame_dim(o, f"output {o.name}")
    # Declared intermediate shapes are stale for any other window; they are
    # optional metadata, so drop them and let the runtime re-infer. (No strict
    # onnx.checker either: the sherpa export itself trips it — loose node
    # ordering — while loading fine everywhere; numbers are the real gate.)
    del g.value_info[:]
    return m


def make_test_windows(count, seed):
    """Realistic [2, 1, 512, 1024] float32 magnitude spectrograms.

    Real |rFFT| magnitudes (via the fixture generator's music-ish signal) plus
    low-range uniform noise; the model computes in fp16, so synthetic inputs
    must stay in-distribution or they saturate to inf regardless of window.
    """
    import importlib.util

    rng = np.random.default_rng(seed)
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("gen", os.path.join(here, "gen_spleeter_fixtures.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    sig = gen.make_signal(480)  # (2, 480*1024) float32
    _, _, mags = gen.analyze(sig)  # (2, 480, 1024)
    rows = mags.shape[1]
    out = []
    for k in range(count):
        if k % 2 == 0:
            if rows >= WINDOW:
                start = int(rng.integers(0, rows - WINDOW + 1))
                out.append(np.ascontiguousarray(mags[:, start:start + WINDOW]).reshape(2, 1, WINDOW, MODEL_BINS))
            else:  # fixture shorter than a window: tile with wraparound
                idx = (np.arange(WINDOW) + int(rng.integers(0, rows))) % rows
                out.append(np.ascontiguousarray(mags[:, idx]).reshape(2, 1, WINDOW, MODEL_BINS))
        else:
            out.append(rng.uniform(0.0, 200.0, size=(2, 1, WINDOW, MODEL_BINS)).astype(np.float32))
    return out


def mask_of(m, v):
    """The streamer's accompaniment soft mask from mixture and vocals estimate."""
    a = m - v
    vs = v * v
    asq = a * a
    return (asq + np.float32(5e-11)) / (vs + asq + np.float32(1e-10))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", required=True, help="dir with/for vocals.onnx; the converted model is written next to it")
    ap.add_argument("--pin", type=int, default=None,
                    help="write a model with a FIXED frame axis N (vocals.wN.onnx) instead of a dynamic one (vocals.dyn.onnx)")
    ap.add_argument("--window", type=int, default=256,
                    help="frame count used for verification (default 256); the masked-region rows must match the 512 reference")
    ap.add_argument("--advance", type=int, default=None,
                    help="frames per pass the streamer will mask (default: window/4, matching the 512/64 ratio)")
    args = ap.parse_args()
    window, advance = args.window, args.advance or max(1, args.window // 4)
    if not 256 <= window <= WINDOW:
        sys.exit(f"--window must be in 256..{WINDOW} (below 256 the up/down path diverges), got {window}")
    if not 1 <= advance <= window:
        sys.exit(f"--advance must be in 1..window, got {advance}")
    if args.pin is not None and not 256 <= args.pin <= WINDOW:
        sys.exit(f"--pin must be in 256..{WINDOW}, got {args.pin}")

    paths = ensure_models(args.models_dir)
    src = paths["vocals.onnx"]

    import onnx
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess_ref = ort.InferenceSession(src, opts, providers=["CPUExecutionProvider"])
    in_name = sess_ref.get_inputs()[0].name

    model = rewrite_window(src, args.pin)
    try:
        sess_n = ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    except Exception as e:
        sys.exit(f"converted model fails to load (graph not time-agnostic after all): {e}")

    tests = make_test_windows(6, seed=0x5EED)

    def mask_stats(sess, w):
        """(bit_exact_at_512, max, p999, energy_rms) of mask error vs the reference at window w."""
        exact, mx, p999, energy = True, 0.0, 0.0, 0.0
        for x in tests:
            y_ref = sess_ref.run(None, {in_name: x})[0][:, 0]          # (2, 512, 1024)
            y_n = sess.run(None, {in_name: x[:, :, -w:, :]})[0][:, 0]  # (2, w, 1024)
            for y in (y_ref, y_n):
                if not np.isfinite(y).all():
                    sys.exit(f"non-finite model output (input peak {x.max():.1f})")
            ref_tail = y_ref[:, -w:, :]                                # same absolute frames
            m = x[:, :, -w:, :][:, 0]                                  # the mixture rows themselves
            mask_ref = mask_of(m, ref_tail)
            mask_n = mask_of(m, y_n)
            err = np.abs(mask_n - mask_ref)[:, -advance:, :]           # the rows the streamer masks
            exact = exact and bool((err == 0.0).all())
            mx = max(mx, float(err.max()))
            p999 = max(p999, float(np.percentile(err, 99.9)))
            energy = max(energy, float(np.sqrt((err ** 2 * m[:, -advance:, :]).sum() / (m[:, -advance:, :] ** 2).sum())))
        return exact, mx, p999, energy

    if args.pin is None:
        exact512, mx512, _, _ = mask_stats(sess_n, WINDOW)
        print(f"gate 1 — window {WINDOW}: {'BIT-EXACT' if exact512 else f'max mask err {mx512:.2e} (NOT bit-exact)'}")
        if not exact512:
            sys.exit("the rewrite changed the graph's numerics at the original window — NOT writing the model")
    else:
        # A pinned model rejects other window sizes by design; the rewrite itself is identical
        # to the dynamic one (gate 1 there), so only the window-N agreement applies here.
        print(f"gate 1 — skipped for the pinned model (input is fixed at {args.pin} frames)")

    exact_w, mx_w, p999_w, energy_w = mask_stats(sess_n, window)
    kind = "bit-exact" if exact_w else f"max {mx_w:.5f}, p99.9 {p999_w:.5f}, energy-RMS {energy_w:.6f}"
    print(f"gate 2 — window {window}, masked advance {advance}: {kind}")
    if p999_w > 2e-3 or energy_w > 5e-4:
        sys.exit(f"masked-region agreement at window {window} too coarse (p99.9 {p999_w:.5f}, energy-RMS {energy_w:.6f}) — "
                 f"model NOT written (try a larger window or smaller advance)")

    out_kind = f"w{args.pin}" if args.pin is not None else "dyn"
    out_path = os.path.join(args.models_dir, f"vocals.{out_kind}.onnx")
    onnx.save(model, out_path)
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)")
    print("Sideload: adb push it to <filesDir>/karaoke/ — KaraokeModelStore.modelPath() prefers it.")


if __name__ == "__main__":
    main()
