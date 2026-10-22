#!/usr/bin/env python3
"""Behavioral reference of SpleeterStream.kt for SpleeterStreamIntegrationTest.

Generates a deterministic stereo test clip (test_in.npy) and its expected
karaoke output (test_ref_out.npy) by replaying the exact streaming algorithm of
app/src/main/java/com/bearinmind/equalizer314/audio/karaoke/SpleeterStream.kt
with NumPy + ONNX Runtime. The integration test streams test_in.npy through
SpleeterStream and requires sample-level agreement with test_ref_out.npy.

Model contract (k2-fsa sherpa-onnx fp16 port of Deezer Spleeter 2stems, MIT):
  input  "x" float32 [2, num_splits, 512, 1024] — |unnormalized rFFT| magnitude
         spectrogram, channel-major (ch0=L, ch1=R), 512 STFT frames x 1024 bins.
  output "y" float32 same shape — sigmoid_mask * input magnitude, i.e. per-stem
         magnitude estimates (NOT raw masks). Only the vocals model is run; the
         accompaniment magnitude is the residual a = m - v (m = the mixture
         magnitude, v = the vocals estimate). The model is downloaded from the
         pinned HuggingFace URL and verified against the pinned SHA-256 + byte
         size when absent from --models-dir.

STFT: n_fft=4096, hop=1024, 44100 Hz, periodic Hann w[n] = 0.5-0.5*cos(2*pi*n/4096),
forward unscaled rFFT (numpy rfft convention), inverse scaled by 1/N. Frame f
covers virtual input samples [(f-3)*1024, (f+1)*1024): the center=false STFT of
x with 3072 zero samples prepended.

Streaming batching (mirrors SpleeterStream.runBatch/applyMasks):
  for target = 0, 64, 128, ...: assemble window rows w = 0..511 from frames
  src = target - 448 + w (rows with src < 0 stay ZERO — zero-padded head);
  magnitude tensor flat index (ch * 512 + w) * 1024 + bin, shape
  [2, 1, 512, 1024]. Run ONLY the vocals model on that window (num_splits=1),
  then mask ONLY frames [target, target+64) = window rows 448..511 using that
  frame's own per-channel vocals estimate v and mixture magnitude m: the
  accompaniment is the residual a = m - v, so the Wiener ratio mask with
  separation exponent 2, EPSILON 1e-10 (Deezer) becomes
      mask_a = ((m-v)^2 + 5e-11) / (v^2 + (m-v)^2 + 1e-10)
  (mask_v + mask_a still sums to 1: v^2/(v^2+a^2) + a^2/(v^2+a^2) = 1),
  applied to the full 2049-bin complex spectrum of the mixture frame. Bins
  1024..2048 (above the model's 1024 bins) reuse the row's bin-1023 mask value
  (deliberate deviation from Deezer's mask_extension:"zeros" — keeps the top
  octave).

WOLA synthesis (mirrors WolaIstft): per masked frame, rebuild the Hermitian
  4096-bin spectrum (re[4096-i]=re[i], im[4096-i]=-im[i] for i=1..2047; bin
  2048 kept as-is), inverse DFT (1/N) x periodic Hann synthesis window,
  overlap-add into a 4096 accumulator with a w^2 envelope accumulator; each
  frame emits exactly 1024 samples acc/wsum over the frame's leading hop region
  (then those slots are zeroed). No extra 2/3 factor; the steady-state w^2
  envelope is 1.5.

Output: maskedFrames = (numInputFrames // 64) * 64 frames x 1024 samples per
channel -> test_ref_out.npy shape (2, maskedFrames*1024) float32.

Usage:
  gen_spleeter_fixtures.py --models-dir DIR --out DIR
"""

import argparse
import hashlib
import os
import sys
import urllib.request

import numpy as np

SAMPLE_RATE = 44100
FRAME = 4096
HOP = 1024
BINS = FRAME // 2 + 1          # 2049
MODEL_BINS = 1024              # the U-Net only models bins 0..1023
WINDOW = 512                   # model input frames per window
ADVANCE = 64                   # frames masked (and emitted) per inference pass; override with --advance
LEFT_CONTEXT = WINDOW - ADVANCE  # 448
EPS = np.float32(1e-10)

CLIP_FRAMES = 480              # 480 * 1024 samples ≈ 11.15 s at 44.1 kHz

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
                print(f"{name}: present, sha256 OK ({got_sha})")
                paths[name] = path
                continue
            print(f"{name}: present but mismatched (sha256={got_sha}, {got_bytes} bytes), re-downloading")
        tmp = path + ".part"
        print(f"{name}: downloading {url}")
        urllib.request.urlretrieve(url, tmp)
        got_sha, got_bytes = sha256_file(tmp), os.path.getsize(tmp)
        if got_sha != want_sha or got_bytes != want_bytes:
            os.remove(tmp)
            sys.exit(
                f"{name}: download mismatch — got sha256={got_sha} {got_bytes} bytes, "
                f"expected sha256={want_sha} {want_bytes} bytes (pins NOT edited)"
            )
        os.replace(tmp, path)
        print(f"{name}: downloaded, sha256 OK ({got_sha})")
        paths[name] = path
    return paths


def hann(frame):
    """Periodic Hann: 0.5 - 0.5*cos(2*pi*n/frame), matching KaraokeDsp.hann."""
    n = np.arange(frame, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / frame)).astype(np.float32)


def make_signal(frames):
    """Deterministic ~11 s music-ish stereo clip: tones, chirp, AM, shaped noise."""
    n = frames * HOP
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    rng = np.random.default_rng(0x5EED)
    left = np.zeros(n)
    right = np.zeros(n)

    # Chord tones with per-tone amplitude modulation and slow panning.
    tones = [
        (220.00, 0.26), (277.18, 0.19), (329.63, 0.17),
        (440.00, 0.13), (554.37, 0.10), (659.25, 0.08),
    ]
    for k, (f, amp) in enumerate(tones):
        am = 0.75 + 0.25 * np.sin(2.0 * np.pi * (0.9 + 0.13 * k) * t + 0.7 * k)
        tone = amp * am * np.sin(2.0 * np.pi * f * t + 0.3 * k)
        pan = 0.5 + 0.4 * np.sin(2.0 * np.pi * 0.05 * t + k)
        left += pan * tone
        right += (1.0 - pan) * tone

    # Slow upward chirp 300 -> 1400 Hz with its own slow AM.
    dur = t[-1]
    inst_phase = 2.0 * np.pi * (300.0 * t + 0.5 * (1400.0 - 300.0) / dur * t * t)
    chirp = 0.13 * np.sin(inst_phase) * (0.8 + 0.2 * np.sin(2.0 * np.pi * 0.23 * t))
    left += chirp
    right += 0.92 * np.roll(chirp, 64)

    # Shaped noise (white noise through a 32-tap moving-average lowpass).
    k_noise = np.ones(32, dtype=np.float64) / 32.0
    left += 0.09 * np.convolve(rng.standard_normal(n), k_noise, mode="same")
    right += 0.09 * np.convolve(rng.standard_normal(n), k_noise, mode="same")

    peak = max(np.abs(left).max(), np.abs(right).max())
    scale = 0.85 / peak
    return np.stack([left * scale, right * scale]).astype(np.float32)


def analyze(x):
    """STFT of x: frame f = hann-windowed samples [(f-3)*1024, (f+1)*1024) of x.

    Returns complex spectra (2, frames, BINS) and magnitudes of bins 0..1023
    (2, frames, MODEL_BINS), all float32, matching SpleeterStream.analyze.
    """
    frames = x.shape[1] // HOP
    xpad = np.concatenate([np.zeros(3072, dtype=np.float32), x[0]]), \
           np.concatenate([np.zeros(3072, dtype=np.float32), x[1]])
    win = hann(FRAME)
    spec_re = np.zeros((2, frames, BINS), dtype=np.float32)
    spec_im = np.zeros((2, frames, BINS), dtype=np.float32)
    mags = np.zeros((2, frames, MODEL_BINS), dtype=np.float32)
    for ch in range(2):
        for f in range(frames):
            buf = xpad[ch][f * HOP:f * HOP + FRAME] * win
            spec = np.fft.rfft(buf)  # unscaled forward DFT, 2049 bins
            spec_re[ch, f] = spec.real.astype(np.float32)
            spec_im[ch, f] = spec.imag.astype(np.float32)
            mags[ch, f] = np.abs(spec[:MODEL_BINS]).astype(np.float32)
    return spec_re, spec_im, mags


def run_reference(model_path, spec_re, spec_im, mags):
    """Replay SpleeterStream's window/advance batching + WOLA; returns (2, masked*1024)."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # single-threaded for deterministic outputs
    opts.inter_op_num_threads = 1
    sess_v = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    in_name = sess_v.get_inputs()[0].name

    frames = mags.shape[1]
    masked_frames = (frames // ADVANCE) * ADVANCE
    masked_re = np.zeros((2, masked_frames, BINS), dtype=np.float32)
    masked_im = np.zeros((2, masked_frames, BINS), dtype=np.float32)

    net_in = np.zeros((2, 1, WINDOW, MODEL_BINS), dtype=np.float32)
    for target in range(0, masked_frames, ADVANCE):
        net_in.fill(0.0)
        for w in range(WINDOW):
            src = target - LEFT_CONTEXT + w
            if src < 0:
                continue  # zero-padded head
            net_in[0, 0, w, :] = mags[0, src]
            net_in[1, 0, w, :] = mags[1, src]
        out_v = sess_v.run(None, {in_name: net_in})[0][:, 0]  # (2, WINDOW, MODEL_BINS)

        # Mask only frames [target, target+ADVANCE) = window rows LEFT_CONTEXT..WINDOW-1.
        sl = slice(target, target + ADVANCE)
        for ch in range(2):
            v = out_v[ch, LEFT_CONTEXT:WINDOW]  # (64, MODEL_BINS) float32
            m = mags[ch, sl]
            a = m - v  # residual accompaniment estimate
            vs = v * v
            asq = a * a
            mask = (asq + EPS / np.float32(2)) / (vs + asq + EPS)
            # bin-1023 mask extended over bins 1024..2048
            last = mask[:, MODEL_BINS - 1:MODEL_BINS]
            ext = np.repeat(last, BINS - MODEL_BINS, axis=1)
            full = np.concatenate([mask, ext], axis=1)  # (64, BINS)
            masked_re[ch, sl] = spec_re[ch, sl] * full
            masked_im[ch, sl] = spec_im[ch, sl] * full

    # WOLA synthesis per channel, mirroring WolaIstft.push exactly.
    win = hann(FRAME)
    out = np.zeros((2, masked_frames * HOP), dtype=np.float32)
    herm = np.arange(1, FRAME // 2)
    for ch in range(2):
        acc = np.zeros(FRAME, dtype=np.float32)
        wsum = np.zeros(FRAME, dtype=np.float32)
        pos = 0
        for f in range(masked_frames):
            re = np.zeros(FRAME, dtype=np.float32)
            im = np.zeros(FRAME, dtype=np.float32)
            re[:BINS] = masked_re[ch, f]
            im[:BINS] = masked_im[ch, f]
            re[FRAME - herm] = re[herm]
            im[FRAME - herm] = -im[herm]
            sig = np.fft.ifft(re + 1j * im).real.astype(np.float32)  # 1/N-scaled IDFT
            off = pos * HOP
            idx = (off + np.arange(FRAME)) % FRAME
            acc[idx] += sig * win
            wsum[idx] += win * win
            seg = slice(off, off + HOP)
            safe = wsum[seg] > np.float32(1e-9)
            hop_out = np.zeros(HOP, dtype=np.float32)
            np.divide(acc[seg], wsum[seg], out=hop_out, where=safe)
            out[ch, f * HOP:(f + 1) * HOP] = hop_out
            acc[seg] = 0.0
            wsum[seg] = 0.0
            pos = (pos + 1) % (FRAME // HOP)
    return out, masked_frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", required=True, help="dir with/for vocals.onnx")
    ap.add_argument("--out", required=True, help="output dir for test_in.npy + test_ref_out.npy")
    ap.add_argument("--advance", type=int, default=64,
                    help="frames masked per inference pass (default 64); LEFT_CONTEXT = window - advance. "
                    "Used to validate SpleeterStream's variable-advance batching (e.g. 32)")
    ap.add_argument("--window", type=int, default=512,
                    help="model input frames per window (default 512; 256 for the dynamic-axis "
                    "export of shrink_window.py, which the stream then runs at that size)")
    ap.add_argument("--model", default=None,
                    help="model file to run the reference with (default: the pinned vocals.onnx); "
                    "pass the dynamic vocals.dyn.onnx together with --window 256")
    args = ap.parse_args()

    global ADVANCE, LEFT_CONTEXT, WINDOW
    ADVANCE = args.advance
    WINDOW = args.window
    LEFT_CONTEXT = WINDOW - ADVANCE

    models = ensure_models(args.models_dir)
    os.makedirs(args.out, exist_ok=True)

    x = make_signal(CLIP_FRAMES)
    assert x.dtype == np.float32 and x.shape == (2, CLIP_FRAMES * HOP)
    assert np.abs(x).max() < 0.9
    np.save(os.path.join(args.out, "test_in.npy"), x)
    print(f"test_in.npy: shape {x.shape} float32, peak {np.abs(x).max():.4f}")

    spec_re, spec_im, mags = analyze(x)
    model_path = args.model or models["vocals.onnx"]
    ref, masked_frames = run_reference(model_path, spec_re, spec_im, mags)
    assert ref.dtype == np.float32 and ref.shape == (2, masked_frames * HOP)
    np.save(os.path.join(args.out, "test_ref_out.npy"), ref)
    print(f"test_ref_out.npy: shape {ref.shape} float32 "
          f"({masked_frames} masked frames of {CLIP_FRAMES} input frames, window {WINDOW}, advance {ADVANCE}, model {os.path.basename(model_path)})")


if __name__ == "__main__":
    main()
