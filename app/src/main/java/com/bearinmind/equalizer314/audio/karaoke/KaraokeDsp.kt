package com.bearinmind.equalizer314.audio.karaoke

import java.util.Arrays
import kotlin.math.PI
import kotlin.math.cos

/** Shared STFT geometry for the karaoke separator (Spleeter 2stems model contract). */
object KaraokeDsp {
    /** The model's rate; capture, inference and replay all run at it (no resampling). */
    const val SAMPLE_RATE = 44100
    const val FRAME = 4096
    const val HOP = 1024
    const val BINS = FRAME / 2 + 1
    /** The U-Net only models bins 0..1023 (~0–11 kHz, a known Spleeter trait). */
    const val MODEL_BINS = 1024
    /** Model input frames per window (baked into the exported ONNX). */
    const val WINDOW = 512
    /** Real frames advanced (and masked) per inference pass; 64 frames ≈ 1.49 s of audio. */
    const val ADVANCE = 64

    /** Periodic Hann (matches tf.signal.stft's default window). */
    fun hann(frame: Int): FloatArray = FloatArray(frame) { n ->
        (0.5 * (1.0 - cos(2.0 * PI * n / frame))).toFloat()
    }
}

/** In-place float radix-2 complex FFT; [inverse] includes the 1/N scaling. */
class RealFft(val size: Int) {
    init { require(size > 1 && size and (size - 1) == 0) { "FFT size must be a power of two, got $size" } }

    private val cosT = FloatArray(size / 2)
    private val sinT = FloatArray(size / 2)
    private val rev = IntArray(size)

    init {
        for (i in 0 until size / 2) {
            val a = -2.0 * PI * i / size
            cosT[i] = cos(a).toFloat()
            sinT[i] = kotlin.math.sin(a).toFloat()
        }
        var bits = 0
        while (1 shl bits < size) bits++
        for (i in 0 until size) {
            var x = i
            var r = 0
            repeat(bits) { r = (r shl 1) or (x and 1); x = x shr 1 }
            rev[i] = r
        }
    }

    fun forward(re: FloatArray, im: FloatArray) {
        for (i in 0 until size) {
            val j = rev[i]
            if (j > i) {
                var t = re[i]; re[i] = re[j]; re[j] = t
                t = im[i]; im[i] = im[j]; im[j] = t
            }
        }
        var len = 2
        while (len <= size) {
            val half = len / 2
            val step = size / len
            var base = 0
            while (base < size) {
                var k = 0
                for (off in 0 until half) {
                    val i = base + off
                    val j = i + half
                    val wr = cosT[k]
                    val wi = sinT[k]
                    val tr = wr * re[j] - wi * im[j]
                    val ti = wr * im[j] + wi * re[j]
                    re[j] = re[i] - tr
                    im[j] = im[i] - ti
                    re[i] += tr
                    im[i] += ti
                    k += step
                }
                base += len
            }
            len = len shl 1
        }
    }

    /** Inverse via the swap identity: IDFT(x) = J(DFT(J(x)))/N with J swapping re/im. */
    fun inverse(re: FloatArray, im: FloatArray) {
        forward(im, re)
        val s = 1f / size
        for (i in 0 until size) {
            re[i] *= s
            im[i] *= s
        }
    }
}

/**
 * Mono weighted overlap-add synthesizer: hann analysis (caller) × hann synthesis with exact
 * per-sample normalization, so reconstruction is exact (up to float error) even at stream start.
 * Emits exactly [KaraokeDsp.HOP] samples per pushed spectrum into [out].
 */
class WolaIstft(
    private val frame: Int = KaraokeDsp.FRAME,
    private val hop: Int = KaraokeDsp.HOP,
) {
    private val fft = RealFft(frame)
    private val window = KaraokeDsp.hann(frame)
    private val acc = FloatArray(frame)
    private val wsum = FloatArray(frame)
    private val re = FloatArray(frame)
    private val im = FloatArray(frame)
    private var pos = 0

    /** Push one masked spectrum ([bins] = frame/2+1 values); the completed hop lands in [out]. */
    fun push(binsRe: FloatArray, binsIm: FloatArray, out: FloatArray) {
        Arrays.fill(re, 0f)
        Arrays.fill(im, 0f)
        System.arraycopy(binsRe, 0, re, 0, frame / 2 + 1)
        System.arraycopy(binsIm, 0, im, 0, frame / 2 + 1)
        var i = 1
        while (i < frame / 2) {
            re[frame - i] = re[i]
            im[frame - i] = -im[i]
            i++
        }
        fft.inverse(re, im)
        val off = pos * hop
        for (j in 0 until frame) {
            val w = window[j]
            var idx = off + j
            if (idx >= frame) idx -= frame
            acc[idx] += re[j] * w
            wsum[idx] += w * w
        }
        for (j in 0 until hop) {
            val idx = off + j
            out[j] = if (wsum[idx] > 1e-9f) acc[idx] / wsum[idx] else 0f
            acc[idx] = 0f
            wsum[idx] = 0f
        }
        pos = (pos + 1) % (frame / hop)
    }

    fun clear() {
        Arrays.fill(acc, 0f)
        Arrays.fill(wsum, 0f)
        pos = 0
    }
}
