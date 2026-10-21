package com.bearinmind.equalizer314.audio.karaoke

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin
import kotlin.random.Random

/** Round-trip checks for the karaoke separator's analysis/synthesis DSP (identity mask). */
class KaraokeDspTest {

    private val frame = KaraokeDsp.FRAME
    private val hop = KaraokeDsp.HOP
    private val bins = KaraokeDsp.BINS

    /** forward ∘ inverse must return the input. */
    @Test
    fun fftInverseRoundTrip() {
        val fft = RealFft(frame)
        val rng = Random(42)
        val re = FloatArray(frame) { rng.nextFloat() * 2 - 1 }
        val im = FloatArray(frame) { rng.nextFloat() * 2 - 1 }
        val re0 = re.copyOf()
        val im0 = im.copyOf()
        fft.forward(re, im)
        fft.inverse(re, im)
        for (i in 0 until frame) {
            assertTrue(Math.abs(re[i] - re0[i]) < 1e-3)
            assertTrue(Math.abs(im[i] - im0[i]) < 1e-3)
        }
    }

    /** hann analysis → WOLA synthesis reconstructs the signal exactly, including at stream start. */
    @Test
    fun stftWolaReconstruction() {
        val fft = RealFft(frame)
        val window = KaraokeDsp.hann(frame)
        val wola = WolaIstft()
        val frames = 12
        val n = frame + (frames - 1) * hop
        val rng = Random(7)
        // Music-ish: sum of tones plus noise.
        val x = FloatArray(n) { i ->
            (0.3 * sin(2.0 * PI * 440 * i / 44100) +
                0.2 * sin(2.0 * PI * 3500 * i / 44100) +
                0.1 * rng.nextFloat()).toFloat()
        }
        val hist = FloatArray(frame)
        val re = FloatArray(frame)
        val im = FloatArray(frame)
        val specRe = FloatArray(bins)
        val specIm = FloatArray(bins)
        val outHop = FloatArray(hop)
        val out = FloatArray(frames * hop)
        for (t in 0 until frames) {
            System.arraycopy(hist, hop, hist, 0, frame - hop)
            System.arraycopy(x, t * hop, hist, frame - hop, hop)
            for (j in 0 until frame) {
                re[j] = hist[j] * window[j]
                im[j] = 0f
            }
            fft.forward(re, im)
            System.arraycopy(re, 0, specRe, 0, bins)
            System.arraycopy(im, 0, specIm, 0, bins)
            wola.push(specRe, specIm, outHop)
            System.arraycopy(outHop, 0, out, t * hop, hop)
        }
        var maxErr = 0.0
        // The streaming analysis zero-pads the first FRAME-HOP samples, so the output
        // carries a 3-hop lead-in: out[t] reconstructs the padded signal, not raw x.
        val pad = frame - hop
        for (i in 0 until frames * hop) {
            val ref = if (i < pad) 0.0 else x[i - pad].toDouble()
            maxErr = maxOf(maxErr, Math.abs(out[i] - ref))
        }
        assertTrue("max reconstruction error $maxErr", maxErr < 1e-3)
    }

    /** An all-zero-mask stream must produce digital silence, including the partial start regions. */
    @Test
    fun zeroMaskSilences() {
        val wola = WolaIstft()
        val zeros = FloatArray(bins)
        val outHop = FloatArray(hop)
        for (t in 0 until 8) {
            wola.push(zeros, zeros, outHop)
            for (v in outHop) assertEquals(0f, v, 1e-9f)
        }
    }
}
