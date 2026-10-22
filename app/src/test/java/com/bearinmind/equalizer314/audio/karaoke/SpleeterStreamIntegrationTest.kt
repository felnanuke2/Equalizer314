package com.bearinmind.equalizer314.audio.karaoke

import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * End-to-end equivalence test: streams a real 11 s music clip (44.1 kHz) through
 * [SpleeterStream] and requires sample-level agreement with a NumPy/ONNX Runtime reference
 * that replays the exact same batching (512-frame windows, 64-frame advance, zero-padded
 * head, bin-1023 mask extension, WOLA emission).
 *
 * Opt-in — needs the two models plus the generated .npy fixtures; set KARAOKE_TEST_MODELS to
 * the directory holding them (vocals.onnx, accompaniment.onnx, test_in.npy, test_ref_out.npy)
 * and run with `-PkaraokeTestOrt` so the desktop ONNX Runtime is on the test classpath.
 * Skipped otherwise: it needs ~39 MB of models and a CPU faster than real time.
 */
class SpleeterStreamIntegrationTest {

    private val hop = KaraokeDsp.HOP

    @Test
    fun matchesStreamingReference() {
        val dir = System.getenv("KARAOKE_TEST_MODELS")
        assumeTrue("set KARAOKE_TEST_MODELS to run", dir != null)
        val d = File(dir!!)
        assumeTrue(
            "fixtures present",
            File(d, "vocals.onnx").isFile && File(d, "accompaniment.onnx").isFile &&
                File(d, "test_in.npy").isFile && File(d, "test_ref_out.npy").isFile,
        )

        val input = readNpy2ch(File(d, "test_in.npy"))
        val ref = readNpy2ch(File(d, "test_ref_out.npy"))
        val frames = input[0].size / hop
        val maskedFrames = (frames / KaraokeDsp.ADVANCE) * KaraokeDsp.ADVANCE

        // Single inference thread, fixed 64-frame advance (adaptation off): matches the fixture
        // generator's session so both sides see identical ONNX Runtime numerics and batching
        // (production keeps KaraokeService.INFER_THREADS and runtime adaptation).
        val sep = SpleeterStream.open(File(d, "vocals.onnx").absolutePath, File(d, "accompaniment.onnx").absolutePath, 1, adaptive = false)
        sep.start()
        val inBuf = FloatArray(hop * 2)
        val outBuf = FloatArray(hop * 2)
        val outL = FloatArray(maskedFrames * hop)
        val outR = FloatArray(maskedFrames * hop)
        val hopNs = 1_000_000_000L * hop / 44100L
        var emitted = 0
        try {
            var nextDeadline = System.nanoTime()
            for (t in 0 until frames) {
                for (j in 0 until hop) {
                    inBuf[j * 2] = input[0][t * hop + j]
                    inBuf[j * 2 + 1] = input[1][t * hop + j]
                }
                sep.push(inBuf, hop)
                while (sep.popOutput(outBuf)) {
                    if (emitted < maskedFrames * hop) {
                        for (j in 0 until hop) {
                            outL[emitted + j] = outBuf[j * 2]
                            outR[emitted + j] = outBuf[j * 2 + 1]
                        }
                        emitted += hop
                    }
                }
                check(sep.failure == null) { "separator failed: ${sep.failure}" }
                // Pace like real time so the lag guard doesn't trip on a fast feed.
                nextDeadline += hopNs
                val wait = nextDeadline - System.nanoTime()
                if (wait > 0) Thread.sleep(wait / 1_000_000, (wait % 1_000_000).toInt())
            }
            // Let the in-flight batch land: release() discards masks computed after it.
            val drainDeadline = System.nanoTime() + 8_000_000_000L
            var quietMs = 0L
            while (System.nanoTime() < drainDeadline && quietMs < 2500 && emitted < maskedFrames * hop) {
                var got = false
                while (sep.popOutput(outBuf)) {
                    if (emitted < maskedFrames * hop) {
                        for (j in 0 until hop) {
                            outL[emitted + j] = outBuf[j * 2]
                            outR[emitted + j] = outBuf[j * 2 + 1]
                        }
                        emitted += hop
                    }
                    got = true
                }
                check(sep.failure == null) { "separator failed: ${sep.failure}" }
                if (got) quietMs = 0 else { Thread.sleep(100); quietMs += 100 }
            }
        } finally {
            sep.release()
        }
        while (sep.popOutput(outBuf)) {
            if (emitted < maskedFrames * hop) {
                for (j in 0 until hop) {
                    outL[emitted + j] = outBuf[j * 2]
                    outR[emitted + j] = outBuf[j * 2 + 1]
                }
                emitted += hop
            }
        }

        assertTrue("only $emitted of ${maskedFrames * hop} samples emitted", emitted >= maskedFrames * hop)
        // Compare from hop 4 on: the first hops ride on partial window normalization and the
        // fp16 model reacts strongly to the zero-padded head — both identical in principle,
        // but compared where the pipeline is in steady state.
        var maxErr = 0.0
        var sumSqErr = 0.0
        var sumSqRef = 0.0
        var corr = 0.0
        val start = 4 * hop
        for (i in start until maskedFrames * hop) {
            val e0 = (outL[i] - ref[0][i]).toDouble()
            val e1 = (outR[i] - ref[1][i]).toDouble()
            maxErr = maxOf(maxErr, abs(e0), abs(e1))
            sumSqErr += e0 * e0 + e1 * e1
            sumSqRef += ref[0][i].toDouble() * ref[0][i] + ref[1][i].toDouble() * ref[1][i]
            corr += outL[i] * ref[0][i].toDouble() + outR[i] * ref[1][i].toDouble()
        }
        corr = corr / Math.sqrt(sumSqRef * sumSqRef) // cosine-ish since both near unit scale
        val relRms = Math.sqrt(sumSqErr / sumSqRef)
        // Diagnostics: best correlation over a range of sample offsets.
        var bestOff = 0
        var bestCorr = -1.0
        for (off in -65536..65536 step 1024) {
            var c = 0.0; var sa = 0.0; var sb = 0.0
            val a0 = (start + off).coerceAtLeast(0)
            val a1 = (maskedFrames * hop - 1024 + off).coerceAtMost(maskedFrames * hop - 1024)
            var i = a0
            while (i < a1) {
                val r = i - off
                if (r in 0 until maskedFrames * hop) {
                    c += outL[i] * ref[0][r].toDouble() + outR[i] * ref[1][r].toDouble()
                    sa += outL[i].toDouble() * outL[i] + outR[i].toDouble() * outR[i]
                    sb += ref[0][r].toDouble() * ref[0][r] + ref[1][r].toDouble() * ref[1][r]
                }
                i += 97
            }
            val nc = if (sa > 0 && sb > 0) c / Math.sqrt(sa * sb) else -1.0
            if (nc > bestCorr) { bestCorr = nc; bestOff = off }
        }
        println("emitted=$emitted maxErr=%.5f relRms=%.5f corr=%.6f bestAlignOff=$bestOff bestCorr=%.6f".format(maxErr, relRms, corr, bestCorr))
        assertTrue("relative RMS error vs reference too high: $relRms", relRms < 0.005)
        assertTrue("correlation with reference too low: $corr", corr > 0.9999)
    }

    /** Minimal NumPy .npy reader: float32, little endian, C-order, shape (2, n). */
    private fun readNpy2ch(f: File): Array<FloatArray> {
        val bytes = f.readBytes()
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val magic = ByteArray(6)
        bb.get(magic)
        check(String(magic, Charsets.ISO_8859_1) == "\u0093NUMPY") { "not an npy file" }
        bb.position(bb.position() + 2) // version
        val headerLen = bb.short.toInt() and 0xffff
        val header = ByteArray(headerLen)
        bb.get(header)
        val h = String(header, Charsets.ISO_8859_1)
        val shape = Regex("'shape': \\((\\d+), (\\d+)").find(h)?.groupValues
            ?: error("can't parse npy header: $h")
        val rows = shape[1].toInt()
        val cols = shape[2].toInt()
        check(rows == 2) { "expected 2 rows, got $rows" }
        val out = arrayOf(FloatArray(cols), FloatArray(cols))
        for (r in 0 until 2) for (c in 0 until cols) out[r][c] = bb.float
        return out
    }
}
