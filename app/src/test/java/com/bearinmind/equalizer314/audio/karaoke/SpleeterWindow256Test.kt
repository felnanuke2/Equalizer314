package com.bearinmind.equalizer314.audio.karaoke

import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Shrunk-window coverage for [SpleeterStream], complementing the fixed-512 reference tests:
 * sample-level agreement with a reference replaying the same streaming algorithm at a
 * 256-frame window (fixtures from `gen_spleeter_fixtures.py --window 256 --model
 * vocals.dyn.onnx`), proving the window-dynamic batching (rings, assembly rows, mask rows,
 * tensor shape) generalizes beyond the exported 512 — at half the per-pass compute.
 *
 * Opt-in like SpleeterStreamIntegrationTest: needs the dynamic-axis model (written by
 * `shrink_window.py`), the fixtures and `-PkaraokeTestOrt`; set KARAOKE_TEST_MODELS_W256.
 */
class SpleeterWindow256Test {

    private val hop = KaraokeDsp.HOP

    @Test
    fun matchesShrunkWindowReference() {
        val dir = System.getenv("KARAOKE_TEST_MODELS_W256")
        assumeTrue("set KARAOKE_TEST_MODELS_W256 to run", dir != null)
        val d = File(dir!!)
        assumeTrue(
            "fixtures present",
            File(d, "vocals.dyn.onnx").isFile && File(d, "test_in.npy").isFile && File(d, "test_ref_out.npy").isFile,
        )

        val (input, ref) = Pair(readNpy2ch(File(d, "test_in.npy")), readNpy2ch(File(d, "test_ref_out.npy")))
        val advance = KaraokeDsp.ADVANCE
        val frames = input[0].size / hop
        val maskedFrames = (frames / advance) * advance

        val sep = SpleeterStream.open(File(d, "vocals.dyn.onnx").absolutePath, 1, window = 256, adaptive = false)
        assumeTrue("dynamic model must run at the requested 256-frame window", sep.window == 256)
        sep.configure(advance)
        sep.start()
        val out = feedAndCollect(sep, input, maskedFrames)
        sep.release()
        assertTrue("only ${out[0].size} of ${maskedFrames * hop} samples emitted", out[0].size >= maskedFrames * hop)

        var maxErr = 0.0
        var sumSqErr = 0.0
        var sumSqRef = 0.0
        var corr = 0.0
        val start = 4 * hop
        for (i in start until maskedFrames * hop) {
            val e0 = (out[0][i] - ref[0][i]).toDouble()
            val e1 = (out[1][i] - ref[1][i]).toDouble()
            maxErr = maxOf(maxErr, kotlin.math.abs(e0), kotlin.math.abs(e1))
            sumSqErr += e0 * e0 + e1 * e1
            sumSqRef += ref[0][i].toDouble() * ref[0][i] + ref[1][i].toDouble() * ref[1][i]
            corr += out[0][i] * ref[0][i].toDouble() + out[1][i] * ref[1][i].toDouble()
        }
        corr = corr / Math.sqrt(sumSqRef * sumSqRef)
        val relRms = Math.sqrt(sumSqErr / sumSqRef)
        println("W256 maxErr=%.5f relRms=%.5f corr=%.6f".format(maxErr, relRms, corr))
        assertTrue("relative RMS error vs reference too high: $relRms", relRms < 0.005)
        assertTrue("correlation with reference too low: $corr", corr > 0.9999)
    }

    /** Like the other tests: real-time paced feed, sample collection, in-flight-batch drain. */
    private fun feedAndCollect(sep: SpleeterStream, input: Array<FloatArray>, maskedFrames: Int): Array<FloatArray> {
        val frames = input[0].size / hop
        val inBuf = FloatArray(hop * 2)
        val outBuf = FloatArray(hop * 2)
        val outL = FloatArray(maskedFrames * hop)
        val outR = FloatArray(maskedFrames * hop)
        var emitted = 0
        val hopNs = 1_000_000_000L * hop / 44100L
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
            nextDeadline += hopNs
            val wait = nextDeadline - System.nanoTime()
            if (wait > 0) Thread.sleep(wait / 1_000_000, (wait % 1_000_000).toInt())
        }
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
        return arrayOf(outL, outR)
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
