package com.bearinmind.equalizer314.audio.karaoke

import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Variable-advance coverage for [SpleeterStream], complementing the 64-frame reference test:
 *
 *  1. matchesAdvance32Reference — sample-level agreement with a reference that replays the
 *     exact streaming algorithm at a 32-frame advance (fixtures from
 *     `gen_spleeter_fixtures.py --advance 32`), proving the window/row math generalizes beyond
 *     the default 64. Runs with adaptation off so the batching is the configured one throughout.
 *
 *  2. adaptsBatchToTheDevice — the runtime adaptation itself: started at an unsustainably small
 *     batch on a slow machine it must enlarge it (not die with TOO_SLOW); started at the full
 *     64 on a machine with spare capacity it must shrink it.
 *
 * Opt-in like SpleeterStreamIntegrationTest: needs the models plus fixtures and
 * `-PkaraokeTestOrt`; skipped otherwise.
 */
class SpleeterAdvance32Test {

    private val hop = KaraokeDsp.HOP

    @Test
    fun matchesAdvance32Reference() {
        val sep = openStream("KARAOKE_TEST_MODELS_A32", adaptive = false) ?: return
        val (input, ref) = readFixtures(System.getenv("KARAOKE_TEST_MODELS_A32")!!)
        val advance = 32
        val frames = input[0].size / hop
        val maskedFrames = (frames / advance) * advance
        sep.configure(advance)
        sep.start()
        val out = feedAndCollect(sep, input, maskedFrames)
        assertTrue("only ${out[0].size} of ${maskedFrames * hop} samples emitted", out[0].size >= maskedFrames * hop)
        val stats = compare(out, ref, maskedFrames)
        println("A32 maxErr=%.5f relRms=%.5f corr=%.6f".format(stats[0], stats[1], stats[2]))
        assertTrue("relative RMS error vs reference too high: ${stats[1]}", stats[1] < 0.005)
        assertTrue("correlation with reference too low: ${stats[2]}", stats[2] > 0.9999)
    }

    @Test
    fun enlargesBatchInsteadOfDyingWhenOverrun() {
        val dir = System.getenv("KARAOKE_TEST_MODELS_A32") ?: return assumeTrue(false)
        val sep = openStream("KARAOKE_TEST_MODELS_A32") ?: return
        try {
            // Only meaningful where advance 8 genuinely overruns; fast machines skip.
            val inferSec = sep.benchmark()
            assumeTrue("advance 8 is sustainable here; nothing to recover from", inferSec * 44100.0 > 8 * KaraokeDsp.HOP)
            sep.configure(8)
            val (input, _) = readFixtures(dir)
            sep.start()
            val out = feedRealtime(sep, input)
            check(sep.failure == null) { "separator failed: ${sep.failure}" }
            assertTrue("batch never enlarged (advance=${sep.advanceFrames})", sep.advanceFrames > 8)
            assertTrue("no output", out > 0)
        } finally {
            sep.release()
        }
    }

    @Test
    fun shrinksBatchWhenThereIsSpareCapacity() {
        val sep = openStream("KARAOKE_TEST_MODELS_A32") ?: return
        try {
            val inferSec = sep.benchmark()
            // Step-down needs the pass well under one 64-frame batch (the shrink threshold).
            assumeTrue("device too slow to spare capacity at advance 64", inferSec * 44100.0 <= 0.35 * 64 * KaraokeDsp.HOP)
            sep.configure(KaraokeDsp.ADVANCE)
            val (input, _) = readFixtures(System.getenv("KARAOKE_TEST_MODELS_A32")!!)
            sep.start()
            feedRealtime(sep, input)
            check(sep.failure == null) { "separator failed: ${sep.failure}" }
            assertTrue("batch never shrunk (advance=${sep.advanceFrames})", sep.advanceFrames < KaraokeDsp.ADVANCE)
        } finally {
            sep.release()
        }
    }

    /** Opens the model from the dir named by [env] (skips the test when unset); caller releases. */
    private fun openStream(env: String, adaptive: Boolean = true): SpleeterStream? {
        val dir = System.getenv(env) ?: run { assumeTrue("set $env to run", false); return null }
        val d = File(dir)
        assumeTrue(
            "fixtures present",
            File(d, "vocals.onnx").isFile && File(d, "test_in.npy").isFile,
        )
        return SpleeterStream.open(File(d, "vocals.onnx").absolutePath, 1, adaptive = adaptive)
    }

    private fun readFixtures(dir: String): Pair<Array<FloatArray>, Array<FloatArray>> {
        val d = File(dir)
        val input = readNpy2ch(File(d, "test_in.npy"))
        val ref = File(d, "test_ref_out.npy").takeIf { it.isFile }?.let { readNpy2ch(it) } ?: input
        return input to ref
    }

    /** Feeds at real-time pace, collects every emitted sample (adaptation test: count only). */
    private fun feedRealtime(sep: SpleeterStream, input: Array<FloatArray>): Int {
        val frames = input[0].size / hop
        val inBuf = FloatArray(hop * 2)
        val outBuf = FloatArray(hop * 2)
        val hopNs = 1_000_000_000L * hop / 44100L
        var emitted = 0
        var nextDeadline = System.nanoTime()
        for (t in 0 until frames) {
            for (j in 0 until hop) {
                inBuf[j * 2] = input[0][t * hop + j]
                inBuf[j * 2 + 1] = input[1][t * hop + j]
            }
            sep.push(inBuf, hop)
            while (sep.popOutput(outBuf)) emitted += hop
            check(sep.failure == null) { "separator failed: ${sep.failure}" }
            nextDeadline += hopNs
            val wait = nextDeadline - System.nanoTime()
            if (wait > 0) Thread.sleep(wait / 1_000_000, (wait % 1_000_000).toInt())
        }
        // Let the in-flight batch land.
        val drainDeadline = System.nanoTime() + 8_000_000_000L
        var quietMs = 0L
        while (System.nanoTime() < drainDeadline && quietMs < 2500) {
            var got = false
            while (sep.popOutput(outBuf)) { emitted += hop; got = true }
            check(sep.failure == null) { "separator failed: ${sep.failure}" }
            if (got) quietMs = 0 else { Thread.sleep(100); quietMs += 100 }
        }
        return emitted
    }

    /** Like [feedRealtime] plus sample collection and the drain semantics of the 64-frame test. */
    private fun feedAndCollect(sep: SpleeterStream, input: Array<FloatArray>, maskedFrames: Int): Array<FloatArray> {
        val frames = input[0].size / hop
        val inBuf = FloatArray(hop * 2)
        val outBuf = FloatArray(hop * 2)
        val outL = FloatArray(maskedFrames * hop)
        val outR = FloatArray(maskedFrames * hop)
        var emitted = 0
        try {
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
        } finally {
            sep.release()
        }
        return arrayOf(outL, outR)
    }

    private fun compare(out: Array<FloatArray>, ref: Array<FloatArray>, maskedFrames: Int): DoubleArray {
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
        return doubleArrayOf(maxErr, Math.sqrt(sumSqErr / sumSqRef), corr)
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
