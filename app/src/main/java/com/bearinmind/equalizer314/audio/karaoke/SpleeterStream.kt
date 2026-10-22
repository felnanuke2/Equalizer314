package com.bearinmind.equalizer314.audio.karaoke

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.sqrt

/**
 * Streaming Spleeter 2-stems accompaniment extractor for the karaoke feature.
 *
 * Model contract (k2-fsa sherpa-onnx fp16 port of Deezer's Spleeter 2stems, both MIT):
 * input  "x" [2, 1, 512, 1024] float32 — magnitude spectrograms, channel-major (L, R),
 *        512 STFT frames of 1024 bins each (n_fft 4096, hop 1024, hann, center=false, 44.1 kHz);
 * output "y" same shape — sigmoid-masked magnitude estimates; the accompaniment soft mask is
 *        a²/(v² + a² + 1e-10) applied to the original complex spectrum (per channel).
 *
 * Streaming: STFT runs on the caller's (audio) thread; a 512-frame window is re-inferred every
 * [advance] new frames (64 by default; a startup [benchmark] lets the service tune the starting
 * point to the device), masking only the newest ones (each with WINDOW-advance frames of left
 * context; the stream starts padded with WINDOW-advance silent frames so the first batch needs
 * no warm-up). Bins above the model's 1024 (~11 kHz) reuse bin 1023's mask instead of the
 * reference's hard zero, so the karaoke output keeps its top octave.
 *
 * The inference loop adapts [advance] at runtime (unless opened with adaptive=false): when the
 * capture runs ahead of the masking (device slower than the batch budget) the batch is enlarged
 * — up to [KaraokeDsp.ADVANCE], the geometry the lag guard is calibrated for — and after several
 * fully caught-up passes it is halved again, trimming the playback delay on devices with spare
 * capacity. A startup benchmark alone can't be trusted: it measures a cold, unloaded system,
 * while the live pipeline contends with the EQ, the capture and the muted app's own decode.
 *
 * Threading: [push]/[popOutput]/[reset] are called from one audio thread; inference runs on its
 * own thread. Synchronization is a single lock guarding the frame counters (which publish the
 * ring buffers' state), with generous ring slack so the audio thread never blocks on inference.
 */
class SpleeterStream private constructor(
    private val env: OrtEnvironment,
    private val vocalsSession: OrtSession,
    private val accompanimentSession: OrtSession,
    private val inputName: String,
    private val shape: LongArray,
    private val adaptive: Boolean,
) {
    enum class Failure { MODEL, TOO_SLOW }

    @Volatile
    var failure: Failure? = null
        private set

    /** Real frames captured minus frames masked — the pipeline's live lag, for diagnostics. */
    @Volatile
    var lagFrames = 0
        private set

    /**
     * Frames masked per inference pass. [configure] sets the starting value before [start];
     * afterwards the inference loop adapts it to the observed capture lag (see class doc).
     */
    @Volatile
    private var advance = KaraokeDsp.ADVANCE

    /** Current batch size in frames; moves as the inference loop adapts. */
    val advanceFrames: Int get() = advance

    companion object {
        /** Complex-spectrum ring slots; masks a frame at most ~100 frames after capture, so 256 gives seconds of slack. */
        private const val COMPLEX_RING = 256
        /** Magnitude ring rows: window 512 + 128 rows (≈3 s) of scheduling slack before the oldest row is overwritten. */
        private const val MAG_RING = KaraokeDsp.WINDOW + 128
        /** Capture lead over masking beyond which the device is declared too slow (≈3.7 s). */
        private const val LAG_LIMIT = 160
        /** Smallest batch the loop adapts down to (≈186 ms of audio); the service's choices match. */
        private const val MIN_ADVANCE = 8
        /** Caught-up passes before the batch is halved; 4 keeps the latency win near-term. */
        private const val STABLE_PASSES = 4
        /** Lag (frames) over one batch that marks the device as falling behind — enlarge the batch. */
        private const val LAG_RECOVER = 24
        private const val EPS = 1e-10f
        private const val TAG = "Karaoke"

        /** Loads both models and returns an idle stream; throws when they can't be opened. */
        fun open(vocalsPath: String, accompanimentPath: String, threads: Int, adaptive: Boolean = true): SpleeterStream {
            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                try {
                    addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
                    addConfigEntry("session.intra_op.allow_spinning", "0")
                    setIntraOpNumThreads(1)
                } catch (t: Throwable) {
                    setIntraOpNumThreads(threads)
                }
                setInterOpNumThreads(1)
            }
            try {
                val vocals = env.createSession(File(vocalsPath).absolutePath, opts)
                val acc = env.createSession(File(accompanimentPath).absolutePath, opts)
                val shape = (vocals.inputInfo.values.first().info as ai.onnxruntime.TensorInfo).shape
                // [2, num_splits, 512, 1024]; only the split axis is dynamic — always feed one.
                require(shape.size == 4 && shape[0] == 2L && shape[2] == KaraokeDsp.WINDOW.toLong() && shape[3] == KaraokeDsp.MODEL_BINS.toLong()) {
                    "unexpected spleeter model input $shape"
                }
                return SpleeterStream(env, vocals, acc, vocals.inputNames.first(), shape, adaptive)
            } catch (t: Throwable) {
                try { opts.close() } catch (_: Throwable) {}
                throw t
            }
        }
    }

    private val lock = Object()

    // Frame counters (guarded by [lock]); frames before the stream start exist only as the
    // silent window padding assembled in runBatch (virtual frames < 0).
    private var stftCount = 0
    private var maskedCount = 0
    private var outputCount = 0
    private var epoch = 0
    private var running = true

    // Complex spectra of real frames; slot = (frame mod COMPLEX_RING + RING) mod RING.
    private val reL = Array(COMPLEX_RING) { FloatArray(KaraokeDsp.BINS) }
    private val imL = Array(COMPLEX_RING) { FloatArray(KaraokeDsp.BINS) }
    private val reR = Array(COMPLEX_RING) { FloatArray(KaraokeDsp.BINS) }
    private val imR = Array(COMPLEX_RING) { FloatArray(KaraokeDsp.BINS) }

    // Magnitudes per channel, rows = frame mod MAG_RING (row overwrite is the batching deadline).
    private val mags = Array(2) { FloatArray(MAG_RING * KaraokeDsp.MODEL_BINS) }

    // Inference scratch (inference thread only).
    private val netIn = FloatArray(2 * KaraokeDsp.WINDOW * KaraokeDsp.MODEL_BINS)
    private val netVocals = FloatArray(2 * KaraokeDsp.WINDOW * KaraokeDsp.MODEL_BINS)
    private val netAccomp = FloatArray(2 * KaraokeDsp.WINDOW * KaraokeDsp.MODEL_BINS)

    // STFT analysis state (audio thread only).
    private val fft = RealFft(KaraokeDsp.FRAME)
    private val window = KaraokeDsp.hann(KaraokeDsp.FRAME)
    private val histL = FloatArray(KaraokeDsp.FRAME)
    private val histR = FloatArray(KaraokeDsp.FRAME)
    private val aRe = FloatArray(KaraokeDsp.FRAME)
    private val aIm = FloatArray(KaraokeDsp.FRAME)
    private val carry = FloatArray(KaraokeDsp.HOP * 2)
    private var carryFrames = 0

    // WOLA synthesis (audio thread only).
    private val wolaL = WolaIstft()
    private val wolaR = WolaIstft()
    private val hopL = FloatArray(KaraokeDsp.HOP)
    private val hopR = FloatArray(KaraokeDsp.HOP)

    private val thread = Thread({
        // Priority just below the audio thread so inference stays responsive without preempting it.
        try { android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO) } catch (_: Throwable) {}
        inferLoop()
    }, "KaraokeInference").apply { isDaemon = true }

    fun start() { thread.start() }

    /** Caller thread, before [start]: starting batch size in frames (1..WINDOW). */
    fun configure(framesPerPass: Int) {
        require(framesPerPass in 1..KaraokeDsp.WINDOW) { "advance must be in 1..${KaraokeDsp.WINDOW}, got $framesPerPass" }
        advance = framesPerPass
    }

    /**
     * Caller thread, before [start]: median seconds for one full pass (both models) on silent
     * input, after warm-up — measures the device so the service can size batches and buffers.
     */
    fun benchmark(warmupPasses: Int = 1, timedPasses: Int = 2): Double {
        java.util.Arrays.fill(netIn, 0f)
        repeat(warmupPasses) { pass() }
        val times = DoubleArray(timedPasses) {
            val t0 = System.nanoTime()
            pass()
            (System.nanoTime() - t0) / 1e9
        }
        return times.sorted()[times.size / 2]
    }

    private fun pass() {
        runModel(vocalsSession, netVocals)
        runModel(accompanimentSession, netAccomp)
        if (failure != null) throw IllegalStateException("spleeter inference failed")
    }

    /** Audio thread: captured interleaved stereo samples; full hops are analyzed, the tail carried. */
    fun push(interleaved: FloatArray, frames: Int) {
        var off = 0
        while (off < frames) {
            val take = minOf(KaraokeDsp.HOP - carryFrames, frames - off)
            System.arraycopy(interleaved, off * 2, carry, carryFrames * 2, take * 2)
            carryFrames += take
            off += take
            if (carryFrames == KaraokeDsp.HOP) {
                carryFrames = 0
                pushHop()
            }
        }
    }

    private fun pushHop() {
        System.arraycopy(histL, KaraokeDsp.HOP, histL, 0, KaraokeDsp.FRAME - KaraokeDsp.HOP)
        System.arraycopy(histR, KaraokeDsp.HOP, histR, 0, KaraokeDsp.FRAME - KaraokeDsp.HOP)
        var i = 0
        while (i < KaraokeDsp.HOP) {
            histL[KaraokeDsp.FRAME - KaraokeDsp.HOP + i] = carry[i * 2]
            histR[KaraokeDsp.FRAME - KaraokeDsp.HOP + i] = carry[i * 2 + 1]
            i++
        }
        val frame: Int
        val slot: Int
        synchronized(lock) {
            if (failure != null || !running) return
            frame = stftCount
            slot = ((frame % COMPLEX_RING) + COMPLEX_RING) % COMPLEX_RING
        }
        analyze(histL, reL[slot], imL[slot], 0, frame)
        analyze(histR, reR[slot], imR[slot], 1, frame)
        synchronized(lock) {
            // Publish only after analyze wrote the frame: the inference thread may
            // assemble window rows (and apply masks) as soon as stftCount moves.
            stftCount = frame + 1
            lagFrames = stftCount - maskedCount
            if (stftCount - maskedCount > LAG_LIMIT) failure = Failure.TOO_SLOW
            lock.notifyAll()
        }
    }

    private fun analyze(hist: FloatArray, reOut: FloatArray, imOut: FloatArray, ch: Int, frame: Int) {
        for (j in 0 until KaraokeDsp.FRAME) {
            aRe[j] = hist[j] * window[j]
            aIm[j] = 0f
        }
        fft.forward(aRe, aIm)
        System.arraycopy(aRe, 0, reOut, 0, KaraokeDsp.BINS)
        System.arraycopy(aIm, 0, imOut, 0, KaraokeDsp.BINS)
        val m = mags[ch]
        val dst = ((frame % MAG_RING) + MAG_RING) % MAG_RING * KaraokeDsp.MODEL_BINS
        var b = 0
        while (b < KaraokeDsp.MODEL_BINS) {
            val r = aRe[b]
            val i2 = aIm[b]
            m[dst + b] = sqrt(r * r + i2 * i2)
            b++
        }
    }

    /** Audio thread: non-blocking; drains every masked frame into interleaved [out] (HOP*2). */
    fun popOutput(out: FloatArray): Boolean {
        val f: Int
        val slot: Int
        synchronized(lock) {
            if (outputCount >= maskedCount) return false
            if (failure != null) return false
            f = outputCount
            slot = f % COMPLEX_RING
            outputCount = f + 1
        }
        wolaL.push(reL[slot], imL[slot], hopL)
        wolaR.push(reR[slot], imR[slot], hopR)
        var i = 0
        while (i < KaraokeDsp.HOP) {
            out[i * 2] = hopL[i]
            out[i * 2 + 1] = hopR[i]
            i++
        }
        return true
    }

    /** Audio thread: drops all pending state (silence pause); inference in flight is discarded via [epoch]. */
    fun reset() {
        synchronized(lock) {
            epoch++
            stftCount = 0
            maskedCount = 0
            outputCount = 0
            lagFrames = 0
            lock.notifyAll()
        }
        carryFrames = 0
        wolaL.clear()
        wolaR.clear()
    }

    fun release() {
        synchronized(lock) {
            running = false
            lock.notifyAll()
        }
        try { thread.join(3000) } catch (_: InterruptedException) {}
        try { vocalsSession.close() } catch (_: Throwable) {}
        try { accompanimentSession.close() } catch (_: Throwable) {}
    }

    private fun inferLoop() {
        while (true) {
            val target: Int
            val adv: Int
            val myEpoch: Int
            synchronized(lock) {
                while (running && failure == null && stftCount < maskedCount + advance) lock.wait()
                if (!running) return
                if (failure != null) return
                target = maskedCount
                adv = advance
                myEpoch = epoch
            }
            if (runBatch(target, adv, myEpoch)) return
        }
    }

    /**
     * One inference pass masking frames [target, target+adv). Returns true when the loop
     * must stop (stopped/failed/superseded); a stale pass leaves no trace.
     */
    private fun runBatch(target: Int, adv: Int, myEpoch: Int): Boolean {
        // Assemble [2, 1, WINDOW, MODEL_BINS] magnitudes: window frames target-(WINDOW-adv)
        // .. target+adv-1, zero-padded before the stream start.
        java.util.Arrays.fill(netIn, 0f)
        for (ch in 0..1) {
            val m = mags[ch]
            val base = ch * KaraokeDsp.WINDOW
            for (w in 0 until KaraokeDsp.WINDOW) {
                val src = target - (KaraokeDsp.WINDOW - adv) + w
                if (src < 0) continue
                System.arraycopy(m, src % MAG_RING * KaraokeDsp.MODEL_BINS, netIn, (base + w) * KaraokeDsp.MODEL_BINS, KaraokeDsp.MODEL_BINS)
            }
        }
        if (!runModel(vocalsSession, netVocals) || !runModel(accompanimentSession, netAccomp)) return true
        synchronized(lock) {
            if (!running || failure != null) return true
            if (epoch != myEpoch) return false // reset mid-run: results are stale, keep looping
            applyMasks(target, adv)
            maskedCount = target + adv
            lagFrames = stftCount - maskedCount
            if (adaptive) adaptAdvance(lagFrames)
        }
        return false
    }

    /** Consecutive fully caught-up passes; guarded by [lock]. */
    private var stablePasses = 0

    /**
     * Keeps [advance] matched to what the device actually sustains while playing (lock held):
     * enlarge the batch as soon as the capture lead grows past one batch (the startup benchmark
     * measured an unloaded system), halve it after [STABLE_PASSES] passes with ≥25% spare
     * capacity. The lag guard still fails the stream only when even [KaraokeDsp.ADVANCE]
     * can't keep up.
     */
    private fun adaptAdvance(lag: Int) {
        if (lag >= advance + LAG_RECOVER) {
            if (advance < KaraokeDsp.ADVANCE) {
                val old = advance
                advance = minOf(KaraokeDsp.ADVANCE, advance * 2)
                try { android.util.Log.d(TAG, "batch $old->$advance (lag $lag)") } catch (_: Throwable) {}
            }
            stablePasses = 0
        } else if (lag * 4 <= advance * 3 && advance > MIN_ADVANCE) {
            if (++stablePasses >= STABLE_PASSES) {
                val old = advance
                advance /= 2
                try { android.util.Log.d(TAG, "batch $old->$advance (lag $lag)") } catch (_: Throwable) {}
                stablePasses = 0
            }
        } else {
            stablePasses = 0
        }
    }

    private fun runModel(session: OrtSession, out: FloatArray): Boolean {
        var tensor: OnnxTensor? = null
        var result: OrtSession.Result? = null
        return try {
            tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(netIn), longArrayOf(shape[0], 1, shape[2], shape[3]))
            result = session.run(mapOf(inputName to tensor))
            (result.get(0) as OnnxTensor).floatBuffer.get(out)
            true
        } catch (t: Throwable) {
            failure = Failure.MODEL
            true
        } finally {
            try { result?.close() } catch (_: Throwable) {}
            try { tensor?.close() } catch (_: Throwable) {}
        }
    }

    /** Apply the accompaniment mask to the ring slots of frames [target, target+adv). Lock held. */
    private fun applyMasks(target: Int, adv: Int) {
        val win = KaraokeDsp.WINDOW
        val mb = KaraokeDsp.MODEL_BINS
        for (j in 0 until adv) {
            val f = target + j
            val slot = f % COMPLEX_RING
            val w = win - adv + j // this frame's row in the inference window
            var ch = 0
            while (ch <= 1) {
                val vo = ch * win * mb + w * mb
                var lastMask = 1f
                var b = 0
                val re = if (ch == 0) reL[slot] else reR[slot]
                val im = if (ch == 0) imL[slot] else imR[slot]
                while (b < mb) {
                    val v = netVocals[vo + b]
                    val a = netAccomp[vo + b]
                    val vs = v * v
                    val as2 = a * a
                    val mask = (as2 + EPS / 2) / (vs + as2 + EPS)
                    re[b] *= mask
                    im[b] *= mask
                    lastMask = mask
                    b++
                }
                // Above the model's band (~11 kHz): extend bin 1023's mask (mask+vocals still sum to 1).
                b = mb
                while (b < KaraokeDsp.BINS) {
                    re[b] *= lastMask
                    im[b] *= lastMask
                    b++
                }
                ch++
            }
        }
    }
}
