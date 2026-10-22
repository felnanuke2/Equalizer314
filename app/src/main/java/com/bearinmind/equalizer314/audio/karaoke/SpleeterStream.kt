package com.bearinmind.equalizer314.audio.karaoke

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Streaming Spleeter 2-stems accompaniment extractor for the karaoke feature.
 *
 * Model contract (k2-fsa sherpa-onnx fp16 port of Deezer's Spleeter 2stems, both MIT):
 *        input "x" [2, 1, W, 1024] float32 — magnitude spectrograms, channel-major (L, R),
 *        W STFT frames of 1024 bins each (n_fft 4096, hop 1024, hann, center=false, 44.1 kHz);
 *        output "y" same shape — sigmoid-masked magnitude estimates. Only the vocals model is run;
 *        the accompaniment magnitude is derived as the residual a = m − v (m = the mixture
 *        magnitude already stored in the ring, v = the vocals estimate), giving the
 *        accompaniment soft mask a²/(v² + a² + 1e-10) applied to the original complex spectrum
 *        (per channel) at half the inference cost; mask_v + mask_a still sums to 1:
 *        v²/(v²+a²) + a²/(v²+a²) = 1.
 *
 * Window: the reference export bakes W = 512, but the U-Net is convolutional in time — a
 * window-rewritten export (tools/karaoke/shrink_window.py, verified bit-exact at 512) accepts
 * any W ≥ 256, halving per-pass compute at W = 256 since only the newest [advance] frames of
 * each window are consumed (measured exactly linear in W). [open] uses the model's fixed axis
 * when it has one, else the caller's [window] choice (clamped to the verified-safe 256..512
 * range).
 *
 * Precision: the fp16 port computes in fp16 throughout — fp16 weights and activations, casts
 * only at the fp32 graph-I/O boundary — so XNNPACK already runs NEON fp16 kernels; converting
 * the boundary to fp16 tensors would only shave a memcpy (a few MB per pass) and is not worth
 * the precision plumbing. INT8 would need retraining to keep mask quality (spectrogram masks
 * are quantization-sensitive); fp16 is the right operating point for this model.
 *
 * Streaming: STFT/ISTFT, the rings and the mask math run in the native engine
 * (cpp/karaoke_jni.c on pffft, bound via [KaraokeNative]); the audio thread feeds captured
 * PCM in and drains masked frames out, while a W-frame window is re-inferred every [advance]
 * new frames (64 by default; a startup [benchmark] lets the service tune the starting point
 * to the device) and only the newest frames are masked (each with W-advance frames of left
 * context; the stream starts padded with W-advance silent frames so the first batch needs no
 * warm-up). Bins above the model's 1024 (~11 kHz) reuse bin 1023's mask instead of the
 * reference's hard zero, so the karaoke output keeps its top octave.
 *
 * The inference loop adapts [advance] at runtime (unless opened with adaptive=false) from the
 * measured pass time against the batch budget (the wall time one batch of [advance] frames
 * covers): a pass eating most of its budget leaves nothing for load spikes, so the batch is
 * enlarged — up to [KaraokeDsp.ADVANCE], the geometry the lag guard is calibrated for — and a
 * pass well under budget means the batch (and with it the playback delay) can be halved again.
 * A startup benchmark alone can't be trusted: it measures a cold, unloaded system,
 * while the live pipeline contends with the EQ, the capture and the muted app's own decode.
 *
 * Threading: [push]/[popOutput]/[reset] are called from one audio thread; inference runs on its
 * own thread. The native engine's counters live behind one mutex (counter ops are O(1)); the
 * batch wait sleeps on the engine's condvar, and the generous ring slack keeps the audio thread
 * off any inference-length critical section.
 */
class SpleeterStream private constructor(
    private val env: OrtEnvironment,
    private val vocalsSession: OrtSession,
    private val inputName: String,
    private val shape: LongArray,
    /** Frames per inference window this stream runs at (from the model, or the caller's choice). */
    val window: Int,
    private val adaptive: Boolean,
    private val native: Long,
    private val netIn: FloatBuffer,
    private val netVocals: FloatBuffer,
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
     * afterwards the inference loop adapts it to the measured pass time (see class doc).
     */
    @Volatile
    private var advance = KaraokeDsp.ADVANCE

    /** Current batch size in frames; moves as the inference loop adapts. */
    val advanceFrames: Int get() = advance

    companion object {
        /** Complex-spectrum ring slots; masks a frame at most ~100 frames after capture, so 256 gives seconds of slack. */
        private const val COMPLEX_RING = 256
        /** Magnitude ring slack rows beyond the window (≈3 s) before the oldest row is overwritten. */
        private const val MAG_SLACK = 128
        /** Capture lead over masking beyond which the device is declared too slow (≈3.7 s). */
        private const val LAG_LIMIT = 160
        /** Smallest batch the loop adapts down to (≈186 ms of audio); the service's choices match. */
        private const val MIN_ADVANCE = 8
        /** Spare-capacity passes before the batch is halved; 4 keeps the latency win near-term. */
        private const val STABLE_PASSES = 4
        /** Consecutive near-budget passes before the batch is enlarged; one spike shouldn't cost latency. */
        private const val OVER_BUDGET_PASSES = 2
        /** Pass time over this fraction of the batch budget and the batch is enlarged. */
        private const val ENLARGE_PCT = 85
        /**
         * Pass time under this fraction of the budget and the batch may halve. Kept low enough
         * that a halved batch stays under [ENLARGE_PCT] of its own budget (2 × 35 < 85), so the
         * loop can't oscillate between two sizes.
         */
        private const val SHRINK_PCT = 35
        /** Verified-safe window range: 256+ matches the 512 reference, shorter windows diverge. */
        const val MIN_WINDOW = 256
        private const val TAG = "Karaoke"

        /**
         * Loads the vocals model and returns an idle stream; throws when it can't be opened.
         * [window] is only a request: a model with a fixed frame axis dictates its own, a
         * dynamic-axis one (shrink_window.py export) runs at the requested size.
         */
        fun open(vocalsPath: String, threads: Int, window: Int = KaraokeDsp.WINDOW, adaptive: Boolean = true): SpleeterStream {
            require(KaraokeNative.available) { "karaoke native engine unavailable (library not loaded)" }
            // `adb shell setprop log.tag.Karaoke VERBOSE` turns on ORT's own logging (node/EP
            // assignment) from the next start; the environment singleton takes the level once.
            val env = if (android.util.Log.isLoggable(TAG, android.util.Log.VERBOSE))
                OrtEnvironment.getEnvironment(ai.onnxruntime.OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE, "equalizer314")
            else OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                try {
                    addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
                    // Spinning XNNPACK workers: sleep/wake at every op barrier costs scheduler
                    // latency under load (the live pipeline contends with EQ, capture, decode),
                    // which shows up as pass-time spikes; karaoke is interactive, so latency wins.
                    addConfigEntry("session.intra_op.allow_spinning", "1")
                    setIntraOpNumThreads(1)
                } catch (t: Throwable) {
                    setIntraOpNumThreads(threads)
                }
                setInterOpNumThreads(1)
            }
            try {
                val vocals = env.createSession(File(vocalsPath).absolutePath, opts)
                val shape = (vocals.inputInfo.values.first().info as ai.onnxruntime.TensorInfo).shape
                // [2, num_splits, W, 1024]; only the split axis and (for rewritten exports) the
                // frame axis are dynamic (<= 0) — always feed one split at the chosen window.
                require(shape.size == 4 && shape[0] == 2L && (shape[2] <= 0L || shape[2] in MIN_WINDOW..KaraokeDsp.WINDOW.toLong()) && shape[3] == KaraokeDsp.MODEL_BINS.toLong()) {
                    "unexpected spleeter model input $shape"
                }
                val win = if (shape[2] > 0L) shape[2].toInt() else window.coerceIn(MIN_WINDOW, KaraokeDsp.WINDOW)
                val native = KaraokeNative.create(
                    KaraokeDsp.FRAME, KaraokeDsp.HOP, KaraokeDsp.MODEL_BINS, win,
                    COMPLEX_RING, win + MAG_SLACK, LAG_LIMIT,
                )
                require(native != 0L) { "karaoke native engine failed to allocate" }
                fun fb(getter: (Long) -> java.nio.ByteBuffer) =
                    getter(native).order(ByteOrder.nativeOrder()).asFloatBuffer()
                return SpleeterStream(
                    env, vocals, vocals.inputNames.first(), shape, win, adaptive,
                    native, fb(KaraokeNative::netInBuffer), fb(KaraokeNative::netVocalsBuffer),
                )
            } catch (t: Throwable) {
                try { opts.close() } catch (_: Throwable) {}
                throw t
            }
        }
    }

    /** Consecutive spare-capacity / near-budget passes; inference thread only. */
    private var stablePasses = 0

    /** Consecutive near-budget passes; inference thread only. */
    private var overBudgetPasses = 0

    private val thread = Thread({
        // Priority just below the audio thread so inference stays responsive without preempting it.
        try { android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO) } catch (_: Throwable) {}
        inferLoop()
    }, "KaraokeInference").apply { isDaemon = true }

    fun start() { thread.start() }

    /** Caller thread, before [start]: starting batch size in frames (1..[window]). */
    fun configure(framesPerPass: Int) {
        require(framesPerPass in 1..window) { "advance must be in 1..$window, got $framesPerPass" }
        advance = framesPerPass
    }

    /**
     * Caller thread, before [start]: median seconds for one full pass (the vocals model) on
     * silent input, after warm-up — measures the device so the service can size batches and
     * buffers. Two warm-ups: the first pass pays ORT's lazy init and the CPU governor's ramp,
     * and sizing the latency budget off that cold number would leave speed unused.
     */
    fun benchmark(warmupPasses: Int = 2, timedPasses: Int = 3): Double {
        netIn.clear()
        while (netIn.hasRemaining()) netIn.put(0f)
        repeat(warmupPasses) { pass() }
        val times = DoubleArray(timedPasses) {
            val t0 = System.nanoTime()
            pass()
            (System.nanoTime() - t0) / 1e9
        }
        return times.sorted()[times.size / 2]
    }

    private fun pass() {
        runModel()
        if (failure != null) throw IllegalStateException("spleeter inference failed")
    }

    /** Audio thread: captured interleaved stereo samples; full hops are analyzed, the tail carried. */
    fun push(interleaved: FloatArray, frames: Int) {
        val r = KaraokeNative.push(native, interleaved, frames)
        if (r < 0) failure = Failure.TOO_SLOW
        lagFrames = KaraokeNative.lag(native)
    }

    /** Audio thread: non-blocking; drains every masked frame into interleaved [out] (HOP*2). */
    fun popOutput(out: FloatArray): Boolean {
        val r = KaraokeNative.popOutput(native, out)
        if (r < 0) failure = Failure.TOO_SLOW
        return r > 0
    }

    /** Audio thread: drops all pending state (silence pause); inference in flight is discarded via the epoch. */
    fun reset() {
        KaraokeNative.reset(native)
        lagFrames = 0
    }

    fun release() {
        KaraokeNative.stop(native)
        try { thread.join(3000) } catch (_: InterruptedException) {}
        try { vocalsSession.close() } catch (_: Throwable) {}
        KaraokeNative.destroy(native)
    }

    private fun inferLoop() {
        while (true) {
            val adv = advance
            val target = KaraokeNative.waitBatch(native, adv, 500)
            if (target == KaraokeNative.STOPPED) {
                if (failure == null && KaraokeNative.failure(native) != 0) failure = Failure.TOO_SLOW
                return
            }
            if (target == -2) continue // wait timeout: re-check failure/stop on the next pass
            if (runBatch(target, adv, KaraokeNative.epoch(native))) return
        }
    }

    /**
     * One inference pass masking frames [target, target+adv). Returns true when the loop
     * must stop (stopped/failed); a stale pass (reset mid-run) leaves no trace.
     */
    private fun runBatch(target: Int, adv: Int, myEpoch: Int): Boolean {
        val t0 = System.nanoTime()
        KaraokeNative.prepareBatch(native, target, adv)
        if (!runModel()) return true
        when (val r = KaraokeNative.applyMasks(native, target, adv, myEpoch)) {
            KaraokeNative.STOPPED -> {
                if (failure == null && KaraokeNative.failure(native) != 0) failure = Failure.TOO_SLOW
                return true
            }
            KaraokeNative.STALE -> return false
            else -> {
                lagFrames = KaraokeNative.lag(native)
                if (adaptive) adaptAdvance((System.nanoTime() - t0) / 1e9)
            }
        }
        return false
    }

    /**
     * Keeps [advance] matched to what the device actually sustains while playing. The batch
     * budget is the wall time between output bursts ([advance] frames of audio), so a pass
     * eating most of it leaves nothing for load spikes — enlarge before the output starves —
     * while a pass well under budget means the batch, and with it the playback delay, can
     * halve. The lag guard still fails the stream only when even [KaraokeDsp.ADVANCE]
     * can't keep up.
     */
    private fun adaptAdvance(passSec: Double) {
        val budget = advance.toDouble() * KaraokeDsp.HOP / KaraokeDsp.SAMPLE_RATE
        if (passSec * 100 >= budget * ENLARGE_PCT) {
            // A pass at or over budget is a definite overrun; below that, one spike may be noise.
            overBudgetPasses = if (passSec >= budget) OVER_BUDGET_PASSES else overBudgetPasses + 1
            stablePasses = 0
            if (overBudgetPasses >= OVER_BUDGET_PASSES && advance < KaraokeDsp.ADVANCE) {
                overBudgetPasses = 0
                val old = advance
                advance = minOf(KaraokeDsp.ADVANCE, advance * 2)
                try { android.util.Log.d(TAG, "batch $old->$advance (pass ${(passSec * 1000).toInt()} ms)") } catch (_: Throwable) {}
            }
        } else {
            overBudgetPasses = 0
            if (passSec * 100 <= budget * SHRINK_PCT && advance > MIN_ADVANCE) {
                if (++stablePasses >= STABLE_PASSES) {
                    val old = advance
                    advance /= 2
                    try { android.util.Log.d(TAG, "batch $old->$advance (pass ${(passSec * 1000).toInt()} ms)") } catch (_: Throwable) {}
                    stablePasses = 0
                }
            } else {
                stablePasses = 0
            }
        }
    }

    private fun runModel(): Boolean {
        var tensor: OnnxTensor? = null
        var result: OrtSession.Result? = null
        return try {
            netIn.rewind()
            // The frame axis is this stream's [window] (the model's declared shape[2] may be dynamic).
            tensor = OnnxTensor.createTensor(env, netIn, longArrayOf(shape[0], 1, window.toLong(), shape[3]))
            result = vocalsSession.run(mapOf(inputName to tensor))
            val out = (result.get(0) as OnnxTensor).floatBuffer
            out.rewind()
            netVocals.rewind()
            netVocals.put(out)
            true
        } catch (t: Throwable) {
            failure = Failure.MODEL
            true
        } finally {
            try { result?.close() } catch (_: Throwable) {}
            try { tensor?.close() } catch (_: Throwable) {}
        }
    }
}
