package com.bearinmind.equalizer314.audio.karaoke

/**
 * JNI binding of the karaoke separator DSP engine (cpp/karaoke_jni.c: STFT/ISTFT, magnitude
 * and mask math on pffft). The engine owns the rings, analysis/synthesis state and the batch
 * counters; the net input/output cross into Java as direct buffers so model tensors never
 * round-trip as arrays. SpleeterStream drives it; numerics mirror KaraokeDsp.kt.
 *
 * On Android the library ships in the APK; desktop (real-model unit tests) loads a host
 * build via the KARAOKE_NATIVE_LIB environment variable (tools/karaoke/build_host_native.sh).
 */
internal object KaraokeNative {
    val available: Boolean

    init {
        var ok = false
        try { System.loadLibrary("karaoke_jni"); ok = true } catch (_: Throwable) {
            val host = System.getenv("KARAOKE_NATIVE_LIB")
            if (host != null) {
                try { System.load(host); ok = true } catch (_: Throwable) {}
            }
        }
        available = ok
    }

    /** Result codes of [applyMasks]: batch applied / superseded by a reset / stream ended. */
    const val STALE = 0
    const val STOPPED = -1

    @JvmStatic external fun create(
        frame: Int, hop: Int, modelBins: Int, window: Int,
        complexRing: Int, magRing: Int, lagLimit: Int,
    ): Long
    @JvmStatic external fun netInBuffer(handle: Long): java.nio.ByteBuffer
    @JvmStatic external fun netVocalsBuffer(handle: Long): java.nio.ByteBuffer

    /** Audio thread; returns the new STFT frame count, or -1 when the stream failed. */
    @JvmStatic external fun push(handle: Long, pcm: FloatArray, frames: Int): Int

    /** Audio thread; 1 = [out] filled (hop*2 interleaved), 0 = none ready, -1 = failed. */
    @JvmStatic external fun popOutput(handle: Long, out: FloatArray): Int

    /** Inference thread: waits for [advance] fresh frames; the batch target, [STOPPED], or -2 (timeout, retry). */
    @JvmStatic external fun waitBatch(handle: Long, advance: Int, timeoutMs: Int): Int
    @JvmStatic external fun epoch(handle: Long): Int
    @JvmStatic external fun lag(handle: Long): Int
    @JvmStatic external fun failure(handle: Long): Int

    /** Inference thread: assemble the window tensor into the net-in buffer. */
    @JvmStatic external fun prepareBatch(handle: Long, target: Int, adv: Int)

    /** Inference thread: mask frames from the net-vocals buffer; new masked count, [STALE], or [STOPPED]. */
    @JvmStatic external fun applyMasks(handle: Long, target: Int, adv: Int, myEpoch: Int): Int

    /** Audio thread (silence pause): drop pending state; in-flight batches go stale via the epoch. */
    @JvmStatic external fun reset(handle: Long)
    @JvmStatic external fun stop(handle: Long)
    @JvmStatic external fun destroy(handle: Long)
}
