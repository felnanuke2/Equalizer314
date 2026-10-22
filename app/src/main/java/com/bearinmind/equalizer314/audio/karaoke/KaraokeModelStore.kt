package com.bearinmind.equalizer314.audio.karaoke

import android.content.Context
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * Local store of the Spleeter 2-stems vocals model (k2-fsa sherpa-onnx fp16 port, MIT) used by
 * the karaoke separator: filesDir/karaoke/vocals.onnx. The accompaniment model of earlier
 * versions is no longer run (the accompaniment is derived as the mixture-minus-vocals
 * residual), so a leftover accompaniment.onnx is deleted whenever the store is touched.
 * Downloads are verified against the pinned SHA-256 hash before being moved into place.
 */
object KaraokeModelStore {

    private data class Model(val file: String, val url: String, val sha256: String, val bytes: Long)

    private val VOCALS = Model(
        "vocals.onnx",
        "https://huggingface.co/csukuangfj/sherpa-onnx-spleeter-2stems-fp16/resolve/main/vocals.fp16.onnx",
        "24cef84aedcd1fe87c0b743ef3370ad34dc1fabf6c9014d6128a75a538c7b668",
        19681017L,
    )

    private fun dir(context: Context) = File(context.filesDir, "karaoke").apply { mkdirs() }

    /** Frees the ~20 MB accompaniment.onnx a previous two-model version left behind. */
    private fun cleanStale(context: Context) {
        File(dir(context), "accompaniment.onnx").delete()
    }

    fun downloaded(context: Context): Boolean {
        cleanStale(context)
        return File(dir(context), VOCALS.file).length() == VOCALS.bytes
    }

    fun vocalsPath(context: Context) = File(dir(context), VOCALS.file).absolutePath

    /**
     * The model the engine should run: a sideloaded window-rewritten export when present —
     * vocals.dyn.onnx (dynamic frame axis, tools/karaoke/shrink_window.py) or the smallest
     * fixed vocals.wNNN.onnx — else the pinned vocals.onnx. A rewritten model halves per-pass
     * compute at a 256-frame window; it is NOT downloaded by the app (publishing one is a
     * hosting decision), so most users stay on the pinned original.
     */
    fun modelPath(context: Context): String {
        val d = dir(context)
        File(d, "vocals.dyn.onnx").takeIf { it.isFile }?.let { return it.absolutePath }
        d.listFiles { f -> Regex("vocals\\.w\\d+\\.onnx").matchEntire(f.name) != null }
            ?.mapNotNull { f -> Regex("vocals\\.w(\\d+)\\.onnx").find(f.name)?.groupValues?.get(1)?.toIntOrNull()?.let { it to f } }
            ?.filter { it.first >= 256 }
            ?.minByOrNull { it.first }?.let { return it.second.absolutePath }
        return vocalsPath(context)
    }

    fun delete(context: Context) {
        dir(context).listFiles()?.forEach { it.delete() }
    }

    /**
     * Fetch the model; [onProgress] gets a 0–100 percentage and returns false to cancel.
     * Throws [IOException] on any network/hash failure; a partial download never lands.
     */
    fun download(context: Context, onProgress: (Int) -> Boolean) {
        cleanStale(context)
        fetch(context, VOCALS, onProgress, 0, 100)
    }

    private fun fetch(context: Context, model: Model, onProgress: (Int) -> Boolean, fromPct: Int, toPct: Int) {
        val part = File(dir(context), model.file + ".part")
        val conn = URL(model.url).openConnection() as HttpURLConnection
        conn.connectTimeout = 15000
        conn.readTimeout = 30000
        conn.instanceFollowRedirects = true
        try {
            val code = conn.responseCode
            if (code !in 200..299) throw IOException("HTTP $code")
            val total = conn.contentLengthLong.takeIf { it > 0 } ?: model.bytes
            conn.inputStream.use { input ->
                part.outputStream().use { out ->
                    val buf = ByteArray(64 * 1024)
                    var done = 0L
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        done += n
                        if (!onProgress((fromPct + (toPct - fromPct) * done / total).toInt().coerceIn(fromPct, toPct))) {
                            part.delete()
                            throw IOException("cancelled")
                        }
                    }
                }
            }
            if (!verify(part, model)) {
                part.delete()
                throw IOException("checksum mismatch")
            }
            val dest = File(dir(context), model.file)
            if (!part.renameTo(dest)) {
                part.delete()
                throw IOException("can't move into place")
            }
        } finally {
            conn.disconnect()
            part.delete()
        }
    }

    private fun verify(file: File, model: Model): Boolean {
        if (file.length() != model.bytes) return false
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(64 * 1024)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) } == model.sha256
    }
}
