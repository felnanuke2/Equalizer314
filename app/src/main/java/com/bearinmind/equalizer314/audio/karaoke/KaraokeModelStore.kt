package com.bearinmind.equalizer314.audio.karaoke

import android.content.Context
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * Local store of the Spleeter 2-stems models (k2-fsa sherpa-onnx fp16 port, MIT) used by the
 * karaoke separator: filesDir/karaoke/{vocals,accompaniment}.onnx. Downloads are verified
 * against pinned SHA-256 hashes before being moved into place.
 */
object KaraokeModelStore {

    private data class Model(val file: String, val url: String, val sha256: String, val bytes: Long)

    private val VOCALS = Model(
        "vocals.onnx",
        "https://huggingface.co/csukuangfj/sherpa-onnx-spleeter-2stems-fp16/resolve/main/vocals.fp16.onnx",
        "24cef84aedcd1fe87c0b743ef3370ad34dc1fabf6c9014d6128a75a538c7b668",
        19681017L,
    )
    private val ACCOMPANIMENT = Model(
        "accompaniment.onnx",
        "https://huggingface.co/csukuangfj/sherpa-onnx-spleeter-2stems-fp16/resolve/main/accompaniment.fp16.onnx",
        "d14cea55793cc531a5875f5f4da08207d1c5ab9292e8e0099a104eecb014fcc0",
        19681024L,
    )

    private fun dir(context: Context) = File(context.filesDir, "karaoke").apply { mkdirs() }

    fun downloaded(context: Context): Boolean =
        File(dir(context), VOCALS.file).length() == VOCALS.bytes &&
            File(dir(context), ACCOMPANIMENT.file).length() == ACCOMPANIMENT.bytes

    fun vocalsPath(context: Context) = File(dir(context), VOCALS.file).absolutePath
    fun accompanimentPath(context: Context) = File(dir(context), ACCOMPANIMENT.file).absolutePath

    fun delete(context: Context) {
        dir(context).listFiles()?.forEach { it.delete() }
    }

    /**
     * Fetch both models; [onProgress] gets a 0–100 percentage and returns false to cancel.
     * Throws [IOException] on any network/hash failure; a partial download never lands.
     */
    fun download(context: Context, onProgress: (Int) -> Boolean) {
        fetch(context, VOCALS, onProgress, 0, 50)
        fetch(context, ACCOMPANIMENT, onProgress, 50, 100)
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
