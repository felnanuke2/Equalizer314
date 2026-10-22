package com.bearinmind.equalizer314.audio

import android.content.pm.PackageManager
import android.os.ParcelFileDescriptor
import android.util.Log
import java.io.BufferedReader
import java.io.InputStreamReader

/** Privileged `dumpsys <service>` transport via Shizuku (https://shizuku.rikka.app): AudioPolicyDumpParser falls back to this when the direct binder dump is denied (Samsung One UI), so session discovery works on any ROM where the user runs Shizuku. Unavailable/not granted → every call just returns null and behavior stays as without it. */
object ShizukuShell {

    private const val TAG = "ShizukuShell"

    /** Runs `dumpsys [service]` as the shell user through Shizuku, feeding each stdout line to [onLine] until EOF or [timeoutMs]; null when Shizuku is unavailable, the process fails, or the dump is refused. */
    fun dumpService(service: String, timeoutMs: Long, onLine: (String) -> Unit): Unit? {
        val binder = try {
            rikka.shizuku.Shizuku.getBinder()
        } catch (t: Throwable) {
            null          // binder dies on reboot/server stop
        } ?: return null
        val granted = try {
            rikka.shizuku.Shizuku.checkSelfPermission() == PackageManager.PERMISSION_GRANTED
        } catch (t: Throwable) {
            false
        }
        if (!granted) return null

        // Shizuku.newProcess() is hidden in v13; the public route is the one rish uses.
        val svc = moe.shizuku.server.IShizukuService.Stub
            .asInterface(rikka.shizuku.ShizukuBinderWrapper(binder))
        val process = try {
            svc.newProcess(arrayOf("dumpsys", service), null, null)
        } catch (t: Throwable) {
            Log.w(TAG, "shizuku dumpsys $service failed to start", t)
            return null
        }
        var lines = 0
        var refused = false
        // Same deadline pattern as the direct pipe read: callers run on a HandlerThread.
        val deadline = System.currentTimeMillis() + timeoutMs
        try {
            BufferedReader(
                InputStreamReader(ParcelFileDescriptor.AutoCloseInputStream(process.inputStream))
            ).use { reader ->
                while (true) {
                    if (System.currentTimeMillis() > deadline) {
                        Log.w(TAG, "$service shizuku dump read timed out after ${timeoutMs}ms")
                        break
                    }
                    val line = reader.readLine() ?: break
                    if (line.startsWith("Permission Denial")) { refused = true; break }
                    onLine(line)
                    lines++
                }
            }
        } catch (t: Throwable) {
            // The Shizuku binder can die mid-stream (server stopped, device rebooted).
            Log.w(TAG, "shizuku dumpsys $service read failed", t)
            return null
        } finally {
            try { process.destroy() } catch (_: Throwable) {}
        }
        if (refused) return null
        if (lines > 0) Log.i(TAG, "audio dump via shizuku ($service)")
        return Unit
    }
}
