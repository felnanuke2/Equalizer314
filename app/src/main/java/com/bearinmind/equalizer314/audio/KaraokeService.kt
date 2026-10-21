package com.bearinmind.equalizer314.audio

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioPlaybackCaptureConfiguration
import android.media.AudioPlaybackConfiguration
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.audiofx.DynamicsProcessing
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Looper
import android.os.Process
import android.os.SystemClock
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.core.app.NotificationCompat
import androidx.core.content.IntentCompat
import com.bearinmind.equalizer314.MainActivity
import com.bearinmind.equalizer314.R
import com.bearinmind.equalizer314.audio.karaoke.KaraokeDsp
import com.bearinmind.equalizer314.audio.karaoke.KaraokeModelStore
import com.bearinmind.equalizer314.audio.karaoke.SpleeterStream
import java.util.concurrent.atomic.AtomicReference

/**
 * Karaoke (experimental): captures other apps' audio, removes the vocals on-device with the
 * Spleeter 2-stems model (sherpa-onnx fp16 port) and replays the instrumental as plain media so
 * the session-0 EQ still applies; each captured app's own session is muted with a -200 dB
 * DynamicsProcessing, as VirtualSurroundService does. Capture and playback run at the model's
 * 44.1 kHz so no resampling is needed. The neural separation adds roughly 2–3 s of delay —
 * fine for singing along, visible against video.
 */
@RequiresApi(Build.VERSION_CODES.Q)
class KaraokeService : Service() {

    enum class Status { OFF, STARTING, RUNNING, NO_SESSIONS, FAILED }

    companion object {
        private const val TAG = "Karaoke"
        const val ACTION_START = "com.bearinmind.equalizer314.KARAOKE_START"
        const val ACTION_STOP = "com.bearinmind.equalizer314.KARAOKE_STOP"
        /** Package-local broadcast on every [status] change. */
        const val ACTION_STATE = "com.bearinmind.equalizer314.KARAOKE_STATE"
        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_RESULT_DATA = "result_data"
        const val ERROR_REVOKED = "revoked"
        const val ERROR_MODEL = "model"
        const val ERROR_CAPTURE = "capture"
        const val ERROR_TOO_SLOW = "too_slow"
        private const val CHANNEL_ID = "karaoke_channel"
        private const val NOTIFICATION_ID = 32
        /** The model's rate; both the capture and the replay run at it. */
        private const val SAMPLE_RATE = 44100
        /** Inference threads for ONNX Runtime; big-core count most phones can spare. */
        private const val INFER_THREADS = 4
        /** Mute-scan heartbeat; playback callbacks trigger faster scans. */
        private const val SCAN_MS = 2000L
        /** Pause the output after this much captured silence (no idle stream keeping the audio HAL awake). */
        private const val IDLE_PAUSE_MS = 3000
        /** Seconds of separated audio to accumulate in the AudioTrack before playback starts; keeps the buffer above one inference-burst period so it never underruns. */
        private const val PREBUFFER_SEC = 3
        /** Without DUMP: how long one recording check listens, the wait before the next, and checks per app until playback changes. */
        private const val PROBE_MS = 2500L
        private const val PROBE_RETRY_MS = 3000L
        private const val MAX_PROBES = 4

        @Volatile var status = Status.OFF
            private set
        @Volatile var mutedApps = 0
            private set
        @Volatile var lastError: String? = null
            private set

        val isRunning: Boolean get() = status == Status.STARTING || status == Status.RUNNING || status == Status.NO_SESSIONS

        fun stop(context: Context) {
            if (isRunning) context.startService(Intent(context, KaraokeService::class.java).setAction(ACTION_STOP))
        }

        private const val PREFS = "karaoke"
        private const val KEY_MUTED = "muted_sessions"
        private const val KEY_BOOT = "muted_boot"

        private fun bootCount(context: Context) =
            android.provider.Settings.Global.getInt(context.contentResolver, android.provider.Settings.Global.BOOT_COUNT, -1)

        /** AudioFlinger keeps a mute pinned on another app's session after our process dies: on the next start (same boot only), take each over, reset its gain and release it. Blocking (retries for ~19 s), so call it off the main thread. */
        fun healStaleMutes(context: Context) {
            val p = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            var pending = p.getStringSet(KEY_MUTED, null)?.mapNotNull { it.toIntOrNull() }.orEmpty()
            if (pending.isEmpty() || isRunning) return
            if (p.getInt(KEY_BOOT, -1) != bootCount(context)) pending = emptyList()
            for (delayMs in longArrayOf(1500, 2000, 3000, 5000, 7000)) {
                if (pending.isEmpty() || isRunning) break
                Thread.sleep(delayMs)
                pending = pending.filter { sid ->
                    runCatching {
                        DynamicsProcessing(Int.MAX_VALUE, sid, null).apply { setInputGainAllChannelsTo(0f); release() }
                        Log.d(TAG, "released stale mute on session $sid")
                    }.onFailure { Log.w(TAG, "stale mute on session $sid not released yet: $it") }.isFailure
                }
            }
            if (!isRunning) p.edit().remove(KEY_MUTED).commit()
        }
    }

    @Volatile private var projection: MediaProjection? = null
    private var audioThread: Thread? = null
    @Volatile private var running = false
    @Volatile private var record: AudioRecord? = null
    /** Apps left out of the capture (they can't be muted); set by the muter, picked up by the audio thread. */
    @Volatile private var excludedUids = IntArray(0)
    /** Without DUMP: capture only these apps (announced and checked); null = every capturable app minus [excludedUids]. */
    @Volatile private var matchUids: IntArray? = null
    @Volatile private var recreateRecord = false
    private var muter: SessionMuter? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> stopSelf()
            ACTION_START -> if (!running) start(intent)
            else -> stopSelf()
        }
        return START_NOT_STICKY
    }

    private fun start(intent: Intent) {
        // Karaoke replaces virtual surround: two capture pipelines can't share the mute dance.
        VirtualSurroundService.stop(this)
        // Foreground first: Android 14+ only hands the projection to a running mediaProjection-type service.
        try {
            createChannel()
            startForeground(NOTIFICATION_ID, buildNotification(), ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION)
        } catch (e: Exception) {
            Log.w(TAG, "startForeground refused", e)
            fail(ERROR_CAPTURE)
            return
        }
        val data = IntentCompat.getParcelableExtra(intent, EXTRA_RESULT_DATA, Intent::class.java)
        val mp = try {
            data?.let { getSystemService(MediaProjectionManager::class.java).getMediaProjection(intent.getIntExtra(EXTRA_RESULT_CODE, 0), it) }
        } catch (e: Exception) {
            Log.w(TAG, "getMediaProjection failed", e)
            null
        }
        if (mp == null) {
            fail(ERROR_REVOKED)
            return
        }
        mp.registerCallback(projectionCallback, Handler(Looper.getMainLooper()))
        projection = mp
        lastError = null
        setStatus(Status.STARTING)
        running = true
        // Capture nothing until the first scan says which apps can be muted.
        matchUids = IntArray(0)
        audioThread = Thread({ runEngine(mp) }, "KaraokeAudio").also { it.start() }
        muter = SessionMuter().also { it.start() }
    }

    // The user (status-bar chip / "Stop sharing") or another recording ended the projection.
    private val projectionCallback = object : MediaProjection.Callback() {
        override fun onStop() {
            if (running) fail(ERROR_REVOKED)
        }
    }

    private fun runEngine(mp: MediaProjection) {
        runCatching { Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO) }
        var sep: SpleeterStream? = null
        var rec: AudioRecord? = null
        var track: AudioTrack? = null
        try {
            sep = try {
                SpleeterStream.open(KaraokeModelStore.vocalsPath(this), KaraokeModelStore.accompanimentPath(this), INFER_THREADS)
            } catch (t: Throwable) {
                Log.w(TAG, "spleeter model failed to load", t)
                null
            }
            if (sep == null) {
                Handler(Looper.getMainLooper()).post { fail(ERROR_MODEL) }
                return
            }
            sep.start()
            separator.set(sep)
            var r = buildRecord(mp)
            rec = r
            record = r
            val t = buildTrack()
            track = t
            r.startRecording()
            // Don't play yet: accumulate enough separated audio to survive the gap between
            // inference bursts (each ≈1.5 s) before starting the output stream.
            val prebufferFrames = SAMPLE_RATE.toLong() * PREBUFFER_SEC
            var outputFrames = 0L
            var playing = false
            val buf = FloatArray(KaraokeDsp.HOP * 2)
            val out = FloatArray(KaraokeDsp.HOP * 2)
            var silentFrames = 0
            var paused = false
            loop@ while (running) {
                if (recreateRecord) {
                    // A new exclusion list only applies to a new AudioRecord (same projection).
                    recreateRecord = false
                    try { r.stop() } catch (_: Throwable) {}
                    r.release()
                    r = buildRecord(mp)
                    rec = r
                    record = r
                    r.startRecording()
                }
                val f = sep.failure
                if (f != null) {
                    Handler(Looper.getMainLooper()).post { fail(if (f == SpleeterStream.Failure.MODEL) ERROR_MODEL else ERROR_TOO_SLOW) }
                    break@loop
                }
                val n = r.read(buf, 0, buf.size, AudioRecord.READ_BLOCKING)
                if (n < 0) throw IllegalStateException("AudioRecord.read = $n")
                if (n == 0 || !running) continue
                val frames = n / 2
                if (isSilent(buf, n)) silentFrames += frames else silentFrames = 0
                if (silentFrames > SAMPLE_RATE * IDLE_PAUSE_MS / 1000) {
                    if (!paused) {
                        t.pause(); t.flush(); sep.reset()
                        paused = true; playing = false; outputFrames = 0
                    }
                    continue
                }
                if (paused) paused = false
                sep.push(buf, frames)
                while (sep.popOutput(out)) {
                    t.write(out, 0, KaraokeDsp.HOP * 2, AudioTrack.WRITE_BLOCKING)
                    if (!playing) {
                        outputFrames += KaraokeDsp.HOP
                        if (outputFrames >= prebufferFrames) {
                            t.play()
                            playing = true
                        }
                    }
                }
            }
        } catch (t: Throwable) {
            Log.w(TAG, "engine stopped", t)
            if (running) Handler(Looper.getMainLooper()).post { fail(ERROR_CAPTURE) }
        } finally {
            separator.getAndSet(null)?.release()
            record = null
            try { rec?.stop() } catch (_: Throwable) {}
            rec?.release()
            try { track?.stop() } catch (_: Throwable) {}
            track?.release()
        }
    }

    private val separator = AtomicReference<SpleeterStream>()

    private fun isSilent(buf: FloatArray, n: Int): Boolean {
        for (i in 0 until n) if (buf[i] > 1e-5f || buf[i] < -1e-5f) return false
        return true
    }

    /** [onlyUids] non-null = capture just those apps (our own uid stands in for "none": our output is never capturable). */
    private fun buildRecord(mp: MediaProjection, onlyUids: IntArray? = matchUids): AudioRecord {
        val config = AudioPlaybackCaptureConfiguration.Builder(mp)
            .addMatchingUsage(AudioAttributes.USAGE_MEDIA)
            .addMatchingUsage(AudioAttributes.USAGE_GAME)
            .addMatchingUsage(AudioAttributes.USAGE_UNKNOWN)
            .apply {
                // Matching and excluding uids can't be mixed in one config.
                if (onlyUids != null) {
                    if (onlyUids.isEmpty()) addMatchingUid(Process.myUid()) else onlyUids.forEach { addMatchingUid(it) }
                } else {
                    excludeUid(Process.myUid())
                    excludedUids.forEach { excludeUid(it) }
                }
            }
            .build()
        val min = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_STEREO, AudioFormat.ENCODING_PCM_FLOAT)
        // 3 s: the separation pipeline emits in bursts; the capture must never overflow.
        return AudioRecord.Builder()
            .setAudioPlaybackCaptureConfig(config)
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_IN_STEREO)
                    .build()
            )
            .setBufferSizeInBytes(maxOf(min, SAMPLE_RATE * 8 * 3))
            .build()
            .also { if (it.state != AudioRecord.STATE_INITIALIZED) { it.release(); throw IllegalStateException("AudioRecord not initialized") } }
    }

    // Plain media stream (the session-0 EQ processes it), never capturable itself, so it can't feed back into the capture.
    private fun buildTrack(): AudioTrack {
        val min = AudioTrack.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_OUT_STEREO, AudioFormat.ENCODING_PCM_FLOAT)
        // 4 s: one inference burst is ≈1.5 s; the buffer must absorb the gap between bursts plus inference jitter.
        return AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                    .setAllowedCapturePolicy(AudioAttributes.ALLOW_CAPTURE_BY_NONE)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                    .build()
            )
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setBufferSizeInBytes(maxOf(min, SAMPLE_RATE * 8 * 4))
            .build()
    }

    private fun fail(error: String) {
        lastError = error
        setStatus(Status.FAILED)
        stopSelf()
    }

    private fun setStatus(s: Status) {
        if (status == s) return
        status = s
        sendBroadcast(Intent(ACTION_STATE).setPackage(packageName))
        if (s == Status.RUNNING || s == Status.NO_SESSIONS) {
            getSystemService(NotificationManager::class.java)?.notify(NOTIFICATION_ID, buildNotification())
        }
    }

    override fun onDestroy() {
        running = false
        muter?.stop()
        muter = null
        // Unblock the pending read, then wait for the thread to release record/track/separator.
        try { record?.stop() } catch (_: Throwable) {}
        audioThread?.join(4000)
        audioThread = null
        projection?.unregisterCallback(projectionCallback)
        projection?.stop()
        projection = null
        stopForeground(STOP_FOREGROUND_REMOVE)
        mutedApps = 0
        if (status != Status.FAILED) setStatus(Status.OFF) else sendBroadcast(Intent(ACTION_STATE).setPackage(packageName))
        super.onDestroy()
    }

    private fun createChannel() {
        val nm = getSystemService(NotificationManager::class.java) ?: return
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, getString(R.string.karaoke_title), NotificationManager.IMPORTANCE_LOW).apply {
                setShowBadge(false)
            }
        )
    }

    private fun buildNotification(): Notification {
        val open = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val stop = PendingIntent.getService(
            this, 0, Intent(this, KaraokeService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_nav_equalizer)
            .setContentTitle(getString(R.string.karaoke_notif_title))
            .setContentText(getString(R.string.karaoke_notif_text))
            .setContentIntent(open)
            .setOngoing(true)
            .setSilent(true)
            .addAction(R.drawable.ic_nav_power, getString(R.string.turn_off), stop)
            .build()
    }

    /** Keeps a -200 dB DynamicsProcessing on every capturable app session so only the instrumental copy is heard. */
    private inner class SessionMuter {
        private val thread = HandlerThread("KaraokeMuter").also { it.start() }
        private val handler = Handler(thread.looper)
        private val effects = HashMap<Int, DynamicsProcessing>()
        private val owners = HashMap<Int, String>()
        private var policies: Map<String, Boolean> = emptyMap()
        private val audioManager = getSystemService(AudioManager::class.java)

        private val callback = object : AudioManager.AudioPlaybackCallback() {
            override fun onPlaybackConfigChanged(configs: MutableList<AudioPlaybackConfiguration>?) {
                onPlaybackChanged()
                schedule(80)
            }
        }
        private val scan = Runnable { scanNow() }
        private val heartbeat = object : Runnable {
            override fun run() {
                schedule(0)
                handler.postDelayed(this, SCAN_MS)
            }
        }

        fun start() {
            audioManager.registerAudioPlaybackCallback(callback, handler)
            AnnouncedSessions.onChange = { schedule(80) }
            handler.post(heartbeat)
        }

        fun stop() {
            AnnouncedSessions.onChange = null
            audioManager.unregisterAudioPlaybackCallback(callback)
            handler.removeCallbacksAndMessages(null)
            handler.post {
                for (sid in effects.keys.toList()) release(sid)
                thread.quitSafely()
            }
        }

        private fun schedule(delayMs: Long) {
            handler.removeCallbacks(scan)
            handler.postDelayed(scan, delayMs)
        }

        private val excluded = HashSet<Int>()

        private fun scanNow() {
            if (!running) return
            val snap = AudioPolicyDumpParser.captureSessions(this@KaraokeService)
            if (snap == null) {
                scanAnnounced()
                return
            }
            if (matchUids != null) {
                // DUMP granted mid-run: back to capturing every app.
                matchUids = null
                recreateRecord = true
            }
            // A player without a session (AAudio, SoundPool, Web Audio) can't be muted: leave that app out of the capture so it plays once, unprocessed.
            if (!excluded.containsAll(snap.sessionlessUids)) {
                excluded += snap.sessionlessUids
                excludedUids = excluded.toIntArray()
                recreateRecord = true
                Log.d(TAG, "left out of the capture (no session): uids $excluded")
            }
            // Apps that opt out of capture must keep their own (unprocessed) stream; re-read the log only for new packages.
            if (snap.sessions.values.any { it.second !in policies }) AudioPolicyDumpParser.capturePolicies()?.let { policies = it }
            val wanted = snap.sessions
                .filterValues { (uid, pkg) -> uid !in excluded && policies[pkg] != false }
                .mapValues { it.value.second }
            for (sid in effects.keys - wanted.keys) release(sid)
            for ((sid, pkg) in wanted) if (sid !in effects) mute(sid, pkg)
            mutedApps = owners.values.toSet().size
            report(Status.RUNNING)
        }

        /** Without DUMP: apps whose audio a capture really gets (only these are muted), apps being checked, checks left per app until playback changes. */
        private val recordable = HashSet<String>()
        private val probing = HashSet<String>()
        private val probesLeft = HashMap<String, Int>()

        fun onPlaybackChanged() = probesLeft.clear()

        /** Without DUMP only announced sessions are known: capture just those apps, so the rest play once, unprocessed. */
        private fun scanAnnounced() {
            val announced = AnnouncedSessions.snapshot()
            val uids = HashMap<String, Int>()
            for (pkg in announced.values.toSet()) {
                runCatching { uids[pkg] = packageManager.getApplicationInfo(pkg, 0).uid }
            }
            // An app can announce yet block recording (Spotify): muting it would silence it, so check it first.
            for ((pkg, uid) in uids) {
                if (pkg in recordable || pkg in probing) continue
                val left = probesLeft.getOrPut(pkg) { MAX_PROBES }
                if (left <= 0) continue
                probesLeft[pkg] = left - 1
                probe(pkg, uid)
            }
            val wanted = announced.filterValues { it in recordable && it in uids }
            val match = wanted.values.toSet().map { uids.getValue(it) }.sorted().toIntArray()
            if (!match.contentEquals(matchUids)) {
                matchUids = match
                recreateRecord = true
            }
            for (sid in effects.keys - wanted.keys) release(sid)
            for ((sid, pkg) in wanted) if (sid !in effects) mute(sid, pkg)
            mutedApps = owners.values.toSet().size
            report(if (wanted.isEmpty()) Status.NO_SESSIONS else Status.RUNNING)
        }

        private fun probe(pkg: String, uid: Int) {
            val mp = projection ?: return
            probing += pkg
            Thread({
                val heard = runCatching { hears(mp, uid) }
                    .onFailure { Log.w(TAG, "recording check for $pkg failed", it) }.getOrDefault(false)
                handler.post {
                    probing -= pkg
                    Log.d(TAG, "recording check $pkg: ${if (heard) "recordable" else "nothing heard"}")
                    if (heard) recordable += pkg
                    schedule(if (heard) 0 else PROBE_RETRY_MS)
                }
            }, "KaraokeProbe").start()
        }

        /** True once [uid]'s audio shows up in a capture of that app alone; an app that blocks recording stays silent. */
        private fun hears(mp: MediaProjection, uid: Int): Boolean {
            val r = buildRecord(mp, intArrayOf(uid))
            try {
                r.startRecording()
                val buf = FloatArray(KaraokeDsp.HOP * 2)
                val end = SystemClock.elapsedRealtime() + PROBE_MS
                while (running && SystemClock.elapsedRealtime() < end) {
                    val n = r.read(buf, 0, buf.size, AudioRecord.READ_BLOCKING)
                    if (n < 0) return false
                    if (n > 0 && !isSilent(buf, n)) return true
                }
                return false
            } finally {
                runCatching { r.stop() }
                r.release()
            }
        }

        private var reported: Pair<Status, Int>? = null

        /** Tell the UI only when the status or the muted-app count changed. */
        private fun report(s: Status) {
            val now = s to mutedApps
            if (now == reported) return
            reported = now
            Handler(Looper.getMainLooper()).post {
                if (!running) return@post
                if (status != s) setStatus(s) else sendBroadcast(Intent(ACTION_STATE).setPackage(packageName))
            }
        }

        private fun mute(sid: Int, pkg: String) {
            try {
                // Null config = the engine's own channel count, so every channel of the stream gets the gain.
                val dp = DynamicsProcessing(Int.MAX_VALUE, sid, null)
                dp.setInputGainAllChannelsTo(-200f)
                dp.enabled = true
                // Another effect app grabbing then returning the session: put the mute back.
                dp.setControlStatusListener { _, granted ->
                    if (granted) handler.post { effects[sid]?.let { e -> runCatching { e.setInputGainAllChannelsTo(-200f); e.enabled = true } } }
                }
                effects[sid] = dp
                owners[sid] = pkg
                persist()
                Log.d(TAG, "muted session $sid ($pkg)")
            } catch (t: Throwable) {
                Log.w(TAG, "could not mute session $sid ($pkg)", t)
            }
        }

        // 0 dB first: another app's handle on the same shared effect would otherwise keep the mute.
        private fun release(sid: Int) {
            effects.remove(sid)?.let { runCatching { it.setInputGainAllChannelsTo(0f) }; runCatching { it.release() } }
            owners.remove(sid)?.let { Log.d(TAG, "unmuted session $sid ($it)") }
            persist()
        }

        /** Record the muted sessions (synchronously) so a killed process can be cleaned up by [healStaleMutes]. */
        private fun persist() {
            getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                .putStringSet(KEY_MUTED, effects.keys.map { it.toString() }.toSet())
                .putInt(KEY_BOOT, bootCount(this@KaraokeService))
                .commit()
        }
    }
}
