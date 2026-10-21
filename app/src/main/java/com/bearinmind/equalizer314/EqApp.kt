package com.bearinmind.equalizer314

import android.app.Activity
import android.app.Application
import android.content.Context
import android.os.Bundle
import androidx.appcompat.app.AppCompatDelegate
import java.util.WeakHashMap

/** Applies the saved theme before any activity inflates (prefs read raw to keep startup light). */
class EqApp : Application() {
    override fun onCreate() {
        super.onCreate()
        applyNightMode(this)
        registerActivityLifecycleCallbacks(amoledHook)
        // TV Mode: app-wide screen tracking (peer nav-follow) + the remote-controlled touch lock on every activity.
        com.bearinmind.equalizer314.remote.RemoteScrim.install(this)
        // A killed virtual-surround or karaoke run leaves other apps muted (issue #128); undo it as soon as the process is back.
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
            Thread {
                com.bearinmind.equalizer314.audio.VirtualSurroundService.healStaleMutes(this)
                com.bearinmind.equalizer314.audio.KaraokeService.healStaleMutes(this)
            }.start()
        }
    }

    // Black (AMOLED) theme: overlay every activity at creation; activities built under an older stamp recreate on resume.
    private val createdUnderStamp = WeakHashMap<Activity, Int>()
    private val amoledHook = object : ActivityLifecycleCallbacks {
        override fun onActivityCreated(activity: Activity, savedInstanceState: Bundle?) {
            if (amoledActive(activity)) activity.theme.applyStyle(R.style.ThemeOverlay_Equalizer314_Amoled, true)
            createdUnderStamp[activity] = themeStamp
        }
        override fun onActivityResumed(activity: Activity) {
            if (createdUnderStamp[activity] != themeStamp) activity.recreate()
        }
        override fun onActivityStarted(activity: Activity) {}
        override fun onActivityPaused(activity: Activity) {}
        override fun onActivityStopped(activity: Activity) {}
        override fun onActivitySaveInstanceState(activity: Activity, outState: Bundle) {}
        override fun onActivityDestroyed(activity: Activity) {}
    }

    companion object {
        /** Bumped by the theme picker so live activities rebuild with the new palette. */
        @Volatile var themeStamp = 0

        // Theme picker values (issue #125).
        const val THEME_LIGHT = "light"
        const val THEME_DARK = "dark"
        const val THEME_AMOLED = "amoled"
        const val THEME_SYSTEM = "system"

        /** Saved theme; installs from before the picker map their old Light / Black AMOLED toggles. */
        fun themeMode(context: Context): String {
            val p = context.getSharedPreferences("eq_settings", Context.MODE_PRIVATE)
            p.getString("themeMode", null)?.let { return it }
            return when {
                p.getBoolean("lightTheme", false) -> THEME_LIGHT
                p.getBoolean("amoledTheme", false) -> THEME_AMOLED
                else -> THEME_DARK
            }
        }

        /** Save the theme; the old toggles are mirrored so backups still restore on older versions. */
        fun saveThemeMode(context: Context, mode: String) {
            context.getSharedPreferences("eq_settings", Context.MODE_PRIVATE).edit()
                .putString("themeMode", mode)
                .putBoolean("lightTheme", mode == THEME_LIGHT)
                .putBoolean("amoledTheme", mode == THEME_AMOLED)
                .apply()
        }

        /** Light → day, Follow system → the phone's setting, Dark / Black → night. */
        fun applyNightMode(context: Context) {
            AppCompatDelegate.setDefaultNightMode(when (themeMode(context)) {
                THEME_LIGHT -> AppCompatDelegate.MODE_NIGHT_NO
                THEME_SYSTEM -> AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM
                else -> AppCompatDelegate.MODE_NIGHT_YES
            })
        }

        fun amoledActive(context: Context): Boolean = themeMode(context) == THEME_AMOLED
    }
}
