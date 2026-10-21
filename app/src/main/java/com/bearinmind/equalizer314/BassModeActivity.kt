package com.bearinmind.equalizer314

import android.os.Bundle
import android.view.View
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.bearinmind.equalizer314.audio.EqService
import com.bearinmind.equalizer314.dsp.ParametricToDpConverter
import com.bearinmind.equalizer314.state.EqPreferencesManager
import com.google.android.material.card.MaterialCardView

/**
 * Bass Mode detail page: master enable toggle + intensity presets
 * ("more or less sub bass"). Stores values into [EqPreferencesManager]
 * only; the live overlay re-apply happens in EqService via
 * [EqService.ACTION_APPLY_BASS_MODE], exactly like the pipeline card's
 * power button. Opened by tapping the Bass Mode card on the pipeline
 * screen.
 */
class BassModeActivity : AppCompatActivity() {

    private lateinit var eqPrefs: EqPreferencesManager
    private lateinit var powerButton: ImageView

    /** One card per intensity level; rebuilt selection state on tap. */
    private data class PresetUi(
        val intensity: ParametricToDpConverter.BassIntensity,
        val card: MaterialCardView,
        val dot: View,
    )
    private val presetViews = mutableListOf<PresetUi>()

    private fun ParametricToDpConverter.BassIntensity.titleRes(): Int = when (this) {
        ParametricToDpConverter.BassIntensity.SUBTLE -> R.string.bass_preset_subtle
        ParametricToDpConverter.BassIntensity.BALANCED -> R.string.bass_preset_balanced
        ParametricToDpConverter.BassIntensity.DEEP -> R.string.bass_preset_deep
        ParametricToDpConverter.BassIntensity.MAX -> R.string.bass_preset_max
    }

    private fun ParametricToDpConverter.BassIntensity.descRes(): Int = when (this) {
        ParametricToDpConverter.BassIntensity.SUBTLE -> R.string.bass_preset_subtle_desc
        ParametricToDpConverter.BassIntensity.BALANCED -> R.string.bass_preset_balanced_desc
        ParametricToDpConverter.BassIntensity.DEEP -> R.string.bass_preset_deep_desc
        ParametricToDpConverter.BassIntensity.MAX -> R.string.bass_preset_max_desc
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_bass_mode)
        eqPrefs = EqPreferencesManager(this)

        findViewById<ImageButton>(R.id.bassBackButton).setOnClickListener { finish() }

        powerButton = findViewById(R.id.bassPowerButton)
        powerButton.setOnClickListener {
            eqPrefs.setBassModeEnabled(!eqPrefs.getBassModeEnabled())
            applyToService()
            paintPower()
        }
        paintPower()

        // Sustain: slow-release low-band compressor (the "prolonged" half).
        // Pinging the service lets it swap/restore the MBC band immediately.
        findViewById<com.google.android.material.materialswitch.MaterialSwitch>(R.id.bassSustainSwitch).apply {
            isChecked = eqPrefs.getBassSustainEnabled()
            setOnCheckedChangeListener { _, checked ->
                eqPrefs.setBassSustainEnabled(checked)
                applyToService()
            }
        }

        buildPresetCards()
    }

    override fun onResume() {
        super.onResume()
        // The pipeline screen's power button can flip the state while we're
        // away (activity stacked underneath) — re-sync the icon and the
        // selected preset on return.
        paintPower()
        repaintSelection()
    }

    private fun buildPresetCards() {
        val container = findViewById<LinearLayout>(R.id.bassPresetContainer)
        val density = resources.displayMetrics.density

        for (level in ParametricToDpConverter.BassIntensity.values()) {
            val card = MaterialCardView(this).apply {
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                ).apply { bottomMargin = (12 * density).toInt() }
                radius = 16 * density
                cardElevation = 0f
                strokeWidth = 0
                val bgAttr = android.util.TypedValue()
                theme.resolveAttribute(
                    com.google.android.material.R.attr.colorSurfaceContainerHigh, bgAttr, true
                )
                setCardBackgroundColor(bgAttr.data)
                isClickable = true
                isFocusable = true
            }

            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = android.view.Gravity.CENTER_VERTICAL
                setPadding(
                    (16 * density).toInt(), (14 * density).toInt(),
                    (16 * density).toInt(), (14 * density).toInt()
                )
            }
            card.addView(row, LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ))

            val textCol = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            }
            val title = TextView(this).apply {
                text = getString(level.titleRes())
                setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_TitleMedium)
                val tc = android.util.TypedValue()
                theme.resolveAttribute(com.google.android.material.R.attr.colorOnSurface, tc, true)
                setTextColor(tc.data)
            }
            textCol.addView(title)
            val desc = TextView(this).apply {
                text = getString(level.descRes())
                setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_BodySmall)
                val tc = android.util.TypedValue()
                theme.resolveAttribute(com.google.android.material.R.attr.colorOnSurfaceVariant, tc, true)
                setTextColor(tc.data)
                setPadding(0, (4 * density).toInt(), 0, 0)
            }
            textCol.addView(desc)
            row.addView(textCol)

            // Selection dot: filled accent circle when selected, hollow when not.
            val dot = View(this).apply {
                layoutParams = LinearLayout.LayoutParams(
                    (10 * density).toInt(), (10 * density).toInt()
                ).apply { marginStart = (12 * density).toInt() }
            }
            row.addView(dot)

            card.setOnClickListener {
                eqPrefs.setBassModeIntensity(level.ordinal)
                // First-ever preset tap with Bass Mode off is almost
                // certainly "I want to hear it" — switch it on for them.
                if (!eqPrefs.getBassModeEnabled()) {
                    eqPrefs.setBassModeEnabled(true)
                    paintPower()
                }
                applyToService()
                repaintSelection()
            }
            container.addView(card)
            presetViews.add(PresetUi(level, card, dot))
        }
        repaintSelection()
    }

    /** Selected card gets a 1.5dp white stroke + filled dot; others collapse both. */
    private fun repaintSelection() {
        val selected = eqPrefs.getBassModeIntensity()
        val density = resources.displayMetrics.density
        for (ui in presetViews) {
            val isSel = ui.intensity.ordinal == selected
            ui.card.strokeWidth = if (isSel) (1.5 * density).toInt() else 0
            ui.card.strokeColor = 0xFFFFFFFF.toInt()
            ui.dot.background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.OVAL
                if (isSel) setColor(0xFFFFFFFF.toInt()) else {
                    setColor(0x00000000)
                    setStroke((1.5 * density).toInt(), 0xFF777777.toInt())
                }
            }
        }
    }

    /** Pipeline-card power-button styling: dark rounded square, white when on. */
    private fun paintPower() {
        val on = eqPrefs.getBassModeEnabled()
        val density = resources.displayMetrics.density
        val iconColor = if (on) 0xFFFFFFFF.toInt() else 0xFF555555.toInt()
        val strokeColor = if (on) 0xFFFFFFFF.toInt() else 0xFF444444.toInt()
        val shape = android.graphics.drawable.GradientDrawable().apply {
            cornerRadius = 12 * density
            setColor(0xFF2A2A2A.toInt())
            setStroke((1 * density).toInt(), strokeColor)
        }
        val mask = android.graphics.drawable.GradientDrawable().apply {
            cornerRadius = 12 * density
            setColor(0xFFFFFFFF.toInt())
        }
        powerButton.background = android.graphics.drawable.RippleDrawable(
            android.content.res.ColorStateList.valueOf(0x33FFFFFF.toInt()), shape, mask
        )
        powerButton.imageTintList = android.content.res.ColorStateList.valueOf(iconColor)
    }

    /** Re-apply ping to the service: rebuilds the overlay + live DP bands. */
    private fun applyToService() {
        try {
            val intent = android.content.Intent(this, EqService::class.java)
                .setAction(EqService.ACTION_APPLY_BASS_MODE)
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                startForegroundService(intent)
            } else {
                startService(intent)
            }
        } catch (_: Throwable) { /* service may not be running — fine */ }
    }
}
