package com.bearinmind.equalizer314.dsp

import org.junit.Assert.assertTrue
import org.junit.Test

/** Bass Mode overlay: soft-sub response math + conversion gains.
 *  Curve is tuned for a BT speaker with a 6.5" woofer / 130 W RMS:
 *  peak where the driver delivers (~57 Hz), guarded below its tuning,
 *  midbass tucked so bass reads "soft", not "punchy". */
class BassModeOverlayTest {

    private fun responseAt(eq: ParametricEqualizer, f: Float): Float = eq.getFrequencyResponse(f)

    @Test
    fun `soft sub curve peaks in the woofer band`() {
        val bass = ParametricToDpConverter.bassModeCurve()
        // Weight concentrated where a 6.5" woofer actually delivers...
        val db60 = responseAt(bass, 60f)
        assertTrue("60 Hz response $db60 dB should be ≥ +6.5 dB", db60 >= 6.5f)
        // ...gently rolled off below tuning (subsonic guard, 6 dB/oct) so
        // the amp's 130 W isn't spent unloading the driver:
        val db30 = responseAt(bass, 30f)
        assertTrue("30 Hz response $db30 dB should be ≤ +4.5 dB (subsonic guarded)", db30 <= 4.5f)
        val db10 = responseAt(bass, 10f)
        assertTrue("10 Hz response $db10 dB should be ≤ -4 dB", db10 <= -4f)
        // ...soft midbass: no punch hump at 150-300 Hz...
        val db220 = responseAt(bass, 220f)
        assertTrue("220 Hz response $db220 dB should be ≤ +1 dB (soft, not punchy)", db220 <= 1f)
        // ...and stay out of the mids/highs.
        assertTrue("1 kHz response ${responseAt(bass, 1000f)} dB must be near flat",
            kotlin.math.abs(responseAt(bass, 1000f)) < 0.5f)
        assertTrue("10 kHz response ${responseAt(bass, 10000f)} dB must be flat",
            kotlin.math.abs(responseAt(bass, 10000f)) < 0.5f)
    }

    @Test
    fun `overlay lifts converted bass band gains`() {
        ParametricToDpConverter.setNumBands(128)
        ParametricToDpConverter.deviceSampleRateHz = 48000f
        ParametricToDConverterSetup()

        val flat = ParametricEqualizer().apply { clearBands(); isEnabled = true }

        ParametricToDpConverter.bassBoostEq = null
        val without = ParametricToDpConverter.convertFeatureAware(flat)

        ParametricToDpConverter.bassBoostEq = ParametricToDpConverter.bassModeCurve()
        val with = ParametricToDpConverter.convertFeatureAware(flat)
        ParametricToDpConverter.bassBoostEq = null

        // Sample the staircase around a target: the gain difference must be ≈ the curve response.
        fun stairDbAt(r: ParametricToDpConverter.ConvertedBands, target: Float): Float {
            var best = 0
            var bestDist = Float.MAX_VALUE
            for (i in r.cutoffs.indices) {
                val d = kotlin.math.abs(r.cutoffs[i] - target)
                if (d < bestDist) { bestDist = d; best = i }
            }
            return r.gains[best]
        }
        val subLift = stairDbAt(with, 60f) - stairDbAt(without, 60f)
        assertTrue("60 Hz stair lift $subLift dB should be ≥ +5.5 dB", subLift >= 5.5f)

        val deepLift = stairDbAt(with, 40f) - stairDbAt(without, 40f)
        assertTrue("40 Hz stair lift $deepLift dB should be ≥ +4.5 dB", deepLift >= 4.5f)

        // Soft midbass: the 150-300 Hz region must NOT get a punch hump.
        val midBassLift = stairDbAt(with, 220f) - stairDbAt(without, 220f)
        assertTrue("220 Hz stair lift $midBassLift dB should be within ±1 dB (no punch hump)",
            kotlin.math.abs(midBassLift) <= 1f)

        // Mid/high stairs must stay flat.
        val midLift = stairDbAt(with, 2000f) - stairDbAt(without, 2000f)
        assertTrue("mid stair lift $midLift dB should be < 1 dB", kotlin.math.abs(midLift) < 1f)

        // Regression guard for respond(): reading the bass EQ's own response directly
        // (no overlay double-count) still shows the strong low end.
        val selfDb = ParametricToDpConverter.run {
            bassBoostEq = bassModeCurve()
            val r = bassBoostEq!!.getFrequencyResponse(60f)
            bassBoostEq = null
            r
        }
        assertTrue("direct bass curve read $selfDb dB should be ≥ +6.5 dB", selfDb >= 6.5f)
    }

    @Test
    fun `intensity presets scale the sub boost and keep the shape soft`() {
        val levels = ParametricToDpConverter.BassIntensity.values()
        // Monotonically more sub weight at the woofer band...
        var prev60 = Float.NEGATIVE_INFINITY
        for (level in levels) {
            val eq = ParametricToDpConverter.bassModeCurve(level)
            val db60 = responseAt(eq, 60f)
            assertTrue("$level 60 Hz response $db60 dB must exceed the previous level's $prev60 dB",
                db60 > prev60)
            prev60 = db60
            // ...while every level keeps the soft shape: no midbass punch
            // hump, flat highs, subsonic rolloff below tuning (proportional
            // to the level's own peak — the fixed 42 Hz guard attenuates
            // less in absolute dB as the boost scales).
            assertTrue("$level 220 Hz ${responseAt(eq, 220f)} dB must stay ≤ +1 dB",
                responseAt(eq, 220f) <= 1f)
            assertTrue("$level 1 kHz ${responseAt(eq, 1000f)} dB must stay flat",
                kotlin.math.abs(responseAt(eq, 1000f)) < 0.5f)
            assertTrue("$level 10 Hz ${responseAt(eq, 10f)} dB must sit ≥ 8 dB under its 60 Hz peak",
                responseAt(eq, 10f) <= responseAt(eq, 60f) - 8f)
        }
        // The persisted default (BALANCED) is the tuned curve.
        assertTrue("BALANCED 60 Hz ${responseAt(ParametricToDpConverter.bassModeCurve(), 60f)} dB should be ≥ +6.5 dB",
            responseAt(ParametricToDpConverter.bassModeCurve(), 60f) >= 6.5f)
    }

    private fun ParametricToDConverterSetup() {
        // Keep engine geometry at defaults; conversion is deterministic given fs + frame.
        ParametricToDpConverter.frameDurationMs = 80f
    }
}
