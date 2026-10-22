/* Karaoke separator DSP engine: the STFT/ISTFT, magnitude and mask math of SpleeterStream.kt,
 * in C with pffft (SIMD) so the audio thread spends its budget in vectorized code instead of
 * the naive Kotlin FFT, and the per-pass window assembly / mask application cross JNI as
 * direct buffers rather than Java arrays. Orchestration (batch adaptation, ONNX inference)
 * stays in Kotlin; counters and the batch-wait live here behind one mutex + condvar.
 *
 * The numerics mirror SpleeterStream.kt / KaraokeDsp.kt exactly:
 *   analysis: periodic hann × frame, unscaled forward rFFT (pffft REAL, N = frame);
 *   magnitudes: sqrt(re² + im²) of bins 0..modelBins-1;
 *   batching: window rows target-(window-adv)..target+adv-1, zero rows before stream start;
 *   masking: a = m − v (m = ring magnitude, v = vocals model output),
 *            mask = (a² + eps/2)/(v² + a² + eps) applied to the complex spectrum,
 *            bins modelBins..bins-1 reuse bin modelBins-1's mask;
 *   synthesis: Hermitian rebuild, 1/N-scaled inverse rFFT, hann × hann weighted overlap-add
 *   with per-sample w² normalization (exact WOLA), hop samples emitted per frame. */
#include <jni.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <spa/support/cpu.h>

#include "pffft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if defined(__aarch64__) || defined(__arm__)
#define EQ314_CPU_FLAG SPA_CPU_FLAG_NEON
#else
#define EQ314_CPU_FLAG SPA_CPU_FLAG_SSE
#endif

#define KN_STOPPED (-1)
#define KN_STALE 0

struct kn {
	int frame, hop, bins, model_bins, window, complex_ring, mag_ring, lag_limit;

	PFFFT_Setup *fft;
	float *hann;			/* [frame] */
	float *fft_io;			/* [frame], aligned */
	float *fft_work;		/* [frame], aligned */

	/* analysis state (audio thread only) */
	float *hist[2];			/* [frame] each */
	float *carry;			/* [hop*2] interleaved */
	int carry_frames;

	/* shared rings */
	float *re[2], *im[2];		/* [complex_ring * bins] each */
	float *mags;			/* [2 * mag_ring * model_bins] */

	/* WOLA synthesis (audio thread only); pos shared, channels pushed in lockstep */
	float *w_acc[2], *w_sum[2];	/* [frame] each */
	int w_pos;
	float *pop[2];			/* [hop] each, JNI staging */
	float *pop_out;			/* [hop*2] interleaved staging */

	/* model I/O exposed to Java as direct buffers */
	float *net_in;			/* [2 * window * model_bins] */
	float *net_vocals;		/* [2 * window * model_bins] */

	pthread_mutex_t mu;
	pthread_cond_t cv;
	int stft_count, masked_count, output_count, epoch, running, failure;
};

static void kn_free(struct kn *k)
{
	if (k == NULL)
		return;
	pffft_destroy_setup(k->fft);
	pffft_aligned_free(k->fft_io);
	pffft_aligned_free(k->fft_work);
	free(k->hann);
	for (int ch = 0; ch < 2; ch++) {
		free(k->hist[ch]);
		free(k->re[ch]);
		free(k->im[ch]);
		free(k->w_acc[ch]);
		free(k->w_sum[ch]);
		free(k->pop[ch]);
	}
	free(k->carry);
	free(k->pop_out);
	free(k->mags);
	free(k->net_in);
	free(k->net_vocals);
	pthread_mutex_destroy(&k->mu);
	pthread_cond_destroy(&k->cv);
	free(k);
}

/* Unpack pffft's ordered REAL forward output ([re0, reN/2, re1, im1, ...]) into the
 * frame's ring slot and the magnitude row; mirrors SpleeterStream.analyze. */
static void kn_analyze(struct kn *k, int ch, int frame)
{
	const float *f = k->fft_io;
	const int bins = k->bins;
	float *re = k->re[ch] + (size_t)(frame % k->complex_ring) * bins;
	float *im = k->im[ch] + (size_t)(frame % k->complex_ring) * bins;
	float *m = k->mags + ((size_t)ch * k->mag_ring + frame % k->mag_ring) * k->model_bins;

	re[0] = f[0];
	im[0] = 0.0f;
	for (int b = 1; b < bins - 1; b++) {
		re[b] = f[2 * b];
		im[b] = f[2 * b + 1];
	}
	re[bins - 1] = f[1];
	im[bins - 1] = 0.0f;
	for (int b = 0; b < k->model_bins; b++)
		m[b] = sqrtf(re[b] * re[b] + im[b] * im[b]);
}

static void kn_push_hop(struct kn *k)
{
	const int frame = k->stft_count;
	const int hop = k->hop;

	memmove(k->hist[0], k->hist[0] + hop, (k->frame - hop) * sizeof(float));
	memmove(k->hist[1], k->hist[1] + hop, (k->frame - hop) * sizeof(float));
	for (int i = 0; i < hop; i++) {
		k->hist[0][k->frame - hop + i] = k->carry[2 * i];
		k->hist[1][k->frame - hop + i] = k->carry[2 * i + 1];
	}

	for (int ch = 0; ch < 2; ch++) {
		for (int j = 0; j < k->frame; j++)
			k->fft_io[j] = k->hist[ch][j] * k->hann[j];
		pffft_transform_ordered(k->fft, k->fft_io, k->fft_io, k->fft_work, PFFFT_FORWARD);
		kn_analyze(k, ch, frame);
	}

	/* Publish only after the frame is written: the inference thread may assemble
	 * window rows (and apply masks) as soon as stft_count moves. */
	pthread_mutex_lock(&k->mu);
	if (k->failure || !k->running) {
		pthread_mutex_unlock(&k->mu);
		return;
	}
	k->stft_count = frame + 1;
	if (k->stft_count - k->masked_count > k->lag_limit)
		k->failure = 1;		/* TOO_SLOW */
	pthread_cond_broadcast(&k->cv);
	pthread_mutex_unlock(&k->mu);
}

/* Exact-WOLA synthesis of one masked spectrum into k->pop[ch]; mirrors WolaIstft.push. */
static void kn_synthesize(struct kn *k, int ch, int frame)
{
	const int bins = k->bins;
	const float *re = k->re[ch] + (size_t)(frame % k->complex_ring) * bins;
	const float *im = k->im[ch] + (size_t)(frame % k->complex_ring) * bins;
	float *f = k->fft_io;

	f[0] = re[0];
	f[1] = re[bins - 1];
	for (int b = 1; b < bins - 1; b++) {
		f[2 * b] = re[b];
		f[2 * b + 1] = im[b];
	}
	pffft_transform_ordered(k->fft, f, f, k->fft_work, PFFFT_BACKWARD);
	const float scale = 1.0f / k->frame;
	for (int j = 0; j < k->frame; j++)
		f[j] *= scale;

	float *acc = k->w_acc[ch];
	float *sum = k->w_sum[ch];
	const int off = k->w_pos * k->hop;
	for (int j = 0; j < k->frame; j++) {
		int idx = off + j;
		if (idx >= k->frame)
			idx -= k->frame;
		const float w = k->hann[j];
		acc[idx] += f[j] * w;
		sum[idx] += w * w;
	}
	for (int j = 0; j < k->hop; j++) {
		const int idx = off + j;
		k->pop[ch][j] = sum[idx] > 1e-9f ? acc[idx] / sum[idx] : 0.0f;
		acc[idx] = 0.0f;
		sum[idx] = 0.0f;
	}
}

JNIEXPORT jlong JNICALL
Java_com_bearinmind_equalizer314_audio_karaoke_KaraokeNative_create(JNIEnv *env, jclass clazz,
		jint frame, jint hop, jint modelBins, jint window, jint complexRing, jint magRing, jint lagLimit)
{
	static int cpu_selected;
	struct kn *k;
	(void)clazz;

	if (!cpu_selected) {
		pffft_select_cpu(EQ314_CPU_FLAG);
		cpu_selected = 1;
	}
	if (frame <= 0 || hop <= 0 || frame % hop != 0 || modelBins <= 0 ||
	    window <= 0 || complexRing <= 0 || magRing <= 0 || lagLimit <= 0)
		return 0;

	k = calloc(1, sizeof(*k));
	if (k == NULL)
		return 0;
	k->frame = frame;
	k->hop = hop;
	k->bins = frame / 2 + 1;
	k->model_bins = modelBins;
	k->window = window;
	k->complex_ring = complexRing;
	k->mag_ring = magRing;
	k->lag_limit = lagLimit;
	k->running = 1;

	if (pthread_mutex_init(&k->mu, NULL) != 0 || pthread_cond_init(&k->cv, NULL) != 0)
		goto fail;
	k->fft = pffft_new_setup(frame, PFFFT_REAL);
	k->fft_io = pffft_aligned_malloc(frame * sizeof(float));
	k->fft_work = pffft_aligned_malloc(frame * sizeof(float));
	k->hann = malloc(frame * sizeof(float));
	for (int ch = 0; ch < 2; ch++) {
		k->hist[ch] = calloc(frame, sizeof(float));
		k->re[ch] = malloc((size_t)complexRing * k->bins * sizeof(float));
		k->im[ch] = malloc((size_t)complexRing * k->bins * sizeof(float));
		k->w_acc[ch] = calloc(frame, sizeof(float));
		k->w_sum[ch] = calloc(frame, sizeof(float));
		k->pop[ch] = malloc(hop * sizeof(float));
	}
	k->carry = calloc(hop * 2, sizeof(float));
	k->pop_out = calloc(hop * 2, sizeof(float));
	k->mags = calloc((size_t)2 * magRing * modelBins, sizeof(float));
	k->net_in = calloc((size_t)2 * window * modelBins, sizeof(float));
	k->net_vocals = calloc((size_t)2 * window * modelBins, sizeof(float));
	if (k->fft == NULL || k->fft_io == NULL || k->fft_work == NULL || k->hann == NULL ||
	    k->mags == NULL || k->carry == NULL || k->pop_out == NULL || k->net_in == NULL || k->net_vocals == NULL)
		goto fail;
	for (int ch = 0; ch < 2; ch++)
		if (k->hist[ch] == NULL || k->re[ch] == NULL || k->im[ch] == NULL ||
		    k->w_acc[ch] == NULL || k->w_sum[ch] == NULL || k->pop[ch] == NULL)
			goto fail;
	for (int n = 0; n < frame; n++)		/* periodic hann, matches KaraokeDsp.hann */
		k->hann[n] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * n / frame));
	return (jlong)(intptr_t)k;
fail:
	kn_free(k);
	return 0;
}

JNIEXPORT jobject JNICALL
Java_com_bearinmind_equalizer314_audio_karaoke_KaraokeNative_netInBuffer(JNIEnv *env, jclass clazz, jlong handle)
{
	struct kn *k = (struct kn *)(intptr_t)handle;
	(void)clazz;
	return k ? (*env)->NewDirectByteBuffer(env, k->net_in,
		(jlong)((size_t)2 * k->window * k->model_bins * sizeof(float))) : NULL;
}

JNIEXPORT jobject JNICALL
Java_com_bearinmind_equalizer314_audio_karaoke_KaraokeNative_netVocalsBuffer(JNIEnv *env, jclass clazz, jlong handle)
{
	struct kn *k = (struct kn *)(intptr_t)handle;
	(void)clazz;
	return k ? (*env)->NewDirectByteBuffer(env, k->net_vocals,
		(jlong)((size_t)2 * k->window * k->model_bins * sizeof(float))) : NULL;
}

/* Audio thread: interleaved stereo samples; returns the new stft count, or -1 on failure. */
JNIEXPORT jint JNICALL
Java_com_bearinmind_equalizer314_audio_karaoke_KaraokeNative_push(JNIEnv *env, jclass clazz, jlong handle,
		jfloatArray pcm, jint frames)
{
	struct kn *k = (struct kn *)(intptr_t)handle;
	jint off = 0;
	(void)clazz;
	if (k == NULL)
		return -1;
	while (off < frames) {
		const int take = k->hop - k->carry_frames < frames - off ? k->hop - k->carry_frames : frames - off;
		(*env)->GetFloatArrayRegion(env, pcm, off * 2, take * 2, k->carry + k->carry_frames * 2);
		k->carry_frames += take;
		off += take;
		if (k->carry_frames == k->hop) {
			k->carry_frames = 0;
			kn_push_hop(k);
		}
	}
	pthread_mutex_lock(&k->mu);
	const int fail = k->failure;
	const int count = k->stft_count;
	pthread_mutex_unlock(&k->mu);
	return fail ? -1 : count;
}

/* Audio thread: drains the next masked frame into [out] (hop*2 interleaved);
 * 1 emitted, 0 none ready, -1 on failure. */
JNIEXPORT jint JNICALL
Java_com_bearinmind_equalizer314_audio_karaoke_KaraokeNative_popOutput(JNIEnv *env, jclass clazz, jlong handle,
		jfloatArray out)
{
	struct kn *k = (struct kn *)(intptr_t)handle;
	(void)clazz;
	if (k == NULL)
		return -1;
	pthread_mutex_lock(&k->mu);
	if (k->failure || k->output_count >= k->masked_count) {
		const int fail = k->failure;
		pthread_mutex_unlock(&k->mu);
		return fail ? -1 : 0;
	}
	const int f = k->output_count++;
	pthread_mutex_unlock(&k->mu);
	/* Safe unlocked: applyMasks only ever touches frames >= masked_count (under mu),
	 * and f < masked_count was observed under mu; both channels share w_pos. */
	kn_synthesize(k, 0, f);
	kn_synthesize(k, 1, f);
	k->w_pos = (k->w_pos + 1) % (k->frame / k->hop);
	for (int i = 0; i < k->hop; i++) {
		k->pop_out[2 * i] = k->pop[0][i];
		k->pop_out[2 * i + 1] = k->pop[1][i];
	}
	(*env)->SetFloatArrayRegion(env, out, 0, k->hop * 2, k->pop_out);
	return 1;
}

/* Inference thread: waits until a full batch of [advance] fresh frames is captured.
 * Returns the batch target (masked count), KN_STOPPED when stopped/failed,
 * or -2 on timeout (caller re-checks and retries). */
JNIEXPORT jint JNICALL
Java_com_bearinmind_equalizer314_audio_karaoke_KaraokeNative_waitBatch(JNIEnv *env, jclass clazz, jlong handle,
		jint advance, jint timeoutMs)
{
	struct kn *k = (struct kn *)(intptr_t)handle;
	jint target;
	struct timespec ts;
	(void)clazz;
	(void)env;
	if (k == NULL)
		return KN_STOPPED;
	clock_gettime(CLOCK_REALTIME, &ts);
	ts.tv_sec += timeoutMs / 1000;
	ts.tv_nsec += (long)(timeoutMs % 1000) * 1000000L;
	if (ts.tv_nsec >= 1000000000L) {
		ts.tv_sec += 1;
		ts.tv_nsec -= 1000000000L;
	}
	pthread_mutex_lock(&k->mu);
	while (k->running && !k->failure && k->stft_count < k->masked_count + advance) {
		if (pthread_cond_timedwait(&k->cv, &k->mu, &ts) == ETIMEDOUT)
			break;
	}
	if (!k->running || k->failure) {
		target = KN_STOPPED;
	} else if (k->stft_count < k->masked_count + advance) {
		target = -2;
	} else {
		target = k->masked_count;
	}
	pthread_mutex_unlock(&k->mu);
	return target;
}

JNIEXPORT jint JNICALL
Java_com_bearinmind_equalizer314_audio_karaoke_KaraokeNative_epoch(JNIEnv *env, jclass clazz, jlong handle)
{
	struct kn *k = (struct kn *)(intptr_t)handle;
	int e;
	(void)clazz;
	(void)env;
	if (k == NULL)
		return 0;
	pthread_mutex_lock(&k->mu);
	e = k->epoch;
	pthread_mutex_unlock(&k->mu);
	return e;
}

JNIEXPORT jint JNICALL
Java_com_bearinmind_equalizer314_audio_karaoke_KaraokeNative_lag(JNIEnv *env, jclass clazz, jlong handle)
{
	struct kn *k = (struct kn *)(intptr_t)handle;
	int lag;
	(void)clazz;
	(void)env;
	if (k == NULL)
		return 0;
	pthread_mutex_lock(&k->mu);
	lag = k->stft_count - k->masked_count;
	pthread_mutex_unlock(&k->mu);
	return lag;
}

JNIEXPORT jint JNICALL
Java_com_bearinmind_equalizer314_audio_karaoke_KaraokeNative_failure(JNIEnv *env, jclass clazz, jlong handle)
{
	struct kn *k = (struct kn *)(intptr_t)handle;
	int f;
	(void)clazz;
	(void)env;
	if (k == NULL)
		return 0;
	pthread_mutex_lock(&k->mu);
	f = k->failure;
	pthread_mutex_unlock(&k->mu);
	return f;
}

/* Inference thread: assembles [2, 1, window, model_bins] magnitudes — window rows
 * target-(window-adv)..target+adv-1, zero rows before the stream start — into net_in. */
JNIEXPORT void JNICALL
Java_com_bearinmind_equalizer314_audio_karaoke_KaraokeNative_prepareBatch(JNIEnv *env, jclass clazz, jlong handle,
		jint target, jint adv)
{
	struct kn *k = (struct kn *)(intptr_t)handle;
	const int mb = k->model_bins;
	(void)clazz;
	(void)env;
	if (k == NULL)
		return;
	memset(k->net_in, 0, (size_t)2 * k->window * mb * sizeof(float));
	for (int ch = 0; ch < 2; ch++) {
		const float *m = k->mags + (size_t)ch * k->mag_ring * mb;
		float *dst = k->net_in + (size_t)ch * k->window * mb;
		for (int w = 0; w < k->window; w++) {
			const int src = target - (k->window - adv) + w;
			if (src < 0)
				continue;
			memcpy(dst + (size_t)w * mb, m + (size_t)(src % k->mag_ring) * mb, mb * sizeof(float));
		}
	}
}

/* Inference thread: applies the residual accompaniment mask to frames [target, target+adv)
 * using net_vocals and the ring's mixture magnitudes; advances masked_count.
 * Returns the new masked count, KN_STALE when a reset superseded the batch
 * (keep looping), KN_STOPPED when stopped/failed. */
JNIEXPORT jint JNICALL
Java_com_bearinmind_equalizer314_audio_karaoke_KaraokeNative_applyMasks(JNIEnv *env, jclass clazz, jlong handle,
		jint target, jint adv, jint myEpoch)
{
	struct kn *k = (struct kn *)(intptr_t)handle;
	const int mb = k->model_bins;
	jint ret;
	(void)clazz;
	(void)env;
	if (k == NULL)
		return KN_STOPPED;
	pthread_mutex_lock(&k->mu);
	if (!k->running || k->failure) {
		ret = KN_STOPPED;
	} else if (k->epoch != myEpoch) {
		ret = KN_STALE;
	} else {
		for (int j = 0; j < adv; j++) {
			const int f = target + j;
			const int slot = f % k->complex_ring;
			const int w = k->window - adv + j;
			for (int ch = 0; ch < 2; ch++) {
				const float *v = k->net_vocals + ((size_t)ch * k->window + w) * mb;
				const float *m = k->mags + ((size_t)ch * k->mag_ring + f % k->mag_ring) * mb;
				float *re = k->re[ch] + (size_t)slot * k->bins;
				float *im = k->im[ch] + (size_t)slot * k->bins;
				float last = 1.0f;
				for (int b = 0; b < mb; b++) {
					const float vs = v[b] * v[b];
					const float a = m[b] - v[b];
					const float as2 = a * a;
					const float mask = (as2 + 5e-11f) / (vs + as2 + 1e-10f);
					re[b] *= mask;
					im[b] *= mask;
					last = mask;
				}
				for (int b = mb; b < k->bins; b++) {
					re[b] *= last;
					im[b] *= last;
				}
			}
		}
		k->masked_count = target + adv;
		pthread_cond_broadcast(&k->cv);
		ret = k->masked_count;
	}
	pthread_mutex_unlock(&k->mu);
	return ret;
}

/* Audio thread (silence pause): drops all pending state; in-flight inference is discarded
 * via the epoch. Also clears the analysis carry and the WOLA accumulators. */
JNIEXPORT void JNICALL
Java_com_bearinmind_equalizer314_audio_karaoke_KaraokeNative_reset(JNIEnv *env, jclass clazz, jlong handle)
{
	struct kn *k = (struct kn *)(intptr_t)handle;
	(void)clazz;
	(void)env;
	if (k == NULL)
		return;
	pthread_mutex_lock(&k->mu);
	k->epoch++;
	k->stft_count = 0;
	k->masked_count = 0;
	k->output_count = 0;
	pthread_cond_broadcast(&k->cv);
	pthread_mutex_unlock(&k->mu);
	k->carry_frames = 0;
	k->w_pos = 0;
	for (int ch = 0; ch < 2; ch++) {
		memset(k->w_acc[ch], 0, k->frame * sizeof(float));
		memset(k->w_sum[ch], 0, k->frame * sizeof(float));
	}
}

JNIEXPORT void JNICALL
Java_com_bearinmind_equalizer314_audio_karaoke_KaraokeNative_stop(JNIEnv *env, jclass clazz, jlong handle)
{
	struct kn *k = (struct kn *)(intptr_t)handle;
	(void)clazz;
	(void)env;
	if (k == NULL)
		return;
	pthread_mutex_lock(&k->mu);
	k->running = 0;
	pthread_cond_broadcast(&k->cv);
	pthread_mutex_unlock(&k->mu);
}

JNIEXPORT void JNICALL
Java_com_bearinmind_equalizer314_audio_karaoke_KaraokeNative_destroy(JNIEnv *env, jclass clazz, jlong handle)
{
	(void)clazz;
	(void)env;
	kn_free((struct kn *)(intptr_t)handle);
}
