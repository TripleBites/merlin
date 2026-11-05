#include "core/tern.h"
#include "core/rng.h"
#include "core/util.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* a_i * b_j has variance RNG_NOISE_STD^4; divide by RNG_NOISE_STD^2 so each
 * entry of the rank-one perturbation has std 1 (times sigma). */
#define NOISE_VAR (RNG_NOISE_STD * RNG_NOISE_STD)

void tmat_init(TMat *w, const char *name, int id, int rows, int cols) {
    w->name = name;
    w->id = id;
    w->rows = rows;
    w->cols = cols;
    size_t n = (size_t)rows * (size_t)cols;
    w->latent = xcalloc(n, 1);
    w->trit = xcalloc(n, 1);
    w->scale = xcalloc((size_t)rows, sizeof(float));
}

void tmat_free(TMat *w) {
    free(w->latent);
    free(w->trit);
    free(w->scale);
    memset(w, 0, sizeof *w);
}

void tmat_randomize(TMat *w, uint64_t seed, float latent_std) {
    uint64_t key = rng_key(seed, RNG_TAG_INIT, (uint64_t)w->id, 0);
    size_t n = (size_t)w->rows * (size_t)w->cols;
    float k = latent_std / RNG_NOISE_STD;
    for (size_t i = 0; i < n; i++)
        w->latent[i] = sat_i8((int)lrintf(k * (float)rng_noise_i8(key, i)));
    tmat_requantize(w);
}

size_t tmat_requantize(TMat *w) {
    size_t changed = 0;
    for (int i = 0; i < w->rows; i++) {
        const int8_t *l = w->latent + (size_t)i * w->cols;
        int8_t *t = w->trit + (size_t)i * w->cols;

        int64_t sum = 0;
        for (int j = 0; j < w->cols; j++) sum += abs(l[j]);
        float gamma = (float)sum / (float)w->cols;
        if (gamma < 1.0f) gamma = 1.0f;   /* all-zero row: avoid divide by 0 */

        for (int j = 0; j < w->cols; j++) {
            int q = clampi((int)lrintf((float)l[j] / gamma), -1, 1);
            changed += (t[j] != q);
            t[j] = (int8_t)q;
        }
        w->scale[i] = gamma / LATENT_ONE;
    }
    return changed;
}

void tmat_add_stochastic(TMat *w, const float *delta, uint64_t key) {
    size_t n = (size_t)w->rows * (size_t)w->cols;
    for (size_t i = 0; i < n; i++) {
        float d = delta[i];
        float fl = floorf(d);
        int step = (int)fl + (rng_uniform(key, i) < (d - fl) ? 1 : 0);
        w->latent[i] = sat_i8((int)w->latent[i] + step);
    }
}

void tmat_trit_counts(const TMat *w, size_t *neg, size_t *zero, size_t *pos) {
    size_t n = (size_t)w->rows * (size_t)w->cols, c[3] = {0, 0, 0};
    for (size_t i = 0; i < n; i++) c[w->trit[i] + 1]++;
    *neg = c[0]; *zero = c[1]; *pos = c[2];
}

int32_t dot_i8(const int8_t *a, const int8_t *b, int n) {
    int32_t acc = 0;
    for (int i = 0; i < n; i++) acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
}

void tmat_matvec(const TMat *w, const int8_t *x, float xs, const Perturb *p, float *y) {
    for (int i = 0; i < w->rows; i++) {
        int32_t acc = dot_i8(w->trit + (size_t)i * w->cols, x, w->cols);
        y[i] = w->scale[i] * xs * (float)acc;
    }
    if (p) {
        /* (W + c a b^T) x = W x + c a (b . x): one extra dot product per matrix. */
        const RankOne *r = &p->r1[w->id];
        float c = p->sigma * xs * (float)dot_i8(r->b, x, w->cols) / (NOISE_VAR * LATENT_ONE);
        for (int i = 0; i < w->rows; i++) y[i] += c * (float)r->a[i];
    }
}

void tmat_row(const TMat *w, int r, const Perturb *p, float *out) {
    const int8_t *t = w->trit + (size_t)r * w->cols;
    float s = w->scale[r];
    for (int j = 0; j < w->cols; j++) out[j] = s * (float)t[j];
    if (p) {
        const RankOne *ro = &p->r1[w->id];
        float c = p->sigma * (float)ro->a[r] / (NOISE_VAR * LATENT_ONE);
        for (int j = 0; j < w->cols; j++) out[j] += c * (float)ro->b[j];
    }
}

float quant_i8(const float *x, int n, int8_t *q) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > m) m = a; }
    float s = m > 0.0f ? m / 127.0f : 1.0f;
    float inv = 1.0f / s;
    for (int i = 0; i < n; i++) q[i] = sat_i8((int)lrintf(x[i] * inv));
    return s;
}

void tern_pack(const int8_t *trit, size_t n, uint8_t *out) {
    memset(out, 0, tern_packed_bytes(n));
    for (size_t i = 0; i < n; i++) {
        uint8_t code = trit[i] == 1 ? 1u : (trit[i] == -1 ? 2u : 0u);
        out[i >> 2] |= (uint8_t)(code << ((i & 3) * 2));
    }
}

void tern_unpack(const uint8_t *in, size_t n, int8_t *trit) {
    static const int8_t lut[4] = {0, 1, -1, 0};
    for (size_t i = 0; i < n; i++)
        trit[i] = lut[(in[i >> 2] >> ((i & 3) * 2)) & 3u];
}
