#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "core/ckpt.h"
#include "core/rng.h"
#include "core/tern.h"

#define NOISE_VAR (RNG_NOISE_STD * RNG_NOISE_STD)

/* Dense reference: real W (+ explicit rank-one E) times dequantised x. */
static void ref_matvec(const TMat *w, const int8_t *x, float xs, const Perturb *p, float *y) {
    for (int i = 0; i < w->rows; i++) {
        double acc = 0;
        for (int j = 0; j < w->cols; j++) {
            double wij = (double)w->scale[i] * w->trit[i * w->cols + j];
            if (p) wij += (double)p->sigma * p->r1[w->id].a[i] * p->r1[w->id].b[j] / (NOISE_VAR * LATENT_ONE);
            acc += wij * x[j] * xs;
        }
        y[i] = (float)acc;
    }
}

int main(void) {
    /* ---- packing round trip ---- */
    int8_t t[1001], back[1001];
    uint8_t packed[tern_packed_bytes(1001)];
    uint64_t k = rng_key(1, 2, 3, 4);
    for (int i = 0; i < 1001; i++) t[i] = (int8_t)((int)(rng_u64(k, (uint64_t)i) % 3) - 1);
    tern_pack(t, 1001, packed);
    tern_unpack(packed, 1001, back);
    CHECK(memcmp(t, back, sizeof t) == 0);
    CHECK(sizeof packed == 251);

    /* ---- ternarize: all trits valid, scale = absmean / LATENT_ONE ---- */
    TMat w;
    tmat_init(&w, "w", 0, 17, 33);
    tmat_randomize(&w, 9, 16.0f);
    int valid = 1;
    for (int i = 0; i < 17 * 33; i++) if (w.trit[i] < -1 || w.trit[i] > 1) valid = 0;
    CHECK(valid);
    double s0 = 0;
    for (int j = 0; j < 33; j++) s0 += abs(w.latent[j]);
    CHECK_NEAR(w.scale[0], (s0 / 33) / LATENT_ONE, 1e-5);
    CHECK(tmat_requantize(&w) == 0);   /* idempotent */

    /* ---- matvec matches dense reference, with and without perturbation ---- */
    int8_t x[33], a[17], b[33];
    for (int j = 0; j < 33; j++) x[j] = (int8_t)((int)(rng_u64(k, 5000 + j) % 255) - 127);
    for (int i = 0; i < 17; i++) a[i] = rng_noise_i8(k, 9000 + i);
    for (int j = 0; j < 33; j++) b[j] = rng_noise_i8(k, 9100 + j);
    RankOne r1 = {a, b};
    Perturb p = {-3.5f, &r1};
    float y[17], yr[17];
    tmat_matvec(&w, x, 0.01f, NULL, y);
    ref_matvec(&w, x, 0.01f, NULL, yr);
    float err = 0;
    for (int i = 0; i < 17; i++) err = fmaxf(err, fabsf(y[i] - yr[i]));
    CHECK_NEAR(err, 0.0, 1e-4);
    tmat_matvec(&w, x, 0.01f, &p, y);
    ref_matvec(&w, x, 0.01f, &p, yr);
    err = 0;
    for (int i = 0; i < 17; i++) err = fmaxf(err, fabsf(y[i] - yr[i]));
    CHECK_NEAR(err, 0.0, 1e-4);

    /* ---- row lookup with perturbation == that row of W + E ---- */
    float row[33];
    tmat_row(&w, 5, &p, row);
    CHECK_NEAR(row[7], w.scale[5] * w.trit[5 * 33 + 7] + p.sigma * a[5] * b[7] / (NOISE_VAR * LATENT_ONE), 1e-6);

    /* ---- activation quant: max maps to 127 and dequant is close ---- */
    float v[5] = {0.1f, -2.0f, 0.5f, 1.0f, 0.0f};
    int8_t q[5];
    float qs = quant_i8(v, 5, q);
    CHECK(q[1] == -127);
    CHECK_NEAR(q[2] * qs, 0.5, qs);

    /* ---- stochastic rounding: adding 0.25 many times moves ~25 per 100 ---- */
    TMat z;
    tmat_init(&z, "z", 0, 1, 1000);
    float *d = calloc(1000, sizeof(float));
    for (int i = 0; i < 1000; i++) d[i] = 0.25f;
    for (int s = 0; s < 100; s++) tmat_add_stochastic(&z, d, rng_key(3, 3, (uint64_t)s, 0));
    double mean = 0;
    for (int i = 0; i < 1000; i++) mean += z.latent[i];
    CHECK_NEAR(mean / 1000, 25.0, 1.0);

    /* ---- checkpoint round trip ---- */
    TMat *mats[1] = {&w};
    ParamSet ps = {mats, 1};
    ckpt_save("test_ckpt.bin", "test", &ps);
    int8_t saved[17 * 33];
    memcpy(saved, w.latent, sizeof saved);
    memset(w.latent, 0, sizeof saved);
    ckpt_load("test_ckpt.bin", "test", &ps);
    CHECK(memcmp(saved, w.latent, sizeof saved) == 0);
    remove("test_ckpt.bin");

    free(d);
    tmat_free(&z);
    tmat_free(&w);
    return CHECK_DONE();
}
