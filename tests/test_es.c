/*
 * ES sanity check: recover a hidden ternary matrix using only fitness.
 * Fitness = -|| W x - T x ||^2 over fixed probe vectors x, where T is the
 * target. No gradients anywhere: if this passes, the ES loop works.
 */
#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "core/rng.h"
#include "core/tern.h"
#include "es/es.h"

#define R 16
#define C 32
#define PROBES 64

typedef struct {
    TMat *w;
    TMat *target;
    int8_t x[PROBES][C];
} Ctx;

static float fit(const Perturb *p, int thread, void *vctx) {
    Ctx *c = vctx;
    float y[R], t[R];
    double err = 0;
    for (int k = 0; k < PROBES; k++) {
        tmat_matvec(c->w, c->x[k], 1.0f / 127, p, y);
        tmat_matvec(c->target, c->x[k], 1.0f / 127, NULL, t);
        for (int i = 0; i < R; i++) err += (double)(y[i] - t[i]) * (y[i] - t[i]);
    }
    return (float)(-err / PROBES);
}

static int agree(const TMat *a, const TMat *b) {
    int n = 0;
    for (int i = 0; i < R * C; i++) n += a->trit[i] == b->trit[i];
    return n;
}

int main(void) {
    TMat w, target;
    tmat_init(&w, "w", 0, R, C);
    tmat_init(&target, "target", 0, R, C);
    tmat_randomize(&w, 1, 16.0f);
    tmat_randomize(&target, 2, 16.0f);

    Ctx c = {.w = &w, .target = &target};
    uint64_t k = rng_key(5, 5, 5, 5);
    for (int p = 0; p < PROBES; p++)
        for (int j = 0; j < C; j++) c.x[p][j] = (int8_t)((int)(rng_u64(k, (uint64_t)(p * C + j)) % 255) - 127);

    TMat *mats[1] = {&w};
    ParamSet ps = {mats, 1};
    ESConfig cfg = {.seed = 7, .pop = 64, .sigma = 4.0f, .lr = 6.0f};

    float f0 = fit(NULL, 0, &c);
    int a0 = agree(&w, &target);
    for (int step = 0; step < 300; step++) es_step(&ps, &cfg, (uint64_t)step, fit, &c, NULL);
    float f1 = fit(NULL, 0, &c);
    int a1 = agree(&w, &target);

    printf("fitness %.4f -> %.4f, trits matching target %d -> %d of %d\n", f0, f1, a0, a1, R * C);
    CHECK(f1 > 0.5f * f0);              /* error at least halved (fitness is negative) */
    CHECK(a1 > a0 + (R * C) / 4);       /* recovered a good chunk of the target */

    /* Determinism: same seed, same start -> bit-identical latent. */
    TMat w2;
    tmat_init(&w2, "w", 0, R, C);
    tmat_randomize(&w2, 1, 16.0f);
    TMat *mats2[1] = {&w2};
    ParamSet ps2 = {mats2, 1};
    Ctx c2 = c;
    c2.w = &w2;
    TMat w3;
    tmat_init(&w3, "w", 0, R, C);
    tmat_randomize(&w3, 1, 16.0f);
    TMat *mats3[1] = {&w3};
    ParamSet ps3 = {mats3, 1};
    Ctx c3 = c;
    c3.w = &w3;
    for (int step = 0; step < 5; step++) {
        es_step(&ps2, &cfg, (uint64_t)step, fit, &c2, NULL);
        es_step(&ps3, &cfg, (uint64_t)step, fit, &c3, NULL);
    }
    CHECK(memcmp(w2.latent, w3.latent, R * C) == 0);

    tmat_free(&w); tmat_free(&w2); tmat_free(&w3); tmat_free(&target);
    return CHECK_DONE();
}
