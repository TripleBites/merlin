#include "es/es.h"
#include "core/rng.h"
#include "core/util.h"

#include <stdlib.h>
#include <string.h>

#ifdef MERLIN_HAVE_OPENMP
#include <omp.h>
#endif

#define NOISE_VAR (RNG_NOISE_STD * RNG_NOISE_STD)

int es_max_threads(void) {
#ifdef MERLIN_HAVE_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static int thread_id(void) {
#ifdef MERLIN_HAVE_OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

void es_rank_one(uint64_t seed, uint64_t step, int pair, const TMat *w,
                 int8_t *a, int8_t *b) {
    uint64_t base = rng_key(seed, RNG_TAG_NOISE, step, (uint64_t)pair);
    uint64_t ka = rng_key(base, (uint64_t)w->id, 'a', 0);
    uint64_t kb = rng_key(base, (uint64_t)w->id, 'b', 0);
    for (int i = 0; i < w->rows; i++) a[i] = rng_noise_i8(ka, (uint64_t)i);
    for (int j = 0; j < w->cols; j++) b[j] = rng_noise_i8(kb, (uint64_t)j);
}

/* Scratch for one thread: a/b vectors for every matrix + the RankOne table. */
typedef struct {
    int8_t  *buf;
    RankOne *r1;
} NoiseScratch;

static void scratch_init(NoiseScratch *s, const ParamSet *ps) {
    size_t total = 0;
    for (int m = 0; m < ps->n_mats; m++) total += (size_t)ps->mats[m]->rows + ps->mats[m]->cols;
    s->buf = xmalloc(total);
    s->r1 = xcalloc((size_t)ps->n_mats, sizeof(RankOne));
    int8_t *p = s->buf;
    for (int m = 0; m < ps->n_mats; m++) {
        const TMat *w = ps->mats[m];
        if (w->id != m) die("ParamSet: matrix '%s' has id %d, expected %d", w->name, w->id, m);
        s->r1[m].a = p; p += w->rows;
        s->r1[m].b = p; p += w->cols;
    }
}

static void scratch_fill(NoiseScratch *s, const ParamSet *ps, uint64_t seed, uint64_t step, int pair) {
    for (int m = 0; m < ps->n_mats; m++)
        es_rank_one(seed, step, pair, ps->mats[m], (int8_t *)s->r1[m].a, (int8_t *)s->r1[m].b);
}

static void scratch_free(NoiseScratch *s) {
    free(s->buf);
    free(s->r1);
}

/* Centered ranks: worst -> -0.5, best -> +0.5. Robust to fitness scale. */
static const float *g_sort_fit;
static int cmp_idx(const void *x, const void *y) {
    float a = g_sort_fit[*(const int *)x], b = g_sort_fit[*(const int *)y];
    return (a > b) - (a < b);
}
static void centered_ranks(const float *fit, int n, float *u) {
    int *idx = xmalloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;
    g_sort_fit = fit;               /* es_step is not re-entrant; fine for now */
    qsort(idx, (size_t)n, sizeof(int), cmp_idx);
    for (int r = 0; r < n; r++) u[idx[r]] = (n > 1) ? (float)r / (float)(n - 1) - 0.5f : 0.0f;
    free(idx);
}

void es_step(ParamSet *ps, const ESConfig *cfg, uint64_t step,
             FitnessFn fn, void *ctx, ESStats *stats) {
    if (cfg->pop < 2 || (cfg->pop & 1)) die("ES population must be even and >= 2 (got %d)", cfg->pop);
    const int pop = cfg->pop, pairs = pop / 2;

    float *fit = xmalloc((size_t)pop * sizeof(float));
    float *u = xmalloc((size_t)pop * sizeof(float));

    /* ---- 1-2. evaluate the population ---- */
    int nthreads = es_max_threads();
    NoiseScratch *scratch = xcalloc((size_t)nthreads, sizeof(NoiseScratch));
    for (int t = 0; t < nthreads; t++) scratch_init(&scratch[t], ps);

#ifdef MERLIN_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int p = 0; p < pairs; p++) {
        int t = thread_id();
        scratch_fill(&scratch[t], ps, cfg->seed, step, p);
        Perturb plus = {+cfg->sigma, scratch[t].r1};
        Perturb minus = {-cfg->sigma, scratch[t].r1};
        fit[2 * p] = fn(&plus, t, ctx);
        fit[2 * p + 1] = fn(&minus, t, ctx);
    }

    /* ---- 3. fitness shaping ---- */
    centered_ranks(fit, pop, u);

    /* ---- 4-5. update each matrix ---- */
    size_t flips = 0, n_weights = 0;
    float *wpair = xmalloc((size_t)pairs * sizeof(float));
    for (int p = 0; p < pairs; p++) wpair[p] = u[2 * p] - u[2 * p + 1];

    for (int m = 0; m < ps->n_mats; m++) {
        TMat *w = ps->mats[m];
        size_t n = (size_t)w->rows * w->cols;
        float *delta = xcalloc(n, sizeof(float));
        int8_t *a = xmalloc((size_t)w->rows), *b = xmalloc((size_t)w->cols);
        float k = cfg->lr / ((float)pairs * NOISE_VAR);

        /* Serial over pairs keeps the sum order (and the result) deterministic.
         * Each outer product is rows*cols work; parallel over rows inside. */
        for (int p = 0; p < pairs; p++) {
            if (wpair[p] == 0.0f) continue;
            es_rank_one(cfg->seed, step, p, w, a, b);
            float c = k * wpair[p];
#ifdef MERLIN_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < w->rows; i++) {
                float ca = c * (float)a[i];
                float *d = delta + (size_t)i * w->cols;
                for (int j = 0; j < w->cols; j++) d[j] += ca * (float)b[j];
            }
        }
        tmat_add_stochastic(w, delta, rng_key(cfg->seed, RNG_TAG_ROUND, step, (uint64_t)m));
        flips += tmat_requantize(w);
        n_weights += n;
        free(delta); free(a); free(b);
    }

    if (stats) {
        float sum = 0.0f, best = fit[0];
        for (int i = 0; i < pop; i++) { sum += fit[i]; if (fit[i] > best) best = fit[i]; }
        stats->mean_fitness = sum / (float)pop;
        stats->best_fitness = best;
        stats->flips = flips;
        stats->n_weights = n_weights;
    }

    for (int t = 0; t < nthreads; t++) scratch_free(&scratch[t]);
    free(scratch); free(fit); free(u); free(wpair);
}
