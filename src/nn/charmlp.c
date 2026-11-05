#include "nn/charmlp.h"
#include "core/util.h"

#include <math.h>
#include <stdlib.h>

#define VOCAB 256

static void rmsnorm_inplace(float *x, int n) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

void charmlp_init(CharMLP *m, int ctx, int d_emb, int hidden, uint64_t seed) {
    m->ctx = ctx;
    m->d_emb = d_emb;
    m->hidden = hidden;
    tmat_init(&m->emb, "emb", 0, VOCAB, d_emb);
    tmat_init(&m->w1, "w1", 1, hidden, ctx * d_emb);
    tmat_init(&m->w2, "w2", 2, VOCAB, hidden);
    tmat_randomize(&m->emb, seed, 16.0f);
    tmat_randomize(&m->w1, seed, 16.0f);
    /* Small head init -> near-uniform predictions (~8 bits/byte) at step 0. */
    tmat_randomize(&m->w2, seed, 4.0f);
    m->mats[0] = &m->emb;
    m->mats[1] = &m->w1;
    m->mats[2] = &m->w2;
    m->ps.mats = m->mats;
    m->ps.n_mats = 3;
}

void charmlp_free(CharMLP *m) {
    tmat_free(&m->emb);
    tmat_free(&m->w1);
    tmat_free(&m->w2);
}

void charmlp_scratch_init(CharMLPScratch *s, const CharMLP *m) {
    int nx = m->ctx * m->d_emb;
    s->x = xmalloc((size_t)nx * sizeof(float));
    s->xq = xmalloc((size_t)nx);
    s->h = xmalloc((size_t)m->hidden * sizeof(float));
    s->hq = xmalloc((size_t)m->hidden);
    s->logits = xmalloc(VOCAB * sizeof(float));
}

void charmlp_scratch_free(CharMLPScratch *s) {
    free(s->x); free(s->xq); free(s->h); free(s->hq); free(s->logits);
}

void charmlp_forward(const CharMLP *m, const uint8_t *context, const Perturb *p,
                     CharMLPScratch *s) {
    for (int k = 0; k < m->ctx; k++)
        tmat_row(&m->emb, context[k], p, s->x + k * m->d_emb);

    float xs = quant_i8(s->x, m->ctx * m->d_emb, s->xq);
    tmat_matvec(&m->w1, s->xq, xs, p, s->h);
    /* RMS-normalise the pre-activation: ternary weights have a fixed magnitude,
     * so without a norm the activation scale depends on fan-in. */
    rmsnorm_inplace(s->h, m->hidden);
    for (int i = 0; i < m->hidden; i++) if (s->h[i] < 0.0f) s->h[i] = 0.0f;

    float hs = quant_i8(s->h, m->hidden, s->hq);
    tmat_matvec(&m->w2, s->hq, hs, p, s->logits);
}

float charmlp_loss(const CharMLP *m, const uint8_t *window, const Perturb *p,
                   CharMLPScratch *s) {
    charmlp_forward(m, window, p, s);
    const float *z = s->logits;
    float mx = z[0];
    for (int i = 1; i < VOCAB; i++) if (z[i] > mx) mx = z[i];
    double sum = 0.0;
    for (int i = 0; i < VOCAB; i++) sum += exp((double)(z[i] - mx));
    return (float)(log(sum) + mx - z[window[m->ctx]]);
}
