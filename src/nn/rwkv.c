#include "nn/rwkv.h"
#include "core/util.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 256

/* =====================================================================
 * Done for you: allocation, init, parameter registration, loss.
 * ===================================================================== */

static float *vec_fill(int n, float v) {
    float *p = xmalloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) p[i] = v;
    return p;
}

void rwkv_init(RWKV *m, const RWKVConfig *cfg, uint64_t seed) {
    const int d = cfg->d_model, L = cfg->n_layer;
    if (d % cfg->head_size) die("d_model %d not divisible by head_size %d", d, cfg->head_size);
    m->cfg = *cfg;

    int n_mats = 2 + 6 * L, id = 0;
    m->mats = xcalloc((size_t)n_mats, sizeof(TMat *));
    m->layers = xcalloc((size_t)L, sizeof(RWKVLayer));

#define ADD(w, name, rows, cols) do { tmat_init((w), (name), id, (rows), (cols)); \
        tmat_randomize((w), seed, 16.0f); m->mats[id++] = (w); } while (0)

    ADD(&m->emb, "emb", VOCAB, d);
    ADD(&m->head, "head", VOCAB, d);
    tmat_randomize(&m->head, seed, 4.0f);   /* small head -> ~8 bits/byte at step 0 */
    for (int l = 0; l < L; l++) {
        RWKVLayer *ly = &m->layers[l];
        ADD(&ly->wr, "wr", d, d);
        ADD(&ly->wk, "wk", d, d);
        ADD(&ly->wv, "wv", d, d);
        ADD(&ly->wo, "wo", d, d);
        ADD(&ly->fk, "fk", cfg->ffn, d);
        ADD(&ly->fv, "fv", d, cfg->ffn);

        ly->mu_r = vec_fill(d, 0.5f);
        ly->mu_k = vec_fill(d, 0.5f);
        ly->mu_v = vec_fill(d, 0.5f);
        ly->mu_f = vec_fill(d, 0.5f);
        /* Spread decays from fast (0.5) to slow (0.999) inside each head, so
         * some channels remember the last byte and some the last ~1000. */
        ly->decay = xmalloc((size_t)d * sizeof(float));
        for (int c = 0; c < d; c++) {
            float t = (float)(c % cfg->head_size) / (float)(cfg->head_size - 1 > 0 ? cfg->head_size - 1 : 1);
            ly->decay[c] = 1.0f - powf(10.0f, -(0.3f + 2.7f * t));
        }
    }
#undef ADD
    m->ps.mats = m->mats;
    m->ps.n_mats = n_mats;
}

void rwkv_free(RWKV *m) {
    for (int i = 0; i < m->ps.n_mats; i++) tmat_free(m->mats[i]);
    for (int l = 0; l < m->cfg.n_layer; l++) {
        RWKVLayer *ly = &m->layers[l];
        free(ly->mu_r); free(ly->mu_k); free(ly->mu_v); free(ly->mu_f); free(ly->decay);
    }
    free(m->layers);
    free(m->mats);
}

void rwkv_scratch_init(RWKVScratch *s, const RWKV *m) {
    const int d = m->cfg.d_model, L = m->cfg.n_layer, N = m->cfg.head_size, H = d / N;
    s->x = xcalloc((size_t)d, sizeof(float));
    s->xx = xcalloc((size_t)d, sizeof(float));
    s->r = xcalloc((size_t)d, sizeof(float));
    s->k = xcalloc((size_t)d, sizeof(float));
    s->v = xcalloc((size_t)d, sizeof(float));
    s->y = xcalloc((size_t)d, sizeof(float));
    s->tmp = xcalloc((size_t)d, sizeof(float));
    s->ffn = xcalloc((size_t)m->cfg.ffn, sizeof(float));
    s->logits = xcalloc(VOCAB, sizeof(float));
    s->q = xcalloc((size_t)d, 1);
    s->qffn = xcalloc((size_t)m->cfg.ffn, 1);
    s->st.tm_prev = xcalloc((size_t)L * d, sizeof(float));
    s->st.cm_prev = xcalloc((size_t)L * d, sizeof(float));
    s->st.S = xcalloc((size_t)L * H * N * N, sizeof(float));
}

void rwkv_scratch_free(RWKVScratch *s) {
    free(s->x); free(s->xx); free(s->r); free(s->k); free(s->v); free(s->y);
    free(s->tmp); free(s->ffn); free(s->logits); free(s->q); free(s->qffn);
    free(s->st.tm_prev); free(s->st.cm_prev); free(s->st.S);
}

void rwkv_state_reset(const RWKV *m, RWKVState *st) {
    const int d = m->cfg.d_model, L = m->cfg.n_layer, N = m->cfg.head_size, H = d / N;
    memset(st->tm_prev, 0, (size_t)L * d * sizeof(float));
    memset(st->cm_prev, 0, (size_t)L * d * sizeof(float));
    memset(st->S, 0, (size_t)L * H * N * N * sizeof(float));
}

/* RMS-normalise x into out (no learned gain yet; adding one is a v1 task). */
static void rmsnorm(const float *x, int n, float *out) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv;
}

float rwkv_seq_loss(const RWKV *m, const uint8_t *seq, int len, const Perturb *p,
                    RWKVScratch *s) {
    rwkv_state_reset(m, &s->st);
    double total = 0.0;
    for (int t = 0; t + 1 < len; t++) {
        rwkv_step(m, &s->st, seq[t], p, s);
        const float *z = s->logits;
        float mx = z[0];
        for (int i = 1; i < VOCAB; i++) if (z[i] > mx) mx = z[i];
        double sum = 0.0;
        for (int i = 0; i < VOCAB; i++) sum += exp((double)(z[i] - mx));
        total += log(sum) + mx - z[seq[t + 1]];
    }
    return (float)(total / (len - 1));
}

/* =====================================================================
 * YOURS. Everything below is the architecture. See docs/rwkv_lite.md.
 *
 * Tools you have:
 *   tmat_row(&m->emb, token, p, out)            embedding lookup (perturbed)
 *   q_scale = quant_i8(x, n, q)                 float -> int8 activations
 *   tmat_matvec(&W, q, q_scale, p, y)           y = W x  (ternary, perturbed)
 *   rmsnorm(x, n, out)
 *   s->...                                       scratch buffers (add more)
 * Always pass `p` through to tmat_*: that is how ES sees your weights.
 * ===================================================================== */

/* YOURS: time mix for layer l. x is the normalised layer input [d];
 * write the block output (before the residual add) into out [d].
 *
 *   xx   = st->tm_prev[l]  (previous token's x), then store x into it
 *   x_r  = lerp(x, xx, mu_r)    likewise x_k, x_v
 *   r, k, v = Wr x_r, Wk x_k, Wv x_v
 *   per head h (N channels):  S_h = diag(w) S_h + k_h v_h^T
 *                             y_h = r_h^T S_h
 *   out  = Wo rmsnorm(y)
 */
static void time_mix(const RWKV *m, int l, RWKVState *st, const float *x,
                     const Perturb *p, RWKVScratch *s, float *out) {
    (void)rmsnorm;
    MERLIN_TODO("time_mix: token shift, r/k/v projections, wkv state update, output projection");
}

/* YOURS: channel mix for layer l.
 *
 *   xx  = st->cm_prev[l], then store x into it
 *   x_k = lerp(x, xx, mu_f)
 *   out = Fv relu(Fk x_k)^2
 */
static void channel_mix(const RWKV *m, int l, RWKVState *st, const float *x,
                        const Perturb *p, RWKVScratch *s, float *out) {
    MERLIN_TODO("channel_mix: token shift, squared-ReLU FFN");
}

/* YOURS: one token through the whole network.
 *
 *   x = emb[token]
 *   for each layer:  x += time_mix(norm(x));  x += channel_mix(norm(x))
 *   logits = head norm(x)          -> s->logits
 */
void rwkv_step(const RWKV *m, RWKVState *st, uint8_t token, const Perturb *p,
               RWKVScratch *s) {
    (void)time_mix;
    (void)channel_mix;
    MERLIN_TODO("rwkv_step: embedding, residual stack of time_mix + channel_mix, head");
}
