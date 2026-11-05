#ifndef MERLIN_CHARMLP_H
#define MERLIN_CHARMLP_H

#include <stdint.h>

#include "core/tern.h"
#include "es/es.h"

/*
 * Phase-1 reference model: a ternary character MLP (Bengio-style, like
 * makemore part 2). It exists to prove the ES machinery end to end before
 * the RWKV block is written. Every matrix is ternary.
 *
 *   e_k    = Emb[c_k]                      k = 1..ctx    (256 x d_emb)
 *   x      = concat(e_1 .. e_ctx)          -> int8
 *   h      = relu(rmsnorm(W1 x))           -> int8       (hidden x ctx*d_emb)
 *   logits = W2 h                                         (256 x hidden)
 */

typedef struct {
    int ctx, d_emb, hidden;
    TMat emb, w1, w2;
    TMat *mats[3];
    ParamSet ps;
} CharMLP;

/* Per-thread working memory for one forward pass. */
typedef struct {
    float  *x, *h, *logits;
    int8_t *xq, *hq;
} CharMLPScratch;

void  charmlp_init(CharMLP *m, int ctx, int d_emb, int hidden, uint64_t seed);
void  charmlp_free(CharMLP *m);
void  charmlp_scratch_init(CharMLPScratch *s, const CharMLP *m);
void  charmlp_scratch_free(CharMLPScratch *s);

/* logits for the byte that follows `context` (ctx bytes). Result in s->logits. */
void  charmlp_forward(const CharMLP *m, const uint8_t *context, const Perturb *p,
                      CharMLPScratch *s);

/* Cross-entropy (nats) of window[ctx] given window[0..ctx-1]. */
float charmlp_loss(const CharMLP *m, const uint8_t *window, const Perturb *p,
                   CharMLPScratch *s);

#endif
