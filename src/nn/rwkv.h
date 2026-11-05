#ifndef MERLIN_RWKV_H
#define MERLIN_RWKV_H

#include <stdint.h>

#include "core/tern.h"
#include "es/es.h"

/*
 * RWKV-lite: the model you are going to write.
 *
 * Allocation, init, parameter registration and the sequence loss are done.
 * The three functions marked YOURS in rwkv.c are the architecture itself.
 * Once rwkv_step works, `merlin_train --model rwkv` trains it with ES.
 *
 * Staged plan (see docs/rwkv_lite.md for the math):
 *   v0  token shift + linear-attention state with a fixed per-channel decay
 *       (RWKV-5/6 style, decay not data-dependent). Float state.
 *   v1  learn decay and token-shift mixes (needs ES over int8 vectors).
 *   v2  RWKV-7 generalized delta rule (data-dependent decay, in-context
 *       learning rate, removal key).
 *   v3  integer state and integer non-linearities.
 *
 * Ternary: every d x d / d x ffn matrix (TMat). Not ternary: decays and
 * token-shift mixes (int8 vectors), the recurrent state (float for now).
 */

typedef struct {
    int d_model;     /* channels                          */
    int n_layer;
    int head_size;   /* n_head = d_model / head_size      */
    int ffn;         /* channel-mix hidden size           */
} RWKVConfig;

typedef struct {
    /* time mix (token mixer) */
    TMat wr, wk, wv, wo;        /* d x d                             */
    float *mu_r, *mu_k, *mu_v;  /* [d] token-shift lerp, 0..1         */
    float *decay;               /* [d] per-channel w in (0,1)         */
    /* channel mix (FFN) */
    TMat fk;                    /* ffn x d                            */
    TMat fv;                    /* d x ffn                            */
    float *mu_f;                /* [d]                                */
} RWKVLayer;

typedef struct {
    RWKVConfig cfg;
    TMat emb;                   /* 256 x d                            */
    TMat head;                  /* 256 x d                            */
    RWKVLayer *layers;
    TMat **mats;                /* every TMat, in id order            */
    ParamSet ps;
} RWKV;

/* Recurrent state for ONE sequence. Reset it at each sequence start. */
typedef struct {
    float *tm_prev;   /* [L][d]       last input to each time mix (token shift)  */
    float *cm_prev;   /* [L][d]       last input to each channel mix             */
    float *S;         /* [L][H][N][N] wkv state, H heads of N x N (N=head_size)  */
} RWKVState;

/* Per-thread working memory. Add fields as you need them. */
typedef struct {
    float  *x, *xx, *r, *k, *v, *y, *tmp, *ffn, *logits;
    int8_t *q, *qffn;
    RWKVState st;
} RWKVScratch;

void  rwkv_init(RWKV *m, const RWKVConfig *cfg, uint64_t seed);
void  rwkv_free(RWKV *m);

void  rwkv_scratch_init(RWKVScratch *s, const RWKV *m);
void  rwkv_scratch_free(RWKVScratch *s);
void  rwkv_state_reset(const RWKV *m, RWKVState *st);

/* YOURS: feed one byte, update state, write next-byte logits to s->logits. */
void  rwkv_step(const RWKV *m, RWKVState *st, uint8_t token, const Perturb *p,
                RWKVScratch *s);

/* Mean cross-entropy (nats) predicting seq[1..len-1] from seq[0..len-2].
 * Resets the state first. */
float rwkv_seq_loss(const RWKV *m, const uint8_t *seq, int len, const Perturb *p,
                    RWKVScratch *s);

#endif
