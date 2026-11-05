#ifndef MERLIN_TERN_H
#define MERLIN_TERN_H

#include <stddef.h>
#include <stdint.h>

/*
 * Ternary matrices.
 *
 * Every learned matrix has two faces:
 *
 *   latent  int8, rows*cols   the training state. ES updates land here.
 *   trit    int8 in {-1,0,+1} what the forward pass uses (derived from latent).
 *   scale   float per row     real value of a +1 trit in that row.
 *
 * Real weight:   W[i][j] = scale[i] * trit[i][j]
 * Ternarize:     gamma_i = mean_j |latent[i][j]|               (BitNet absmean)
 *                trit    = clamp(round(latent / gamma_i), -1, 1)
 *                scale_i = gamma_i / LATENT_ONE
 *
 * LATENT_ONE says which latent value means "real 1.0". It also converts ES
 * perturbations (latent units) into real-valued weight changes.
 *
 * trit is kept unpacked (1 byte per weight) because int8 dot products
 * auto-vectorise well. tern_pack() gives the 2-bit format for storage; a
 * packed add/subtract kernel is a good experiment for later.
 */

#define LATENT_ONE 32.0f

typedef struct {
    const char *name;
    int rows, cols;
    int id;          /* index into a Perturb's rank-one table */
    int8_t *latent;  /* [rows*cols] */
    int8_t *trit;    /* [rows*cols] */
    float  *scale;   /* [rows]      */
} TMat;

/* One member's rank-one perturbation of one matrix: E = a b^T / NOISE_VAR.
 * a has `rows` entries, b has `cols` entries, both from rng_noise_i8. */
typedef struct {
    const int8_t *a;
    const int8_t *b;
} RankOne;

/* Everything a forward pass needs to evaluate one population member.
 * Pass NULL for the unperturbed ("center") model. */
typedef struct {
    float sigma;          /* noise std in latent units, sign already applied */
    const RankOne *r1;    /* indexed by TMat.id */
} Perturb;

void   tmat_init(TMat *w, const char *name, int id, int rows, int cols);
void   tmat_free(TMat *w);
/* Fill latent with integer noise of roughly the given std (latent units). */
void   tmat_randomize(TMat *w, uint64_t seed, float latent_std);
/* Recompute trit and scale from latent. Returns how many trits changed. */
size_t tmat_requantize(TMat *w);
/* latent += delta with stochastic rounding (so tiny updates still move it on
 * average) and saturation to [-127, 127]. delta has rows*cols entries. */
void   tmat_add_stochastic(TMat *w, const float *delta, uint64_t key);
/* Count of -1 / 0 / +1 trits. */
void   tmat_trit_counts(const TMat *w, size_t *neg, size_t *zero, size_t *pos);

/*
 * y = W x  (+ perturbation)
 *   x  : int8 activations, cols entries, real value = x * xs
 *   y  : float, rows entries
 */
void tmat_matvec(const TMat *w, const int8_t *x, float xs, const Perturb *p, float *y);

/* out = row `r` of W as real values (+ perturbation). Used for embeddings. */
void tmat_row(const TMat *w, int r, const Perturb *p, float *out);

/* ---- activation quantisation (BitNet absmax, per vector) ---- */
/* q = round(x / s), s = max|x| / 127. Returns s. */
float quant_i8(const float *x, int n, int8_t *q);

/* ---- integer helpers ---- */
int32_t dot_i8(const int8_t *a, const int8_t *b, int n);

/* ---- 2-bit packing: 4 trits per byte, 0 -> 00, +1 -> 01, -1 -> 10 ---- */
static inline size_t tern_packed_bytes(size_t n) { return (n + 3) / 4; }
void tern_pack(const int8_t *trit, size_t n, uint8_t *out);
void tern_unpack(const uint8_t *in, size_t n, int8_t *trit);

#endif
