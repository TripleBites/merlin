#ifndef MERLIN_ES_H
#define MERLIN_ES_H

#include <stddef.h>
#include <stdint.h>

#include "core/tern.h"

/*
 * Rank-one evolution strategies (EGGROLL-style) over a set of ternary matrices.
 *
 * One step:
 *   1. For each antithetic pair p, draw a_p (rows) and b_p (cols) per matrix
 *      from the counter RNG keyed by (seed, step, p, matrix).
 *   2. Member 2p evaluates W + sigma*E_p, member 2p+1 evaluates W - sigma*E_p,
 *      where E_p = a_p b_p^T / 4 (entries have std 1).
 *   3. Fitnesses are turned into centered ranks in [-0.5, 0.5].
 *   4. latent += lr / P * sum_p (u_2p - u_2p+1) * E_p   (stochastic rounding)
 *   5. Re-ternarize.
 *
 * The model never sees ES internals: it gets a Perturb and returns a fitness.
 * Nothing about the noise is stored; step 4 regenerates it from the seed.
 */

typedef struct {
    TMat **mats;
    int    n_mats;
} ParamSet;

typedef struct {
    uint64_t seed;
    int      pop;     /* population size, must be even (antithetic pairs) */
    float    sigma;   /* perturbation std, latent units                   */
    float    lr;      /* update size, latent units per step               */
} ESConfig;

/* Evaluate one member. Higher is better. Called concurrently from several
 * threads: use `thread` to pick per-thread scratch memory. */
typedef float (*FitnessFn)(const Perturb *p, int thread, void *ctx);

typedef struct {
    float  mean_fitness;
    float  best_fitness;
    size_t flips;          /* trits that changed sign/zero this step */
    size_t n_weights;
} ESStats;

/* How many threads es_step may call the fitness function from. */
int  es_max_threads(void);

void es_step(ParamSet *ps, const ESConfig *cfg, uint64_t step,
             FitnessFn fn, void *ctx, ESStats *stats);

/* Regenerate pair p's rank-one vectors for one matrix (exposed for tests). */
void es_rank_one(uint64_t seed, uint64_t step, int pair, const TMat *w,
                 int8_t *a, int8_t *b);

#endif
