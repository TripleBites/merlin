#ifndef MERLIN_RNG_H
#define MERLIN_RNG_H

#include <stdint.h>

/*
 * Counter-based randomness.
 *
 * ES never stores its noise. Every random number is a pure function of a key
 * and an index, so any member's perturbation can be regenerated at any time
 * (during evaluation, during the update, or next week from a log line).
 *
 *   key  = rng_key(seed, step, member, matrix, ...)   -- which stream
 *   x    = rng_u64(key, i)                            -- i-th number in it
 *
 * This is the same idea as Philox / Threefry, built on the SplitMix64
 * finalizer, which is plenty for ES noise (it is not cryptographic).
 */

static inline uint64_t rng_mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* Fold several integers into one stream key. Order matters. */
static inline uint64_t rng_key(uint64_t seed, uint64_t a, uint64_t b, uint64_t c) {
    uint64_t k = rng_mix64(seed);
    k = rng_mix64(k ^ a);
    k = rng_mix64(k ^ b);
    k = rng_mix64(k ^ c);
    return k;
}

static inline uint64_t rng_u64(uint64_t key, uint64_t i) {
    return rng_mix64(key ^ rng_mix64(i));
}

/* Uniform float in [0, 1). */
static inline float rng_uniform(uint64_t key, uint64_t i) {
    return (float)(rng_u64(key, i) >> 40) * (1.0f / 16777216.0f);
}

/*
 * Integer "Gaussian" noise: popcount of 16 random bits minus 8.
 * Binomial(16, 0.5) - 8  ->  values in [-8, 8], mean 0, std 2.
 * Cheap, integer-only, and close enough to normal for ES.
 */
static inline int8_t rng_noise_i8(uint64_t key, uint64_t i) {
    return (int8_t)(__builtin_popcount((unsigned)(rng_u64(key, i) & 0xFFFFu)) - 8);
}
#define RNG_NOISE_STD 2.0f

/* Uniform integer in [0, n). Slight modulo bias is fine for sampling batches. */
static inline uint32_t rng_below(uint64_t key, uint64_t i, uint32_t n) {
    return (uint32_t)(rng_u64(key, i) % n);
}

/* Domain tags so different uses of the same (seed, step) never collide. */
enum {
    RNG_TAG_INIT  = 0x494E4954, /* weight init        */
    RNG_TAG_NOISE = 0x4E4F4953, /* ES perturbations   */
    RNG_TAG_BATCH = 0x42415443, /* minibatch indices  */
    RNG_TAG_ROUND = 0x524F554E, /* stochastic rounding*/
    RNG_TAG_SAMPL = 0x53414D50  /* text sampling      */
};

#endif
