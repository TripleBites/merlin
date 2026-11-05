#include "check.h"
#include "core/rng.h"

int main(void) {
    /* Same key + index -> same number, always. This is what lets ES regenerate noise. */
    uint64_t k = rng_key(42, RNG_TAG_NOISE, 7, 3);
    CHECK(rng_u64(k, 100) == rng_u64(rng_key(42, RNG_TAG_NOISE, 7, 3), 100));
    CHECK(rng_u64(k, 100) != rng_u64(k, 101));
    CHECK(rng_key(42, 1, 2, 3) != rng_key(42, 1, 3, 2));   /* order matters */

    /* Integer noise: range [-8, 8], mean ~0, std ~2. */
    const int N = 200000;
    double sum = 0, sq = 0;
    int lo = 0, hi = 0;
    for (int i = 0; i < N; i++) {
        int x = rng_noise_i8(k, (uint64_t)i);
        sum += x; sq += (double)x * x;
        if (x < lo) lo = x;
        if (x > hi) hi = x;
    }
    double mean = sum / N, var = sq / N - mean * mean;
    CHECK_NEAR(mean, 0.0, 0.02);
    CHECK_NEAR(sqrt(var), RNG_NOISE_STD, 0.02);
    CHECK(lo >= -8 && hi <= 8);

    /* Uniform floats in [0,1) with mean ~0.5. */
    double us = 0;
    int ok = 1;
    for (int i = 0; i < N; i++) { float u = rng_uniform(k, (uint64_t)i); us += u; if (u < 0 || u >= 1) ok = 0; }
    CHECK(ok);
    CHECK_NEAR(us / N, 0.5, 0.005);

    return CHECK_DONE();
}
