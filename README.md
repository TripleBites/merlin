# Merlin

A ternary {-1, 0, +1} language model in C, trained with evolution strategies
instead of backpropagation. CPU first, Vulkan later.

## Quick start

```
scripts/get_data.sh                       # Tiny Shakespeare -> data/
cmake -S . -B build && cmake --build build -j
(cd build && ctest --output-on-failure)   # 4 test suites
./build/merlin_train --data data/tinyshakespeare.txt --steps 3000 --prompt "ROMEO:"
```

CMake options: `-DMERLIN_NATIVE=ON` (`-march=native`, the default),
`-DMERLIN_OPENMP=ON` (the default), `-DMERLIN_SANITIZE=ON` (ASan + UBSan).
Threads: `OMP_NUM_THREADS=8 ./build/merlin_train ...`

## What works today

The phase-1 char MLP (ctx 4, emb 16, hidden 128, 45k ternary weights) trains
end to end with rank-one ES. Reference run on a 2-core cloud VM:

| | bits/byte (val) |
| --- | --- |
| Uniform over 256 bytes | 8.00 |
| Step 0 | 8.48 |
| Unigram (train entropy) | 4.77 |
| **ES, 3000 steps, ~200 s** (`--pop 256 --batch 64 --sigma 4 --lr 12`) | **4.49** |
| Smoothed bigram | 3.59 |

So it beats a unigram model and doesn't yet beat a bigram, using no gradients
at all. Closing that gap is experiment #1.

## Layout

```
src/core/rng.h      counter-based RNG: noise is regenerated, never stored
src/core/tern.*     TMat (int8 latent -> trits + row scale), matvec with rank-one
                    perturbation, int8 activation quant, 2-bit packing
src/core/data.*     byte-level dataset, deterministic minibatch windows
src/core/ckpt.*     save/load latents
src/es/es.*         EGGROLL-style rank-one ES with antithetic pairs + centered ranks
src/nn/charmlp.*    phase-1 reference model (done)
src/nn/rwkv.*       RWKV-lite: scaffolding done, architecture is YOURS
tools/merlin_train.c  trainer CLI (mlp | rwkv), eval, sampling
tests/              rng, tern, data, es (ES recovers a hidden ternary matrix)
docs/rwkv_lite.md   equations and staged plan for the RWKV block
```

## How one ES step works

1. For each antithetic pair `p`, regenerate `a_p` and `b_p` for every matrix
   from `rng_key(seed, NOISE, step, p)`.
2. Member `2p` runs with `W + sigma*a b^T`, member `2p+1` with `W - sigma*a b^T`.
   `tmat_matvec` adds `sigma * a * (b·x)`, which is one extra dot product per matrix.
3. Every member scores the same minibatch, and fitness is `-mean loss`.
4. Fitnesses become centered ranks, and `latent += lr/P * Σ (u+ - u-) a b^T`
   with stochastic rounding.
5. Re-ternarize: `trit = clamp(round(latent / absmean_row), -1, 1)`.

A run is a pure function of its seed: same seed, same binary, same result.

## Your part: the RWKV block

Read `docs/rwkv_lite.md`, then fill in `time_mix`, `channel_mix` and
`rwkv_step` in `src/nn/rwkv.c`. The trainer, ES hooks, state, scratch
buffers and loss are already wired up. Train it with:

```
./build/merlin_train --data data/tinyshakespeare.txt --model rwkv --pop 128 --batch 16
```

## Experiments to try

1. **Beat the bigram (3.59).** Sweep `--sigma`/`--lr`, add an lr decay,
   try `--ctx 8 --emb 24 --hidden 256`. Note: 4× the population didn't help
   per step in early tests, so the bottleneck may be the update rule or the
   int8 latent resolution rather than noise.
2. **Latent precision.** Switch `TMat.latent` to int16 and see whether
   learning speeds up.
3. **Sparse-flip ES.** Perturb the latents and evaluate the *real* ternary
   model, instead of ternary + dense rank-one.
4. **Pure GA on trits.** Mutation flips k trits, with truncation selection.
   Same fitness, same budget.
5. **Packed kernels.** A 2-bit add/subtract matvec vs the int8 dot. Measure
   whether it's faster.
6. **Share the base matmul.** Members only differ by the rank-one term, so for
   the *first* layer `W·x` is identical across the population. Compute it once.
7. **Architecture genome (phase 4).** Function-preserving "add block" works
   because an all-zero output projection is a valid ternary matrix.
