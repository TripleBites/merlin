# RWKV-lite: what you're building

`src/nn/rwkv.c` has three functions marked **YOURS**: `time_mix`, `channel_mix`
and `rwkv_step`. Everything around them already works: allocation, init, the
ES hooks, the sequence loss, sampling and checkpoints. When `rwkv_step` stops
calling `MERLIN_TODO`, this command trains your model:

```
./build/merlin_train --data data/tinyshakespeare.txt --model rwkv \
    --pop 128 --batch 16 --seqlen 32
```

v0 below is a deliberately simplified RWKV, closest to RWKV-5/6 with a
*static* decay. It is a Merlin design choice, not an official RWKV version.
v2 is where the real RWKV-7 ideas come in.

## Notation

- `d` channels, `H` heads of `N = head_size` channels each (`d = H*N`)
- `x_t` is the layer input at time t, and `xx` is the same layer's input at t-1
  (token shift)
- `lerp(a, b, mu) = a + mu*(b - a)`, per channel
- Every `W` is a `TMat`, applied as `tmat_matvec(&W, q, qs, p, y)` on int8
  activations from `quant_i8`

## v0: static-decay linear attention

**rwkv_step(token)**

```
x = emb[token]                                  tmat_row(&m->emb, token, p, x)
for l in layers:
    x = x + time_mix_l(rmsnorm(x))
    x = x + channel_mix_l(rmsnorm(x))
logits = head · rmsnorm(x)
```

**time_mix_l(x)**

```
xx       = tm_prev[l];   tm_prev[l] = x
r        = Wr · lerp(x, xx, mu_r)
k        = Wk · lerp(x, xx, mu_k)
v        = Wv · lerp(x, xx, mu_v)
for each head h (channels h*N .. h*N+N-1):
    S_h[i][j] = w[i] * S_h[i][j] + k[i] * v[j]      # N x N state, decay per key channel
    y[j]      = sum_i r[i] * S_h[i][j]
out      = Wo · rmsnorm(y)
```

`w = layer->decay`, already initialised from 0.5 (fast) to 0.999 (slow)
within each head.

**channel_mix_l(x)**

```
xx  = cm_prev[l];   cm_prev[l] = x
out = Fv · relu(Fk · lerp(x, xx, mu_f))^2
```

Tip: quantise right before every matvec (`quant_i8` into `s->q`, or into
`s->qffn` for the ffn-sized vector), and always pass `p` through. If you skip
`p`, ES can't see that matrix and it never learns.

**Done when:** step-0 val is about 8 bits/byte and falls steadily. Expect it
to reach the low 6s within ~100 steps at `--pop 128 --batch 16`.

## v1: learn the small parameters

`decay` and `mu_*` are plain float vectors that ES doesn't touch yet.
Options, roughly from simplest to cleverest:

1. Add a `PVec` type to `es.c`: full-rank Gaussian noise on small vectors. It's
   cheap because they are only `d` long.
2. Store them as int8 and map them through a lookup table (`decay = LUT[q]`),
   which also prepares for v3.
3. Make `decay` data-dependent: `w_t = f(Ww · x)` with a small ternary `Ww`.
   This is RWKV-6's step.

Add a learnable RMSNorm gain here too.

## v2: RWKV-7 generalized delta rule

RWKV-7 replaces "decay and add" with a delta rule: the state can *remove*
what it stored at a key before writing a new value. Per head, roughly:

```
S_t = S_{t-1} · (diag(w_t) - kappa_t^T (a_t ⊙ kappa_t)) + v_t^T k_t
```

Here `w_t` is a data-dependent decay, `a_t` an in-context learning rate,
`kappa_t` a normalised removal key, and `k_t` the replacement key. Check
the exact form (and the low-rank MLPs that produce `w`, `a`, `g`) in
[the RWKV-7 paper](https://arxiv.org/abs/2503.14456) before writing it.
Keep its small low-rank matrices int8 rather than ternary.

## v3: integer everything

- State in int32 with a fixed-point decay (multiply, then shift).
- `rmsnorm` with an integer rsqrt from a lookup table.
- Squared ReLU and lerp in int16.

Once this works, the forward pass contains no floats, which is EGG's goal and
the right moment to port to Vulkan.

## Testing tip

Write `tests/test_rwkv.c` that runs a 1-layer, d=16 model on a short sequence
and compares against a 40-line NumPy version of the same equations, using
exported trits and scales. Bugs in recurrent code are silent: the loss still
goes down a bit, just worse than it should.
