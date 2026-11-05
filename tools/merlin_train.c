/*
 * merlin_train: train a ternary model with rank-one evolution strategies.
 *
 *   merlin_train --data data/tinyshakespeare.txt --model mlp --steps 2000
 *   merlin_train --data data/tinyshakespeare.txt --model rwkv   (after you write rwkv_step)
 *
 * Run with no arguments for the full option list.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/ckpt.h"
#include "core/data.h"
#include "core/rng.h"
#include "core/util.h"
#include "es/es.h"
#include "nn/charmlp.h"
#include "nn/rwkv.h"

#ifdef MERLIN_HAVE_OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/* A tiny model interface so the trainer doesn't care which net it is. */

typedef struct Model Model;
struct Model {
    const char *tag;
    ParamSet   *ps;
    int         window;          /* bytes per training example */
    int         n_threads;
    void       *impl;
    void      **scratch;         /* one per thread */
    float     (*loss)(Model *m, const uint8_t *win, const Perturb *p, int thread);
    void      (*sample)(Model *m, const char *prompt, int n, float temp, uint64_t seed);
};

static int sample_logits(const float *z, float temp, uint64_t key, uint64_t i) {
    float mx = z[0];
    for (int c = 1; c < 256; c++) if (z[c] > mx) mx = z[c];
    double p[256], sum = 0.0;
    for (int c = 0; c < 256; c++) { p[c] = exp((double)(z[c] - mx) / temp); sum += p[c]; }
    double r = (double)rng_uniform(key, i) * sum;
    for (int c = 0; c < 256; c++) { r -= p[c]; if (r <= 0.0) return c; }
    return 255;
}

static void put_byte(int c) {
    if (c == '\n' || (c >= 32 && c < 127)) putchar(c);
    else putchar('?');
}

/* ---- char MLP adapter ---- */
static float mlp_loss(Model *m, const uint8_t *win, const Perturb *p, int thread) {
    return charmlp_loss(m->impl, win, p, m->scratch[thread]);
}
static void mlp_sample(Model *m, const char *prompt, int n, float temp, uint64_t seed) {
    CharMLP *net = m->impl;
    CharMLPScratch *s = m->scratch[0];
    uint8_t ctx[64];
    memset(ctx, '\n', sizeof ctx);
    for (const char *c = prompt; *c; c++) {
        memmove(ctx, ctx + 1, (size_t)net->ctx - 1);
        ctx[net->ctx - 1] = (uint8_t)*c;
        put_byte((uint8_t)*c);
    }
    uint64_t key = rng_key(seed, RNG_TAG_SAMPL, 0, 0);
    for (int i = 0; i < n; i++) {
        charmlp_forward(net, ctx, NULL, s);
        int c = sample_logits(s->logits, temp, key, (uint64_t)i);
        put_byte(c);
        memmove(ctx, ctx + 1, (size_t)net->ctx - 1);
        ctx[net->ctx - 1] = (uint8_t)c;
    }
    putchar('\n');
}

/* ---- RWKV adapter ---- */
static float rwkv_loss_fn(Model *m, const uint8_t *win, const Perturb *p, int thread) {
    return rwkv_seq_loss(m->impl, win, m->window, p, m->scratch[thread]);
}
static void rwkv_sample(Model *m, const char *prompt, int n, float temp, uint64_t seed) {
    RWKV *net = m->impl;
    RWKVScratch *s = m->scratch[0];
    rwkv_state_reset(net, &s->st);
    const char *pr = *prompt ? prompt : "\n";
    for (const char *c = pr; *c; c++) { rwkv_step(net, &s->st, (uint8_t)*c, NULL, s); put_byte((uint8_t)*c); }
    uint64_t key = rng_key(seed, RNG_TAG_SAMPL, 0, 0);
    for (int i = 0; i < n; i++) {
        int c = sample_logits(s->logits, temp, key, (uint64_t)i);
        put_byte(c);
        rwkv_step(net, &s->st, (uint8_t)c, NULL, s);
    }
    putchar('\n');
}

/* ------------------------------------------------------------------ */

typedef struct {
    Model         *m;
    const Dataset *ds;
    uint64_t       batch_key;
    int            batch;
    uint8_t      **buf;          /* per thread window buffer */
} FitCtx;

/* Fitness = -mean loss on this step's minibatch. Every member sees the same batch. */
static float fitness(const Perturb *p, int thread, void *vctx) {
    FitCtx *c = vctx;
    double total = 0.0;
    for (int i = 0; i < c->batch; i++) {
        data_window(c->ds, SPLIT_TRAIN, c->batch_key, (uint64_t)i, c->m->window, c->buf[thread]);
        total += c->m->loss(c->m, c->buf[thread], p, thread);
    }
    return (float)(-total / c->batch);
}

/* Mean loss of the unperturbed model on a fixed set of validation windows. */
static float eval_loss(Model *m, const Dataset *ds, int n, uint8_t **buf) {
    const uint64_t key = rng_key(0xE7A1, RNG_TAG_BATCH, 0, 0);   /* same windows every eval */
    double total = 0.0;
#ifdef MERLIN_HAVE_OPENMP
#pragma omp parallel for reduction(+:total) schedule(static)
#endif
    for (int i = 0; i < n; i++) {
        int t = 0;
#ifdef MERLIN_HAVE_OPENMP
        t = omp_get_thread_num();
#endif
        data_window(ds, SPLIT_VAL, key, (uint64_t)i, m->window, buf[t]);
        total += m->loss(m, buf[t], NULL, t);
    }
    return (float)(total / n);
}

/* ------------------------------------------------------------------ */

typedef struct {
    const char *data, *model, *save, *load, *prompt;
    int steps, pop, batch, eval_every, eval_n, sample_n;
    float sigma, lr, temp;
    uint64_t seed;
    int ctx, emb, hidden;                 /* mlp  */
    int d, layers, head, ffn, seqlen;     /* rwkv */
} Args;

static void usage(void) {
    fprintf(stderr,
        "usage: merlin_train --data FILE [options]\n"
        "  --model mlp|rwkv   (mlp)     --steps N (1000)     --pop N (256, even)\n"
        "  --batch N (64)               --sigma F (4.0)      --lr F (12.0)\n"
        "  --seed N (1)                 --eval-every N (100) --eval-n N (auto)\n"
        "  --sample N (200)             --temp F (0.8)       --prompt STR (\"\")\n"
        "  --save FILE                  --load FILE\n"
        " mlp:  --ctx N (4) --emb N (16) --hidden N (128)\n"
        " rwkv: --d N (64) --layers N (2) --head N (16) --ffn N (256) --seqlen N (32)\n"
        " threads: set OMP_NUM_THREADS\n");
    exit(2);
}

static Args parse(int argc, char **argv) {
    Args a = {.data = NULL, .model = "mlp", .save = NULL, .load = NULL, .prompt = "",
              .steps = 1000, .pop = 256, .batch = 64, .eval_every = 100, .eval_n = 0,
              .sample_n = 200, .sigma = 4.0f, .lr = 12.0f, .temp = 0.8f, .seed = 1,
              .ctx = 4, .emb = 16, .hidden = 128,
              .d = 64, .layers = 2, .head = 16, .ffn = 256, .seqlen = 32};
    for (int i = 1; i < argc; i++) {
        const char *k = argv[i];
        if (i + 1 >= argc) usage();
        const char *v = argv[++i];
#define S(name, field) if (!strcmp(k, name)) { a.field = v; continue; }
#define I(name, field) if (!strcmp(k, name)) { a.field = atoi(v); continue; }
#define F(name, field) if (!strcmp(k, name)) { a.field = (float)atof(v); continue; }
        S("--data", data) S("--model", model) S("--save", save) S("--load", load) S("--prompt", prompt)
        I("--steps", steps) I("--pop", pop) I("--batch", batch) I("--eval-every", eval_every)
        I("--eval-n", eval_n) I("--sample", sample_n)
        F("--sigma", sigma) F("--lr", lr) F("--temp", temp)
        I("--ctx", ctx) I("--emb", emb) I("--hidden", hidden)
        I("--d", d) I("--layers", layers) I("--head", head) I("--ffn", ffn) I("--seqlen", seqlen)
        if (!strcmp(k, "--seed")) { a.seed = strtoull(v, NULL, 10); continue; }
#undef S
#undef I
#undef F
        fprintf(stderr, "unknown option %s\n", k);
        usage();
    }
    if (!a.data) usage();
    return a;
}

int main(int argc, char **argv) {
    Args a = parse(argc, argv);

    Dataset ds;
    data_load(&ds, a.data, 0.1f);

    Model m = {0};
    m.n_threads = es_max_threads();
    m.scratch = xcalloc((size_t)m.n_threads, sizeof(void *));
    CharMLP mlp;
    RWKV rwkv;

    if (!strcmp(a.model, "mlp")) {
        charmlp_init(&mlp, a.ctx, a.emb, a.hidden, a.seed);
        m.tag = "charmlp-v1"; m.impl = &mlp; m.ps = &mlp.ps; m.window = a.ctx + 1;
        m.loss = mlp_loss; m.sample = mlp_sample;
        for (int t = 0; t < m.n_threads; t++) {
            CharMLPScratch *s = xmalloc(sizeof *s);
            charmlp_scratch_init(s, &mlp);
            m.scratch[t] = s;
        }
    } else if (!strcmp(a.model, "rwkv")) {
        RWKVConfig cfg = {.d_model = a.d, .n_layer = a.layers, .head_size = a.head, .ffn = a.ffn};
        rwkv_init(&rwkv, &cfg, a.seed);
        m.tag = "rwkv-lite-v0"; m.impl = &rwkv; m.ps = &rwkv.ps; m.window = a.seqlen + 1;
        m.loss = rwkv_loss_fn; m.sample = rwkv_sample;
        for (int t = 0; t < m.n_threads; t++) {
            RWKVScratch *s = xmalloc(sizeof *s);
            rwkv_scratch_init(s, &rwkv);
            m.scratch[t] = s;
        }
    } else {
        die("unknown model '%s' (mlp or rwkv)", a.model);
    }
    if (a.load) ckpt_load(a.load, m.tag, m.ps);
    if (a.eval_n <= 0) a.eval_n = m.window > 8 ? 512 : 2048;   /* ~8-16k predicted bytes */

    size_t n_weights = 0;
    for (int i = 0; i < m.ps->n_mats; i++) n_weights += (size_t)m.ps->mats[i]->rows * m.ps->mats[i]->cols;

    printf("merlin | model %s | %zu ternary weights in %d matrices | %d threads\n",
           m.tag, n_weights, m.ps->n_mats, m.n_threads);
    printf("data %s: %zu train / %zu val bytes\n", a.data, ds.n_train, ds.n - ds.n_train);
    printf("ES: pop %d  batch %d  sigma %.2f  lr %.2f  seed %llu\n",
           a.pop, a.batch, a.sigma, a.lr, (unsigned long long)a.seed);
    printf("reference: uniform over 256 bytes = 8.00 bits/byte\n\n");

    uint8_t **buf = xcalloc((size_t)m.n_threads, sizeof(uint8_t *));
    for (int t = 0; t < m.n_threads; t++) buf[t] = xmalloc((size_t)m.window);

    ESConfig es = {.seed = a.seed, .pop = a.pop, .sigma = a.sigma, .lr = a.lr};
    FitCtx fc = {.m = &m, .ds = &ds, .batch = a.batch, .buf = buf};
    const float LN2 = 0.69314718f;

    float v = eval_loss(&m, &ds, a.eval_n, buf);
    printf("step %5d | val %.3f bits/byte\n", 0, v / LN2);

    double t0 = now_seconds();
    for (int step = 1; step <= a.steps; step++) {
        double ts = now_seconds();
        fc.batch_key = rng_key(a.seed, RNG_TAG_BATCH, (uint64_t)step, 0);
        ESStats st;
        es_step(m.ps, &es, (uint64_t)step, fitness, &fc, &st);
        double dt = now_seconds() - ts;

        if (step % 10 == 0 || step == 1)
            printf("step %5d | train %.3f bits/byte (pop mean) | flips %6.3f%% | %.2fs/step\n",
                   step, -st.mean_fitness / LN2, 100.0 * st.flips / st.n_weights, dt);

        if (step % a.eval_every == 0 || step == a.steps) {
            v = eval_loss(&m, &ds, a.eval_n, buf);
            size_t neg = 0, zero = 0, pos = 0;
            for (int i = 0; i < m.ps->n_mats; i++) {
                size_t n0, n1, n2;
                tmat_trit_counts(m.ps->mats[i], &n0, &n1, &n2);
                neg += n0; zero += n1; pos += n2;
            }
            printf("step %5d | val %.3f bits/byte | trits -1/0/+1: %.0f%%/%.0f%%/%.0f%% | %.0fs elapsed\n",
                   step, v / LN2, 100.0 * neg / n_weights, 100.0 * zero / n_weights,
                   100.0 * pos / n_weights, now_seconds() - t0);
            if (a.sample_n > 0) {
                printf("---- sample ----\n");
                m.sample(&m, a.prompt, a.sample_n, a.temp, a.seed + (uint64_t)step);
                printf("----------------\n");
            }
            if (a.save) ckpt_save(a.save, m.tag, m.ps);
            fflush(stdout);
        }
    }

    for (int t = 0; t < m.n_threads; t++) {
        if (m.impl == &mlp) charmlp_scratch_free(m.scratch[t]);
        else rwkv_scratch_free(m.scratch[t]);
        free(m.scratch[t]);
        free(buf[t]);
    }
    free(m.scratch);
    free(buf);
    if (m.impl == &mlp) charmlp_free(&mlp);
    else rwkv_free(&rwkv);
    data_free(&ds);
    return 0;
}
