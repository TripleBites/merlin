#include "core/ckpt.h"
#include "core/util.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CKPT_MAGIC "MRLN"
#define CKPT_VERSION 1u
#define TAG_LEN 64

static void wr(FILE *f, const void *p, size_t n) {
    if (fwrite(p, 1, n, f) != n) die("checkpoint write failed");
}
static void rd(FILE *f, void *p, size_t n) {
    if (fread(p, 1, n, f) != n) die("checkpoint truncated");
}

void ckpt_save(const char *path, const char *tag, const ParamSet *ps) {
    FILE *f = fopen(path, "wb");
    if (!f) die("cannot write checkpoint '%s'", path);
    char t[TAG_LEN] = {0};
    strncpy(t, tag, TAG_LEN - 1);
    uint32_t ver = CKPT_VERSION, n = (uint32_t)ps->n_mats;
    wr(f, CKPT_MAGIC, 4); wr(f, &ver, 4); wr(f, t, TAG_LEN); wr(f, &n, 4);
    for (int m = 0; m < ps->n_mats; m++) {
        const TMat *w = ps->mats[m];
        int32_t rc[2] = {w->rows, w->cols};
        wr(f, rc, sizeof rc);
        wr(f, w->latent, (size_t)w->rows * w->cols);
    }
    fclose(f);
}

void ckpt_load(const char *path, const char *tag, ParamSet *ps) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot read checkpoint '%s'", path);
    char magic[4], t[TAG_LEN];
    uint32_t ver, n;
    rd(f, magic, 4); rd(f, &ver, 4); rd(f, t, TAG_LEN); rd(f, &n, 4);
    t[TAG_LEN - 1] = 0;
    if (memcmp(magic, CKPT_MAGIC, 4) || ver != CKPT_VERSION) die("'%s' is not a Merlin v%u checkpoint", path, CKPT_VERSION);
    if (strcmp(t, tag)) die("checkpoint is for '%s', model is '%s'", t, tag);
    if ((int)n != ps->n_mats) die("checkpoint has %u matrices, model has %d", n, ps->n_mats);
    for (int m = 0; m < ps->n_mats; m++) {
        TMat *w = ps->mats[m];
        int32_t rc[2];
        rd(f, rc, sizeof rc);
        if (rc[0] != w->rows || rc[1] != w->cols)
            die("matrix %d (%s): checkpoint %dx%d, model %dx%d", m, w->name, rc[0], rc[1], w->rows, w->cols);
        rd(f, w->latent, (size_t)w->rows * w->cols);
        tmat_requantize(w);
    }
    fclose(f);
}
