#include "core/data.h"
#include "core/rng.h"
#include "core/util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void data_load(Dataset *ds, const char *path, float val_frac) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open dataset '%s' (run scripts/get_data.sh?)", path);
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) die("dataset '%s' is empty", path);

    ds->bytes = xmalloc((size_t)sz);
    if (fread(ds->bytes, 1, (size_t)sz, f) != (size_t)sz) die("short read on '%s'", path);
    fclose(f);

    ds->n = (size_t)sz;
    ds->n_train = ds->n - (size_t)llround((double)ds->n * (double)val_frac);
}

void data_free(Dataset *ds) {
    free(ds->bytes);
    memset(ds, 0, sizeof *ds);
}

size_t data_split_len(const Dataset *ds, Split split) {
    return split == SPLIT_TRAIN ? ds->n_train : ds->n - ds->n_train;
}

static const uint8_t *split_base(const Dataset *ds, Split split) {
    return split == SPLIT_TRAIN ? ds->bytes : ds->bytes + ds->n_train;
}

void data_window(const Dataset *ds, Split split, uint64_t key, uint64_t i,
                 int len, uint8_t *out) {
    size_t n = data_split_len(ds, split);
    if ((size_t)len > n) die("window of %d bytes is longer than the split", len);
    size_t pos = (size_t)(rng_u64(key, i) % (uint64_t)(n - (size_t)len + 1));
    memcpy(out, split_base(ds, split) + pos, (size_t)len);
}

void data_window_at(const Dataset *ds, Split split, size_t pos, int len, uint8_t *out) {
    size_t n = data_split_len(ds, split);
    if (pos + (size_t)len > n) die("window [%zu, +%d) past end of split", pos, len);
    memcpy(out, split_base(ds, split) + pos, (size_t)len);
}
