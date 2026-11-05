#ifndef MERLIN_DATA_H
#define MERLIN_DATA_H

#include <stddef.h>
#include <stdint.h>

/* Byte-level dataset: the vocabulary is the 256 byte values, no tokenizer. */

typedef enum { SPLIT_TRAIN = 0, SPLIT_VAL = 1 } Split;

typedef struct {
    uint8_t *bytes;
    size_t   n;
    size_t   n_train;   /* bytes [0, n_train) train, [n_train, n) validation */
} Dataset;

/* Load a whole file; the last `val_frac` of it becomes the validation split. */
void data_load(Dataset *ds, const char *path, float val_frac);
void data_free(Dataset *ds);

/*
 * Copy `len` consecutive bytes from a random position in `split` into out.
 * The position is a pure function of (key, i), so all population members
 * see the exact same batch when given the same key.
 */
void data_window(const Dataset *ds, Split split, uint64_t key, uint64_t i,
                 int len, uint8_t *out);

/* Deterministic window starting at byte `pos` of the split (for full evals). */
void data_window_at(const Dataset *ds, Split split, size_t pos, int len, uint8_t *out);

size_t data_split_len(const Dataset *ds, Split split);

#endif
