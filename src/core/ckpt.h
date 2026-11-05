#ifndef MERLIN_CKPT_H
#define MERLIN_CKPT_H

#include "es/es.h"

/*
 * Checkpoints store the latent int8 weights of every matrix in a ParamSet,
 * plus a free-form model tag so you can't load an MLP into an RWKV.
 *
 *   "MRLN" u32 version | tag[64] | u32 n_mats | per matrix: i32 rows, i32 cols, latent bytes
 *
 * Little-endian, no padding. Trits and scales are rebuilt on load.
 */

void ckpt_save(const char *path, const char *tag, const ParamSet *ps);
/* Loads into an already-built ParamSet; shapes and tag must match. */
void ckpt_load(const char *path, const char *tag, ParamSet *ps);

#endif
