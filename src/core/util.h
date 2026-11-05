#ifndef MERLIN_UTIL_H
#define MERLIN_UTIL_H

#include <stddef.h>
#include <stdint.h>

/* Allocation that never returns NULL: on failure it prints and exits. */
void *xmalloc(size_t n);
void *xcalloc(size_t count, size_t size);

/* Print a printf-style message to stderr and exit(1). */
void die(const char *fmt, ...);

/* Monotonic wall clock in seconds. */
double now_seconds(void);

/* Mark code you still have to write. Aborts with file:line when reached. */
#define MERLIN_TODO(what) die("TODO (%s:%d): %s", __FILE__, __LINE__, (what))

static inline int clampi(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
static inline int8_t sat_i8(int x) { return (int8_t)clampi(x, -127, 127); }

#endif
