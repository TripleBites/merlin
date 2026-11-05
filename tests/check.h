/* Minimal test helpers: no framework, just counters and a non-zero exit. */
#ifndef MERLIN_CHECK_H
#define MERLIN_CHECK_H

#include <math.h>
#include <stdio.h>

static int g_checks, g_fails;

#define CHECK(cond) do { g_checks++; if (!(cond)) { g_fails++; \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)

#define CHECK_NEAR(a, b, tol) do { g_checks++; double _a = (a), _b = (b); \
    if (fabs(_a - _b) > (tol)) { g_fails++; \
    fprintf(stderr, "FAIL %s:%d: %s = %g, expected %g (tol %g)\n", \
            __FILE__, __LINE__, #a, _a, _b, (double)(tol)); } } while (0)

#define CHECK_DONE() (printf("%s: %d checks, %d failed\n", __FILE__, g_checks, g_fails), g_fails ? 1 : 0)

#endif
