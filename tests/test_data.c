#include <stdio.h>
#include <string.h>

#include "check.h"
#include "core/data.h"

int main(void) {
    FILE *f = fopen("test_data.txt", "wb");
    for (int i = 0; i < 1000; i++) fputc('a' + i % 26, f);
    fclose(f);

    Dataset ds;
    data_load(&ds, "test_data.txt", 0.1f);
    CHECK(ds.n == 1000);
    CHECK(ds.n_train == 900);
    CHECK(data_split_len(&ds, SPLIT_VAL) == 100);

    /* Same (key, i) -> same window: every ES member sees the same batch. */
    uint8_t w1[16], w2[16];
    data_window(&ds, SPLIT_TRAIN, 123, 7, 16, w1);
    data_window(&ds, SPLIT_TRAIN, 123, 7, 16, w2);
    CHECK(memcmp(w1, w2, 16) == 0);
    /* Windows are contiguous text. */
    int contiguous = 1;
    for (int i = 1; i < 16; i++) if ((w1[i] - 'a') != (w1[i - 1] - 'a' + 1) % 26) contiguous = 0;
    CHECK(contiguous);

    /* Validation starts after the training bytes. */
    data_window_at(&ds, SPLIT_VAL, 0, 4, w1);
    CHECK(w1[0] == 'a' + 900 % 26);

    data_free(&ds);
    remove("test_data.txt");
    return CHECK_DONE();
}
