#include <stdint.h>
#include "printf.h"
#include "kws_config.h"

#if !KWS_BURN
#include "aps256xx.h"

#define W4_BLKS    8475u
#define W4_ROW_MAX 134u

static int8_t w4_row[W4_ROW_MAX * 29u] __attribute__((section(".wram1")));
static unsigned int w4_row_ch = 0xFFFFFFFFu;
static unsigned int w4_row_off = 0u;

static unsigned int w4_phys_addr(unsigned int b) {
    unsigned int g = b / 63u, idx = b % 63u;
    return (g * 64u + 1u + idx) * 32u;
}

static void w4_fetch(unsigned int ch) {
    unsigned int start = ch * 3840u;
    unsigned int b0 = start / 29u;
    unsigned int nb = (start % 29u + 3840u + 28u) / 29u;
    unsigned int done = 0u;
    while (done < nb) {
        unsigned int b = b0 + done;
        unsigned int idx = b % 63u;
        unsigned int run = 63u - idx;
        if (run > nb - done) run = nb - done;
        aps256xx_read((unsigned char *)w4_row + done * 29u, w4_phys_addr(b), run * 29u);
        done += run;
    }
    w4_row_off = start % 29u;
    w4_row_ch = ch;
}

int32_t W4_AT(int32_t k) {
    unsigned int ch = (unsigned int)k / 3840u;
    if (ch != w4_row_ch) w4_fetch(ch);
    return (int32_t)w4_row[w4_row_off + ((unsigned int)k - ch * 3840u)];
}

static uint32_t crc32_byte(uint32_t crc, uint8_t b) {
    int i;
    crc ^= b;
    for (i = 0; i < 8; i++)
        crc = (crc >> 1) ^ (0xEDB88320u & (uint32_t)-(int32_t)(crc & 1u));
    return crc;
}

void kws_w4_selfcheck(void) {
    unsigned int b, j;
    uint8_t buf[29];
    uint32_t crc = 0xFFFFFFFFu;
    for (b = 0u; b < W4_BLKS; b++) {
        aps256xx_read(buf, w4_phys_addr(b), 29u);
        for (j = 0u; j < 29u; j++) crc = crc32_byte(crc, buf[j]);
    }
    crc ^= 0xFFFFFFFFu;
    printf("W4 PSRAM crc32 = %08lX\r\n", (unsigned long)crc);
}
#endif
