#include <stdint.h>
#include "printf.h"
#include "kws_config.h"

#if KWS_BURN
#include "aps256xx.h"
#include "kws_weights.h"

#define W4_TOTAL 245760u
#define W4_BLKS  8475u

static uint32_t crc32_byte(uint32_t crc, uint8_t b) {
    int i;
    crc ^= b;
    for (i = 0; i < 8; i++)
        crc = (crc >> 1) ^ (0xEDB88320u & (uint32_t)-(int32_t)(crc & 1u));
    return crc;
}

static uint8_t w4_byte(unsigned int k) {
    if (k < 120000u) return (uint8_t)W4_A[k];
    if (k < W4_TOTAL) return (uint8_t)W4_B[k - 120000u];
    return 0u;
}

int main(void) {
    unsigned int b, j;
    uint8_t buf[29];
    uint32_t crc_s = 0xFFFFFFFFu, crc_p = 0xFFFFFFFFu;
    printf("kws burn start\r\n");
    aps256xx_init();
    for (b = 0u; b < W4_BLKS; b++) {
        unsigned int g = b / 63u, idx = b % 63u;
        unsigned int pa = (g * 64u + 1u + idx) * 32u;
        for (j = 0u; j < 29u; j++) {
            buf[j] = w4_byte(b * 29u + j);
            crc_s = crc32_byte(crc_s, buf[j]);
        }
        aps256xx_write(buf, pa, 29u);
        if ((b & 0x3FFu) == 0u) printf("write %d/8475\r\n", (int)b);
    }
    crc_s ^= 0xFFFFFFFFu;
    printf("W4 SRAM  crc32 = %08lX\r\n", (unsigned long)crc_s);
    for (b = 0u; b < W4_BLKS; b++) {
        unsigned int g = b / 63u, idx = b % 63u;
        unsigned int pa = (g * 64u + 1u + idx) * 32u;
        aps256xx_read(buf, pa, 29u);
        for (j = 0u; j < 29u; j++) crc_p = crc32_byte(crc_p, buf[j]);
        if ((b & 0x3FFu) == 0u) printf("read %d/8475\r\n", (int)b);
    }
    crc_p ^= 0xFFFFFFFFu;
    printf("W4 PSRAM crc32 = %08lX\r\n", (unsigned long)crc_p);
    if (crc_s == crc_p) printf("BURN PASS\r\n");
    else printf("BURN FAIL\r\n");
    while (1) { }
    return 0;
}

#elif !KWS_LIVE
#include "aps256xx.h"
#include "kws_decide.h"
#include "gpio_reg.h"
#include "base_addr.h"

#define AON_AC_REG8   (SCU_BASE + 0x320)

static void kws_wake_c908(void) {
    unsigned int v;
    v = *(volatile unsigned int *)(GPIO_SWPORTA_DDR);
    *(volatile unsigned int *)(GPIO_SWPORTA_DDR) = v | (1u << 2);
    v = *(volatile unsigned int *)(GPIO_SWPORTA_DR);
    *(volatile unsigned int *)(GPIO_SWPORTA_DR) = v | (1u << 2);
    v = *(volatile unsigned int *)(AON_AC_REG8);
    *(volatile unsigned int *)(AON_AC_REG8) = v & ~(1u << 2);
    printf("wake c908 done\r\n");
}

void kws_board_run(const int16_t *pcm, uint8_t out8[8]);
void kws_w4_selfcheck(void);

static int16_t inject_pcm[16000];
volatile unsigned int kws_go = 0;

static inline unsigned int kws_cycles(void) {
    unsigned int c;
    __asm__ __volatile__("rdcycle %0" : "=r"(c));
    return c;
}

static const char *kws_name(int k) {
    switch (k) {
        case 1: return "dakai";
        case 2: return "guanbi";
        case 3: return "huanxing";
        default: return "none";
    }
}

int main(void) {
    int i, j;
    uint8_t out8[8];
    kws_decide_t dec;
    unsigned int win = 0;
    unsigned int c0, c1, c2;
    kws_decide_init(&dec, 0.4f, 0.1f, 0.0f);
    printf("kws inject ready\r\n");
    aps256xx_init();
    kws_w4_selfcheck();
    printf("kws inject loop\r\n");
    while (1) {
        if (kws_go) {
            float probs[4], conf, gap;
            int hit;
            kws_go = 0;
            c0 = kws_cycles();
            kws_board_run(inject_pcm, out8);
            c1 = kws_cycles();
            printf("t30:");
            for (i = 0; i < 4; i++) printf(" %d", (int8_t)out8[i]);
            printf("  t31:");
            for (i = 4; i < 8; i++) printf(" %d", (int)(int8_t)out8[i] + 128);
            printf("\r\n");
            for (j = 0; j < 4; j++)
                probs[j] = (float)((int)(int8_t)out8[4 + j] + 128) / 255.0f;
            hit = kws_decide_feed(&dec, probs, (float)(++win), &conf, &gap);
            if (hit == 3) kws_wake_c908();  /* 3 = huanxing */
            c2 = kws_cycles();
            if (hit)
                printf("decide: HIT %s conf=%d gap=%d\r\n", kws_name(hit),
                       (int)(conf * 1000.0f), (int)(gap * 1000.0f));
            else
                printf("decide: none conf=%d gap=%d\r\n",
                       (int)(conf * 1000.0f), (int)(gap * 1000.0f));
            printf("latency: infer=%d us decide_wake=%d us total=%d us\r\n",
                   (int)(c1 - c0) / 100, (int)(c2 - c1) / 100, (int)(c2 - c0) / 100);
        }
    }
    return 0;
}
#endif
