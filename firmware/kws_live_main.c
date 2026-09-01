#include <stdint.h>
#include "printf.h"
#include "kws_config.h"

#if !KWS_BURN && KWS_LIVE

#include "common.h"
#include "vad.h"
#include "aps256xx.h"
#include "kws_decim3.h"
#include "kws_decide.h"
#include "gpio_reg.h"
#include "base_addr.h"

/* batch switch: 1 = capture + energy gate print only; 2 = full chain to wake c908 */
#define LIVE_STAGE 2

#define AON_AC_REG8   (SCU_BASE + 0x320)
#define EXT_STAT        (SCU_BASE + 0x22C) /* 0x1000622C: [2]=i_c908_ready [3]=PAD_FLASH_PGM (spec 17.8) */

extern void wm8731_init(void);
extern void i2c_transmit_status_check(void);
extern void iic_master_init(unsigned char dev_id);
int  vad_read_frame(int16_t *dst);
extern volatile unsigned int vad_last_dn;
#if LIVE_STAGE >= 2
void kws_board_run(const int16_t *pcm, uint8_t out8[8]);
void kws_w4_selfcheck(void);
#endif

static int16_t frame48[4096];
#if LIVE_STAGE >= 2
static int16_t live_pcm[16000];
#endif

/* energy gate v2 (BPGATE): trigger on first-order-difference energy to reject
 * mains hum / low-freq rumble; scratch (ret 2) still judged on raw signal.
 * o_mean/o_c150 now carry DIFF stats (used by LIVE_STAGE==1 calibration print) */
static int kws_gate_check(const int16_t *s, int n, int *o_peak, int *o_mean, int *o_c150)
{
    int i, dc, peak = 0, c150 = 0, meanv, c150_d = 0, meanv_d;
    long sum = 0, absum = 0, absum_d = 0;
    for (i = 0; i < n; i++) sum += s[i];
    dc = (int)(sum / n);
    for (i = 0; i < n; i++) {
        int a = s[i] - dc;
        if (a < 0) a = -a;
        if (a > peak) peak = a;
        absum += a;
        if (a > 150) c150++;
    }
    for (i = 1; i < n; i++) {
        int d = s[i] - s[i - 1];
        if (d < 0) d = -d;
        absum_d += d;
        if (d > 150) c150_d++;
    }
    meanv = (int)(absum / n);
    meanv_d = (int)(absum_d / (n - 1));
    *o_peak = peak; *o_mean = meanv_d; *o_c150 = c150_d;
    if (peak > 20000 || (peak > 8000 && meanv > 700)) return 2;
    if (meanv_d >= 90 && c150_d >= 500) return 1; /* BPGATE: calibrated on board wavs 2026-08-11 */
    return 0;
}

#if LIVE_STAGE >= 2
static void kws_wake_c908(void) {
    unsigned int v;
    long to;
    v = *(volatile unsigned int *)(GPIO_SWPORTA_DDR);
    *(volatile unsigned int *)(GPIO_SWPORTA_DDR) = v | (1u << 2);
    v = *(volatile unsigned int *)(GPIO_SWPORTA_DR);
    *(volatile unsigned int *)(GPIO_SWPORTA_DR) = v | (1u << 2);
    /* FIX20260812 spec17.8: acreg8[2]=c908_poweroff, 1=powerON, 0=off; old &=~ was powerOFF (wrong dir) */
    v = *(volatile unsigned int *)(AON_AC_REG8);
    *(volatile unsigned int *)(AON_AC_REG8) = v | (1u << 2);
    for (to = 0; to < 2000000; to++) {
        if (*(volatile unsigned int *)(EXT_STAT) & (1u << 2)) break;
    }
    if (*(volatile unsigned int *)(EXT_STAT) & (1u << 2)) printf("wake c908: READY ack\r\n");
    else printf("wake c908: power-on sent, ready TIMEOUT\r\n");
}

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

/* drain backlog frames after inference (READ_FINISH only at full frames) */
static void vad_flush(void) {
    int guard = 0;
    while (read(VAD_BASE + 0x4020) >= 4096u && guard < 40) {
        vad_read_frame(frame48);
        guard++;
    }
}
#endif

int main(void)
{
    int peak, mean, c150;
    unsigned int alive = 0;
#if LIVE_STAGE >= 2
    kws_decide_t dec;
    decim3_t d3;
    unsigned int win = 0;
    int capturing = 0;
    int post = 0;    /* PLANB: frames since trigger */
    int silcnt = 0;  /* PLANB: consecutive silent frames while capturing */
#endif
    *(volatile unsigned int *)(0x1000631C) = 0xFFEFAAAA; /* REG7 pull up disable, per working i2c_test main */
    iic_master_init(0x1A);
    wm8731_init();
    i2c_transmit_status_check();
    wm8731_init();
    i2c_transmit_status_check();
    vad_init();
    printf("vad r0=%08x r4=%08x r24=%08x r34=%08x\r\n",
           (unsigned)read(VAD_BASE + 0x4000), (unsigned)read(VAD_BASE + 0x4004),
           (unsigned)read(VAD_BASE + 0x4024), (unsigned)read(VAD_BASE + 0x4034));
    printf("vad r8=%08x rC=%08x r10=%08x i2smode=%08x\r\n",
           (unsigned)read(VAD_BASE + 0x4008), (unsigned)read(VAD_BASE + 0x400C),
           (unsigned)read(VAD_BASE + 0x4010), (unsigned)read(VAD_BASE + 0x403C));
#if LIVE_STAGE >= 2
    aps256xx_init();
    kws_w4_selfcheck();
    kws_decide_init(&dec, 0.4f, 0.1f, 0.0f);
#endif
    printf("kws live ready, stage=%d\r\n", LIVE_STAGE);

    while (1) {
        int g;
        if (!vad_read_frame(frame48)) {
            if (++alive >= 2000000) {
                printf("alive dn=%d st=%x\r\n", (int)read(VAD_BASE + 0x4020), (unsigned)read(VAD_BASE + 0x4028));
                alive = 0;
            }
            continue;
        }
        g = kws_gate_check(frame48, 4096, &peak, &mean, &c150);
#if LIVE_STAGE == 1
        printf("LIVE peak=%d mean=%d c150=%d gate=%c\r\n",
               peak, mean, c150, (g == 2) ? 'X' : (g == 1) ? 'S' : '.');
        { volatile int dly; for (dly = 0; dly < 2000000; dly++); }
#else
        /* PLANB: slide window every frame; live_pcm always holds the last 16000 samples */
        { int k2;
          for (k2 = 0; k2 < 16000 - 4096; k2++) live_pcm[k2] = live_pcm[k2 + 4096];
          for (k2 = 0; k2 < 4096; k2++) live_pcm[16000 - 4096 + k2] = frame48[k2]; }
        if (!capturing) {
            if (g == 1) {
                capturing = 1;
                post = 0;
                silcnt = 0;
                printf("CAP start\r\n");
            }
        } else {
            int do_infer = 0;
            post++;
            if (g == 0) silcnt++; else silcnt = 0;
            if (silcnt >= 2) do_infer = 1; /* PLANB: word end = 2 consecutive silent frames */
            if (post >= 6) do_infer = 1;   /* PLANB: safety bound, max 6 frames */
            { int fp = 0, k;
              for (k = 0; k < 4096; k++) { int av = frame48[k] < 0 ? -frame48[k] : frame48[k]; if (av > fp) fp = av; }
              printf("CF post=%d sil=%d pk=%d\r\n", post, silcnt, fp); }
            if (do_infer) {
                uint8_t out8[8];
                float probs[4], conf, gap;
                int hit, i;
                unsigned int c0, c1, c2;
                { long agc_acc = 0; int agc_i, agc_g = 256, av;
                  for (agc_i = 0; agc_i < 16000; agc_i++) { av = live_pcm[agc_i] < 0 ? -live_pcm[agc_i] : live_pcm[agc_i]; agc_acc += av; }
                  agc_acc = agc_acc / 16000;
                  if (agc_acc > 0 && agc_acc < 1200) {
                      agc_g = (int)((1200 * 256) / agc_acc); /* AGC: normalize mean|x| to 1200, max 8x */
                      if (agc_g > 2048) agc_g = 2048;
                      for (agc_i = 0; agc_i < 16000; agc_i++) {
                          int v = ((int)live_pcm[agc_i] * agc_g) >> 8;
                          if (v > 32767) v = 32767; if (v < -32768) v = -32768;
                          live_pcm[agc_i] = (int16_t)v; } }
                  printf("AGC ma=%d g=%d ; ", (int)agc_acc, agc_g); }
                c0 = kws_cycles();
                kws_board_run(live_pcm, out8);
                c1 = kws_cycles();
                printf("t30:");
                for (i = 0; i < 4; i++) printf(" %d", (int8_t)out8[i]);
                printf("  t31:");
                for (i = 4; i < 8; i++) printf(" %d", (int)(int8_t)out8[i] + 128);
                printf("\r\n");
                printf("PCM_BEGIN\r\n");
                for (i = 0; i < 16000; i += 16) {
                    int j;
                    for (j = 0; j < 16; j++) printf(" %d", (int)live_pcm[i + j]);
                    printf("\r\n");
                }
                printf("PCM_END\r\n");
                for (i = 0; i < 4; i++)
                    probs[i] = (float)((int)(int8_t)out8[4 + i] + 128) / 255.0f;
                hit = kws_decide_feed(&dec, probs, (float)(++win), &conf, &gap);
                if (hit == 3) kws_wake_c908();
                c2 = kws_cycles();
                if (hit)
                    printf("decide: HIT %s conf=%d gap=%d\r\n", kws_name(hit),
                           (int)(conf * 1000.0f), (int)(gap * 1000.0f));
                else
                    printf("decide: none conf=%d gap=%d\r\n",
                           (int)(conf * 1000.0f), (int)(gap * 1000.0f));
                printf("latency: infer=%d us decide_wake=%d us\r\n",
                       (int)(c1 - c0) / 100, (int)(c2 - c1) / 100);
                capturing = 0;
                vad_flush();
                printf("RDY\r\n");
            }
        }
#endif
    }
    return 0;
}
#endif /* !KWS_BURN && KWS_LIVE */
