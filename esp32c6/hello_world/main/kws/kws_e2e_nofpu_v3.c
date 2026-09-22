#include "kws_config.h"
#if !KWS_BURN
/* kws_e2e_nofpu_v3.c — kws_e2e.c 的无 FPU 版本 v3（修复 exp_q16 移位 UB）
 * 改动: softmax 由 double exp 改为定点查表 exp_q16 + 整数除法，
 *       运行时路径（extract_feat_int8 + infer）零浮点，可直接上 E906。
 *       init_tables 中的 libm 浮点仅上电初始化执行一次，保留不动。
 * 输出约定不变: out.bin 8 字节 = t30[4] int8 + t31[4] int8，t31 仍舍入 1/256 格子。
 * 已知误差: exp 查表误差 ~1e-5 量级，t31 可能偶发 ±1 LSB（与 fc2 同级，有界不扩散）。
 * 编译: gcc -O2 -o kws_e2e kws_e2e_nofpu_v2.c -lm   （需要同目录 kws_weights.h）
 * 用法: ./kws_e2e audio.s16 out.bin
 *   audio.s16: 16000 个 int16（1 秒窗）
 *   out.bin  : 8 字节 = t30[4] int8 + t31[4] int8 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "kws_weights.h"
int32_t W4_AT(int32_t k) {
    return (int32_t)((k >= 0 && k < 120000) ? W4_A[k] : W4_B[k - 120000]);
}

#define SEC_WRAM0
#define SEC_WRAM1

#define SAMPLE_RATE 16000
#define FRAME_LEN   480
#define FRAME_STEP  320
#define NFFT        512
#define NUM_MEL     40
#define NUM_FRAMES  49
#define NBINS       (NFFT/2+1)

/* v4 量化参数 */
#define IN_SCALE  0.03337131068110466
#define IN_ZERO   40

/* v4 归一化参数（float 常量，仅用于初始化时生成定点表） */
static const float MEAN_F[NUM_MEL] = {
-15.943873f,-14.16633f,-12.998032f,-12.462342f,-12.318329f,-12.704891f,-12.857549f,-12.660684f,
-12.195224f,-12.259253f,-12.342042f,-12.198825f,-12.3698f,-12.4610615f,-12.394521f,-12.323932f,
-12.414016f,-12.5005245f,-12.309226f,-12.138959f,-12.1677f,-12.141489f,-11.859203f,-11.67147f,
-11.778436f,-11.855022f,-11.824253f,-11.767101f,-11.634663f,-11.674359f,-11.811094f,-11.844134f,
-11.938671f,-12.157752f,-12.199463f,-12.262533f,-12.318928f,-12.540266f,-13.091838f,-14.084435f };
static const float STD_F[NUM_MEL] = {
3.8046782f,3.9481246f,4.2083883f,4.525759f,4.7511187f,4.661246f,4.578508f,4.601244f,
4.7056847f,4.707778f,4.671908f,4.708175f,4.6861043f,4.6325207f,4.5717707f,4.504851f,
4.4351234f,4.398801f,4.389547f,4.370672f,4.34508f,4.331155f,4.330351f,4.3580027f,
4.3851905f,4.380036f,4.3976364f,4.4038687f,4.4182887f,4.4240932f,4.4209657f,4.413164f,
4.455687f,4.505014f,4.5425143f,4.6010294f,4.6517916f,4.72842f,4.982264f,5.4091825f };

static int32_t TWR[NFFT], TWI[NFFT];     /* Q31 旋转因子 */
static int16_t WIN_Q15[FRAME_LEN];       /* Q15 汉宁窗 */
static int16_t FB_Q15[NUM_MEL][NBINS] SEC_WRAM1;   /* Q15 Mel 滤波器组 */
static int32_t MEAN_Q16[NUM_MEL];        /* Q16 mean */
static int32_t A_Q14[NUM_MEL];           /* Q14: 1/(std*in_s) */
static uint16_t LOG2_TAB[256];           /* log2(1+i/256) 的 Q16（0..65436） */
static uint32_t EXP2_TAB[256];           /* 2^(i/256) 的 Q16（65536..131071） */
static int32_t  T30S_Q16;                /* logits scale 的 Q16 */

static void init_tables(void) {
    int n, m, k;
    for (n = 0; n < NFFT; n++) {
        int64_t vr = llround(cos(-2.0*M_PI*n/NFFT) * 2147483648.0);
        int64_t vi = llround(sin(-2.0*M_PI*n/NFFT) * 2147483648.0);
        if (vr > INT32_MAX) vr = INT32_MAX;
        if (vi > INT32_MAX) vi = INT32_MAX;
        TWR[n] = (int32_t)vr;
        TWI[n] = (int32_t)vi;
    }
    for (n = 0; n < FRAME_LEN; n++) {
        int64_t w = lround((0.54 - 0.46*cos(2.0*M_PI*n/(FRAME_LEN-1))) * 32768.0);
        if (w > 32767) w = 32767;
        WIN_Q15[n] = (int16_t)w;
    }
    {
        double low = 0.0, high = 2595.0 * log10(1.0 + SAMPLE_RATE / 1400.0);
        int bins[NUM_MEL + 2];
        for (m = 0; m < NUM_MEL + 2; m++) {
            double pts = low + (high - low) * m / (NUM_MEL + 1);
            double hz = 700.0 * (pow(10.0, pts / 2595.0) - 1.0);
            int b = (int)floor((NFFT + 1) * hz / SAMPLE_RATE);
            if (b < 0) b = 0; if (b > NFFT/2) b = NFFT/2;
            bins[m] = b;
        }
        memset(FB_Q15, 0, sizeof(FB_Q15));
        for (m = 1; m <= NUM_MEL; m++) {
            int den;
            den = bins[m] - bins[m-1]; if (den < 1) den = 1;
            for (k = bins[m-1]; k < bins[m]; k++) {
                int64_t v = lround((double)(k - bins[m-1]) / den * 32768.0);
                if (v > 32767) v = 32767;
                FB_Q15[m-1][k] = (int16_t)v;
            }
            den = bins[m+1] - bins[m]; if (den < 1) den = 1;
            for (k = bins[m]; k < bins[m+1]; k++) {
                int64_t v = lround((double)(bins[m+1] - k) / den * 32768.0);
                if (v > 32767) v = 32767;
                FB_Q15[m-1][k] = (int16_t)v;
            }
        }
    }
    for (m = 0; m < NUM_MEL; m++) {
        MEAN_Q16[m] = (int32_t)lround(MEAN_F[m] * 65536.0);
        A_Q14[m]    = (int32_t)lround(16384.0 / (STD_F[m] * IN_SCALE));
    }
    for (k = 0; k < 256; k++)
        LOG2_TAB[k] = (uint16_t)lround(log2(1.0 + k/256.0) * 65536.0);
    for (k = 0; k < 256; k++)
        EXP2_TAB[k] = (uint32_t)lround(pow(2.0, k / 256.0) * 65536.0);
    T30S_Q16 = (int32_t)lround(KWS_T30_S * 65536.0);
}

static int32_t log_q16(int64_t e, int off) {
    int b = 63 - __builtin_clzll((unsigned long long)e);
    int64_t m12 = (b >= 12) ? (e >> (b - 12)) & 0xFFF : (e << (12 - b)) & 0xFFF;
    int idx = (int)(m12 >> 4), rem = (int)(m12 & 0xF);
    int32_t v0 = LOG2_TAB[idx];
    int32_t v1 = (idx == 255) ? 65536 : LOG2_TAB[idx + 1];
    int32_t frac = v0 + ((v1 - v0) * rem >> 4);
    int32_t log2_q16 = ((b - off) << 16) + frac;
    return (int32_t)(((int64_t)log2_q16 * 45426) >> 16);
}

static void fft_fixed(int32_t *re, int32_t *im, int n) {
    int i, j, k, m;
    for (i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { int32_t t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (m = 2; m <= n; m <<= 1) {
        int step = NFFT / m;
        for (k = 0; k < n; k += m) {
            for (j = 0; j < m/2; j++) {
                int u = k + j, v = k + j + m/2;
                int32_t twr = TWR[j*step], twi = TWI[j*step];
                int32_t tr = (int32_t)(((int64_t)re[v]*twr - (int64_t)im[v]*twi + (1<<30)) >> 31);
                int32_t ti = (int32_t)(((int64_t)re[v]*twi + (int64_t)im[v]*twr + (1<<30)) >> 31);
                re[v] = re[u] - tr; im[v] = im[u] - ti;
                re[u] += tr;        im[u] += ti;
            }
        }
    }
}

void extract_feat_int8(const int16_t *pcm, int8_t *out) {
    static int32_t pe[SAMPLE_RATE] SEC_WRAM0;
    static int32_t re[NFFT], im[NFFT];
    static int64_t p2[NBINS];
    int i, j, k, f;
    const int32_t EPS_LN_Q16 = (int32_t)lround(-15.942384719848633 * 65536.0);
#define PE_K 10
    pe[0] = (int32_t)pcm[0] << PE_K;
    for (i = 1; i < SAMPLE_RATE; i++)
        pe[i] = ((int32_t)pcm[i] << PE_K)
              - (int32_t)((((int64_t)pcm[i-1] * 31785 << PE_K) + (1<<14)) >> 15);
    int nf = 1 + (SAMPLE_RATE - FRAME_LEN) / FRAME_STEP;
    if (nf > NUM_FRAMES) nf = NUM_FRAMES;
    for (f = 0; f < nf; f++) {
        int32_t W = 0;
        int s, m, off;
        int64_t sump2 = 0, pmax = 0;
        for (i = 0; i < FRAME_LEN; i++) {
            re[i] = (int32_t)(((int64_t)pe[f*FRAME_STEP + i] * WIN_Q15[i] + (1<<14)) >> 15);
            int32_t a = re[i] < 0 ? -re[i] : re[i];
            if (a > W) W = a;
        }
        for (; i < NFFT; i++) re[i] = 0;
        memset(im, 0, sizeof(im));
        if (W == 0) {
            for (j = 0; j < NUM_MEL; j++) {
                int64_t t = (int64_t)(EPS_LN_Q16 - MEAN_Q16[j]) * A_Q14[j];
                int32_t q = (int32_t)((t + (1<<29)) >> 30) + IN_ZERO;
                if (q > 127) q = 127; if (q < -128) q = -128;
                out[f*NUM_MEL + j] = (int8_t)q;
            }
            continue;
        }
        {
            int c = 32 - __builtin_clz((unsigned)W);
            s = 20 - c; if (s < 0) s = 0;
        }
        if (s > 0)
            for (i = 0; i < FRAME_LEN; i++) re[i] *= (1 << s);
        fft_fixed(re, im, NFFT);
        for (k = 0; k < NBINS; k++) {
            p2[k] = (int64_t)re[k]*re[k] + (int64_t)im[k]*im[k];
            sump2 += p2[k];
            if (p2[k] > pmax) pmax = p2[k];
        }
        m = (63 - __builtin_clzll((unsigned long long)sump2)) - 47;
        if (m < 0) m = 0;
        off = 54 + 2*PE_K + 2*s - m;
        for (j = 0; j < NUM_MEL; j++) {
            int64_t e = 0;
            for (k = 0; k < NBINS; k++) {
                int64_t pv = (m > 0) ? ((p2[k] + ((int64_t)1 << (m-1))) >> m) : p2[k];
                e += pv * FB_Q15[j][k];
            }
            int32_t lq = (e == 0) ? (int32_t)(-(int64_t)off * 45426) : log_q16(e, off);
            int64_t t = (int64_t)(lq - MEAN_Q16[j]) * A_Q14[j];
            int32_t q = (int32_t)((t + (1<<29)) >> 30) + IN_ZERO;
            if (q > 127) q = 127; if (q < -128) q = -128;
            out[f*NUM_MEL + j] = (int8_t)q;
        }
    }
    for (; f < NUM_FRAMES; f++)
        for (j = 0; j < NUM_MEL; j++) out[f*NUM_MEL + j] = 0;
}

/* ================= 推理部分（与 kws_infer.c 逐字一致） ================= */
static int8_t T21[25*20*16], T22[25*20*16], T24[12*10*32], T29[64];
static int8_t T23[25*20*32] SEC_WRAM1;

static inline int32_t rdbpot(int32_t x, int s) {
    if (s <= 0) return x;
    int32_t mask = (1 << s) - 1;
    int32_t rem = x & mask;
    int32_t thr = (mask >> 1) + (x < 0);
    return (x >> s) + (rem > thr);
}
static inline int32_t srdhm(int32_t a, int32_t q) {
    int64_t p = (int64_t)a * q;
    int64_t nudge = (p >= 0) ? (1LL << 30) : (1LL - (1LL << 30));
    int64_t r = (p + nudge) >> 31;
    if (r > INT32_MAX) return INT32_MAX;
    if (r < INT32_MIN) return INT32_MIN;
    return (int32_t)r;
}
static inline int8_t requant(int32_t acc, int32_t mult, int32_t shift, int32_t out_z) {
    if (shift < 0) acc = (int32_t)((int64_t)acc << (-shift));
    int32_t v = rdbpot(srdhm(acc, mult), shift > 0 ? shift : 0) + out_z;
    if (v < -128) v = -128;
    if (v > 127)  v = 127;
    return (int8_t)v;
}

/* exp(x)，x 为 Q16 且 <= 0，返回 Q16；纯整数查表+线性插值，误差 ~1e-5
 * v3 修复移位 UB（x86 截断移位次数、RISC-V 出错）双保险：
 *   1) x <= -21 直接返回 0（exp(-21)≈7.6e-10，对 1/256 格子无影响）；
 *   2) k <= -31 直接返回 0，保证 val >> (-k) 的移位次数恒在 [0,30]。
 * 验证: gcc -O2 下 200 万随机 t30 输入，与浮点参考最大偏差 1 LSB。 */
static uint32_t exp_q16(int32_t x) {
    int32_t y, k, frac, idx, rem;
    uint32_t v0, v1, val;
    if (x <= -(21 << 16)) return 0;                      /* 保险 1：小尾巴钳位 */
    y = (int32_t)(((int64_t)x * 94549) >> 16);           /* x*log2(e) → log2 域 Q16 */
    k = y >> 16;                                         /* 整数部分（算术右移=floor） */
    if (k <= -31) return 0;                              /* 保险 2：移位边界钳位 */
    frac = y & 0xFFFF;                                   /* 小数部分 [0,65536) */
    idx = frac >> 8; rem = frac & 0xFF;
    v0 = EXP2_TAB[idx];
    v1 = (idx == 255) ? 131072u : EXP2_TAB[idx + 1];
    val = v0 + (((v1 - v0) * (uint32_t)rem) >> 8);       /* 2^frac, Q16 */
    return val >> (-k);                                  /* -k 范围 [0,30]，安全 */
}

static void infer(const int8_t x[49*40], int8_t t30[4], int8_t t31[4]) {
    int oy, ox, ky, kx, c, i, iy, ix;
    for (oy = 0; oy < 25; oy++) for (ox = 0; ox < 20; ox++) for (c = 0; c < 16; c++) {
        int32_t acc = B1[c];
        for (ky = 0; ky < 10; ky++) {
            iy = oy*2 + ky - 4; if (iy < 0 || iy >= 49) continue;
            for (kx = 0; kx < 8; kx++) {
                ix = ox*2 + kx - 3; if (ix < 0 || ix >= 40) continue;
                acc += ((int32_t)x[iy*40+ix] - KWS_IN_Z) * W1[c*80 + ky*8 + kx];
            }
        }
        T21[(oy*20+ox)*16+c] = requant(acc, MULT1[c], SHIFT1[c], KWS_Z21);
    }
    for (oy = 0; oy < 25; oy++) for (ox = 0; ox < 20; ox++) for (c = 0; c < 16; c++) {
        int32_t acc = B2[c];
        for (ky = 0; ky < 3; ky++) {
            iy = oy + ky - 1; if (iy < 0 || iy >= 25) continue;
            for (kx = 0; kx < 3; kx++) {
                ix = ox + kx - 1; if (ix < 0 || ix >= 20) continue;
                acc += ((int32_t)T21[(iy*20+ix)*16+c] - KWS_Z21) * W2[(ky*3+kx)*16+c];
            }
        }
        T22[(oy*20+ox)*16+c] = requant(acc, MULT2[c], SHIFT2[c], KWS_Z22);
    }
    for (oy = 0; oy < 25; oy++) for (ox = 0; ox < 20; ox++) for (c = 0; c < 32; c++) {
        int32_t acc = B3[c];
        for (i = 0; i < 16; i++)
            acc += ((int32_t)T22[(oy*20+ox)*16+i] - KWS_Z22) * W3[c*16+i];
        T23[(oy*20+ox)*32+c] = requant(acc, MULT3[c], SHIFT3[c], KWS_Z23);
    }
    for (oy = 0; oy < 12; oy++) for (ox = 0; ox < 10; ox++) for (c = 0; c < 32; c++) {
        int32_t s = (int32_t)T23[((oy*2)*20   + ox*2  )*32+c] + T23[((oy*2)*20   + ox*2+1)*32+c]
                  + (int32_t)T23[((oy*2+1)*20 + ox*2  )*32+c] + T23[((oy*2+1)*20 + ox*2+1)*32+c];
        int32_t v = rdbpot(s, 2);
        if (v < -128) v = -128;
        if (v > 127)  v = 127;
        T24[(oy*10+ox)*32+c] = (int8_t)v;
    }
    for (c = 0; c < 64; c++) {
        int32_t acc = B4[c];
        for (i = 0; i < 3840; i++)
            acc += ((int32_t)T24[i] - KWS_Z23) * W4_AT(c*3840+i);
        T29[c] = requant(acc, MULT4[c], SHIFT4[c], KWS_Z29);
    }
    for (c = 0; c < 4; c++) {
        int32_t acc = B5[c];
        for (i = 0; i < 64; i++)
            acc += ((int32_t)T29[i] - KWS_Z29) * W5[c*64+i];
        t30[c] = requant(acc, MULT5[c], SHIFT5[c], KWS_Z30);
    }
    /* softmax 定点版：exp 查表 + 整数除法，输出仍舍入到 1/256 格子 */
    int32_t lq[4], mxq = INT32_MIN;
    uint32_t e[4]; uint64_t sum = 0;
    for (c = 0; c < 4; c++) {
        lq[c] = (t30[c] - KWS_Z30) * T30S_Q16;   /* logits → Q16 */
        if (lq[c] > mxq) mxq = lq[c];
    }
    for (c = 0; c < 4; c++) { e[c] = exp_q16(lq[c] - mxq); sum += e[c]; }
    for (c = 0; c < 4; c++) {
        int32_t v = (int32_t)(((uint64_t)e[c] * 256 + (sum >> 1)) / sum) + KWS_Z31;
        if (v < -128) v = -128;
        if (v > 127)  v = 127;
        t31[c] = (int8_t)v;
    }
}

int kws_pc_main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s audio.s16 out.bin\n", argv[0]); return 1; }
    static int16_t pcm[SAMPLE_RATE] SEC_WRAM0;
    static int8_t feat[NUM_FRAMES*NUM_MEL];
    int8_t t30[4], t31[4], obuf[8];
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) { perror("open audio"); return 1; }
    size_t n = fread(pcm, sizeof(int16_t), SAMPLE_RATE, fp);
    fclose(fp);
    if (n != SAMPLE_RATE) fprintf(stderr, "warn: read %zu samples, expect %d\n", n, SAMPLE_RATE);
    init_tables();
    extract_feat_int8(pcm, feat);
    infer(feat, t30, t31);
    memcpy(obuf, t30, 4); memcpy(obuf + 4, t31, 4);
    fp = fopen(argv[2], "wb");
    if (!fp) { perror("open out"); return 1; }
    fwrite(obuf, 1, 8, fp);
    fclose(fp);
    return 0;
}

/* ---- board injection wrapper (appended 0801, sealed logic untouched) ---- */
static int g_kws_inited = 0;
void kws_board_run(const int16_t *pcm, uint8_t out8[8]) {
    static int8_t feat[NUM_FRAMES*NUM_MEL];
    int8_t t30[4], t31[4];
    if (!g_kws_inited) { init_tables(); g_kws_inited = 1; }
    extract_feat_int8(pcm, feat);
    infer(feat, t30, t31);
    memcpy(out8, t30, 4); memcpy(out8 + 4, t31, 4);
}

/* ---- ESP32-C6 inject compare entry: feat + logits ---- */
void kws_c6_run(const int16_t *pcm, int8_t *feat, int8_t t30[4], int8_t t31[4]) {
    if (!g_kws_inited) { init_tables(); g_kws_inited = 1; }
    extract_feat_int8(pcm, feat);
    infer(feat, t30, t31);
}
#endif /* !KWS_BURN */
