/* KWS ESP32-C6 阶段四：实时麦克风唤醒 (KWS_LIVE=1) / 注入对拍回归 (KWS_LIVE=0)
 * 实时链路：INMP441(I2S STD 16k/32bit/Philips/左声道) -> int16 环形缓冲(16000)
 *          -> 每 4000 新样本触发 extract_feat_int8 + infer + kws_decide
 * 决策换算与 GX-I2502 参考一致：probs = (t31 + 128) / 255.0f，thr=0.4 margin=0.1 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"

#define KWS_LIVE 1

#define NUM_FRAMES 49
#define NUM_MEL    40

void kws_c6_run(const int16_t *pcm, int8_t *feat, int8_t t30[4], int8_t t31[4]);

#if KWS_LIVE

#include "driver/i2s_std.h"
#include "kws/kws_decide.h"

#define I2S_BCLK      10
#define I2S_WS        11
#define I2S_DIN       2
#define SAMPLE_RATE   16000
#define WIN_LEN       16000   /* 1 秒窗 */
#define STRIDE        4000    /* 0.25s 步进 */
#define I2S_CHUNK     512     /* 每次 DMA 读的 32bit 样本数 */

static i2s_chan_handle_t rx_chan;
static volatile int16_t ring[WIN_LEN];
static volatile int ring_w = 0;          /* 写指针（单调位置对 WIN_LEN 取模由快照处处理） */
static volatile int new_samples = 0;     /* 自上次推理以来新写入的样本数 */

static int8_t feat[NUM_FRAMES * NUM_MEL];

static const char *kws_name(int k) {
    switch (k) {
        case 1: return "dakai";
        case 2: return "guanbi";
        case 3: return "huanxing";
        default: return "none";
    }
}

static void i2s_init(void) {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_chan));
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT,
                                                        I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_BCLK,
            .ws   = I2S_WS,
            .dout = I2S_GPIO_UNUSED,
            .din  = I2S_DIN,
            .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
        },
    };
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;  /* L/R 接地 = 左声道 */
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_chan, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));
}

/* INMP441: 24bit 数据位于 32bit 槽高 24 位，>>16 取高 16 位。
 * 阶段二实测 raw peak 上百万级，>>16 后语音幅度落在 int16 千级，再经 AGC 归一。 */
static void audio_task(void *arg) {
    int32_t buf[I2S_CHUNK];
    size_t n = 0;
    while (1) {
        if (i2s_channel_read(rx_chan, buf, sizeof(buf), &n, portMAX_DELAY) == ESP_OK && n) {
            int cnt = n / sizeof(int32_t);
            for (int i = 0; i < cnt; i++) {
                ring[ring_w] = (int16_t)(buf[i] >> 16);
                ring_w = (ring_w + 1) % WIN_LEN;
            }
            new_samples += cnt;
        }
    }
}

/* 能量门限：原始 mean|x| 低于下限视为环境底噪，跳过决策（防 AGC 放大底噪误触发） */
#define GATE_MA_MIN 150

/* 与 kws_live_main.c 相同的 AGC：mean|x| 归一到 1200，最大 8x；返回原始 mean|x| */
static int kws_agc(int16_t *pcm) {
    long acc = 0;
    for (int i = 0; i < WIN_LEN; i++) { int a = pcm[i] < 0 ? -pcm[i] : pcm[i]; acc += a; }
    acc /= WIN_LEN;
    if (acc > 0 && acc < 1200) {
        int g = (int)((1200 * 256) / acc);
        if (g > 2048) g = 2048;
        for (int i = 0; i < WIN_LEN; i++) {
            int v = ((int)pcm[i] * g) >> 8;
            if (v > 32767) v = 32767; if (v < -32768) v = -32768;
            pcm[i] = (int16_t)v;
        }
    }
    return (int)acc;
}

static void infer_task(void *arg) {
    static int16_t pcm[WIN_LEN];
    kws_decide_t dec;
    int8_t t30[4], t31[4];
    unsigned int win = 0;
    int64_t t_feat = 0, t_infer = 0;

    kws_decide_init(&dec, 0.4f, 0.1f, 4.0f);  /* refractory=4 窗=1s 去重 */
    printf("kws live ready (C6, -O2)\n");
    while (1) {
        if (new_samples < STRIDE) { vTaskDelay(pdMS_TO_TICKS(20)); continue; }
        new_samples = 0;
        /* 快照最近 16000 样本（最旧在前） */
        int w = ring_w;
        for (int i = 0; i < WIN_LEN; i++)
            pcm[i] = ring[(w + i) % WIN_LEN];
        int ma = kws_agc(pcm);
        int64_t t0 = esp_timer_get_time();
        kws_c6_run(pcm, feat, t30, t31);
        int64_t t1 = esp_timer_get_time();
        t_infer = t1 - t0; t_feat = t_infer;
        float probs[4], conf, gap;
        for (int i = 0; i < 4; i++)
            probs[i] = (float)((int)t31[i] + 128) / 255.0f;
        int hit = (ma >= GATE_MA_MIN)
                ? kws_decide_feed(&dec, probs, (float)(++win), &conf, &gap) : 0;
        printf("win=%u ma=%d t30: %d %d %d %d  t31: %d %d %d %d  infer=%lld us",
               win, ma, (int)t30[0], (int)t30[1], (int)t30[2], (int)t30[3],
               (int)t31[0] + 128, (int)t31[1] + 128, (int)t31[2] + 128, (int)t31[3] + 128,
               (long long)t_infer);
        if (hit)
            printf("  KWS_EVT %s conf=%d gap=%d", kws_name(hit),
                   (int)(conf * 1000.0f), (int)(gap * 1000.0f));
        printf("\n");
        (void)t_feat;
    }
}

void app_main(void) {
    esp_task_wdt_deinit();
    i2s_init();
    xTaskCreate(audio_task, "audio", 4096, NULL, 10, NULL);
    xTaskCreate(infer_task, "infer", 8192, NULL, 5, NULL);
}

#else /* !KWS_LIVE：注入对拍回归模式 */

#include "kws/kws_inject_audio.h"

static int8_t feat[NUM_FRAMES * NUM_MEL];

void app_main(void) {
    int clip, r, c, i;
    int8_t t30[4], t31[4];

    esp_task_wdt_deinit();
    vTaskDelay(pdMS_TO_TICKS(200));
    printf("KWS_BEGIN\n");
    for (clip = 0; clip < KWS_INJECT_NUM; clip++) {
        int64_t t0 = esp_timer_get_time();
        memset(feat, 0x55, sizeof(feat));
        kws_c6_run(kws_clips[clip], feat, t30, t31);
        int64_t t1 = esp_timer_get_time();
        printf("CLIP %d infer=%lld us\n", clip, (long long)(t1 - t0));
        for (r = 0; r < NUM_FRAMES; r++)
            for (c = 0; c < NUM_MEL; c++)
                printf("feat %d %d %d %d\n", clip, r, c, (int)feat[r * NUM_MEL + c]);
        printf("t30 %d", clip);
        for (i = 0; i < 4; i++) printf(" %d", (int)t30[i]);
        printf("\nt31 %d", clip);
        for (i = 0; i < 4; i++) printf(" %d", (int)t31[i]);
        printf("\n");
    }
    printf("KWS_END\n");
    fflush(stdout);
    while (1) vTaskDelay(pdMS_TO_TICKS(1000));
}

#endif /* KWS_LIVE */
