#include "kws_config.h"
#if !KWS_BURN
#include "kws_decim3.h"

/* Hamming FIR, fs=48k, fc=6.8kHz, Q15 */
static const int16_t COEF[DECIM3_TAPS] = {
    39, -7, -81, -139, -79, 162, 456, 477, -46, -960,
    -1552, -900, 1440, 4900, 8028, 9288, 8028, 4900, 1440, -900,
    -1552, -960, -46, 477, 456, 162, -79, -139, -81, -7, 39
};

void decim3_init(decim3_t *s)
{
    int i;
    for (i = 0; i < DECIM3_TAPS; i++) s->buf[i] = 0;
    s->idx = 0;
    s->phase = 0;
}

int decim3_put(decim3_t *s, int16_t x, int16_t *out)
{
    s->buf[s->idx] = x;
    s->idx = (s->idx + 1) % DECIM3_TAPS;

    if (s->phase != 2) {
        s->phase++;
        return 0;
    }
    s->phase = 0;

    int32_t acc = 1 << 14;
    int i;
    for (i = 0; i < (DECIM3_TAPS - 1) / 2; i++) {
        int p = (s->idx + i) % DECIM3_TAPS;
        int q = (s->idx + DECIM3_TAPS - 1 - i) % DECIM3_TAPS;
        acc += (int32_t)COEF[i] * ((int32_t)s->buf[p] + s->buf[q]);
    }
    {
        int c = (s->idx + (DECIM3_TAPS - 1) / 2) % DECIM3_TAPS;
        acc += (int32_t)COEF[(DECIM3_TAPS - 1) / 2] * s->buf[c];
    }

    int32_t y = acc >> 15;
    if (y > 32767) y = 32767;
    if (y < -32768) y = -32768;
    *out = (int16_t)y;
    return 1;
}

int decim3_process(decim3_t *s, const int16_t *in, int n, int16_t *out)
{
    int cnt = 0, i;
    for (i = 0; i < n; i++) {
        int16_t y;
        if (decim3_put(s, in[i], &y)) out[cnt++] = y;
    }
    return cnt;
}

#endif /* !KWS_BURN */
