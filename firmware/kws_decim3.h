#ifndef KWS_DECIM3_H
#define KWS_DECIM3_H

#include <stdint.h>

/* 48kHz -> 16kHz ÷3 decimation, 31 taps Q15 anti-alias FIR
 * passband 0-6kHz flat, -16.6dB at 8kHz fold point, -59dB above 10kHz
 * int32 accumulator verified no overflow (max 1.57e9 < 2.15e9) */

#define DECIM3_TAPS 31

typedef struct {
    int16_t buf[DECIM3_TAPS];
    uint8_t idx;
    uint8_t phase;
} decim3_t;

void decim3_init(decim3_t *s);
int decim3_put(decim3_t *s, int16_t x, int16_t *out);
int decim3_process(decim3_t *s, const int16_t *in, int n, int16_t *out);

#endif
