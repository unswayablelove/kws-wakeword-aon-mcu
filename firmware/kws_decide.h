#ifndef KWS_DECIDE_H
#define KWS_DECIDE_H

#define KWS_NCLASS 4   /* 0=negative 1=dakai 2=guanbi 3=huanxing */

typedef struct {
    float thr;
    float margin;
    float refractory;
    float last_t;
    int   last_label;
} kws_decide_t;

void kws_decide_init(kws_decide_t *d, float thr, float margin, float refractory);
int kws_decide_feed(kws_decide_t *d, const float probs[KWS_NCLASS], float t,
                    float *conf, float *gap);

#endif
