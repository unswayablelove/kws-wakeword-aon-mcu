#include "kws_config.h"
#if !KWS_BURN
#include "kws_decide.h"

void kws_decide_init(kws_decide_t *d, float thr, float margin, float refractory)
{
    d->thr = thr;
    d->margin = margin;
    d->refractory = refractory;
    d->last_t = 0.0f;
    d->last_label = -1;
}

int kws_decide_feed(kws_decide_t *d, const float probs[KWS_NCLASS], float t,
                    float *conf, float *gap)
{
    int k = 1, j;
    float second, g;

    for (j = 2; j < KWS_NCLASS; j++)
        if (probs[j] > probs[k]) k = j;

    second = probs[0];
    for (j = 1; j < KWS_NCLASS; j++)
        if (j != k && probs[j] > second) second = probs[j];

    g = probs[k] - second;
    if (conf) *conf = probs[k];
    if (gap)  *gap  = g;
    if (probs[k] < d->thr || g < d->margin) return 0;

    if (d->last_label == k && (t - d->last_t) <= d->refractory) return 0;

    d->last_label = k;
    d->last_t = t;
    return k;
}

#endif /* !KWS_BURN */
