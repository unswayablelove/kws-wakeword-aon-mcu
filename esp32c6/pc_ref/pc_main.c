/* PC 参考：直接 include 同一份 kws_e2e_nofpu_v3.c，输出格式与 C6 串口一致 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "kws_inject_audio.h"
int32_t W4_AT(int32_t k);
#include "kws_e2e_nofpu_v3.c"
int32_t W4_AT(int32_t k) {
    return (int32_t)((k >= 0 && k < 120000) ? W4_A[k] : W4_B[k - 120000]);
}
int main(void) {
    static int8_t feat[49*40];
    int8_t t30[4], t31[4];
    int clip, r, c, i;
    init_tables();
    printf("KWS_BEGIN\n");
    for (clip = 0; clip < KWS_INJECT_NUM; clip++) {
        memset(feat, 0x55, sizeof(feat));
        extract_feat_int8(kws_clips[clip], feat);
        infer(feat, t30, t31);
        printf("CLIP %d\n", clip);
        for (r = 0; r < 49; r++)
            for (c = 0; c < 40; c++)
                printf("feat %d %d %d %d\n", clip, r, c, (int)feat[r*40+c]);
        printf("t30 %d", clip);
        for (i = 0; i < 4; i++) printf(" %d", (int)t30[i]);
        printf("\nt31 %d", clip);
        for (i = 0; i < 4; i++) printf(" %d", (int)t31[i]);
        printf("\n");
    }
    printf("KWS_END\n");
    return 0;
}
