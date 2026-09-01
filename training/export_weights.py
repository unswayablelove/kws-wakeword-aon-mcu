#!/usr/bin/env python3
# export_weights.py — 把 tflite 模型的权重/偏置/requant 参数导出成 kws_weights.h
# requant 系数用 float32 单精度预算（与 TFLite C++ 内部一致，探针已验证）
# 用法: python3 export_weights.py kimi_kws_0724_4cls_v4_int8.tflite
# 产物: kws_weights.h（与 kws_infer.c 同目录即可）
import sys, math
import numpy as np

def get_interpreter(path):
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter(model_path=path)
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter(model_path=path)
    except ImportError:
        pass
    import tensorflow as tf
    return tf.lite.Interpreter(model_path=path)

def quant_mult(m):                     # 与探针逐字一致：m=f*2^e, f∈[0.5,1) → q=f*2^31
    if m == 0.0: return 0, 0
    f, e = math.frexp(m)
    q = int(round(f * (1 << 31)))
    if q == (1 << 31): q >>= 1; e += 1
    return np.int32(q), e

def rq_params(ws, in_s, out_s):
    """每通道 (mult, shift)：m 用 float32 精度算；shift = -e（>0 右移，<0 左移）"""
    mult = np.zeros(len(ws), dtype=np.int64)
    shift = np.zeros(len(ws), dtype=np.int64)
    for c in range(len(ws)):
        m = np.float32(np.float32(in_s) * np.float32(ws[c]))
        m = np.float32(m / np.float32(out_s))
        q, e = quant_mult(float(m))
        mult[c], shift[c] = int(q), -e
    return mult, shift

def emit_arr(f, name, ctype, vals, per_line=16):
    flat = np.asarray(vals).reshape(-1)
    f.write(f"static const {ctype} {name}[{flat.size}] = {{\n")
    for i in range(0, flat.size, per_line):
        f.write("    " + ",".join(str(int(v)) for v in flat[i:i+per_line]) + ",\n")
    f.write("};\n\n")

def main():
    it = get_interpreter(sys.argv[1]); it.allocate_tensors()
    det = {d['index']: d for d in it.get_tensor_details()}
    def q(idx):  return det[idx]['quantization']
    def qs(idx): return np.asarray(det[idx]['quantization_parameters']['scales'], dtype=np.float64)
    def w(idx):  return it.get_tensor(idx)

    # 张量 ID 来自 stage4_dump 解剖结果；加载后逐一核对形状
    W1, B1, WS1 = w(16), w(15), qs(16)
    W2, B2, WS2 = w(14), w(13), qs(14)
    W3, B3, WS3 = w(12), w(11), qs(12)
    W4, B4, WS4 = w(10), w(9),  qs(10)
    W5, B5, WS5 = w(8),  w(7),  qs(8)
    assert W1.shape == (16, 10, 8, 1),  W1.shape
    assert W2.shape == (1, 3, 3, 16),   W2.shape
    assert W3.shape == (32, 1, 1, 16),  W3.shape
    assert W4.shape == (64, 3840),      W4.shape
    assert W5.shape == (4, 64),         W5.shape
    IN_S, IN_Z = q(0); S21, Z21 = q(21); S22, Z22 = q(22); S23, Z23 = q(23)
    S29, Z29 = q(29); S30, Z30 = q(30); S31, Z31 = q(31)

    M1, SH1 = rq_params(WS1, IN_S, S21)
    M2, SH2 = rq_params(WS2, S21, S22)
    M3, SH3 = rq_params(WS3, S22, S23)
    M4, SH4 = rq_params(WS4, S23, S29)
    M5, SH5 = rq_params(WS5, S29, S30)
    for name, sh in [('1', SH1), ('2', SH2), ('3', SH3), ('4', SH4), ('5', SH5)]:
        if (sh < 0).any():
            print(f"警告: 第{name}层出现左移（e>0），shift={sh[sh<0]}（C 侧已通用处理，但请把本行发我）")

    with open('kws_weights.h', 'w') as f:
        f.write("/* kws_weights.h — export_weights.py 自动生成，勿手改\n")
        f.write("   requant 系数 float32 精度预算（TFLite C++ 行为）；shift>0=右移, <0=左移 */\n")
        f.write("#ifndef KWS_WEIGHTS_H\n#define KWS_WEIGHTS_H\n#include <stdint.h>\n\n")
        f.write(f"#define KWS_IN_Z  ({IN_Z})\n")
        f.write(f"#define KWS_Z21   ({Z21})\n#define KWS_Z22   ({Z22})\n#define KWS_Z23   ({Z23})\n")
        f.write(f"#define KWS_Z29   ({Z29})\n#define KWS_Z30   ({Z30})\n#define KWS_Z31   ({Z31})\n")
        f.write(f"#define KWS_T30_S ({S30:.17g})\n\n")
        emit_arr(f, 'W1', 'int8_t',  W1);  emit_arr(f, 'B1', 'int32_t', B1)
        emit_arr(f, 'MULT1', 'int32_t', M1, 8); emit_arr(f, 'SHIFT1', 'int32_t', SH1, 8)
        emit_arr(f, 'W2', 'int8_t',  W2);  emit_arr(f, 'B2', 'int32_t', B2)
        emit_arr(f, 'MULT2', 'int32_t', M2, 8); emit_arr(f, 'SHIFT2', 'int32_t', SH2, 8)
        emit_arr(f, 'W3', 'int8_t',  W3);  emit_arr(f, 'B3', 'int32_t', B3)
        emit_arr(f, 'MULT3', 'int32_t', M3, 8); emit_arr(f, 'SHIFT3', 'int32_t', SH3, 8)
        emit_arr(f, 'W4', 'int8_t',  W4);  emit_arr(f, 'B4', 'int32_t', B4)
        emit_arr(f, 'MULT4', 'int32_t', M4, 8); emit_arr(f, 'SHIFT4', 'int32_t', SH4, 8)
        emit_arr(f, 'W5', 'int8_t',  W5);  emit_arr(f, 'B5', 'int32_t', B5)
        emit_arr(f, 'MULT5', 'int32_t', M5, 8); emit_arr(f, 'SHIFT5', 'int32_t', SH5, 8)
        f.write("#endif\n")

    import os
    print(f"kws_weights.h 已生成（{os.path.getsize('kws_weights.h')/1024:.0f} KB）")
    print(f"shift 范围: 1[{SH1.min()},{SH1.max()}] 2[{SH2.min()},{SH2.max()}] 3[{SH3.min()},{SH3.max()}] "
          f"4[{SH4.min()},{SH4.max()}] 5[{SH5.min()},{SH5.max()}]（应全 >0）")
    print(f"T30_S={S30:.17g} Z30={Z30} | IN_S={IN_S:.6f} IN_Z={IN_Z}")

if __name__ == "__main__":
    main()
