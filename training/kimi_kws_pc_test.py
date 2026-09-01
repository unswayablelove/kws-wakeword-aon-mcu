import os, glob, argparse
import numpy as np
import librosa

SAMPLE_RATE = 16000
FRAME_LEN = 480; FRAME_STEP = 320; NFFT = 512
NUM_MEL_BINS = 40; NUM_FRAMES = 49
LABELS = ["negative", "打开", "关闭", "唤醒"]

def extract_log_mel(audio):
    audio = np.append(audio[0], audio[1:]-0.97*audio[:-1])
    nf = 1+(len(audio)-FRAME_LEN)//FRAME_STEP
    if nf > NUM_FRAMES: nf = NUM_FRAMES
    frames = np.zeros((nf, FRAME_LEN))
    win = np.hamming(FRAME_LEN)
    for i in range(nf):
        frames[i] = audio[i*FRAME_STEP:i*FRAME_STEP+FRAME_LEN]*win
    ps = (np.abs(np.fft.rfft(frames, n=NFFT))**2)/NFFT
    low, high = 0, 2595*np.log10(1+SAMPLE_RATE/1400)
    pts = np.linspace(low, high, NUM_MEL_BINS+2)
    hz = 700*(10**(pts/2595)-1)
    bins = np.floor((NFFT+1)*hz/SAMPLE_RATE).astype(int); bins = np.clip(bins, 0, NFFT//2)
    fb = np.zeros((NUM_MEL_BINS, NFFT//2+1))
    for m in range(1, NUM_MEL_BINS+1):
        for k in range(bins[m-1], bins[m]): fb[m-1, k] = (k-bins[m-1])/max(bins[m]-bins[m-1], 1)
        for k in range(bins[m], bins[m+1]): fb[m-1, k] = (bins[m+1]-k)/max(bins[m+1]-bins[m], 1)
    return np.log(np.where((e := np.dot(ps, fb.T)) == 0, np.finfo(float).eps, e)).astype(np.float32)

def load_interpreter(model_path):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
    it = Interpreter(model_path=model_path)
    it.allocate_tensors()
    return it

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--dir", default=".")
    ap.add_argument("--thr", type=float, default=0.4)
    ap.add_argument("--margin", type=float, default=0.0, help="top1 - runnerup gap; 0 disables")
    ap.add_argument("--agc", type=float, default=0.0)
    args = ap.parse_args()

    nz = np.load(args.norm)
    mean, std = nz["mean"], nz["std"]
    interp = load_interpreter(args.model)
    inp = interp.get_input_details()[0]; out = interp.get_output_details()[0]
    in_s, in_z = inp["quantization"]; out_s, out_z = out["quantization"]

    def infer(window):
        feat = extract_log_mel(window)
        feat = (feat - mean) / (std + 1e-8)
        q = np.clip(np.round(feat/in_s + in_z), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q.reshape(1, NUM_FRAMES, NUM_MEL_BINS))
        interp.invoke()
        raw = interp.get_tensor(out["index"])[0]
        if out["dtype"] == np.int8: raw = (raw.astype(np.float32) - out_z) * out_s
        e = np.exp(raw - np.max(raw))
        return e / e.sum()

    files = sorted(sum([glob.glob(os.path.join(args.dir, f"*.{ext}"))
                        for ext in ("wav", "m4a", "mp3")], []))
    hop = SAMPLE_RATE // 4
    for f in files:
        y, _ = librosa.load(f, sr=SAMPLE_RATE, mono=True)
        if len(y) < SAMPLE_RATE: y = np.pad(y, (0, SAMPLE_RATE-len(y)))
        if args.agc > 0:
            r = np.sqrt(np.mean(y**2)) + 1e-8
            y = (y * (args.agc / r)).astype(np.float32)
        events = []
        maxu = 0.0
        for st in range(0, len(y)-SAMPLE_RATE+1, hop):
            probs = infer(y[st:st+SAMPLE_RATE])
            maxu = max(maxu, probs[0])
            k = 1 + int(np.argmax(probs[1:]))
            second = max([probs[0]] + [probs[j] for j in range(1, 4) if j != k])
            gap = probs[k] - second
            if probs[k] >= args.thr and gap >= args.margin:
                t = st / SAMPLE_RATE
                if not events or LABELS[k] != events[-1][1] or t - events[-1][0] > 1.2:
                    events.append((t, LABELS[k], probs[k], gap))
        print(f"\n=== {os.path.basename(f)} ({len(y)/SAMPLE_RATE:.1f}s) ===")
        counts = {}
        for t, w, p, g in events:
            counts[w] = counts.get(w, 0) + 1
            print(f"  {t:6.2f}s  {w}  conf={p:.2f}  gap={g:.2f}")
        print("  events:", counts if counts else "none")
        print(f"  max negative prob: {maxu:.2f}")

if __name__ == "__main__":
    main()
