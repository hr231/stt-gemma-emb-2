"""Incremental V1 mel features: Gemma4AudioFeatureExtractor (common.mel) computed as audio arrives.

    from stream_mel import IncrementalMel
    im = IncrementalMel()
    for chunk in chunks:            # 16 kHz mono float32, any sizes (including 0 or 1 sample)
        new = im.feed(chunk)        # float32 [n, 128]: the newly complete mel frames, in order
    im.finish()                     # float32 [0, 128]: nothing is left (see below); im.n_valid = V
    im.reset()                      # new stream

The extractor (transformers gemma4/feature_extraction_gemma4.py, V1's preprocessor_config.json):
prepend 160 zeros (semicausal pad), unfold 321-sample windows with hop 160 and keep the first 320
(preemphasis 0), multiply by the float32 periodic Hann window, np.fft.rfft(n=512) (complex64 for
float32 input in numpy 2), abs, matmul with the float64 HTK mel matrix (128 filters, 0-8000 Hz,
norm None), log(mel + 0.001), cast to float32, zero the invalid frames. Mel frame i reads original
samples [160i - 160, 160i + 160) and is valid iff sample 160i + 160 exists, so after N samples the
valid frames are V = (N - 161) // 160 + 1 (N >= 161) and every valid frame only reads real samples:
the extractor's right padding (pad_to_multiple_of=128) only adds frames that are zeroed. Hence
feed() emits frame i as soon as sample 160i + 160 arrives, and finish() has nothing to add: the
frames the offline extractor returns beyond V are all zeros (input_features_mask = 0).

The same numpy ops run here row by row on the buffered samples; the offline extractor runs them on
the whole utterance at once. Equality is bitwise when the float64 matmul gives each row the same
bits whatever the number of rows; `python coreml/stream_mel.py` checks it on FAST (S1).
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, SR, feature_extractor, load_audio, load_list, mel  # noqa: E402

HOP = 160
N_FEAT = 128


class IncrementalMel:
    def __init__(self, fe=None):
        fe = fe or feature_extractor()
        if fe.dither != 0.0 or fe.preemphasis != 0.0 or fe.input_scale_factor != 1.0 \
                or fe.per_bin_mean is not None or fe.per_bin_stddev is not None:
            raise ValueError("IncrementalMel reproduces V1's extractor settings only "
                             "(no dither, preemphasis, scaling or per-bin statistics)")
        if (fe.frame_length, fe.hop_length, fe.fft_length, fe.feature_size) != (320, HOP, 512, N_FEAT):
            raise ValueError("unexpected extractor geometry")
        self.frame = fe.frame_length                  # 320 samples used per frame
        self.pad_left = fe.frame_length // 2          # 160
        self.fft = fe.fft_length                      # 512
        self.window = fe.window                       # float32 [320]
        self.filters = fe.mel_filters                 # float64 [257, 128]
        self.floor = fe.mel_floor                     # float64 0.001
        self.reset()

    def reset(self):
        self.buf = np.zeros(self.pad_left, np.float32)   # padded stream from padded index buf_start
        self.buf_start = 0
        self.n = 0                                        # original samples received
        self.n_valid = 0                                  # mel frames emitted

    def valid_frames(self, n_samples):
        """V for n_samples of audio: frame i is valid iff sample 160i + 160 exists."""
        return 0 if n_samples < self.frame // 2 + 1 else (n_samples - (self.frame // 2 + 1)) // HOP + 1

    def feed(self, samples):
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(x):
            self.buf = np.concatenate([self.buf, x])
            self.n += len(x)
        v = self.valid_frames(self.n)
        if v <= self.n_valid:
            return np.zeros((0, N_FEAT), np.float32)
        i0 = self.n_valid
        a = i0 * HOP - self.buf_start                     # buffer offset of frame i0
        span = self.buf[a:a + (v - i0 - 1) * HOP + self.frame]
        frames = np.lib.stride_tricks.as_strided(span, shape=(v - i0, self.frame),
                                                 strides=(span.strides[0] * HOP, span.strides[0]))
        spec = np.abs(np.fft.rfft(frames * self.window, n=self.fft, axis=-1))
        out = np.log(np.matmul(spec, self.filters) + self.floor).astype(np.float32)
        self.n_valid = v
        drop = v * HOP - self.buf_start                   # frame v starts at padded index 160 v
        self.buf = self.buf[drop:]
        self.buf_start += drop
        return out

    def finish(self):
        """End of stream: every valid frame was already returned by feed()."""
        return np.zeros((0, N_FEAT), np.float32)


# ---------------------------------------------------------------- S1: parity with common.mel()
def _chunk_sizes(rng, n):
    """Random chunk sizes covering 1-sample feeds, sub-hop, around-hop and large chunks."""
    sizes, tot = [], 0
    while tot < n:
        kind = rng.random()
        s = (rng.randint(0, 3) if kind < 0.15 else rng.randint(1, 159) if kind < 0.4
             else rng.randint(150, 700) if kind < 0.7 else rng.randint(700, 20000))
        sizes.append(s)
        tot += s
    return sizes


def parity(n_utts=50, seed=0, log=print):
    utts = sorted(random.Random(seed).sample(load_list("fast"), n_utts), key=lambda u: u["id"])
    d = OUT / "reference" / "fast"
    ref_idx = {u["id"]: u for u in json.loads((d / "index.json").read_text())["utts"]}
    ref_mel = np.load(d / "mel.npy", mmap_mode="r")
    rng = random.Random(seed)
    im = IncrementalMel()
    rows, t_stream, t_off = [], 0.0, 0.0
    for u in utts:
        w = load_audio(u["audio"])
        t0 = time.perf_counter()
        m_off, v_off = mel(w)
        t_off += time.perf_counter() - t0
        sizes = _chunk_sizes(rng, len(w))
        im.reset()
        parts, pos = [], 0
        t0 = time.perf_counter()
        for s in sizes:
            parts.append(im.feed(w[pos:pos + s]))
            pos += s
        parts.append(im.finish())
        t_stream += time.perf_counter() - t0
        m_st = np.concatenate(parts)
        r = ref_idx[u["id"]]
        m_ref = np.asarray(ref_mel[r["mel_off"]:r["mel_off"] + r["mel_valid"]])
        same_len = len(m_st) == v_off == im.n_valid == r["mel_valid"]
        n = min(len(m_st), v_off)
        d_off = float(np.abs(m_st[:n] - m_off[:n]).max()) if n else 0.0
        n2 = min(len(m_st), len(m_ref))
        rows.append({"id": u["id"], "samples": len(w), "chunks": len(sizes), "V": len(m_st), "V_offline": v_off,
                     "V_ref": r["mel_valid"], "same_len": same_len,
                     "bitwise_offline": bool(same_len and np.array_equal(m_st, m_off[:v_off])),
                     "bitwise_reference": bool(same_len and np.array_equal(m_st, m_ref[:n2])),
                     "offline_tail_zero": bool(not m_off[v_off:].any()),
                     "max_abs_diff_offline": d_off})
        log(json.dumps(rows[-1]))
    audio_s = sum(r["samples"] for r in rows) / SR
    summary = {"utts": len(rows), "chunks": sum(r["chunks"] for r in rows),
               "frames": sum(r["V"] for r in rows), "same_len": sum(r["same_len"] for r in rows),
               "bitwise_offline": sum(r["bitwise_offline"] for r in rows),
               "bitwise_reference": sum(r["bitwise_reference"] for r in rows),
               "offline_tail_zero": sum(r["offline_tail_zero"] for r in rows),
               "max_abs_diff_offline": max(r["max_abs_diff_offline"] for r in rows),
               "audio_s": audio_s, "stream_ms_per_s_audio": 1e3 * t_stream / audio_s,
               "offline_ms_per_s_audio": 1e3 * t_off / audio_s, "numpy": np.__version__}
    return summary, rows


def main():
    import argparse
    ap = argparse.ArgumentParser(description="S1: IncrementalMel vs common.mel() on FAST, random chunk sizes")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    summary, rows = parity(args.n, args.seed)
    out = OUT / "stream"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mel_parity.json").write_text(json.dumps({"summary": summary, "utts": rows}, indent=1))
    print(json.dumps(summary))
    ok = summary["bitwise_offline"] == summary["bitwise_reference"] == summary["utts"]
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
