#!/usr/bin/env python3
"""V1 on Core ML: audio -> mel (CPU, V1's feature extractor, truncation=False) -> V1 window model
(Core ML, default CPU_AND_NE) -> ctc-beam-decoder with the V1 subword 6-gram and settings.

  ~/.venv-coreml/bin/python coreml/transcribe.py AUDIO [AUDIO ...] [--compute-units CPU_AND_NE] [--json]

Audio up to 35 s runs in the smallest window that fits (5 / 10 / 20 / 35 s). Longer audio is cut
at pauses: each segment ends at the quietest point (mean log-mel energy, 100 ms moving average) in
the last 10 s of what fits; every segment after the first gets 2 s of left context whose output
frames are dropped; no right overlap. Cuts fall on the 4-mel-frame grid, so the segments' output
frames line up with one global frame sequence, which is decoded at once.
"""
import argparse
import json
import sys
import time

import numpy as np

from cmlrun import V1CoreML
from common import WINDOWS, beam_decoder, check_v1_files, greedy_text, load_audio, mel, norm_text

MAX_MEL = max(WINDOWS.values())
CONTEXT_MEL = 200                   # 2 s of left context = 50 output frames
SEARCH_MEL = 1000                   # look for the cut in the last 10 s of a segment
SMOOTH = 10                         # 100 ms


def plan_segments(m, max_mel=MAX_MEL, context=CONTEXT_MEL, search=SEARCH_MEL):
    """[(start with context, keep from, end)] in mel frames; all multiples of 4 except the last end."""
    n = len(m)
    if n <= max_mel:
        return [(0, 0, n)]
    energy = m.mean(1)
    smooth = np.convolve(energy, np.ones(SMOOTH) / SMOOTH, mode="same")
    segs, cur = [], 0
    while cur < n:
        ctx = min(context, cur)
        span = max_mel - ctx
        if n - cur <= span:
            segs.append((cur - ctx, cur, n))
            break
        lo, hi = cur + max(span - search, 4), cur + span
        cands = np.arange(lo - lo % 4 + 4, hi - hi % 4 + 1, 4)
        end = int(cands[np.argmin(smooth[cands - 1])]) if len(cands) else hi - hi % 4
        segs.append((cur - ctx, cur, end))
        cur = end
    return segs


class Transcriber:
    def __init__(self, variant="faithful", precision="fp16", compute_units="CPU_AND_NE", cml=None):
        check_v1_files()
        t0 = time.time()
        self.cml = cml if cml is not None else V1CoreML(variant, precision, compute_units)
        self.decoder = beam_decoder()
        self.load_s = time.time() - t0

    def logprobs(self, wav):
        """(log-probs [frames, 257] float32, segments, timing)."""
        t0 = time.perf_counter()
        m, n_valid = mel(wav)
        m = m[:n_valid]
        t1 = time.perf_counter()
        segs = plan_segments(m)
        parts = []
        for start, keep, end in segs:
            lp, _ = self.cml.predict(m[start:end], end - start)
            parts.append(lp[(keep - start) // 4:])
        t2 = time.perf_counter()
        return np.concatenate(parts) if parts else np.zeros((0, 257), np.float32), segs, {"mel_s": t1 - t0, "model_s": t2 - t1}

    def transcribe(self, wav, greedy=False):
        lp, segs, timing = self.logprobs(wav)
        t0 = time.perf_counter()
        text = greedy_text(lp) if greedy else norm_text(self.decoder.decode(lp))
        timing["decode_s"] = time.perf_counter() - t0
        timing["audio_s"] = len(wav) / 16000
        return text, {"segments": segs, **timing}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--variant", default="faithful")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--compute-units", default="CPU_AND_NE")
    ap.add_argument("--greedy", action="store_true", help="greedy CTC instead of beam + LM")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    tr = Transcriber(args.variant, args.precision, args.compute_units)
    print(f"[load] {tr.load_s:.2f} s ({args.variant}/{args.precision}, {args.compute_units})", file=sys.stderr)
    for path in args.audio:
        text, info = tr.transcribe(load_audio(path), greedy=args.greedy)
        print(json.dumps({"audio": path, "text": text, **info}) if args.json else text, flush=True)


if __name__ == "__main__":
    main()
