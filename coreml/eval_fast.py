#!/usr/bin/env python3
"""Numerics and transcript parity of a Core ML build on FAST vs the PyTorch fp32 reference.

  ~/.venv-coreml/bin/python coreml/eval_fast.py --variant faithful --precision fp16 [--compute-units CPU_AND_NE]

Each utterance runs in its smallest fitting window. Per window: L11 cosine (per frame: mean, min),
max |d| of L11 and log-probs, utterances with NaN/Inf; overall: greedy and beam+LM (V1 decoder)
transcripts vs the reference (identical share, WER). Writes OUT/eval/<variant>_<precision>_<cu>.json.
"""
import argparse
import json
import time

import numpy as np

from cmlrun import V1CoreML, cos_rows, load_reference, ref_slices
from common import OUT, beam_decoder, greedy_text, norm_text
from wer import corpus_wer


def evaluate(cml, utts, arrays, with_beam=True):
    dec = beam_decoder() if with_beam else None
    per = {}
    rows = []
    for u in utts:
        mel, rl, rp = ref_slices(u, arrays)
        w = cml.window_for(len(mel))
        lp, l11 = cml.predict(mel, u["mel_valid"], w)
        s = per.setdefault(w, {"utts": 0, "cos": [], "l11_maxabs": 0.0, "lp_maxabs": 0.0, "nonfinite_utts": 0})
        fin = bool(np.isfinite(lp).all() and np.isfinite(l11).all())
        s["utts"] += 1
        s["nonfinite_utts"] += not fin
        if fin:
            s["cos"].append(cos_rows(l11, rl))
            s["l11_maxabs"] = max(s["l11_maxabs"], float(np.abs(l11 - rl).max()))
            s["lp_maxabs"] = max(s["lp_maxabs"], float(np.abs(lp - rp).max()))
        g = greedy_text(lp) if fin else ""
        b = norm_text(dec.decode(lp)) if (with_beam and fin) else ""
        rows.append({"id": u["id"], "window": w, "greedy": g, "beam": b})
    out = {"windows": {}}
    for w, s in per.items():
        c = np.concatenate(s["cos"]) if s["cos"] else np.array([np.nan])
        out["windows"][w] = {"utts": s["utts"], "nonfinite_utts": s["nonfinite_utts"], "l11_cos_mean": float(c.mean()),
                             "l11_cos_min": float(c.min()), "l11_maxabs": s["l11_maxabs"], "lp_maxabs": s["lp_maxabs"]}
    refs = [u["ref"] for u in utts]
    for kind in ("greedy", "beam") if with_beam else ("greedy",):
        hyp = [r[kind] for r in rows]
        ref_hyp = [u[kind] for u in utts]
        w, e, n = corpus_wer(refs, hyp)
        wr, er, _ = corpus_wer(refs, ref_hyp)
        out[kind] = {"wer": w, "edits": e, "ref_wer": wr, "ref_edits": er, "words": n,
                     "identical": sum(a == b for a, b in zip(hyp, ref_hyp)), "utts": len(utts)}
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="faithful")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--compute-units", default="CPU_AND_NE")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-beam", action="store_true")
    args = ap.parse_args()
    utts, arrays = load_reference("fast")
    utts = utts[: args.limit] if args.limit else utts
    t0 = time.time()
    cml = V1CoreML(args.variant, args.precision, args.compute_units)
    load_s = time.time() - t0
    res, rows = evaluate(cml, utts, arrays, with_beam=not args.no_beam)
    res.update({"variant": args.variant, "precision": args.precision, "compute_units": args.compute_units,
                "load_s": load_s, "seconds": round(time.time() - t0, 1)})
    d = OUT / "eval"
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{args.variant}_{args.precision}_{args.compute_units}" + (f"_n{args.limit}" if args.limit else "")
    (d / f"{stem}.json").write_text(json.dumps(res, indent=1))
    (d / f"{stem}_hyps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
