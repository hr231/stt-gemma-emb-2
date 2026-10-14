#!/usr/bin/env python3
"""Cost of the embedding on top of V1's transcript (embed/onepass.py, PyTorch MPS bf16).

  ~/.venv-gemma-stt/bin/python embed/cost.py [--n 100] [--warmup 5] [--out ~/gemma_stt/embed/cost.json]

Runs two child processes (so peak RSS and MPS memory are per configuration), model load excluded:
  asr      OnePass(load_language_model=False): audio tower + V1 head + beam/LM decoder only;
           per clip wall time of transcribe(wav);
  onepass  full OnePass (tower + embed_audio + language model): per clip wall time of transcribe(wav)
           and of transcribe_and_embed(wav), alternating which runs first, plus its stage timings.
Clips: the first --n of V1's FAST list (dev-clean, seed 0); --warmup further FAST clips run first
and are not counted. Waveforms are read before timing. Memory: process peak RSS
(getrusage ru_maxrss) and torch.mps.current_allocated_memory / driver_allocated_memory after load
and the maximum seen after each clip.
"""
import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

GB = 1 << 30


def mem():
    import torch
    return {"peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GB,   # bytes on macOS
            "mps_current_gb": torch.mps.current_allocated_memory() / GB,
            "mps_driver_gb": torch.mps.driver_allocated_memory() / GB}


def pct(v):
    v = np.asarray(v, np.float64)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "sum": float(v.sum())}


def phase(name, n, warmup):
    import torch
    from onepass import OnePass                    # puts coreml/ (common) on sys.path
    from common import load_audio, load_list
    clips = load_list("fast")
    if n + warmup > len(clips):
        raise SystemExit(f"FAST has {len(clips)} clips < n + warmup")
    run = [load_audio(u["audio"]) for u in clips[:n]]
    warm = [load_audio(u["audio"]) for u in clips[n:n + warmup]]
    m0 = mem()
    t0 = time.time()
    op = OnePass(load_language_model=(name == "onepass"))
    load_s = time.time() - t0
    m_load = mem()
    fns = {"asr": [("transcribe", op.transcribe)],
           "onepass": [("transcribe", op.transcribe), ("transcribe_and_embed", op.transcribe_and_embed)]}[name]
    for w in warm:
        for _, f in fns:
            f(w)
    wall = {k: [] for k, _ in fns}
    stages = []
    m_max = mem()
    for i, w in enumerate(run):
        order = fns if i % 2 == 0 else fns[::-1]
        for k, f in order:
            torch.mps.synchronize()
            t = time.perf_counter()
            r = f(w)
            wall[k].append(time.perf_counter() - t)
            if k == "transcribe_and_embed":
                stages.append(r["timings"])
        m = mem()
        m_max = {k: max(m_max[k], m[k]) for k in m}
    audio = [len(w) / 16000 for w in run]
    out = {"phase": name, "n": n, "warmup": warmup, "audio_s": pct(audio), "load_s": load_s,
           "mem_before_load": m0, "mem_after_load": m_load, "mem_max_during_run": m_max,
           "wall_s": {k: pct(v) for k, v in wall.items()},
           "rtf": {k: float(np.sum(v) / np.sum(audio)) for k, v in wall.items()},
           "per_clip": {k: v for k, v in wall.items()}, "audio_per_clip": audio}
    if stages:
        out["stages_s"] = {k: pct([s[k] for s in stages]) for k in stages[0] if k != "audio_s"}
        extra = np.asarray(wall["transcribe_and_embed"]) - np.asarray(wall["transcribe"])
        out["embed_extra_s"] = pct(extra)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default=str(Path.home() / "gemma_stt/embed/cost.json"))
    ap.add_argument("--phase", choices=["asr", "onepass"], help="internal: run one child configuration")
    args = ap.parse_args()
    if args.phase:
        print(json.dumps(phase(args.phase, args.n, args.warmup)), flush=True)
        return
    res = {}
    for name in ("asr", "onepass"):
        p = subprocess.run([sys.executable, __file__, "--phase", name, "--n", str(args.n),
                            "--warmup", str(args.warmup)], capture_output=True, text=True)
        if p.returncode:
            sys.stderr.write(p.stderr)
            raise SystemExit(f"phase {name} failed")
        res[name] = json.loads(p.stdout.strip().splitlines()[-1])
    a, o = res["asr"], res["onepass"]
    res["summary"] = {
        "clips": args.n, "mean_audio_s": a["audio_s"]["mean"],
        "transcript_only_s (no LM loaded)": a["wall_s"]["transcribe"]["mean"],
        "transcript_only_s (LM loaded)": o["wall_s"]["transcribe"]["mean"],
        "transcript_plus_embedding_s": o["wall_s"]["transcribe_and_embed"]["mean"],
        "embedding_extra_s (paired, same process)": o["embed_extra_s"]["mean"],
        "embedding_extra_vs_asr_process_s": o["wall_s"]["transcribe_and_embed"]["mean"] - a["wall_s"]["transcribe"]["mean"],
        "rtf_transcript": a["rtf"]["transcribe"], "rtf_onepass": o["rtf"]["transcribe_and_embed"],
        "peak_rss_gb": {"asr": a["mem_max_during_run"]["peak_rss_gb"], "onepass": o["mem_max_during_run"]["peak_rss_gb"]},
        "mps_current_gb_after_load": {"asr": a["mem_after_load"]["mps_current_gb"],
                                      "onepass": o["mem_after_load"]["mps_current_gb"]},
        "mps_driver_gb_max": {"asr": a["mem_max_during_run"]["mps_driver_gb"],
                              "onepass": o["mem_max_during_run"]["mps_driver_gb"]},
    }
    Path(args.out).expanduser().write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
