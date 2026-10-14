#!/usr/bin/env python3
"""T10 speed benchmark over full dev-clean (2703 utterances, 5.4 h), Mac otherwise idle.

  ~/.venv-coreml/bin/python coreml/bench.py [--configs fp16:CPU_AND_NE fp16:CPU_AND_GPU fp16:CPU_ONLY fp32:CPU_AND_GPU]
                                            [--limit N] [--keep-cache]

Every configuration (build precision : compute units) runs in its own child process:
  - all audio is read into memory first (not timed); the models are loaded (load time recorded);
  - 20 files are processed as warm-up (not recorded);
  - then every file is timed end to end, from 16 kHz samples in memory to text, as the Phonon-2
    harness does: mel (CPU, V1's feature extractor) + Core ML encoder+head + ctc-beam-decoder
    (V1 LM and settings), with the per-stage times; RTFx = audio duration / processing time;
  - WER over the transcripts and the child's peak RSS (ru_maxrss) are recorded.
Load time: before the first child the Core ML compile cache (~/Library/Caches/org.python.python/
com.apple.e5rt.e5bundlecache) is moved aside (renamed, nothing deleted), so each configuration's
first load is cold (includes the Neural Engine / GPU compile); a second, load-only child per
configuration then measures the warm load. Writes OUT/bench/bench_<time>.{json,md}.
"""
import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

from common import OUT, V1_WER, load_list

HERE = Path(__file__).resolve().parent
CACHE = Path.home() / "Library/Caches/org.python.python/com.apple.e5rt.e5bundlecache"
CONFIGS = ["fp16:CPU_AND_NE", "fp16:CPU_AND_GPU", "fp16:CPU_ONLY", "fp32:CPU_AND_GPU"]
WARMUP = 20


def child(variant, precision, units, limit, load_only):
    from cmlrun import V1CoreML
    from common import load_audio, norm_text
    from transcribe import Transcriber
    from wer import corpus_wer
    utts = [u for u in load_list("full") if u["split"] == "dev-clean"]
    utts = utts[:limit] if limit else utts
    wavs = None if load_only else [load_audio(u["audio"]) for u in utts]
    t0 = time.perf_counter()
    cml = V1CoreML(variant, precision, units)
    load_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    tr = Transcriber(cml=cml)
    out = {"variant": variant, "precision": precision, "compute_units": units, "load_s": load_s,
           "decoder_setup_s": time.perf_counter() - t1}
    if load_only:
        out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        return out
    for w in wavs[:WARMUP]:
        tr.transcribe(w)
    hyps, tot = [], {"mel_s": 0.0, "model_s": 0.0, "decode_s": 0.0, "e2e_s": 0.0, "audio_s": 0.0}
    for w in wavs:
        a = time.perf_counter()
        text, info = tr.transcribe(w)
        tot["e2e_s"] += time.perf_counter() - a
        for k in ("mel_s", "model_s", "decode_s", "audio_s"):
            tot[k] += info[k]
        hyps.append(text)
    wer, edits, words = corpus_wer([norm_text(u["ref"]) for u in utts], hyps)
    out.update(tot)
    out.update({"utts": len(utts), "warmup": WARMUP, "wer": wer, "edits": edits, "words": words,
                "rtfx_e2e": tot["audio_s"] / tot["e2e_s"], "rtfx_model": tot["audio_s"] / tot["model_s"],
                "rtfx_decoder": tot["audio_s"] / tot["decode_s"], "rtfx_mel": tot["audio_s"] / tot["mel_s"],
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20})
    return out


def run_child(variant, precision, units, limit, load_only):
    cmd = [sys.executable, str(HERE / "bench.py"), "--child", variant, precision, units, str(limit)]
    if load_only:
        cmd.append("--load-only")
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
    line = [l for l in p.stdout.splitlines() if l.startswith("BENCH_JSON:")]
    if p.returncode or not line:
        raise RuntimeError(f"{' '.join(cmd)} failed ({p.returncode}):\n{p.stderr[-3000:]}")
    return json.loads(line[-1][len("BENCH_JSON:"):])


def run_all(configs=CONFIGS, variant="faithful", limit=0, keep_cache=False):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    moved = None
    if not keep_cache and CACHE.exists():
        moved = OUT / "bench" / f"e5rt_cache_moved_{stamp}"
        moved.parent.mkdir(parents=True, exist_ok=True)
        CACHE.rename(moved)
    load = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    runs = []
    for c in configs:
        precision, units = c.split(":")
        r = run_child(variant, precision, units, limit, False)
        r["cold_load_s"] = r.pop("load_s") if moved else None
        r["warm_load_s"] = run_child(variant, precision, units, limit, True)["load_s"]
        runs.append(r)
        print(f"[bench] {precision} {units}: e2e {r['rtfx_e2e']:.1f}x | encoder+head {r['rtfx_model']:.1f}x | decoder "
              f"{r['rtfx_decoder']:.0f}x | WER {100 * r['wer']:.3f}% | RSS {r['peak_rss_mb']:.0f} MB | load cold "
              f"{r['cold_load_s']} warm {r['warm_load_s']:.2f} s", flush=True)
    res = {"time": stamp, "uptime_before": load, "uptime_after": subprocess.run(["uptime"], capture_output=True,
           text=True).stdout.strip(), "cache_moved_to": str(moved) if moved else None, "limit": limit,
           "v1_dev_clean_wer": V1_WER["dev-clean"], "runs": runs}
    d = OUT / "bench"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"bench_{stamp}.json").write_text(json.dumps(res, indent=1))
    lines = [f"# T10 speed, full dev-clean ({runs[0]['utts']} utterances, {runs[0]['audio_s'] / 3600:.2f} h), {stamp}", "",
             "RTFx = audio duration / processing time, after a 20-file warm-up, model load excluded; end to end = "
             "16 kHz samples in memory -> text (same method as the Phonon-2 harness).", "",
             "| build | compute units | end-to-end RTFx | encoder+head RTFx | decoder RTFx | mel RTFx | WER % | peak RSS MB | "
             "cold load s | warm load s |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        lines.append(f"| {r['precision']} | {r['compute_units']} | {r['rtfx_e2e']:.1f} | {r['rtfx_model']:.1f} | "
                     f"{r['rtfx_decoder']:.0f} | {r['rtfx_mel']:.0f} | {100 * r['wer']:.3f} | {r['peak_rss_mb']:.0f} | "
                     f"{r['cold_load_s'] if r['cold_load_s'] is None else round(r['cold_load_s'], 1)} | {r['warm_load_s']:.2f} |")
    lines += ["", f"V1 (PyTorch pipeline) dev-clean WER {100 * V1_WER['dev-clean']:.3f} %. Load: `{load}`."]
    (d / f"bench_{stamp}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        variant, precision, units, limit = sys.argv[2:6]
        r = child(variant, precision, units, int(limit), "--load-only" in sys.argv)
        print("BENCH_JSON:" + json.dumps(r))
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--variant", default="faithful")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-cache", action="store_true", help="do not move the compile cache (no cold load)")
    args = ap.parse_args()
    run_all(args.configs, args.variant, args.limit, args.keep_cache)


if __name__ == "__main__":
    main()
