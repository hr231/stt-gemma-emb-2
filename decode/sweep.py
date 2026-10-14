#!/usr/bin/env python3
"""Beam sweep: every decoder + LM combination at several beams, with the alpha/beta that tune.py
found on dev-clean (<results>/tuned.json); no-LM combinations use alpha = beta = 0 (greedy runs once).

  python decode/sweep.py --split dev-clean
  python decode/sweep.py --split test-clean --beams 16 --combos pyctc:4g subword:sp6
  python decode/sweep.py --split dev-clean --combos pyctc:4g --opts '{"fix_break": false}'
  python decode/sweep.py --split dev-clean --weights-from 4g=3g    # 4-gram with the 3-gram's tuned weights

Combos are decoder:lm with lm in none, 3g, 4g, sp6 (cbd/pyctc with sp6 run with units=piece).
Run tags: sweep_<split>_<decoder>_<lm>_beam<B>[_opts][_w<SRC>]; finished runs are reused unless --force.
--jobs 1 (default) gives the cleanest timing; the CPU seconds are per process either way.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tune import (LOGPROBS, RESULTS, add_python_arg, combo_opts, decode_cmd, num, opts_tag,  # noqa: E402
                  run_grid, set_pythons, tuned_key)

BEAMS = [1, 4, 8, 16, 32, 64]
COMBOS = ["greedy:none", "subword:none", "subword:sp6", "cbd:none", "cbd:3g", "cbd:4g", "cbd:sp6",
          "pyctc:none", "pyctc:3g", "pyctc:4g", "pyctc:sp6"]


def plan(args, tuned):
    """[(key, cmd, out_dir, force, nice)], skipped combos."""
    results = Path(args.results_dir).expanduser()
    user = json.loads(args.opts) if args.opts else {}
    weights_from = dict(s.split("=", 1) for s in args.weights_from)
    tasks, skipped = [], []
    for combo in args.combos:
        decoder, lm = combo.split(":", 1)
        lm_name = Path(lm).stem if "/" in lm else lm
        opts = combo_opts(decoder, lm_name, user) if decoder != "greedy" else {}
        wtag = ""
        if lm == "none":
            alpha = beta = 0.0
        else:
            src = weights_from.get(lm_name, lm_name)
            key = tuned_key(decoder, src, opts)
            t = tuned.get(key)
            if t is None:
                skipped.append(f"{combo} (no tuned.json entry {key}; run tune.py)")
                continue
            if t["split"] != "dev-clean":
                raise SystemExit(f"{combo}: tuned on {t['split']}, not dev-clean")
            if t.get("limit"):
                print(f"[sweep] warning: {combo} was tuned on the first {t['limit']} utterances only")
            alpha, beta = t["alpha"], t["beta"]
            if src != lm_name:
                wtag = f"_w{src}"
        for beam in ([1] if decoder == "greedy" else args.beams):
            tag = (f"sweep_{args.split}_{decoder}_{lm_name}_beam{beam}{opts_tag(opts)}{wtag}"
                   + (f"_n{args.limit}" if args.limit else ""))
            cmd = decode_cmd(decoder, args.split, lm, alpha, beta, beam, opts, args.limit,
                             args.logprobs_dir, results, tag)
            tasks.append(((combo, beam), cmd, results / tag, args.force, args.nice))
    return tasks, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--beams", type=int, nargs="+", default=BEAMS)
    ap.add_argument("--combos", nargs="+", default=COMBOS, help="decoder:lm pairs")
    ap.add_argument("--opts", default="", help="JSON options added to every non-greedy combo")
    ap.add_argument("--weights-from", nargs="+", default=[], metavar="LM=SRC",
                    help="take LM's alpha/beta from SRC's tuned.json entry, e.g. 4g=3g (tuned once with the "
                         "pruned 3-gram); run tags get _w<SRC>")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--nice", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--logprobs-dir", default=str(LOGPROBS))
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--tuned", default=None, help="default <results-dir>/tuned.json")
    add_python_arg(ap)
    args = ap.parse_args(argv)
    set_pythons(args.python)

    tuned_path = Path(args.tuned).expanduser() if args.tuned else Path(args.results_dir).expanduser() / "tuned.json"
    tuned = json.loads(tuned_path.read_text()) if tuned_path.exists() else {}
    tasks, skipped = plan(args, tuned)
    for s in skipped:
        print(f"[sweep] skipping {s}", flush=True)
    print(f"[sweep] {args.split}: {len(tasks)} runs, {args.jobs} jobs", flush=True)
    res, failed = run_grid(args.jobs, tasks)
    for (combo, beam), m in sorted(res.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        print(f"  {combo:<14} beam {beam:3d} | alpha {num(m['alpha'])} beta {num(m['beta'])} | "
              f"WER {100 * m['wer']:6.2f}% | {m['xrt_per_core']:9.1f}x RT/core | "
              f"RSS {m['peak_rss_bytes'] / 2**20:6.0f} MB | load {m['lm_load_s']:.2f}s | fix_break {m.get('fix_break')}")
    if failed:
        print(f"[sweep] {len(failed)} runs failed: {sorted(failed)}")
    print(f"[sweep] done; python decode/report.py --split {args.split}")


if __name__ == "__main__":
    main()
