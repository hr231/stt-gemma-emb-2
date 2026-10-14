#!/usr/bin/env python3
"""Grid-search alpha (LM weight) and beta (insertion bonus) for one decoder + LM at a fixed beam,
on dev-clean only (test sets and dev-other are refused).

  python decode/tune.py --decoder pyctc --lm 4g --beam 16 --jobs 4
  python decode/tune.py --decoder subword --lm sp6 --beam 16 --alphas 0.1 0.2 0.3 0.5 --betas 0 0.5 1 2

Each grid point is one run_decode.py process (single-threaded), --jobs at a time; finished points
(metrics.json present) are reused unless --force. Prints and writes the WER grid
(<results>/tune_<...>.md / .json) and records the best point in <results>/tuned.json under
"<decoder>:<lm>", which sweep.py reads.
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_decode import fix_break  # noqa: E402

RESULTS = Path.home() / "gemma_stt/decode/results"
LOGPROBS = Path.home() / "gemma_stt/decode/logprobs"
HELD_OUT = ("dev-other", "test-clean", "test-other")
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
BETAS = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def default_opts(decoder, lm):
    """Options a decoder + LM pair needs: the token LM is used by cbd / pyctc with piece units."""
    return {"units": "piece"} if decoder in ("cbd", "pyctc") and lm == "sp6" else {}


def opts_tag(opts):
    return "".join(f"_{k}-{v}" for k, v in sorted(opts.items())
                   if not isinstance(v, (list, dict)) and "/" not in str(v))


def tuned_key(decoder, lm_name, opts):
    """tuned.json key: decoder:lm plus the options that change decoding (paths left out)."""
    return f"{decoder}:{lm_name}{opts_tag(opts)}"


def combo_opts(decoder, lm, user_opts):
    """default_opts for the pair, overridden by user options."""
    return {**default_opts(decoder, lm), **(user_opts or {})}


def num(x):
    return f"{x:g}"


def run_one(cmd, out_dir, force, nice):
    """Run run_decode.py unless out_dir/metrics.json exists; returns its metrics."""
    mf = out_dir / "metrics.json"
    if not force and mf.exists():
        return json.loads(mf.read_text())
    full = (["nice", "-n", str(nice)] if nice else []) + cmd
    p = subprocess.run(full, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[-3000:]}")
    print(p.stdout.strip(), flush=True)
    return json.loads(mf.read_text())


# Interpreter per decoder: ctc-beam-decoder is installed in the training venv, pyctcdecode and the
# kenlm module (needed by subword with an LM) in ~/.venv-pyctcdecode. Override with --python DEC=PATH.
PYTHONS = {"cbd": Path.home() / ".venv-gemma-stt/bin/python",
           "pyctc": Path.home() / ".venv-pyctcdecode/bin/python",
           "subword": Path.home() / ".venv-pyctcdecode/bin/python"}


def set_pythons(pairs):
    for p in pairs or []:
        dec, path = p.split("=", 1)
        PYTHONS[dec] = Path(path).expanduser()


def python_for(decoder):
    p = PYTHONS.get(decoder)
    return str(p) if p is not None and Path(p).exists() else sys.executable


def decode_cmd(decoder, split, lm, alpha, beta, beam, opts, limit, logprobs_dir, results_dir, tag):
    cmd = [python_for(decoder), str(HERE / "run_decode.py"), "--decoder", decoder, "--split", split,
           "--lm", lm, "--alpha", num(alpha), "--beta", num(beta), "--beam", str(beam),
           "--logprobs-dir", str(logprobs_dir), "--results-dir", str(results_dir), "--tag", tag]
    if opts:
        cmd += ["--opts", json.dumps(opts)]
    if limit:
        cmd += ["--limit", str(limit)]
    return cmd


def run_grid(jobs, tasks):
    """tasks: list of (key, cmd, out_dir, force, nice) -> ({key: metrics}, {key: error text})."""
    with ThreadPoolExecutor(max(1, jobs)) as pool:
        futs = {k: pool.submit(run_one, cmd, od, force, nice) for k, cmd, od, force, nice in tasks}
        done, failed = {}, {}
        for k, f in futs.items():
            try:
                done[k] = f.result()
            except RuntimeError as e:
                failed[k] = str(e)
                print(f"[error] {k}: {e}", flush=True)
        return done, failed


def add_python_arg(ap):
    ap.add_argument("--python", nargs="+", default=None, metavar="DEC=PATH",
                    help="interpreter for a decoder (defaults: " +
                    ", ".join(f"{k}={v}" for k, v in PYTHONS.items()) + "; else this interpreter)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decoder", required=True, choices=("cbd", "pyctc", "subword"))
    ap.add_argument("--lm", required=True, help="LM label (3g, 4g, sp6) or path")
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    ap.add_argument("--betas", type=float, nargs="+", default=BETAS)
    ap.add_argument("--opts", default="", help="JSON decoder options, added to the per-pair defaults")
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--limit", type=int, default=0, help="first N utterances only (quick look; 0 = all)")
    ap.add_argument("--jobs", type=int, default=4, help="parallel single-threaded decode processes")
    ap.add_argument("--nice", type=int, default=0, help="run children under nice -n N")
    ap.add_argument("--force", action="store_true", help="re-run points that already have results")
    ap.add_argument("--logprobs-dir", default=str(LOGPROBS))
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--tuned", default=None, help="tuned.json to update (default <results-dir>/tuned.json)")
    add_python_arg(ap)
    args = ap.parse_args(argv)
    set_pythons(args.python)

    if args.split in HELD_OUT or args.split.startswith("test"):
        raise SystemExit(f"tuning on {args.split} is not allowed: tune on dev-clean only")
    opts = combo_opts(args.decoder, args.lm, json.loads(args.opts) if args.opts else {})
    fb = fix_break(args.decoder, opts)
    lm_name = Path(args.lm).stem if "/" in args.lm else args.lm
    results = Path(args.results_dir).expanduser()
    stem = (f"tune_{args.split}_{args.decoder}_{lm_name}_beam{args.beam}{opts_tag(opts)}"
            + (f"_n{args.limit}" if args.limit else ""))
    tasks = []
    for a in args.alphas:
        for b in args.betas:
            tag = f"{stem}_a{num(a)}_b{num(b)}"
            cmd = decode_cmd(args.decoder, args.split, args.lm, a, b, args.beam, opts, args.limit,
                             args.logprobs_dir, results, tag)
            tasks.append(((a, b), cmd, results / tag, args.force, args.nice))
    t0 = time.time()
    print(f"[tune] {stem}: {len(tasks)} points, {args.jobs} jobs", flush=True)
    res, failed = run_grid(args.jobs, tasks)
    if failed:
        raise SystemExit(f"[tune] {len(failed)} of {len(tasks)} points failed; nothing recorded")

    (best_a, best_b), best = min(res.items(), key=lambda kv: (kv[1]["wer"], kv[0][0], kv[0][1]))
    head = "| alpha \\ beta | " + " | ".join(num(b) for b in args.betas) + " |"
    lines = [f"# {stem}", "",
             f"WER % on {args.split} ({best['n_utts']} utts, {best['words']} words), beam {args.beam}, "
             f"opts {json.dumps(opts)}, fix_break {fb}. Best: alpha {num(best_a)}, beta {num(best_b)} -> "
             f"{100 * best['wer']:.2f}% (marked *).", "", head,
             "|" + "---|" * (len(args.betas) + 1)]
    for a in args.alphas:
        cells = []
        for b in args.betas:
            w = f"{100 * res[(a, b)]['wer']:.2f}"
            cells.append(w + ("*" if (a, b) == (best_a, best_b) else ""))
        lines.append(f"| {num(a)} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print("\n" + table + f"\n\n[tune] {len(tasks)} points in {time.time() - t0:.0f}s", flush=True)
    (results / f"{stem}.md").write_text(table + "\n")
    (results / f"{stem}.json").write_text(json.dumps(
        {"decoder": args.decoder, "lm": args.lm, "beam": args.beam, "split": args.split, "opts": opts,
         "fix_break": fb, "limit": args.limit, "best": {"alpha": best_a, "beta": best_b, "wer": best["wer"]},
         "grid": [{"alpha": a, "beta": b, "wer": m["wer"], "xrt_per_core": m["xrt_per_core"]}
                  for (a, b), m in res.items()]}, indent=1))

    tuned_path = Path(args.tuned).expanduser() if args.tuned else results / "tuned.json"
    tuned = json.loads(tuned_path.read_text()) if tuned_path.exists() else {}
    tuned[tuned_key(args.decoder, lm_name, opts)] = {
        "alpha": best_a, "beta": best_b, "wer": best["wer"], "beam": args.beam, "split": args.split,
        "opts": opts, "fix_break": fb, "lm": args.lm, "limit": args.limit, "source": stem,
        "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    tuned_path.write_text(json.dumps(tuned, indent=1))
    print(f"[tune] best alpha {num(best_a)} beta {num(best_b)} WER {100 * best['wer']:.2f}% -> {tuned_path}")


if __name__ == "__main__":
    main()
