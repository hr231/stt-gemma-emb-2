#!/usr/bin/env python3
"""Summarize decoding runs of one split: markdown + CSV table and a WER-vs-speed plot.

  python decode/report.py --split dev-clean                # sweep_* runs in ~/gemma_stt/decode/results
  python decode/report.py --split test-clean --prefix sweep_ --out ~/gemma_stt/decode/results/report_test-clean

Columns: WER, x real time per CPU core (audio seconds / single-thread decode CPU seconds), peak RSS,
decoder + LM load time, fix_break (pyctcdecode word-break fix; cbd reproduces stock = False).
Knee: the fastest run whose WER is within 0.05 WER points (absolute, e.g. 5.40% -> <= 5.45%) of the
best WER, per decoder + LM (+ options) ("knee" column, star in the plot) and over all runs ("overall"
column, ringed in the plot). The plot (PNG) needs matplotlib (~/.venv-pyctcdecode); without it only
the tables are written. Runs with --limit are left out unless --include-limited.
"""
import argparse
import csv
import json
from pathlib import Path

RESULTS = Path.home() / "gemma_stt/decode/results"
KNEE_WER_POINTS = 0.05
RULE = (f"Knee = the fastest run (highest x real time per core) whose WER is within {KNEE_WER_POINTS} WER "
        f"points (absolute percentage points) of the best WER: per decoder + LM (+ options), and overall.")
COLS = ["decoder", "lm", "opts", "fix_break", "alpha", "beta", "beam", "n_utts", "wer_pct",
        "xrt_per_core", "decode_cpu_s", "peak_rss_mb", "lm_load_s", "best", "knee", "overall", "tag"]


def opts_label(opts):
    return ",".join(f"{k}={v}" for k, v in sorted(opts.items()) if not isinstance(v, (list, dict)))


def group_key(m):
    return (m["decoder"], m["lm"], opts_label(m.get("opts") or {}))


def load_runs(results, split, prefix, include_limited):
    runs = []
    for mf in sorted(Path(results).glob(f"{prefix}*/metrics.json")):
        m = json.loads(mf.read_text())
        if m["split"] != split or (m.get("limit") and not include_limited):
            continue
        runs.append(m)
    return runs


def best_and_knee(ms):
    """(lowest-WER run, fastest run within KNEE_WER_POINTS of it)."""
    best = min(ms, key=lambda m: (m["wer"], -m["xrt_per_core"]))
    ok = [m for m in ms if 100 * m["wer"] <= 100 * best["wer"] + KNEE_WER_POINTS + 1e-9]
    return best, max(ok, key=lambda m: (m["xrt_per_core"], -m["wer"]))


def mark_knees(runs):
    """Sets m['best'] and m['knee'] per group and m['overall'] over all runs;
    returns ({group: (best run, knee run)}, (overall best, overall knee))."""
    groups = {}
    for m in runs:
        groups.setdefault(group_key(m), []).append(m)
    out = {}
    for g, ms in groups.items():
        best, knee = best_and_knee(ms)
        for m in ms:
            m["best"], m["knee"] = m is best, m is knee
        out[g] = (best, knee)
    overall = best_and_knee(runs)
    for m in runs:
        m["overall"] = "+".join(s for s, r in zip(("best", "knee"), overall) if m is r)
    return out, overall


def row(m):
    return {"decoder": m["decoder"], "lm": m["lm"], "opts": opts_label(m.get("opts") or {}),
            "fix_break": "" if m.get("fix_break") is None else m["fix_break"],
            "alpha": m["alpha"], "beta": m["beta"], "beam": m["beam"], "n_utts": m["n_utts"],
            "wer_pct": round(100 * m["wer"], 3), "xrt_per_core": round(m["xrt_per_core"], 1),
            "decode_cpu_s": round(m["decode_cpu_s"], 2), "peak_rss_mb": round(m["peak_rss_bytes"] / 2**20, 1),
            "lm_load_s": round(m["lm_load_s"], 3), "best": "best" if m["best"] else "",
            "knee": "knee" if m["knee"] else "", "overall": m["overall"], "tag": m["tag"]}


def markdown(split, rows, knees, overall, runs):
    g = runs[0]
    lines = [f"# Decoding on {split}", "",
             f"{len(runs)} runs; head {g.get('head')}; model's own greedy WER (fp32 dump) "
             f"{100 * g['greedy_wer']:.2f}%, from the fp16 store {100 * g['greedy_wer_fp16']:.2f}%."
             if g.get("greedy_wer") is not None else f"{len(runs)} runs.", "",
             "x RT/core = audio seconds / decode CPU seconds of one single-threaded process; "
             "peak RSS = ru_maxrss of that process; load = decoder construction incl. LM load.", "",
             RULE, "", "| " + " | ".join(COLS[:-1]) + " |", "|" + "---|" * (len(COLS) - 1)]
    lines += ["| " + " | ".join(str(r[c]) for c in COLS[:-1]) + " |" for r in rows]
    lines += ["", "## Knee per decoder + LM", "", "| decoder | lm | opts | best WER % (beam) | knee WER % | knee beam | knee x RT/core |",
              "|---|---|---|---|---|---|---|"]
    for (dec, lm, o), (best, knee) in sorted(knees.items()):
        lines.append(f"| {dec} | {lm} | {o} | {100 * best['wer']:.2f} ({best['beam']}) | "
                     f"{100 * knee['wer']:.2f} | {knee['beam']} | {knee['xrt_per_core']:.1f} |")
    ob, ok = overall
    lines += ["", "## Overall knee", "",
              f"Best: {ob['decoder']}+{ob['lm']} beam {ob['beam']}, {100 * ob['wer']:.2f}% at "
              f"{ob['xrt_per_core']:.1f}x RT/core. Knee: {ok['decoder']}+{ok['lm']} "
              f"{opts_label(ok.get('opts') or {})} beam {ok['beam']} (alpha {ok['alpha']:g}, beta {ok['beta']:g}), "
              f"{100 * ok['wer']:.2f}% at {ok['xrt_per_core']:.1f}x RT/core, "
              f"{ok['peak_rss_bytes'] / 2**20:.0f} MB RSS, {ok['lm_load_s']:.2f} s load."]
    return "\n".join(lines) + "\n"


# Plot styles: colour = LM, marker + line = decoder variant, so every group is distinguishable.
LM_COLORS = {"none": "0.4", "3g": "tab:blue", "4g": "tab:orange", "sp6": "tab:green"}
VARIANT_STYLES = {"greedy": ("X", "-"), "cbd": ("o", "-"), "pyctc": ("s", "--"),
                  "pyctc fix_break=False": ("D", ":"), "subword": ("^", "-."),
                  "subword merge=sum": ("v", (0, (3, 1, 1, 1, 1, 1)))}


def plot_style(g):
    """(colour, marker, linestyle) for group (decoder, lm, opts label); units=piece is implied by sp6."""
    variant = " ".join([g[0]] + [o for o in g[2].split(",") if o and o != "units=piece"])
    marker, ls = VARIANT_STYLES.get(variant, ("P", (0, (1, 1))))
    return LM_COLORS.get(g[1], "tab:purple"), marker, ls


def plot(split, runs, knees, overall, path):
    try:
        import matplotlib
    except ImportError:
        print("[report] matplotlib not installed here: no plot (run under ~/.venv-pyctcdecode)")
        return False
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for g, (best, knee) in sorted(knees.items()):
        ms = sorted((m for m in runs if group_key(m) == g), key=lambda m: m["beam"])
        label = f"{g[0]}+{g[1]}" + (f" [{g[2]}]" if g[2] else "")
        color, marker, ls = plot_style(g)
        ax.plot([m["xrt_per_core"] for m in ms], [100 * m["wer"] for m in ms], marker=marker, linestyle=ls,
                color=color, ms=5, label=label)
        for m in ms:
            ax.annotate(str(m["beam"]), (m["xrt_per_core"], 100 * m["wer"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.plot([knee["xrt_per_core"]], [100 * knee["wer"]], "*", ms=16, color=color, markeredgecolor="black")
    knee = overall[1]
    ax.plot([knee["xrt_per_core"]], [100 * knee["wer"]], "o", ms=26, markerfacecolor="none",
            markeredgecolor="black", markeredgewidth=1.5, label="overall knee")
    ax.set_xscale("log")
    ax.set_xlabel("x real time per CPU core (log scale; right = faster)")
    ax.set_ylabel(f"WER % ({split})")
    ax.set_title(f"CTC decoding on {split}: WER vs speed (labels = beam, star = knee, ring = overall knee)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), title="colour = LM, marker = decoder",
              title_fontsize=8)
    fig.text(0.01, 0.01, RULE, fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--prefix", default="sweep_", help="run folder prefix to include")
    ap.add_argument("--out", default=None, help="output stem (default <results-dir>/report_<split>)")
    ap.add_argument("--include-limited", action="store_true")
    args = ap.parse_args(argv)

    results = Path(args.results_dir).expanduser()
    runs = load_runs(results, args.split, args.prefix, args.include_limited)
    if not runs:
        raise SystemExit(f"no {args.prefix}* runs for {args.split} in {results}")
    knees, overall = mark_knees(runs)
    runs.sort(key=lambda m: (group_key(m), m["beam"]))
    rows = [row(m) for m in runs]
    stem = Path(args.out).expanduser() if args.out else results / f"report_{args.split}"
    with open(f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    md = markdown(args.split, rows, knees, overall, runs)
    Path(f"{stem}.md").write_text(md)
    print(md)
    made = plot(args.split, runs, knees, overall, f"{stem}.png")
    print(f"[report] -> {stem}.md, {stem}.csv" + (f", {stem}.png" if made else ""))


if __name__ == "__main__":
    main()
