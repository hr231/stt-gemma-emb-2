#!/usr/bin/env python3
"""Summarize ~/gemma_stt/coreml/results.jsonl into ~/gemma_stt/coreml/summary.md: one section per
run id (newest first) with each test's pass/fail and its key metrics.

  ~/.venv-coreml/bin/python coreml/summary.py
"""
import collections
import json

from common import OUT

KEYS = {"test_t1": ("utt_cos_mean", "utt_cos_min", "frame_cos_mean", "frame_cos_min", "logprob_max_abs_diff"),
        "test_t2": ("identical", "identical_share", "noise_floor_v1_bf16_identical_share", "wer", "ref_wer"),
        "test_t3": ("identical", "wer", "ref_wer", "wer_diff_points"),
        "test_t4": ("max_l11_before", "max_head_before", "min_l11_after"),
        "test_t5": ("l11_cos_min", "logprob_cos_min", "greedy_identical"),
        "test_t6": ("ane_cost_share", "threshold"),
        "test_t7": ("identical_logprobs", "identical_l11", "utts"),
        "test_t8": ("audio_s", "wer_concatenated", "wer_separate", "diff_points")}


def fmt(v):
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def key_metrics(test, m):
    for prefix, keys in KEYS.items():
        if test.startswith(prefix):
            return ", ".join(f"{k} {fmt(m[k])}" for k in keys if k in m)
    if test.startswith("test_t9"):
        return "; ".join(f"{s} WER {100 * v['wer']:.3f} % (V1 {100 * v['v1_wer']:.3f} %, diff {v['diff_points']:+.3f})"
                         for s, v in m.items())
    if test.startswith("test_t10"):
        return "; ".join(f"{r.get('variant', '?')}/{r['precision']} {r['compute_units']}: e2e {r['rtfx_e2e']:.1f}x, enc+head {r['rtfx_model']:.1f}x, "
                         f"WER {100 * r['wer']:.3f} %, RSS {r['peak_rss_mb']:.0f} MB" for r in m["runs"])
    if test.startswith("test_t11"):
        return "; ".join(f"{s} Core ML {100 * v['coreml_wer']:.3f} % / V1 {100 * v['v1_pytorch_wer']:.3f} % / "
                         f"Phonon-2 {100 * v['phonon2_wer']:.3f} %" for s, v in m["splits"].items())
    return ""


def main():
    rows = [json.loads(l) for l in (OUT / "results.jsonl").read_text().splitlines() if l.strip()]
    runs = collections.OrderedDict()
    for r in rows:
        runs.setdefault(r["run"], []).append(r)
    out = ["# Core ML test results", "", f"{len(rows)} test records in {len(runs)} runs (results.jsonl).", ""]
    for run, rs in sorted(runs.items(), reverse=True):
        r0 = rs[0]
        n_pass = sum(r["pass"] for r in rs)
        out += [f"## Run {run}: {n_pass}/{len(rs)} passed", "",
                f"Build {r0['variant']}/{r0['precision']} under {r0['compute_units']}; git {r0['git']['commit'][:8]}"
                f"{' (dirty)' if r0['git']['dirty'] else ''}; coremltools {r0['versions']['coremltools']}, torch "
                f"{r0['versions']['torch']}.", "", "| test | marker | result | key metrics |", "|---|---|---|---|"]
        for r in rs:
            out.append(f"| {r['test']} | {r['marker']} | {'PASS' if r['pass'] else 'FAIL'} | {key_metrics(r['test'], r['metrics'])} |")
        out.append("")
    (OUT / "summary.md").write_text("\n".join(out) + "\n")
    print(f"[summary] {len(runs)} runs -> {OUT / 'summary.md'}")


if __name__ == "__main__":
    main()
