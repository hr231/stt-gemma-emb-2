#!/usr/bin/env python3
"""The release package: V1 as ONE Core ML package. The four fp16safe window programs (5 / 10 / 20 /
35 s) become the functions win_5s, win_10s, win_20s, win_35s of a single multifunction mlpackage with
shared weights (coremltools save_multifunction deduplicates identical constants). Multifunction
packages are spec 9, so the release needs macOS 15 / iOS 18. The fp32 build and the per-window
packages remain internal references.

  ~/.venv-coreml/bin/python coreml/export_release.py [--source fp16safe/fp16] [--verify-utts 60]

Writes OUT/models/release/fp16/v1.mlpackage (metadata: V1 file hashes, windows, decoder settings) and
verifies it: every function's outputs vs its source window package (CPU_AND_NE, FAST utterances),
Neural Engine placement per function, and size vs the four separate packages. Report:
OUT/models/release/fp16/release_check.json.
"""
import argparse
import json
import shutil
import subprocess
import time

import coremltools as ct
import numpy as np

from cmlrun import RELEASE, V1CoreML, function_name, load_reference, package, ref_slices
from common import MODEL_ID, MODEL_REVISION, OUT, V1_DECODER, V1_SHA256, WINDOWS, sha256
from placement import plan_for, summarize


def du_mb(path):
    return int(subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True).stdout.split()[0]) / 1024


def build(src_variant, src_precision, dst):
    desc = ct.utils.MultiFunctionDescriptor()
    for w in WINDOWS:
        desc.add_function(str(package(src_variant, src_precision, w)), "main", function_name(w))
    desc.default_function_name = function_name("10s")
    tmp = dst.with_name("v1.partial.mlpackage")
    if tmp.exists():
        tmp.rename(dst.with_name(f"v1.stale-{int(time.time())}.mlpackage"))
    ct.utils.save_multifunction(desc, str(tmp))
    m = ct.models.MLModel(str(tmp), skip_model_load=True)
    m.short_description = ("V1 speech recognition: EmbeddingGemma 2 audio tower layers 0-11 + int8 round trip + "
                           "standardization + 160 ms-lookahead CTC head; one function per input window.")
    meta = {"functions": json.dumps({function_name(w): {"mel_frames": t, "seconds": t / 100} for w, t in WINDOWS.items()}),
            "inputs": "mel float32 [1,T_mel,128] (Gemma4AudioFeatureExtractor, 100 frames/s, truncation=False); "
                      "valid_len int32 [1] = valid mel frames",
            "outputs": "log_probs [1,T_mel/4,257] (natural log, blank 256, 25 frames/s); l11 [1,T_mel/4,1024]",
            "encoder": f"{MODEL_ID}@{MODEL_REVISION} audio_tower.layers[0..11]",
            "v1_files_sha256": json.dumps({p.name: h for p, h in V1_SHA256.items()}),
            "decoder": json.dumps({"ctc_beam_decoder": "0.2.0", **V1_DECODER, "lm": "sp256_6gram.bin"}),
            "source_build": f"{src_variant}/{src_precision}"}
    m.user_defined_metadata.update(meta)
    final = dst.with_name("v1.meta.mlpackage")
    m.save(str(final))
    shutil.move(str(tmp), str(dst.with_name(f"v1.nometa-{int(time.time())}.mlpackage")))
    if dst.exists():
        dst.rename(dst.with_name(f"v1.old-{int(time.time())}.mlpackage"))
    final.rename(dst)


def verify(src_variant, src_precision, n_utts):
    utts, arrays = load_reference("fast")
    rel = V1CoreML(RELEASE, "fp16", "CPU_AND_NE")
    src = V1CoreML(src_variant, src_precision, "CPU_AND_NE")
    per = {}
    for u in utts[:n_utts]:
        mel, _, _ = ref_slices(u, arrays)
        w = rel.window_for(len(mel))
        a_lp, a_l11 = rel.predict(mel, u["mel_valid"], w)
        b_lp, b_l11 = src.predict(mel, u["mel_valid"], w)
        s = per.setdefault(w, {"utts": 0, "bitwise_identical": 0, "lp_max_abs_diff": 0.0, "l11_max_abs_diff": 0.0})
        s["utts"] += 1
        s["bitwise_identical"] += bool(np.array_equal(a_lp, b_lp) and np.array_equal(a_l11, b_l11))
        s["lp_max_abs_diff"] = max(s["lp_max_abs_diff"], float(np.abs(a_lp - b_lp).max()))
        s["l11_max_abs_diff"] = max(s["l11_max_abs_diff"], float(np.abs(a_l11 - b_l11).max()))
    placement = {}
    for w in WINDOWS:
        s = summarize(plan_for(package(RELEASE, "fp16"), "CPU_AND_NE", function_name(w)))
        placement[w] = {"ane_cost_share": s["ane_cost_share"],
                        "not_on_ane": sorted({f"{r['type']}@{r['device']}" for r in s["not_on_ane"]})}
    return {"outputs_vs_source": per, "placement": placement, "load_s": rel.load_s}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify-only", action="store_true", help="check the existing release package, don't rebuild it")
    ap.add_argument("--source", default="fp16safe/fp16")
    ap.add_argument("--verify-utts", type=int, default=60)
    args = ap.parse_args()
    src_variant, src_precision = args.source.split("/")
    dst = package(RELEASE, "fp16")
    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not args.verify_only:
        build(src_variant, src_precision, dst)
    built_s = None if args.verify_only else time.time() - t0
    res = {"time": time.strftime("%F %T"), "source": args.source, "package": str(dst),
           "build_s": None if built_s is None else round(built_s, 1),
           "size_mb": du_mb(dst), "sources_total_mb": sum(du_mb(package(src_variant, src_precision, w)) for w in WINDOWS),
           "weight_sha256": sha256(dst / "Data/com.apple.CoreML/weights/weight.bin"),
           "functions": [function_name(w) for w in WINDOWS], "coremltools": ct.__version__}
    res.update(verify(src_variant, src_precision, args.verify_utts))
    report = dst.parent / "release_check.json"
    if report.exists():
        report.rename(report.with_name(f"release_check.{int(report.stat().st_mtime)}.json"))
    report.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
