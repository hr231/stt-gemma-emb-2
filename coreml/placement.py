#!/usr/bin/env python3
"""Phase 1 step 5: Core ML compute plan (MLComputePlan) of each window package under CPU_AND_NE.

  ~/.venv-coreml/bin/python coreml/placement.py --variant faithful --precision fp16 [--windows 5s ...]

Per window: share of the estimated cost on the Neural Engine, op counts per device, and every op
that is not placed on the Neural Engine (type, name, device, cost share). Writes
OUT/placement/<variant>_<precision>.json.
"""
import argparse
import collections
import json
import tempfile
from pathlib import Path

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

from cmlrun import package
from common import OUT, WINDOWS


def device_name(d):
    n = type(d).__name__
    return {"MLNeuralEngineComputeDevice": "ANE", "MLGPUComputeDevice": "GPU", "MLCPUComputeDevice": "CPU"}.get(n, n)


def plan_for(path, compute_units="CPU_AND_NE"):
    tmp = Path(tempfile.mkdtemp(prefix="v1plan_"))
    compiled = ct.utils.compile_model(str(path), str(tmp / (Path(path).stem + ".mlmodelc")))
    plan = MLComputePlan.load_from_path(path=str(compiled), compute_units=getattr(ct.ComputeUnit, compute_units))
    main = plan.model_structure.program.functions["main"]
    rows = []
    for op in main.block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        cost = plan.get_estimated_cost_for_mlprogram_operation(op)
        rows.append({"type": op.operator_name, "name": op.outputs[0].name if op.outputs else "",
                     "device": device_name(usage.preferred_compute_device) if usage else "none",
                     "supported": [device_name(d) for d in usage.supported_compute_devices] if usage else [],
                     "cost": cost.weight if cost else 0.0})
    return rows


def summarize(rows):
    total = sum(r["cost"] for r in rows) or 1.0
    by_dev = collections.defaultdict(lambda: {"ops": 0, "cost": 0.0})
    for r in rows:
        by_dev[r["device"]]["ops"] += 1
        by_dev[r["device"]]["cost"] += r["cost"]
    off = [r for r in rows if r["device"] != "ANE" and r["type"] not in ("const",)]
    return {"ops": len(rows), "ane_cost_share": by_dev["ANE"]["cost"] / total,
            "by_device": {d: {"ops": v["ops"], "cost_share": v["cost"] / total} for d, v in by_dev.items()},
            "not_on_ane": [{**r, "cost_share": r["cost"] / total} for r in off]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="faithful")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--windows", nargs="+", default=list(WINDOWS))
    ap.add_argument("--compute-units", default="CPU_AND_NE")
    args = ap.parse_args()
    out = {}
    for w in args.windows:
        s = summarize(plan_for(package(args.variant, args.precision, w), args.compute_units))
        out[w] = s
        types = collections.Counter((r["type"], r["device"]) for r in s["not_on_ane"])
        print(f"[placement] {args.variant}/{args.precision} {w}: {s['ops']} ops, ANE share of estimated cost "
              f"{100 * s['ane_cost_share']:.2f}% | by device "
              + ", ".join(f"{d} {v['ops']} ops {100 * v['cost_share']:.2f}%" for d, v in s["by_device"].items())
              + f" | not on ANE (non-const): {dict(types.most_common(12))}", flush=True)
    d = OUT / "placement"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{args.variant}_{args.precision}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
