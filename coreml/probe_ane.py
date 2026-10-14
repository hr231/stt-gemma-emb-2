"""Convert a small module (fp16, macOS 14) and report where Core ML places its ops under CPU_AND_NE,
plus whether the Neural Engine compile succeeds (Core ML prints ANECCompile errors on stderr).

  from probe_ane import probe
  probe("attn", module, [("h", (1, 125, 1024), np.float32)])
"""
import collections
import subprocess
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

CHILD = """
import sys, collections, json, tempfile, coremltools as ct
from coremltools.models.compute_plan import MLComputePlan
path = sys.argv[1]
tmp = tempfile.mkdtemp()
c = ct.utils.compile_model(path, tmp + "/m.mlmodelc")
plan = MLComputePlan.load_from_path(path=str(c), compute_units=ct.ComputeUnit.CPU_AND_NE)
ops = plan.model_structure.program.functions["main"].block.operations
cnt = collections.Counter()
for op in ops:
    if op.operator_name == "const":
        continue
    u = plan.get_compute_device_usage_for_mlprogram_operation(op)
    d = type(u.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "") if u else "none"
    cnt[(op.operator_name, d)] += 1
print("\\nPLACEMENT_JSON:" + json.dumps([[k[0], k[1], v] for k, v in sorted(cnt.items())]))
"""

def probe(name, module, inputs, quiet=False):
    ex = tuple(torch.from_numpy(np.full(s, 3 if d == np.int32 else 0.5, dtype=d)) for _, s, d in inputs)
    with torch.no_grad():
        tr = torch.jit.trace(module.eval(), ex, check_trace=False)
    ml = ct.convert(tr, inputs=[ct.TensorType(name=n, shape=s, dtype=d) for n, s, d in inputs],
                    convert_to="mlprogram", minimum_deployment_target=ct.target.macOS14,
                    compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_ONLY)
    path = Path(tempfile.mkdtemp(prefix="v1probe_")) / f"{name}.mlpackage"
    ml.save(str(path))
    p = subprocess.run([sys.executable, "-c", CHILD, str(path)], capture_output=True, text=True)
    out = p.stdout + p.stderr
    failed = "ANECCompile" in out or "failed to compile ANE" in out
    import json
    marker = [l for l in p.stdout.splitlines() if l.startswith("PLACEMENT_JSON:")]
    rows = json.loads(marker[-1][len("PLACEMENT_JSON:"):]) if marker else []
    by_dev = collections.Counter()
    for _, d, v in rows:
        by_dev[d] += v
    off = {f"{t}@{d}": v for t, d, v in rows if d != "NeuralEngine"}
    if not quiet:
        print(f"[probe] {name}: ANE compile {'FAILED' if failed else 'ok'} | ops by device {dict(by_dev)} | "
              f"not on ANE: {off}", flush=True)
    return failed, dict(by_dev), off
