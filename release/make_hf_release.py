#!/usr/bin/env python3
"""Assemble a Hugging Face model-repo folder for V1 on Core ML. Dry run by default: prints the layout,
sizes and sha256 that would be written; nothing is copied without --write.

  ~/.venv-coreml/bin/python release/make_hf_release.py [--build faithful/fp16] [--include-fp32]
      [--include-pytorch-head] [--repo-name gemma-stt-v1-coreml] [--hf-user USER] [--author NAME]
      [--github-url URL] [--out DIR] [--no-hash] [--preview] [--write]

Layout (under --out, default <data root>/release/hf/<repo-name>/):
  coreml/v1_{5s,10s,20s,35s}.mlpackage       the chosen build (<variant>/<precision> under <data root>/coreml/models)
  coreml_fp32/v1_*.mlpackage                 --include-fp32: the exact fp32 build of the same variant
  tokenizer/spm_unigram256.model             SentencePiece unigram, 256 pieces (blank = id 256)
  lm/sp256_6gram.bin, lm/README.md           KenLM subword 6-gram (OpenSLR SLR11 text) and its attribution / build
  pytorch/head_F0-0-0-4c.pt                  --include-pytorch-head: the V1 CTC head (PyTorch)
  preprocessor_config.json                   google/embeddinggemma-2's feature extractor config, unchanged
  config.json                                inputs / outputs / windows / features / decoder settings
  README.md                                  model card (release/hf_model_card_template.md + results from the reports)
  NOTICE                                     Gemma Terms of Use notice and upstream attribution
  SHA256SUMS                                 every file above (`shasum -a 256 -c SHA256SUMS`)

Paths come from coreml/common.py (data root ~/gemma_stt). The V1 head, tokenizer and LM must match
V1.md's sha256. Refused: the Discord webhook file and any file containing a Discord webhook URL;
generated text must not contain the local home directory.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "coreml"))
from common import (HEAD, LM, MODEL_ID, MODEL_REVISION, OUT, SPM, SR, SUBSAMPLE, V1_DECODER,  # noqa: E402
                    V1_SHA256, WINDOWS)

STT_HOME = OUT.parent                           # common.OUT = <data root>/coreml
MODELS = OUT / "models"
RESULTS = OUT / "results.jsonl"
V1_MD = STT_HOME / "V1.md"
PHONON2 = STT_HOME / "phonon2/results"
TEMPLATE = HERE / "hf_model_card_template.md"
WEBHOOK_FILE = STT_HOME / "runs/.discord_webhook"
_HOOK = b".com/api/" + b"webhooks"              # split so the repo's own secret grep does not match this file
SECRET_PATTERNS = (b"discord" + _HOOK, b"discordapp" + _HOOK)
LOCAL_HOME = str(Path.home()).encode()
CHUNK = 1 << 22
SKIP_NAMES = {".DS_Store"}
SPEC_TARGET = {7: "macOS 13", 8: "macOS 14", 9: "macOS 15"}   # Core ML specification version -> minimum macOS
HEAD_ARCH = {"d_in": 1024, "n_out": 257, "future": [0, 0, 0, 4], "causal_conv": True, "params": 13283585}  # V1.md
FRAME_RATE_HZ = 25
LOOKAHEAD_MS = 160
DECODER_EXTRA = dict(unk_score_offset=-10.0, lm_score_boundary=True, prune_history=True)  # dec_cbd defaults (V1.md)


# ---------------------------------------------------------------- safety
def refuse(msg):
    raise SystemExit(f"REFUSED: {msg}")


def check_source(path):
    p = Path(path).resolve()
    if p == WEBHOOK_FILE.resolve() or "webhook" in p.name.lower():
        refuse(f"{path} is (or looks like) the Discord webhook secret")
    return p


def check_text(name, text):
    b = text.encode()
    for pat in SECRET_PATTERNS:
        if pat in b:
            refuse(f"generated {name} contains a Discord webhook URL")
    if LOCAL_HOME in b:
        refuse(f"generated {name} contains the local home directory {LOCAL_HOME.decode()}")


def scan_and_hash(path):
    """(sha256, warnings) in one pass; refuses files that contain a Discord webhook URL."""
    h, tail, home_hit = hashlib.sha256(), b"", False
    keep = max(len(p) for p in SECRET_PATTERNS + (LOCAL_HOME,)) - 1
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
            window = tail + chunk
            for pat in SECRET_PATTERNS:
                if pat in window:
                    refuse(f"{path} contains a Discord webhook URL")
            home_hit = home_hit or LOCAL_HOME in window
            tail = window[-keep:]
    return h.hexdigest(), ([f"contains the local path {LOCAL_HOME.decode()}"] if home_hit else [])


# ---------------------------------------------------------------- plan
class Item:
    """One file of the release: copied from `src`, or generated `text`."""

    def __init__(self, dest, src=None, text=None, group=None):
        self.dest, self.src, self.text, self.group = dest, src, text, group or dest
        self.size = len(text.encode()) if text is not None else src.stat().st_size
        self.sha256, self.warnings = None, []

    def hash(self):
        if self.text is not None:
            self.sha256 = hashlib.sha256(self.text.encode()).hexdigest()
        else:
            self.sha256, self.warnings = scan_and_hash(self.src)


def package_files(pkg, dest_dir):
    pkg = check_source(pkg)
    if not (pkg / "Manifest.json").is_file():
        raise SystemExit(f"missing Core ML package {pkg}")
    items = []
    for f in sorted(pkg.rglob("*")):
        if f.is_dir() or f.name in SKIP_NAMES:
            continue
        check_source(f)
        items.append(Item(f"{dest_dir}/{pkg.name}/{f.relative_to(pkg).as_posix()}", src=f, group=f"{dest_dir}/{pkg.name}"))
    return items


def read_spec(pkg):
    """(specification version, {name: (shape, dtype)}) from the package's model.mlmodel."""
    from coremltools.proto import FeatureTypes_pb2, Model_pb2
    spec = Model_pb2.Model()
    spec.ParseFromString((pkg / "Data/com.apple.CoreML/model.mlmodel").read_bytes())
    io = {}
    for d in list(spec.description.input) + list(spec.description.output):
        t = d.type.multiArrayType
        io[d.name] = (list(t.shape), FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Name(t.dataType).lower())
    return spec.specificationVersion, io


def window_specs(build_dir):
    out = {}
    for w, t_mel in WINDOWS.items():
        version, io = read_spec(build_dir / f"v1_{w}.mlpackage")
        want = {"mel": [1, t_mel, 128], "valid_len": [1], "log_probs": [1, t_mel // SUBSAMPLE, 257],
                "l11": [1, t_mel // SUBSAMPLE, 1024]}
        got = {k: v[0] for k, v in io.items()}
        if got != want:
            raise SystemExit(f"{build_dir}/v1_{w}.mlpackage: shapes {got} != {want}")
        out[w] = (version, io)
    return out


# ---------------------------------------------------------------- reports
def md_section(text, heading):
    m = re.search(rf"^{re.escape(heading)}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not m:
        raise SystemExit(f"{V1_MD}: no section {heading!r}")
    return m.group(1)


def md_table(section):
    rows = [ln for ln in section.splitlines() if ln.startswith("|")]
    if len(rows) < 3:
        raise SystemExit("no markdown table in section")
    return "\n".join(rows)


def v1_wers(v1_text):
    """{split: (words, greedy %, greedy edits, beam %, beam edits)} from V1.md's WER table."""
    out = {}
    for ln in md_table(md_section(v1_text, "## WER (%)")).splitlines()[2:]:
        c = [x.strip().strip("*") for x in ln.strip("|").split("|")]
        g = re.match(r"([\d.]+)\s*\(([\d,]+)\)", c[2])
        b = re.match(r"\**([\d.]+)\**\s*\(([\d,]+)\)", c[3])
        out[c[0]] = (int(c[1].replace(",", "")), float(g[1]), int(g[2].replace(",", "")),
                     float(b[1]), int(b[2].replace(",", "")))
    return out


def latest_record(test, variant, precision):
    if not RESULTS.exists():
        return None
    recs = [r for r in map(json.loads, RESULTS.read_text().splitlines())
            if r.get("test") == test and r.get("variant") == variant and r.get("precision") == precision]
    return recs[-1] if recs else None


def v1_table(v1):
    lines = ["| Split | Words | Greedy WER % (edits) | Beam + LM WER % (edits) |", "|---|---|---|---|"]
    for s, (n, g, ge, b, be) in v1.items():
        lines.append(f"| {s} | {n:,} | {g:.3f} ({ge:,}) | **{b:.3f}** ({be:,}) |")
    return "\n".join(lines)


def t9_table(rec, build):
    if rec is None:
        return (f"_No T9 record for `{build}` yet. Run it with `V1_VARIANT=… V1_PRECISION=… make test-full` "
                "in the code repo, then re-run make_hf_release.py._")
    lines = ["| Split | Utterances | Words | Core ML WER % (edits) | V1 PyTorch WER % | Difference (points) |",
             "|---|---|---|---|---|---|"]
    for s in ("dev-clean", "dev-other"):
        m = rec["metrics"][s]
        lines.append(f"| {s} | {m['utts']:,} | {m['words']:,} | **{100 * m['wer']:.3f}** ({m['edits']:,}) | "
                     f"{100 * m['v1_wer']:.3f} | {m['diff_points']:+.3f} |")
    status = "PASS" if rec["pass"] else "FAIL"
    lines += ["", f"Gate: within 0.1 points of V1 on both splits: **{status}** (run `{rec['run']}`, "
              f"{rec['compute_units']}, coremltools {rec['versions']['coremltools']}, {rec['time'][:10]})."]
    return "\n".join(lines)


def phonon2_table(v1, t11):
    p2 = {s: json.loads((PHONON2 / f"score_harness_{s}.json").read_text()) for s in ("test-clean", "test-other")}
    head = "| Split | Words | Phonon-2 WER % (edits) | V1 PyTorch WER % (edits) |"
    if t11:
        head += " This Core ML build WER % (edits) |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for s, r in p2.items():
        b, be = v1[s][3], v1[s][4]
        row = f"| {s} | {r['punct']['words']:,} | {100 * r['punct']['wer']:.3f} ({r['punct']['edits']:,}) | {b:.3f} ({be:,}) |"
        if t11:
            m = t11["metrics"]["splits"][s]
            row += f" {100 * m['coreml_wer']:.3f} ({m['coreml_edits']:,}) |"
        lines.append(row)
    rtf = ", ".join(f"{s} {r['rtfx']:.1f}×" for s, r in p2.items())
    lines += ["", f"Phonon-2 speed on the same Mac (Apple M4 Pro, samples in memory → text, model load excluded): {rtf} real time."]
    if not t11:
        lines += ["The Core ML build itself has not been run on the test sets yet (test T11); "
                  "its dev-set WER matches V1 (T9, above)."]
    return "\n".join(lines)


def speed_section(t10):
    if t10 is None:
        return ""
    lines = ["**Speed** (test T10, full dev-clean, model load excluded, after warm-up; RTFx = audio seconds / "
             "processing seconds):", "",
             "| Build : compute units | End to end | Encoder + head | Decoder | WER % | Peak RSS MB |",
             "|---|---|---|---|---|---|"]
    for r in t10["metrics"]["runs"]:
        lines.append(f"| {r['precision']} : {r['compute_units']} | {r['rtfx_e2e']:.1f}× | {r['rtfx_model']:.1f}× | "
                     f"{r['rtfx_decoder']:.1f}× | {100 * r['wer']:.3f} | {r['peak_rss_mb']:.0f} |")
    broken = [f"{r['precision']} : {r['compute_units']}" for r in t10["metrics"]["runs"] if r["wer"] > 0.5]
    if broken:
        lines += ["", f"**Do not use {', '.join(broken)}:** this configuration produces garbage (WER above 50 %). "
                  "Use `CPU_AND_NE` (recommended) or `CPU_AND_GPU`."]
    return "\n".join(lines)


# ---------------------------------------------------------------- generated files
def preprocessor_config():
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(MODEL_ID, "preprocessor_config.json", revision=MODEL_REVISION)
    if not isinstance(p, str):
        raise SystemExit(f"{MODEL_ID}@{MODEL_REVISION} preprocessor_config.json is not in the local HF cache")
    return check_source(p)


def make_config(variant, precision, specs, fp32, fe_cfg, include_head):
    version, io = specs["5s"]
    shape = lambda n: ["1", "T_mel", "128"] if n == "mel" else ["1"] if n == "valid_len" else \
        ["1", "T_mel/4", str(io[n][0][2])]
    cfg = {
        "name": "V1 (EmbeddingGemma 2 audio tower L11 + causal CTC head)",
        "version": "v1",
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "component": "audio_tower", "layers": [0, 11]},
        "build": {"variant": variant, "precision": precision, "format": "mlprogram",
                  "specification_version": version, "minimum_deployment_target": SPEC_TARGET.get(version, f"spec {version}"),
                  "recommended_compute_units": "CPU_AND_NE", "dir": "coreml"},
        "pipeline": ["log-mel (preprocessor_config.json)", "audio_tower.layers[0..11]",
                     "per-frame int8 round trip (absmax / 127)", "standardize (head mean / std)",
                     "causal CTC head", "log_softmax"],
        "inputs": [{"name": n, "shape": shape(n), "dtype": io[n][1],
                    "description": d} for n, d in (("mel", "log-mel frames, zero-padded to the window"),
                                                   ("valid_len", "number of valid mel frames"))],
        "outputs": [{"name": n, "shape": shape(n), "dtype": io[n][1], "description": d}
                    for n, d in (("log_probs", "CTC log-probabilities, 256 pieces + blank"),
                                 ("l11", "layer-11 features after the int8 round trip"))],
        "valid_output_frames": "ceil(valid_len / 4)",
        "windows": [{"name": w, "seconds": t // 100, "mel_frames": t, "output_frames": t // SUBSAMPLE,
                     "file": f"coreml/v1_{w}.mlpackage"} for w, t in WINDOWS.items()],
        "feature_extractor": {"config": "preprocessor_config.json", **fe_cfg,
                              "call": {"sampling_rate": SR, "padding": "longest", "truncation": False,
                                       "valid_frames": "sum(input_features_mask)"},
                              "mel_frame_rate_hz": SR // fe_cfg["hop_length"]},
        "frame_rate_hz": FRAME_RATE_HZ,
        "subsampling": SUBSAMPLE,
        "lookahead_ms": LOOKAHEAD_MS,
        "encoder_lookahead_ms": 0,
        "head": dict(HEAD_ARCH),
        "vocab": {"tokenizer": "tokenizer/spm_unigram256.model", "type": "sentencepiece-unigram", "size": 256,
                  "blank_id": 256, "num_classes": 257},
        "decoder": {"library": "ctc-beam-decoder", "library_version": "0.2.0", "lm": "lm/sp256_6gram.bin",
                    "lm_order": 6, "piece_word_prefix": "t", **V1_DECODER, **DECODER_EXTRA},
        "long_audio": {"max_mel_frames": max(WINDOWS.values()), "left_context_s": 2.0,
                       "cut": "quietest point in the last 10 s (coreml/transcribe.py)"},
        "text_normalization": "lowercase, collapse whitespace",
    }
    if fp32:
        cfg["fp32_build"] = {"dir": "coreml_fp32", "precision": "fp32", "recommended_compute_units": "CPU_AND_GPU"}
    if include_head:
        cfg["pytorch_head"] = "pytorch/head_F0-0-0-4c.pt"
    return json.dumps(cfg, indent=2) + "\n"


def lm_readme(v1_text):
    sec = md_section(v1_text, "## Subword 6-gram build")
    cmd = re.search(r"```bash\n(.*?)```", sec, re.S)
    counts = re.search(r"\*\*N-grams kept \(orders 1 to 6\):\*\* (.*)", sec)
    return f"""# Subword 6-gram LM (`sp256_6gram.bin`)

KenLM binary (trie, 8-bit quantized) over the 256 SentencePiece pieces of `../tokenizer/spm_unigram256.model`.
Every piece id `i` is the LM word `t<i>`; decode with `units="piece"` (see `../config.json`, `decoder`).

- **Training text:** all 40,418,260 lines of `librispeech-lm-norm.txt.gz` from the
  [LibriSpeech language-model resources (OpenSLR SLR11)](https://www.openslr.org/11/), public domain
  (normalized text of Project Gutenberg books). Each line was lowercased, whitespace-collapsed and
  encoded with the tokenizer: 1,894,497,415 tokens, 0 `<unk>`. No LibriSpeech transcripts were added.
- **Toolkit:** [KenLM](https://github.com/kpu/kenlm) commit `4cb443e60b7bf2c0ddf3c745378f76cb59e254e5`,
  `MAX_ORDER=6` (LGPL-2.1; no KenLM code is included here).
- **N-grams kept (orders 1 to 6):** {counts[1] if counts else "see the code repo"}
- `--discount_fallback` is needed: the 1-gram Kneser-Ney discounts are undefined with only 256 symbols.

Build command (code repo, `decode/build_subword_lm.py`):

```bash
{cmd[1].rstrip() if cmd else "see decode/build_subword_lm.py"}
```
"""


def notice(variant, precision, author, fp32, include_head):
    pal = "pal" in precision
    dirs = "coreml/" + (" and coreml_fp32/" if fp32 else "")
    return f"""NOTICE

This repository combines components under different terms.

1. Core ML packages ({dirs}) - derived from google/embeddinggemma-2

   They contain layers 0-11 of the audio tower of google/embeddinggemma-2
   (https://huggingface.co/google/embeddinggemma-2, revision {MODEL_REVISION}),
   converted to Core ML (build {variant}/{precision}) together with the CTC head below.

   Gemma is provided under and subject to the Gemma Terms of Use found at
   ai.google.dev/gemma/terms
   Use is also subject to the Gemma Prohibited Use Policy:
   ai.google.dev/gemma/prohibited_use_policy

   Modifications made by {author}:
   - the audio tower truncated after layer 11 and exported as fixed-window Core ML programs;
   - blocked local attention rewritten as dense banded attention (identical in fp32);
   - log-softmax rewritten in a numerically stable form; static shapes;
   - a per-frame int8 quantize/dequantize step, standardization and a CTC head appended;
   - weights stored in {"fp16, encoder weights 6-bit palettized" if pal else precision}{" (and fp32 in coreml_fp32/)" if fp32 else ""}.

   preprocessor_config.json is copied unchanged from google/embeddinggemma-2 at the same revision.

2. CTC head{" (pytorch/head_F0-0-0-4c.pt, also inside the Core ML packages)" if include_head else " (inside the Core ML packages)"} and tokenizer (tokenizer/spm_unigram256.model)

   Trained by {author} on the LibriSpeech ASR corpus (OpenSLR SLR12,
   https://www.openslr.org/12/, CC BY 4.0): V. Panayotov, G. Chen, D. Povey and
   S. Khudanpur, "LibriSpeech: an ASR corpus based on public domain audio books",
   ICASSP 2015.

3. Language model (lm/sp256_6gram.bin)

   Built with KenLM (https://github.com/kpu/kenlm) from the LibriSpeech LM corpus
   normalized text, librispeech-lm-norm.txt.gz (OpenSLR SLR11,
   https://www.openslr.org/11/), public domain. See lm/README.md.

4. Decoder (not included; used by the usage example)

   ctc-beam-decoder 0.2.0 (https://pypi.org/project/ctc-beam-decoder/),
   Apache License 2.0.
"""


def files_table(items):
    desc = {"coreml": "Core ML packages, one per window (encoder + CTC head)",
            "coreml_fp32": "exact fp32 build (reference; CPU/GPU, not the Neural Engine)",
            "tokenizer/spm_unigram256.model": "SentencePiece unigram tokenizer, 256 pieces",
            "lm/sp256_6gram.bin": "KenLM subword 6-gram (trie, 8-bit quantized)",
            "lm/README.md": "LM attribution and build command",
            "pytorch/head_F0-0-0-4c.pt": "V1 CTC head, PyTorch (13.3M parameters)",
            "preprocessor_config.json": "feature extractor config (from google/embeddinggemma-2)",
            "config.json": "inputs, outputs, windows, features, decoder settings",
            "NOTICE": "license notice and attribution"}
    groups = {}
    for it in items:
        groups[it.group] = groups.get(it.group, 0) + it.size
    lines = ["| Path | Size | Contents |", "|---|---|---|"]
    for g, size in groups.items():
        top = g.split("/")[0] if g.startswith("coreml") else g
        lines.append(f"| `{g}` | {human(size)} | {desc.get(top, '')} |")
    lines += ["| `README.md` | | this model card |", "| `SHA256SUMS` | | sha256 of every file |"]
    return "\n".join(lines)


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000


def render(template, values):
    out = template
    for k, v in values.items():
        out = out.replace("{{" + k + "}}", v)
    left = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", out)))
    if left:
        raise SystemExit(f"template placeholders not filled: {left}")
    return re.sub(r"\n{3,}", "\n\n", out)


def git_state():
    run = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "--short", "HEAD"), bool(run("status", "--porcelain"))


# ---------------------------------------------------------------- main
def build_plan(args):
    variant, _, precision = args.build.partition("/")
    build_dir = check_source(MODELS / args.build)
    if not precision or not build_dir.is_dir():
        raise SystemExit(f"--build {args.build}: no directory {build_dir} (want <variant>/<precision>)")
    fp32_dir = MODELS / variant / "fp32"
    if args.include_fp32:
        if precision == "fp32":
            raise SystemExit("--include-fp32 with an fp32 build: nothing to add")
        if not fp32_dir.is_dir():
            raise SystemExit(f"--include-fp32: no {fp32_dir}")
    specs = window_specs(build_dir)
    if args.include_fp32:
        window_specs(fp32_dir)

    items = []
    for w in WINDOWS:
        items += package_files(build_dir / f"v1_{w}.mlpackage", "coreml")
    if args.include_fp32:
        for w in WINDOWS:
            items += package_files(fp32_dir / f"v1_{w}.mlpackage", "coreml_fp32")
    items.append(Item("tokenizer/spm_unigram256.model", src=check_source(SPM)))
    items.append(Item("lm/sp256_6gram.bin", src=check_source(LM)))
    if args.include_pytorch_head:
        items.append(Item("pytorch/head_F0-0-0-4c.pt", src=check_source(HEAD)))
    pre = preprocessor_config()
    items.append(Item("preprocessor_config.json", src=pre))

    v1_text = V1_MD.read_text()
    v1 = v1_wers(v1_text)
    fe_cfg = json.loads(Path(pre).read_text())
    texts = {
        "lm/README.md": lm_readme(v1_text),
        "config.json": make_config(variant, precision, specs, args.include_fp32, fe_cfg, args.include_pytorch_head),
        "NOTICE": notice(variant, precision, args.author, args.include_fp32, args.include_pytorch_head),
    }
    for dest, t in texts.items():
        items.append(Item(dest, text=t))

    version = specs["5s"][0]
    readme_size_guess = 12_000
    total = sum(i.size for i in items) + readme_size_guess
    head_link = ("ln -s $ROOT/pytorch/head_F0-0-0-4c.pt $GEMMA_STT_HOME/runs/full_160ms/" if args.include_pytorch_head
                 else "# also needs $GEMMA_STT_HOME/runs/full_160ms/head_F0-0-0-4c.pt (checked by sha256; not in this repo)")
    pal = "pal" in precision
    build_desc = ("fp16 compute, encoder weights 6-bit palettized" if pal else
                  "fp16 compute and weights, runs on the Neural Engine" if precision == "fp16" else
                  "fp32, exact reference" if precision == "fp32" else precision)
    commit, dirty = git_state()
    values = {
        "MODEL_NAME": args.repo_name, "REPO_NAME": args.repo_name, "REPO_ID": f"{args.hf_user}/{args.repo_name}",
        "GITHUB_URL": args.github_url, "AUTHOR": args.author, "BUILD": args.build, "BUILD_DESC": build_desc,
        "VARIANT": variant, "PRECISION": precision, "MIN_OS": SPEC_TARGET.get(version, f"Core ML spec {version}"),
        "MODEL_REVISION": MODEL_REVISION, "MODEL_REVISION_SHORT": MODEL_REVISION[:7],
        "TOTAL_SIZE": human(total), "HEAD_LINK": head_link,
        "V1_WER_TABLE": v1_table(v1),
        "T9_TABLE": t9_table(latest_record("test_t9_wer_vs_v1", variant, precision), args.build),
        "PHONON2_TABLE": phonon2_table(v1, latest_record("test_t11_release", variant, precision)),
        "SPEED_SECTION": speed_section(latest_record("test_t10_speed", variant, precision)),
    }
    values["FILES_TABLE"] = files_table(items)
    readme = render(TEMPLATE.read_text(), values)
    readme += f"\n<!-- generated by release/make_hf_release.py from code commit {commit}{' (dirty)' if dirty else ''} -->\n"
    items.append(Item("README.md", text=readme))
    for it in items:
        if it.text is not None:
            check_text(it.dest, it.text)
    return items, build_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default="faithful/fp16", help="<variant>/<precision> directory under coreml/models")
    ap.add_argument("--include-fp32", action="store_true", help="also ship <variant>/fp32 as coreml_fp32/")
    ap.add_argument("--include-pytorch-head", action="store_true", help="also ship the PyTorch CTC head")
    ap.add_argument("--repo-name", default="gemma-stt-v1-coreml")
    ap.add_argument("--hf-user", default="YOUR_HF_USER", help="Hugging Face user or org (README usage)")
    ap.add_argument("--author", default="the repository author", help="name used in NOTICE / README")
    ap.add_argument("--github-url", default="https://github.com/YOUR_GITHUB_USER/emb_gemma2")
    ap.add_argument("--out", type=Path, default=None, help="default <data root>/release/hf/<repo-name>")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="write", action="store_false", default=False, help="print the plan (default)")
    mode.add_argument("--write", dest="write", action="store_true", help="copy the files and write the folder")
    ap.add_argument("--no-hash", action="store_true", help="dry run: skip sha256 / secret scan of copied files")
    ap.add_argument("--preview", action="store_true", help="also print the generated text files")
    args = ap.parse_args()
    if args.write and args.no_hash:
        ap.error("--write always hashes")
    out = (args.out or STT_HOME / "release/hf" / args.repo_name).expanduser()
    if out.resolve() == REPO or REPO in out.resolve().parents:
        ap.error(f"--out {out} is inside the code repo")

    items, build_dir = build_plan(args)
    for it in items:
        if args.no_hash and it.text is None:
            continue
        if it.src is not None:
            print(f"  hashing {it.dest} ({human(it.size)})", file=sys.stderr, flush=True)
        it.hash()
    sums = "".join(f"{it.sha256}  {it.dest}\n" for it in sorted(items, key=lambda i: i.dest) if it.sha256)
    if not args.no_hash:
        items.append(Item("SHA256SUMS", text=sums))
        items[-1].hash()

    print(f"{'WRITE' if args.write else 'DRY RUN (nothing written; --write to copy)'}: {out}")
    print(f"build {args.build}  ({build_dir})")
    print(f"{'size':>10}  {'sha256':<64}  path  <-  source")
    for it in sorted(items, key=lambda i: i.dest):
        src = "generated" if it.text is not None else str(it.src).replace(str(Path.home()), "~")
        print(f"{human(it.size):>10}  {it.sha256 or '-' * 64:<64}  {it.dest}  <-  {src}")
        for w in it.warnings:
            print(f"{'':>10}  WARNING: {it.dest} {w}")
    groups = {}
    for it in items:
        top = it.dest.split("/")[0] if "/" in it.dest else "(root)"
        groups[top] = groups.get(top, 0) + it.size
    print("totals: " + ", ".join(f"{k} {human(v)}" for k, v in groups.items())
          + f"; all {human(sum(i.size for i in items))} in {len(items)} files")
    if not args.no_hash:
        for it in items:
            if it.src is not None and it.dest in ("tokenizer/spm_unigram256.model", "lm/sp256_6gram.bin",
                                                   "pytorch/head_F0-0-0-4c.pt"):
                want = V1_SHA256[{"tokenizer/spm_unigram256.model": SPM, "lm/sp256_6gram.bin": LM}.get(it.dest, HEAD)]
                if it.sha256 != want:
                    refuse(f"{it.src}: sha256 {it.sha256} != V1 {want}")
        print("V1 files match V1.md's sha256; no Discord webhook URL found.")
    placeholders = [k for k, v in (("--hf-user", args.hf_user == "YOUR_HF_USER"),
                                   ("--github-url", "YOUR_GITHUB_USER" in args.github_url),
                                   ("--author", args.author == "the repository author")) if v]
    if placeholders:
        print("note: README / NOTICE use placeholders; set " + ", ".join(placeholders))
    if args.preview:
        for it in items:
            if it.text is not None and it.dest != "SHA256SUMS":
                print(f"\n===== {it.dest} =====\n{it.text}")
    if not args.write:
        return

    if out.exists():
        raise SystemExit(f"{out} exists; remove it or choose another --out")
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    for it in items:
        dest = tmp / it.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        if it.text is not None:
            dest.write_text(it.text)
            continue
        h = hashlib.sha256()
        with open(it.src, "rb") as fi, open(dest, "wb") as fo:
            while chunk := fi.read(CHUNK):
                h.update(chunk)
                fo.write(chunk)
        if h.hexdigest() != it.sha256:
            raise SystemExit(f"{it.src} changed while copying (sha256 {h.hexdigest()} != {it.sha256})")
    tmp.rename(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
