#!/usr/bin/env python3
"""Assemble the Hugging Face model-repo folder for the V1 release: ONE multifunction Core ML package
(fp16, functions win_5s / win_10s / win_20s / win_35s), the tokenizer, the subword 6-gram LM and the
pinned decoder settings. Dry run by default: prints the layout, sizes, sha256 and every missing
prerequisite; nothing is copied without --write, which refuses while anything is missing.

  ~/.venv-coreml/bin/python release/make_hf_release.py [--hf-user USER] [--author NAME] [--github-url URL]
      [--repo-name gemma-stt-v1-coreml] [--license apache-2.0 [--license-file FILE]] [--out DIR]
      [--no-hash] [--preview] [--write]

Layout (under --out, default <data root>/release/hf/<repo-name>/):
  v1.mlpackage/              <data root>/coreml/models/release/fp16/v1.mlpackage (coreml/export_release.py)
  spm_unigram256.model       SentencePiece unigram tokenizer, 256 pieces (blank = id 256)
  sp256_6gram.bin            KenLM subword 6-gram (OpenSLR SLR11 text)
  config.json                windows -> functions, inputs / outputs, front end, decoder, minimum OS
  LICENSE                    --license-file (default: release/LICENSE-2.0.txt, the Apache License 2.0)
  LICENSE-APACHE-2.0         only when --license is not apache-2.0: the encoder weights are Apache-2.0
  NOTICE                     upstream attribution, modifications, data sources, decoder
  README.md                  model card: release/hf_model_card_template.md + the release package's own results
  SHA256SUMS                 every file above (`shasum -a 256 -c SHA256SUMS`)

Results come only from <data root>/coreml/results.jsonl rows with variant == "release" recorded for this
exact package (model_weight_sha256 = its weight.bin): T9 (dev WER) and T10 (speed, WER, RSS, load under
CPU_AND_NE, CPU_AND_GPU and CPU_ONLY) are required and must have passed; T11 (test WER next to V1 and
Phonon-2) and T6 (Neural Engine share) are shown when present. Refused: the Discord webhook file, any file
containing a Discord webhook URL, and generated text containing the local home directory.
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
from cmlrun import RELEASE, function_name, package  # noqa: E402
from common import LM, MODEL_ID, MODEL_REVISION, OUT, SPM, SR, SUBSAMPLE, V1_DECODER, V1_SHA256, WINDOWS  # noqa: E402

PRECISION = "fp16"
PACKAGE = package(RELEASE, PRECISION)
SOURCE_BUILD = ("fp16safe", "fp16")                # export_release.py's source; used only to preview config.json
STT_HOME = OUT.parent                               # common.OUT = <data root>/coreml
RESULTS = OUT / "results.jsonl"
V1_MD = STT_HOME / "V1.md"
VERSIONS_MD = OUT / "VERSIONS.md"
TEMPLATE = HERE / "hf_model_card_template.md"
APACHE = HERE / "LICENSE-2.0.txt"
APACHE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"   # apache.org/licenses/LICENSE-2.0.txt
WEBHOOK_FILE = STT_HOME / "runs/.discord_webhook"
_HOOK = b".com/api/" + b"webhooks"              # split so the repo's own secret grep does not match this file
SECRET_PATTERNS = (b"discord" + _HOOK, b"discordapp" + _HOOK)
LOCAL_HOME = str(Path.home()).encode()
CHUNK = 1 << 22
SKIP_NAMES = {".DS_Store"}
SPEC_TARGET = {8: "macOS 14 / iOS 17", 9: "macOS 15 / iOS 18"}   # Core ML specification version -> minimum OS
COMPUTE_UNITS = ("CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY")       # recommended first; all three must pass T10
DEFAULT_FUNCTION = function_name("10s")
HEAD_PARAMS = 13_283_585                                          # V1.md
FRAME_RATE_HZ = 25
LOOKAHEAD_MS = 160
DECODER_EXTRA = dict(unk_score_offset=-10.0, lm_score_boundary=True, prune_history=True)  # dec_cbd defaults (V1.md)
MISSING = "MISSING"


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
    if any(pat in b for pat in SECRET_PATTERNS):
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
            if any(pat in window for pat in SECRET_PATTERNS):
                refuse(f"{path} contains a Discord webhook URL")
            home_hit = home_hit or LOCAL_HOME in window
            tail = window[-keep:]
    return h.hexdigest(), ([f"contains the local path {LOCAL_HOME.decode()}"] if home_hit else [])


# ---------------------------------------------------------------- plan items
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


def package_files(pkg):
    items = []
    for f in sorted(check_source(pkg).rglob("*")):
        if f.is_dir() or f.name in SKIP_NAMES:
            continue
        check_source(f)
        items.append(Item(f"{pkg.name}/{f.relative_to(pkg).as_posix()}", src=f, group=pkg.name))
    return items


# ---------------------------------------------------------------- the package
def _load_spec(pkg):
    from coremltools.proto import Model_pb2
    spec = Model_pb2.Model()
    spec.ParseFromString((pkg / "Data/com.apple.CoreML/model.mlmodel").read_bytes())
    return spec


def _io(features):
    from coremltools.proto import FeatureTypes_pb2
    name = FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Name
    return {d.name: (list(d.type.multiArrayType.shape), name(d.type.multiArrayType.dataType).lower()) for d in features}


def _want_shapes(t_mel):
    return {"mel": [1, t_mel, 128], "valid_len": [1], "log_probs": [1, t_mel // SUBSAMPLE, 257],
            "l11": [1, t_mel // SUBSAMPLE, 1024]}


def release_spec(problems):
    """{spec_version, default_function, functions: {fn: io}, metadata} of the release package, validated."""
    spec = _load_spec(PACKAGE)
    d = spec.description
    funcs = {f.name: {**_io(f.input), **_io(f.output)} for f in d.functions}
    want = [function_name(w) for w in WINDOWS]
    if sorted(funcs) != sorted(want):
        problems.append(f"package functions {sorted(funcs)} != {want}")
    if d.defaultFunctionName != DEFAULT_FUNCTION:
        problems.append(f"default function {d.defaultFunctionName!r} != {DEFAULT_FUNCTION!r}")
    for w, t in WINDOWS.items():
        io = funcs.get(function_name(w), {})
        if {k: v[0] for k, v in io.items()} != _want_shapes(t):
            problems.append(f"{function_name(w)}: shapes {io} != {_want_shapes(t)}")
    meta = dict(d.metadata.userDefined)
    want_hashes = {p.name: h for p, h in V1_SHA256.items()}
    if json.loads(meta.get("v1_files_sha256", "{}")) != want_hashes:
        problems.append("package metadata v1_files_sha256 does not match V1.md")
    if json.loads(meta.get("decoder", "{}")) != {"ctc_beam_decoder": "0.2.0", **V1_DECODER, "lm": LM.name}:
        problems.append("package metadata decoder settings differ from V1")
    if spec.specificationVersion not in SPEC_TARGET:
        problems.append(f"unexpected Core ML specification version {spec.specificationVersion}")
    return {"spec_version": spec.specificationVersion, "default_function": d.defaultFunctionName,
            "functions": funcs, "source_build": meta.get("source_build"), "provisional": False}


def preview_spec():
    """Before the release package exists: the I/O of its source window packages (export_release.py copies the
    programs unchanged into the functions); config.json is then marked provisional and --write refuses."""
    variant, precision = SOURCE_BUILD
    funcs = {}
    for w in WINDOWS:
        spec = _load_spec(package(variant, precision, w))
        funcs[function_name(w)] = {**_io(spec.description.input), **_io(spec.description.output)}
    return {"spec_version": 9, "default_function": DEFAULT_FUNCTION, "functions": funcs,
            "source_build": f"{variant}/{precision}", "provisional": True}


# ---------------------------------------------------------------- results (release rows only)
def release_records():
    recs = [json.loads(l) for l in RESULTS.read_text().splitlines()] if RESULTS.exists() else []
    return [r for r in recs if r.get("variant") == RELEASE and r.get("precision") == PRECISION]


def latest(recs, test, weight_sha):
    """Newest row for `test` whose weights are this package's (all windows share one weight.bin)."""
    rows = [r for r in recs if r["test"] == test]
    if weight_sha is not None:
        rows = [r for r in rows if set(r.get("model_weight_sha256", {}).values()) == {weight_sha}]
    return rows[-1] if rows else None


def gather_results(weight_sha):
    """{key: record or None} plus the list of required-but-missing / failed results."""
    recs = release_records()
    res = {"t9": latest(recs, "test_t9_wer_vs_v1", weight_sha), "t10": latest(recs, "test_t10_speed", weight_sha),
           "t11": latest(recs, "test_t11_release", weight_sha),
           "t6": {w: latest(recs, f"test_t6_placement[{w}]", weight_sha) for w in WINDOWS}}
    which = "for this package's weight.bin" if weight_sha else "(weights not checked: --no-hash or no package)"
    missing = []
    for key, test, cmd in (("t9", "T9 dev WER", "make test-full"), ("t10", "T10 speed / WER per compute unit", "make test-full")):
        r = res[key]
        if r is None:
            missing.append(f"{test}: no results.jsonl row with variant == release {which} "
                           f"(run `V1_VARIANT=release V1_PRECISION=fp16 {cmd}`)")
        elif not r["pass"]:
            missing.append(f"{test}: newest release row (run {r['run']}) FAILED")
    if res["t10"] is not None:
        units = {u["compute_units"] for u in t10_runs(res["t10"])}
        for u in COMPUTE_UNITS:
            if u not in units:
                missing.append(f"T10: no release run under {u} in run {res['t10']['run']}")
    return res, missing


def t10_runs(rec):
    return [r for r in rec["metrics"]["runs"] if r["variant"] == RELEASE and r["precision"] == PRECISION]


def missing_note(test, cmd):
    return (f"> **{MISSING}: {test}.** No `results.jsonl` row with `variant == \"release\"` for this package yet "
            f"(`V1_VARIANT=release V1_PRECISION=fp16 {cmd}` in the code repo). `--write` refuses until it exists.")


def t9_table(rec):
    if rec is None:
        return missing_note("T9 (dev-clean / dev-other WER)", "make test-full")
    lines = ["| Split | Utterances | Words | This package: WER % (edits) | V1 PyTorch reference WER % | Difference (points) |",
             "|---|---|---|---|---|---|"]
    for s in ("dev-clean", "dev-other"):
        m = rec["metrics"][s]
        lines.append(f"| {s} | {m['utts']:,} | {m['words']:,} | **{100 * m['wer']:.3f}** ({m['edits']:,}) | "
                     f"{100 * m['v1_wer']:.3f} | {m['diff_points']:+.3f} |")
    lines += ["", f"Gate (within 0.1 points of V1 on both splits): **{'PASS' if rec['pass'] else 'FAIL'}**; run "
              f"`{rec['run']}`, {rec['compute_units']}, coremltools {rec['versions']['coremltools']}, {rec['time'][:10]}."]
    return "\n".join(lines)


def machine():
    m = re.search(r"\*\*Machine:\*\*\s*(.+)", VERSIONS_MD.read_text()) if VERSIONS_MD.exists() else None
    return m[1].strip().rstrip(".") if m else None


def t10_table(rec):
    if rec is None:
        return missing_note("T10 (speed, WER, memory and load time under CPU_AND_NE, CPU_AND_GPU and CPU_ONLY)",
                            "make test-full")
    fmt = lambda x, f: "n/a" if x is None else f.format(x)
    runs = {r["compute_units"]: r for r in t10_runs(rec)}
    lines = ["| Compute units | End to end RTFx | Encoder + head RTFx | Decoder RTFx | dev-clean WER % | Peak RSS MB | "
             "Cold load s | Warm load s |", "|---|---|---|---|---|---|---|---|"]
    for u in COMPUTE_UNITS:
        r = runs.get(u)
        if r is None:
            lines.append(f"| {u} | {MISSING} | | | | | | |")
            continue
        lines.append(f"| {u}{' (recommended)' if u == COMPUTE_UNITS[0] else ''} | {r['rtfx_e2e']:.1f}× | "
                     f"{r['rtfx_model']:.1f}× | {r['rtfx_decoder']:.0f}× | {100 * r['wer']:.3f} | {r['peak_rss_mb']:.0f} | "
                     f"{fmt(r.get('cold_load_s'), '{:.1f}')} | {fmt(r.get('warm_load_s'), '{:.2f}')} |")
    mach = machine()
    lines += ["", f"Full dev-clean ({next(iter(runs.values()))['utts']:,} utterances) one at a time, after a "
              f"{next(iter(runs.values()))['warmup']}-file warm-up; RTFx = audio seconds / processing seconds; end to end "
              "= 16 kHz samples in memory → text, model load excluded. Run "
              f"`{rec['run']}`, {rec['time'][:10]}{f', {mach}' if mach else ''}. "
              f"Gate (every compute unit within 0.1 WER points of V1): **{'PASS' if rec['pass'] else 'FAIL'}**."]
    return "\n".join(lines)


def t11_table(rec):
    if rec is None:
        return ("> Not run yet: T11 (test-clean / test-other once, next to V1 and Phonon-2; "
                "`V1_VARIANT=release V1_PRECISION=fp16 make test-release`). Optional for `--write`.")
    lines = ["| Split | Words | This package: WER % (edits) | V1 PyTorch reference WER % | Phonon-2: WER % (edits) |",
             "|---|---|---|---|---|"]
    for s, m in rec["metrics"]["splits"].items():
        lines.append(f"| {s} | {m['words']:,} | **{100 * m['coreml_wer']:.3f}** ({m['coreml_edits']:,}) | "
                     f"{100 * m['v1_pytorch_wer']:.3f} | {100 * m['phonon2_wer']:.3f} ({m['phonon2_edits']:,}) |")
    rtf = ", ".join(f"{s} {m['phonon2_rtfx_our_method']:.1f}×" for s, m in rec["metrics"]["splits"].items())
    lines += ["", "Phonon-2 = FluidAudio `FluidInference/phonon-2-coreml` (Core ML, Neural Engine), scored with the same "
              "references and corpus scorer; punctuation except apostrophes is mapped to spaces for all systems (a no-op "
              f"on this model's output). Phonon-2 speed, same method on the same Mac: {rtf}. Run `{rec['run']}`, "
              f"{rec['time'][:10]}."]
    return "\n".join(lines)


def placement_line(t6):
    have = {w: r for w, r in t6.items() if r is not None}
    if not have:
        return ""
    share = ", ".join(f"`{function_name(w)}` {100 * r['metrics']['ane_cost_share']:.2f} %" for w, r in have.items())
    return (f"- **Neural Engine placement** (share of the estimated cost under `CPU_AND_NE`, test T6): {share}. "
            "The rest are integer mask ops on the CPU.")


# ---------------------------------------------------------------- generated files
def preprocessor_config():
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(MODEL_ID, "preprocessor_config.json", revision=MODEL_REVISION)
    if not isinstance(p, str):
        raise SystemExit(f"{MODEL_ID}@{MODEL_REVISION} preprocessor_config.json is not in the local HF cache")
    return json.loads(check_source(p).read_text())


def make_config(spec, fe_cfg, license_id):
    io = spec["functions"][DEFAULT_FUNCTION]
    sym = {"mel": ["1", "T_mel", "128"], "valid_len": ["1"], "log_probs": ["1", "T_mel/4", "257"],
           "l11": ["1", "T_mel/4", "1024"]}
    desc = {"mel": "log-mel frames (see feature_extractor), zero-padded to the function's window",
            "valid_len": "number of valid mel frames", "log_probs": "CTC log-probabilities (natural log), 256 pieces + blank",
            "l11": "layer-11 features after the int8 round trip"}
    cfg = {
        "name": "V1: EmbeddingGemma 2 audio tower (layers 0-11) + causal CTC head, Core ML",
        "version": "v1",
        "license": license_id,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "license": "apache-2.0",
                       "component": "audio_tower", "layers": [0, 11]},
        "package": {"file": PACKAGE.name, "format": "mlprogram (multifunction, shared weights)",
                    "precision": PRECISION, "source_build": spec["source_build"],
                    "specification_version": spec["spec_version"],
                    "minimum_os": SPEC_TARGET.get(spec["spec_version"]),
                    "compute_units": {"recommended": COMPUTE_UNITS[0], "supported": list(COMPUTE_UNITS)},
                    "default_function": spec["default_function"]},
        "windows": [{"function": function_name(w), "seconds": t / 100, "mel_frames": t, "output_frames": t // SUBSAMPLE}
                    for w, t in WINDOWS.items()],
        "window_choice": "smallest window with mel_frames >= valid mel frames",
        "inputs": [{"name": n, "shape": sym[n], "dtype": io[n][1], "description": desc[n]} for n in ("mel", "valid_len")],
        "outputs": [{"name": n, "shape": sym[n], "dtype": io[n][1], "description": desc[n]} for n in ("log_probs", "l11")],
        "valid_output_frames": "ceil(valid_len / 4)",
        "feature_extractor": {"class": fe_cfg["feature_extractor_type"], "preprocessor_config": fe_cfg,
                              "source": f"{MODEL_ID}@{MODEL_REVISION}/preprocessor_config.json (unchanged)",
                              "call": {"sampling_rate": SR, "padding": "longest", "truncation": False},
                              "valid_mel_frames": "sum(input_features_mask)",
                              "mel_frame_rate_hz": SR // fe_cfg["hop_length"]},
        "frame_rate_hz": FRAME_RATE_HZ,
        "subsampling": SUBSAMPLE,
        "lookahead_ms": LOOKAHEAD_MS,
        "encoder_lookahead_ms": 0,
        "ctc_head": {"params": HEAD_PARAMS, "future_frames": 4, "causal_conv": True},
        "vocab": {"tokenizer": SPM.name, "type": "sentencepiece-unigram", "size": 256, "blank_id": 256,
                  "num_classes": 257},
        "decoder": {"library": "ctc-beam-decoder", "library_version": "0.2.0", "lm": LM.name, "lm_order": 6,
                    "piece_word_prefix": "t", **V1_DECODER, **DECODER_EXTRA},
        "long_audio": {"max_mel_frames": max(WINDOWS.values()), "left_context_s": 2.0,
                       "cut": "quietest point in the last 10 s (code repo: coreml/transcribe.py)"},
        "text_normalization": "lowercase, collapse whitespace",
    }
    return json.dumps(cfg, indent=2) + "\n"


def notice(author, license_id):
    own = ("The repository as a whole is licensed under the Apache License, Version 2.0 (LICENSE)."
           if license_id == "apache-2.0" else
           f"Files authored for this repository are licensed under {license_id} (LICENSE). The Core ML package "
           "contains weights derived from google/embeddinggemma-2 and remains subject to the Apache License, "
           "Version 2.0 (LICENSE-APACHE-2.0).")
    return f"""V1 speech recognition on Core ML
Copyright {author}

{own}

1. Core ML package (v1.mlpackage), derived from google/embeddinggemma-2

   It contains layers 0-11 of the audio tower of EmbeddingGemma 2 by Google DeepMind,
   https://huggingface.co/google/embeddinggemma-2 (revision {MODEL_REVISION}),
   licensed under the Apache License, Version 2.0
   (https://ai.google.dev/gemma/docs/gemma_4_license).

   Modifications made by {author}:
   - the audio tower truncated after layer 11 and converted to Core ML ML programs,
     one function per fixed input window (5, 10, 20, 35 s) sharing one set of weights;
   - blocked local attention rewritten as dense banded attention (identical in fp32);
   - RMSNorm inputs scaled by powers of two so fp16 norms on the CPU / GPU do not
     overflow, and constant padding fill (identical in fp32);
   - log-softmax rewritten in a numerically stable form; static shapes;
   - a per-frame int8 quantize/dequantize step, standardization and a CTC head appended;
   - weights stored in float16.

   The upstream model card states that deployments must adhere to the Gemma Prohibited
   Use Policy: https://ai.google.dev/gemma/prohibited_use_policy

   config.json embeds the upstream preprocessor_config.json unchanged.

2. CTC head (inside v1.mlpackage) and tokenizer (spm_unigram256.model)

   Trained by {author} on the LibriSpeech ASR corpus, OpenSLR SLR12,
   https://www.openslr.org/12/ (license stated there: CC BY 4.0).
   V. Panayotov, G. Chen, D. Povey and S. Khudanpur, "LibriSpeech: an ASR corpus based
   on public domain audio books", ICASSP 2015.

3. Language model (sp256_6gram.bin)

   Built with KenLM (https://github.com/kpu/kenlm; no KenLM code is included) from
   librispeech-lm-norm.txt.gz of "LibriSpeech language models, vocabulary and G2P models",
   OpenSLR SLR11, https://www.openslr.org/11/ (license stated there: Public domain).

4. Decoder (not included; used by the usage example)

   ctc-beam-decoder 0.2.0, https://pypi.org/project/ctc-beam-decoder/,
   Apache License, Version 2.0.
"""


def lm_section(v1_text):
    sec = re.search(r"^## Subword 6-gram build\s*$(.*?)(?=^## |\Z)", v1_text, re.M | re.S)
    if not sec:
        raise SystemExit(f"{V1_MD}: no section '## Subword 6-gram build'")
    cmd = re.search(r"```bash\n(.*?)```", sec[1], re.S)
    counts = re.search(r"\*\*N-grams kept \(orders 1 to 6\):\*\* (.*)", sec[1])
    if not (cmd and counts):
        raise SystemExit(f"{V1_MD}: LM build command or n-gram counts not found")
    return counts[1].strip(), cmd[1].rstrip()


def files_table(items, license_id):
    desc = {PACKAGE.name: "Core ML package: functions `win_5s`, `win_10s` (default), `win_20s`, `win_35s` (encoder + CTC head)",
            SPM.name: "SentencePiece unigram tokenizer, 256 pieces (blank = id 256)",
            LM.name: "KenLM subword 6-gram (trie, 8-bit quantized)",
            "config.json": "windows → functions, inputs / outputs, front end, decoder settings, minimum OS",
            "LICENSE": f"license ({license_id})", "LICENSE-APACHE-2.0": "Apache License 2.0 (encoder weights)",
            "NOTICE": "attribution, modifications, data sources"}
    groups = {}
    for it in items:
        groups[it.group] = groups.get(it.group, 0) + it.size
    lines = ["| Path | Size | Contents |", "|---|---|---|"]
    lines += [f"| `{g}` | {human(s)} | {desc.get(g, '')} |" for g, s in groups.items()]
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
        if not v:
            out = out.replace("{{" + k + "}}\n", "")
        out = out.replace("{{" + k + "}}", v)
    left = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", out)))
    if left:
        raise SystemExit(f"template placeholders not filled: {left}")
    return re.sub(r"\n{3,}", "\n\n", out)


def git_state():
    run = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "--short", "HEAD"), bool(run("status", "--porcelain"))


# ---------------------------------------------------------------- plan
def build_plan(args):
    problems = []                       # anything here makes --write refuse
    have_pkg = (PACKAGE / "Manifest.json").is_file()
    items = []
    if have_pkg:
        spec = release_spec(problems)
        items += package_files(PACKAGE)
    else:
        problems.append(f"release package {PACKAGE} does not exist (built by coreml/export_release.py)")
        spec = preview_spec()

    weight_sha = None
    if have_pkg and not args.no_hash:
        print(f"  hashing {PACKAGE.name} weights", file=sys.stderr, flush=True)
        w = next(i for i in items if i.dest.endswith("weights/weight.bin"))
        w.hash()
        weight_sha = w.sha256
    results, missing = gather_results(weight_sha)
    problems += missing

    items.append(Item(SPM.name, src=check_source(SPM)))
    items.append(Item(LM.name, src=check_source(LM)))

    lic = check_source(args.license_file)
    if args.license == "apache-2.0" and hashlib.sha256(lic.read_bytes()).hexdigest() != APACHE_SHA256:
        problems.append(f"--license apache-2.0 but {lic} is not the Apache License 2.0 text")
    items.append(Item("LICENSE", src=lic))
    if args.license != "apache-2.0":
        items.append(Item("LICENSE-APACHE-2.0", src=check_source(APACHE)))

    fe_cfg = preprocessor_config()
    texts = {"config.json": make_config(spec, fe_cfg, args.license), "NOTICE": notice(args.author, args.license)}
    for dest, t in texts.items():
        items.append(Item(dest, text=t))

    counts, lm_cmd = lm_section(V1_MD.read_text())
    commit, dirty = git_state()
    total = sum(i.size for i in items) + len(TEMPLATE.read_bytes()) + 4_000    # + README, SHA256SUMS
    values = {
        "MODEL_NAME": args.repo_name, "REPO_NAME": args.repo_name, "REPO_ID": f"{args.hf_user}/{args.repo_name}",
        "GITHUB_URL": args.github_url, "AUTHOR": args.author, "LICENSE_ID": args.license,
        "LICENSE_TEXT": ("the Apache License 2.0 (`LICENSE`)" if args.license == "apache-2.0" else
                         f"`{args.license}` (`LICENSE`); the Core ML package's encoder weights stay under the Apache "
                         "License 2.0 (`LICENSE-APACHE-2.0`)"),
        "MIN_OS": SPEC_TARGET.get(spec["spec_version"], f"Core ML spec {spec['spec_version']}"),
        "MODEL_REVISION": MODEL_REVISION, "MODEL_REVISION_SHORT": MODEL_REVISION[:7],
        "TOTAL_SIZE": human(total) if have_pkg else f"{MISSING} (the package is not built yet)",
        "FILES_TABLE": files_table(items, args.license),
        "PLACEMENT": placement_line(results["t6"]),
        "T9_TABLE": t9_table(results["t9"]), "T10_TABLE": t10_table(results["t10"]), "T11_TABLE": t11_table(results["t11"]),
        "LM_NGRAMS": counts, "LM_BUILD": lm_cmd,
    }
    readme = render(TEMPLATE.read_text(), values)
    readme += f"\n<!-- generated by release/make_hf_release.py from code commit {commit}{' (dirty)' if dirty else ''} -->\n"
    items.append(Item("README.md", text=readme))
    for it in items:
        if it.text is not None:
            check_text(it.dest, it.text)
    return items, problems, spec, results, have_pkg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-name", default="gemma-stt-v1-coreml")
    ap.add_argument("--hf-user", default="YOUR_HF_USER", help="Hugging Face user or org (README usage)")
    ap.add_argument("--author", default="the repository author", help="name used in NOTICE / README")
    ap.add_argument("--github-url", default="https://github.com/YOUR_GITHUB_USER/emb_gemma2")
    ap.add_argument("--license", default="apache-2.0", help="Hugging Face license id of this release (model card, config.json)")
    ap.add_argument("--license-file", type=Path, default=APACHE, help="becomes LICENSE (default: Apache License 2.0 text)")
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

    items, problems, spec, results, have_pkg = build_plan(args)
    if not args.no_hash:
        for it in items:
            if it.sha256 is None:
                if it.src is not None:
                    print(f"  hashing {it.dest} ({human(it.size)})", file=sys.stderr, flush=True)
                it.hash()
        for it in items:
            want = {SPM.name: V1_SHA256[SPM], LM.name: V1_SHA256[LM]}.get(it.dest)
            if want and it.sha256 != want:
                refuse(f"{it.src}: sha256 {it.sha256} != V1 {want}")
        sums = "".join(f"{it.sha256}  {it.dest}\n" for it in sorted(items, key=lambda i: i.dest))
        items.append(Item("SHA256SUMS", text=sums))
        items[-1].hash()

    print(f"{'WRITE' if args.write else 'DRY RUN (nothing written; --write to copy)'}: {out}")
    print(f"package {PACKAGE.name}: " + (f"{PACKAGE}".replace(str(Path.home()), "~") if have_pkg else
                                          f"{MISSING} ({str(PACKAGE).replace(str(Path.home()), '~')})"))
    if spec["provisional"]:
        print(f"  config.json is a preview: I/O read from the source build {spec['source_build']}, minimum OS "
              f"{SPEC_TARGET[9]} assumed (multifunction = spec 9)")
    print(f"{'size':>10}  {'sha256':<64}  path  <-  source")
    if not have_pkg:
        print(f"{MISSING:>10}  {'-' * 64}  {PACKAGE.name}/  <-  coreml/export_release.py")
    for it in sorted(items, key=lambda i: i.dest):
        src = "generated" if it.text is not None else str(it.src).replace(str(Path.home()), "~")
        print(f"{human(it.size):>10}  {it.sha256 or '-' * 64:<64}  {it.dest}  <-  {src}")
        for w in it.warnings:
            print(f"{'':>10}  WARNING: {it.dest} {w}")
    groups = {}
    for it in items:
        top = it.dest.split("/")[0]
        groups[top] = groups.get(top, 0) + it.size
    print("totals: " + ", ".join(f"{k} {human(v)}" for k, v in groups.items())
          + f"; all {human(sum(i.size for i in items))} in {len(items)} files"
          + ("" if have_pkg else f" (without {PACKAGE.name}: {MISSING})"))
    print("results (release rows): " + ", ".join(
        f"{k.upper()} {'run ' + r['run'] if r else MISSING}" for k, r in results.items() if k != "t6")
        + f", T6 {sum(r is not None for r in results['t6'].values())}/{len(WINDOWS)} windows")
    if not args.no_hash:
        print("tokenizer and LM match V1.md's sha256; no Discord webhook URL found.")
    placeholders = [k for k, v in (("--hf-user", args.hf_user == "YOUR_HF_USER"),
                                   ("--github-url", "YOUR_GITHUB_USER" in args.github_url),
                                   ("--author", args.author == "the repository author")) if v]
    if placeholders:
        print("note: README / NOTICE use placeholders; set " + ", ".join(placeholders))
    for p in problems:
        print(f"BLOCKER (--write refuses): {p}")
    if args.preview:
        for it in items:
            if it.text is not None and it.dest != "SHA256SUMS":
                print(f"\n===== {it.dest} =====\n{it.text}")
    if not args.write:
        return

    if problems:
        refuse(f"{len(problems)} prerequisite(s) missing or failed (listed above)")
    if placeholders:
        refuse("placeholders left: set " + ", ".join(placeholders))
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
