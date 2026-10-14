# Publishing V1: GitHub (code) and Hugging Face (model)

Nothing here has been run against GitHub or Hugging Face. Each step is a command for you to run
when you are ready. The check outputs quoted below were produced on 2026-10-14 at commit `70fe4ec`
(the working tree also had uncommitted work in progress, listed in A.1).

**Decisions that are yours:**
- **Repo names:** GitHub `emb_gemma2` or another name; Hugging Face `<user>/gemma-stt-v1-coreml`
  (`--repo-name`).
- **License of the release:** set with `make_hf_release.py --license` (default `apache-2.0`).
  - The base model, [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) at
    `914f7f8`, is Apache-2.0. Its card links https://ai.google.dev/gemma/docs/gemma_4_license and says
    deployments must adhere to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
  - With the default, `LICENSE` is the Apache License 2.0 text (`release/LICENSE-2.0.txt`, the same
    bytes as apache.org's `LICENSE-2.0.txt`).
  - With another id, pass `--license-file`. The release then also carries `LICENSE-APACHE-2.0` for the
    encoder weights.
- **Code license:** the GitHub repo has no `LICENSE` yet. `release/LICENSE-2.0.txt` can be copied to
  `LICENSE` if you choose Apache-2.0 for the code too.
- **Git identity:** whether to rewrite the 9 local commits.
  - They are authored `gemma-stt <gemma-stt@localhost>`.
  - The annotated tag `v1` has tagger `Harshit Brahmbhatt <harshit@Harshits-MacBook-Pro.local>`, which includes a local hostname.
  - See A.5.
- **Tokenizer path:** the released tokenizer embeds a local path (see A.3).

---

## A. GitHub (code repo `~/emb_gemma2`)

### A.1 What is in the repo

| Path | Contents |
|---|---|
| `README.md`, `setup.sh`, `download_librispeech*.{sh,py}` | setup, LibriSpeech download (OpenSLR layout) |
| `cache_feats.py`, `featstore.py` | mel + Gemma audio tower features, int8 shard store |
| `probe.py`, `diag.py`, `head.py`, `eval_head.py`, `train_tokenizer.py`, `watch.py`, `test_*.py` | layer probe, CTC head training / eval, tokenizer, log watcher, tests |
| `decode/` | decoding study: `dec_*.py` decoders, `run_decode.py`, `sweep.py`, `tune.py`, `report.py`, `streaming.py`, `stream_study.py`, `build_subword_lm.py`, pinned `requirements-*.txt` |
| `coreml/` | Core ML export (`common.py`, `wrapper.py`, `export.py`, `export_release.py` = the release package, `palettize.py`), runtime (`cmlrun.py`, `transcribe.py`, streaming / live runners), evaluation (`eval_fast.py`, `placement.py`, `bench.py`, `summary.py`), `tests/` (T1–T11) |
| `embed/`, `demo/` | one-pass transcript + EmbeddingGemma 2 embedding, demo CLI |
| `Makefile` | `make test-fast / test-full / test-release / bench / summary` |
| `release/` | this file, `make_hf_release.py`, `hf_model_card_template.md`, `LICENSE-2.0.txt` (Apache License 2.0 text) |

At the time of writing (the main session commits them): untracked `coreml/LIVE.md`, `coreml/export_release.py`, `coreml/live.py`, `coreml/live_latency.py`, `coreml/livecore.py`, `coreml/palettize.py`, `coreml/stream_latency.py`, `coreml/stream_mel.py`, `coreml/stream_runner.py`, `coreml/stream_step.py`, `coreml/tests/test_stream.py`, `release/LICENSE-2.0.txt`; modified `coreml/bench.py`, `coreml/cmlrun.py`, `coreml/placement.py`, `coreml/summary.py`, `coreml/tests/conftest.py`, `coreml/tests/test_fast.py`, `coreml/tests/test_full.py`, `release/hf_model_card_template.md`, `release/make_hf_release.py`.
Tracked files: 64, 592K in total, so the repo itself is tiny.

**Excluded by `.gitignore`:**

```
__pycache__/
*.pyc
.DS_Store
.pytest_cache/
*.mlpackage/
```

**Not in the repo at all:** everything under the data root `~/gemma_stt`. That covers LibriSpeech,
feature caches, checkpoints, the tokenizer, the LMs, Core ML packages, logs and reports. It also
covers `runs/.discord_webhook`, the Discord webhook secret, which must never be copied anywhere.

**Suggested extra ignore rules.** They guard against copying artifacts in later; this is a proposal,
not applied:

```gitignore
# large artifacts and local state: these live under $GEMMA_STT_HOME, never in git
*.pt
*.bin
*.arpa
*.npy
*.mlmodelc/
*.partial/
.venv*/
probe_results/
.discord_webhook
```

### A.2 Secrets check (run before every push)

```bash
cd ~/emb_gemma2
# webhook URLs, HF / GitHub / OpenAI / AWS tokens, private keys: working tree incl. untracked files
git grep -n -I --untracked -E 'discord(app)?\.com/api/webhooks|hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY' -- . ':!release/PUBLISHING.md'
# the same over the whole history
git log --all -p | grep -c -E 'discord(app)?\.com/api/webhooks|hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}'
# literal absolute home paths
git grep -n -I --untracked '/Users/' -- . ':!release/PUBLISHING.md'
```

(This file is excluded because it quotes the patterns.)

Current output:
- Working tree: nothing (exit 1).
- History: `0`.
- `/Users/`: nothing.

- **Webhook:** the webhook is read only by `~/gemma_stt/runs/after_training.sh`, which is outside
  the repo. Do not copy that script (or any `runs/*.sh`) into the repo without removing the notifier.
- **Release folder:** `release/make_hf_release.py` refuses the webhook file and any file that
  contains a Discord webhook URL.

### A.3 Hardcoded paths

Command:

```bash
git grep -n -I --untracked -E '/Users/|gemma_stt|emb_gemma2|\.venv-(coreml|gemma-stt)|Path\.home\(\)|expanduser\("~' -- . ':!release'
```

**Code defaults that point into the data root** (these break on another machine or another root;
the patch below fixes them):

| file:line | code |
|---|---|
| `cache_feats.py:302` | `ap.add_argument("--data-root", default="~/gemma_stt/data")` |
| `coreml/common.py:18` | `HOME = Path.home()` |
| `coreml/common.py:19` | `REPO = HOME / "emb_gemma2"` |
| `coreml/common.py:25` | `HEAD = HOME / "gemma_stt/runs/full_160ms/head_F0-0-0-4c.pt"` |
| `coreml/common.py:26` | `SPM = HOME / "gemma_stt/feats/full/spm_unigram256.model"` |
| `coreml/common.py:27` | `LM = HOME / "gemma_stt/decode/lm/sp256_6gram.bin"` |
| `coreml/common.py:28` | `DATA = HOME / "gemma_stt/data/LibriSpeech"` |
| `coreml/common.py:29` | `LOGPROBS = HOME / "gemma_stt/decode/logprobs"` |
| `coreml/common.py:30` | `OUT = HOME / "gemma_stt/coreml"` |
| `coreml/tests/test_release.py:15` | `PHONON2 = Path.home() / "gemma_stt/phonon2/results"` |
| `decode/build_subword_lm.py:24` | `HOME = Path.home()` |
| `decode/build_subword_lm.py:25` | `KENLM_BIN = HOME / "gemma_stt/tools/kenlm/build/bin"` |
| `decode/build_subword_lm.py:75` | `ap.add_argument("--corpus", type=Path, default=HOME / "gemma_stt/decode/lm/src/librispeech-lm-norm.txt.gz")` |
| `decode/build_subword_lm.py:76` | `ap.add_argument("--spm", type=Path, default=HOME / "gemma_stt/feats/full/spm_unigram256.model")` |
| `decode/build_subword_lm.py:77` | `ap.add_argument("--out", type=Path, default=HOME / "gemma_stt/decode/lm/sp256_6gram.bin")` |
| `decode/build_subword_lm.py:78` | `ap.add_argument("--work", type=Path, default=HOME / "gemma_stt/decode/lm/sp256_work")` |
| `decode/dump_logprobs.py:31` | `HOME = Path.home()` |
| `decode/dump_logprobs.py:98` | `ap.add_argument("--head", default=str(HOME / "gemma_stt/runs/full_160ms/head_F0-0-0-4c.pt"))` |
| `decode/dump_logprobs.py:99` | `ap.add_argument("--feats", default=str(HOME / "gemma_stt/feats/full"), help="dir with one cache per split")` |
| `decode/dump_logprobs.py:101` | `ap.add_argument("--out", default=str(HOME / "gemma_stt/decode/logprobs"))` |
| `decode/report.py:19` | `RESULTS = Path.home() / "gemma_stt/decode/results"` |
| `decode/run_decode.py:38` | `HOME = Path.home()` |
| `decode/run_decode.py:39` | `LOGPROBS = HOME / "gemma_stt/decode/logprobs"` |
| `decode/run_decode.py:40` | `RESULTS = HOME / "gemma_stt/decode/results"` |
| `decode/run_decode.py:41` | `LM_DIR = HOME / "gemma_stt/decode/lm"` |
| `decode/stream_study.py:40` | `HOME = Path.home()` |
| `decode/stream_study.py:41` | `LOGPROBS = HOME / "gemma_stt/decode/logprobs"` |
| `decode/stream_study.py:42` | `SP6 = HOME / "gemma_stt/decode/lm/sp256_6gram.bin"` |
| `decode/stream_study.py:43` | `RESULTS = HOME / "gemma_stt/decode/results/streaming"` |
| `decode/stream_study.py:44` | `V1_HYPS = HOME / "gemma_stt/decode/results/sweep_dev-clean_cbd_sp6_beam32_units-piece/hyps.jsonl"` |
| `decode/test_dec_cbd.py:18` | `SPM = Path(os.environ.get("SPM_MODEL", "~/gemma_stt/feats/full/spm_unigram256.model")).expanduser()` |
| `decode/test_dec_pyctc.py:18` | `SPM = os.path.expanduser("~/gemma_stt/feats/full/spm_unigram256.model")` |
| `decode/test_harness.py:27` | `SPM = Path.home() / "gemma_stt/feats/full/spm_unigram256.model"` |
| `decode/tune.py:25` | `RESULTS = Path.home() / "gemma_stt/decode/results"` |
| `decode/tune.py:26` | `LOGPROBS = Path.home() / "gemma_stt/decode/logprobs"` |
| `test_cache_plan.py:13` | `TRAIN_CLEAN_100 = Path("~/gemma_stt/data/LibriSpeech/train-clean-100").expanduser()` |
| `watch.py:191` | `ap.add_argument("log", nargs="?", default="~/gemma_stt/logs",` |
| `watch.py:192` | `help="log file, or a directory to follow its newest *.log (default ~/gemma_stt/logs)")` |

**Kept on purpose** (not the data root):
- `Makefile:8`: `PY ?= $(HOME)/.venv-coreml/bin/python`, overridable with `make PY=…`.
- `setup.sh:5`: `VENV="${VENV:-$HOME/.venv-gemma-stt}"`, already overridable.
- `coreml/bench.py:30`: the macOS Core ML compile cache.
- `decode/tune.py:71-73`: the per-decoder venv interpreters.

**Documentation and usage strings** (docstrings, comments, `README.md`): they show `~/gemma_stt/...`
and `~/.venv-*` as examples. They are harmless; you could rewrite them to `$GEMMA_STT_HOME/...`.

<details><summary>Full grep output (170 lines)</summary>

```
Makefile:7:#   make summary        rebuild ~/gemma_stt/coreml/summary.md from results.jsonl
Makefile:8:PY ?= $(HOME)/.venv-coreml/bin/python
README.md:8:| `setup.sh` | Python 3.12 venv at `~/.venv-gemma-stt` (separate from vllm-metal) |
README.md:24:source ~/.venv-gemma-stt/bin/activate
README.md:33:bash download_librispeech.sh ~/gemma_stt/data train-clean-100 dev-clean
README.md:35:python download_librispeech_hf.py ~/gemma_stt/data train-clean-100 dev-clean
README.md:46:    --taps L6 L9 L11 --out ~/gemma_stt/feats/probe/train --verify
README.md:48:    --taps L6 L9 L11 --out ~/gemma_stt/feats/probe/dev
README.md:64:python probe.py --feats ~/gemma_stt/feats/probe --runs L6 L9 L11 L11:int8
README.md:80:caffeinate -dims python diag.py --feats ~/gemma_stt/feats/probe              # uses probe_results/head_L11.pt
README.md:81:caffeinate -dims python head.py --feats ~/gemma_stt/feats/probe --lookahead 0 4
README.md:100:bash download_librispeech.sh ~/gemma_stt/data train-clean-360 train-other-500 dev-other test-clean test-other
README.md:102:    --taps L11 --int8 --out ~/gemma_stt/feats/full/train
README.md:104:  caffeinate -dims python cache_feats.py --splits $s --taps L11 --int8 --out ~/gemma_stt/feats/full/$s
README.md:106:ln -sfn dev-clean ~/gemma_stt/feats/full/dev        # head.py evaluates on <feats>/dev
README.md:117:python train_tokenizer.py --feats ~/gemma_stt/feats/full --compare ~/gemma_stt/feats/probe/spm_unigram256.model
README.md:118:PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25 caffeinate -dims python head.py --feats ~/gemma_stt/feats/full --stream \
README.md:121:    --out ~/gemma_stt/runs/full_160ms --resume 2>&1 | tee -a ~/gemma_stt/logs/full_160ms.log
README.md:122:python eval_head.py --head ~/gemma_stt/runs/full_160ms/head_F0-0-0-4c.pt --feats ~/gemma_stt/feats/full
cache_feats.py:14:      --taps L6 L9 L11 --out ~/gemma_stt/feats/probe/train --verify
cache_feats.py:16:      --taps L6 L9 L11 --out ~/gemma_stt/feats/probe/dev
cache_feats.py:21:      --taps L11 --int8 --out ~/gemma_stt/feats/full/train
cache_feats.py:302:    ap.add_argument("--data-root", default="~/gemma_stt/data")
coreml/LIVE.md:10:Everything is in `~/.venv-coreml`. The app needs one extra package, `sounddevice` (its macOS wheel
coreml/LIVE.md:12:already built into `~/.venv-coreml` (kenlm 0.2.0, from the pinned source, recipe in the versions file).
coreml/LIVE.md:13:Versions added to the env are listed in `~/gemma_stt/coreml/VERSIONS_ADDED.md`.
coreml/LIVE.md:16:~/.venv-coreml/bin/pip install sounddevice==0.5.6
coreml/LIVE.md:17:~/.venv-coreml/bin/python -c "import sounddevice, kenlm; print(sounddevice.__version__)"
coreml/LIVE.md:24:From the repo root (`~/emb_gemma2`):
coreml/LIVE.md:28:~/.venv-coreml/bin/python coreml/live.py
coreml/LIVE.md:31:~/.venv-coreml/bin/python coreml/live.py --mode A --device "MacBook Pro Microphone"
coreml/LIVE.md:32:~/.venv-coreml/bin/python coreml/live.py --list-devices
coreml/LIVE.md:35:~/.venv-coreml/bin/python coreml/live.py --mode C --replay ~/gemma_stt/coreml/live_sessions/dev-clean_6345-93306_0021-0025.flac
coreml/LIVE.md:38:~/.venv-coreml/bin/python coreml/live.py --replay FILE --fast --exit-at-end --session ~/gemma_stt/coreml/live_sessions/try1
coreml/LIVE.md:43:`~/gemma_stt/coreml/compiled/` (about a minute); later starts load the compiled models. In replay
coreml/LIVE.md:72:| `c` | save the transcript to `~/gemma_stt/coreml/live_sessions/transcript-YYYYMMDD-HHMMSS.txt` (or the `--session` folder) and copy it to the clipboard |
coreml/LIVE.md:153:| Mode C ends with `engine stopped: ModuleNotFoundError: No module named 'kenlm'` | The LM fusion decoder needs `kenlm` in `~/.venv-coreml` (see the build recipe in `VERSIONS_ADDED.md`); mode A works without it. |
coreml/LIVE.md:179:  script -q ~/gemma_stt/coreml/live_sessions/take1.ansi \
coreml/LIVE.md:180:      ~/.venv-coreml/bin/python coreml/live.py --mode C --replay FILE --exit-at-end \
coreml/LIVE.md:181:      --session ~/gemma_stt/coreml/live_sessions/take1
coreml/LIVE.md:189:Sample sessions are in `~/gemma_stt/coreml/live_sessions/`:
coreml/bench.py:4:  ~/.venv-coreml/bin/python coreml/bench.py [--configs fp16:CPU_AND_NE fp16:CPU_AND_GPU fp16:CPU_ONLY fp32:CPU_AND_GPU]
coreml/bench.py:30:CACHE = Path.home() / "Library/Caches/org.python.python/com.apple.e5rt.e5bundlecache"
coreml/bisect_ane.py:5:  ~/.venv-coreml/bin/python coreml/bisect_ane.py [--parts layer sscp head int8] [--variant faithful]
coreml/check_wrapper.py:4:  ~/.venv-coreml/bin/python coreml/check_wrapper.py [--variant faithful|fp16safe] [--limit N]
coreml/common.py:3:tests; works in ~/.venv-gemma-stt (reference, V1's torch) and ~/.venv-coreml (conversion, tests).
coreml/common.py:18:HOME = Path.home()
coreml/common.py:19:REPO = HOME / "emb_gemma2"
coreml/common.py:25:HEAD = HOME / "gemma_stt/runs/full_160ms/head_F0-0-0-4c.pt"
coreml/common.py:26:SPM = HOME / "gemma_stt/feats/full/spm_unigram256.model"
coreml/common.py:27:LM = HOME / "gemma_stt/decode/lm/sp256_6gram.bin"
coreml/common.py:28:DATA = HOME / "gemma_stt/data/LibriSpeech"
coreml/common.py:29:LOGPROBS = HOME / "gemma_stt/decode/logprobs"
coreml/common.py:30:OUT = HOME / "gemma_stt/coreml"
coreml/eval_fast.py:4:  ~/.venv-coreml/bin/python coreml/eval_fast.py --variant faithful --precision fp16 [--compute-units CPU_AND_NE]
coreml/export.py:4:  ~/.venv-coreml/bin/python coreml/export.py --windows 5s 10s 20s 35s --variant faithful --precision fp16
coreml/export_release.py:8:  ~/.venv-coreml/bin/python coreml/export_release.py [--source fp16safe/fp16] [--verify-utts 60]
coreml/fp16_layers.py:5:  ~/.venv-coreml/bin/python coreml/fp16_layers.py --variant faithful [--n 12]
coreml/live.py:5:  ~/.venv-coreml/bin/python coreml/live.py [--mode A|C] [--device NAME|INDEX]
coreml/live.py:6:  ~/.venv-coreml/bin/python coreml/live.py --mode C --replay FILE [--fast] [--exit-at-end]
coreml/live.py:34:SESSIONS = Path.home() / "gemma_stt/coreml/live_sessions"
coreml/live.py:716:    home = str(Path.home())
coreml/live_latency.py:6:  ~/.venv-coreml/bin/python coreml/live_latency.py [--limit N] [--shard I --nshards N]   # simulate
coreml/live_latency.py:7:  ~/.venv-coreml/bin/python coreml/live_latency.py --calibrate [--quick]                 # idle machine
coreml/live_latency.py:8:  ~/.venv-coreml/bin/python coreml/live_latency.py --report                             # tables
coreml/live_latency.py:80:V1_HYPS = {s: Path.home() / f"gemma_stt/decode/results/{p}_{s}_cbd_sp6_beam32_units-piece/hyps.jsonl"
coreml/make_reference.py:2:"""Fixed test lists and the PyTorch fp32 CPU reference outputs for FAST (run in ~/.venv-gemma-stt).
coreml/make_reference.py:4:  ~/.venv-gemma-stt/bin/python coreml/make_reference.py
coreml/palettize.py:4:  ~/.venv-coreml/bin/python coreml/palettize.py --builds g8 g16 pt [--windows 5s 10s 20s 35s] [--device mps]
coreml/placement.py:4:  ~/.venv-coreml/bin/python coreml/placement.py --variant faithful --precision fp16 [--windows 5s ...]
coreml/stream_latency.py:4:  ~/.venv-coreml/bin/python coreml/stream_latency.py [--split dev-clean] [--limit N]
coreml/stream_latency.py:19:(~/gemma_stt/decode/logprobs/<split>.npy, V1's PyTorch log-probs). Step j returns frames [4j-4, 4j),
coreml/stream_latency.py:27:Outputs: ~/gemma_stt/coreml/stream/latency_<split>_<model>_<seg>_<timing>.{json,md,utts.jsonl}.
coreml/stream_latency.py:45:V1_HYPS = {"dev-clean": Path.home() / "gemma_stt/decode/results/sweep_dev-clean_cbd_sp6_beam32_units-piece/hyps.jsonl",
coreml/stream_latency.py:46:           "dev-other": Path.home() / "gemma_stt/decode/results/apply_dev-other_cbd_sp6_beam32_units-piece/hyps.jsonl"}
coreml/stream_step.py:5:Plan: ~/gemma_stt/coreml/STREAMING_PLAN.md (sections 2, 3, 5). Call j (0-based, C = 4 L11 frames per
coreml/stream_step.py:19:Packages: ~/gemma_stt/coreml/models/stream/<variant>_<precision>/v1_step_<mech>.mlpackage (+ a
coreml/stream_step.py:22:  ~/.venv-coreml/bin/python coreml/stream_step.py spec                    # state table from the config
coreml/stream_step.py:23:  ~/.venv-coreml/bin/python coreml/stream_step.py export --mech state io  # convert + compile
coreml/stream_step.py:24:  ~/.venv-coreml/bin/python coreml/stream_step.py check-torch [--n 200]   # fp32 StepTorch vs offline
coreml/stream_step.py:25:  ~/.venv-coreml/bin/python coreml/stream_step.py placement --mech state io
coreml/stream_step.py:26:  ~/.venv-coreml/bin/python coreml/stream_step.py bench --mech state      # idle machine only (SLOT)
coreml/summary.py:2:"""Summarize ~/gemma_stt/coreml/results.jsonl into ~/gemma_stt/coreml/summary.md: one section per
coreml/summary.py:5:  ~/.venv-coreml/bin/python coreml/summary.py
coreml/tests/conftest.py:5:~/gemma_stt/coreml/results.jsonl with the run id, versions, model weight hashes and git commit.
coreml/tests/test_release.py:15:PHONON2 = Path.home() / "gemma_stt/phonon2/results"
coreml/transcribe.py:5:  ~/.venv-coreml/bin/python coreml/transcribe.py AUDIO [AUDIO ...] [--compute-units CPU_AND_NE] [--json]
decode/build_subword_lm.py:10:Run with ~/.venv-gemma-stt/bin/python (needs sentencepiece only).
decode/build_subword_lm.py:24:HOME = Path.home()
decode/build_subword_lm.py:25:KENLM_BIN = HOME / "gemma_stt/tools/kenlm/build/bin"
decode/build_subword_lm.py:75:    ap.add_argument("--corpus", type=Path, default=HOME / "gemma_stt/decode/lm/src/librispeech-lm-norm.txt.gz")
decode/build_subword_lm.py:76:    ap.add_argument("--spm", type=Path, default=HOME / "gemma_stt/feats/full/spm_unigram256.model")
decode/build_subword_lm.py:77:    ap.add_argument("--out", type=Path, default=HOME / "gemma_stt/decode/lm/sp256_6gram.bin")
decode/build_subword_lm.py:78:    ap.add_argument("--work", type=Path, default=HOME / "gemma_stt/decode/lm/sp256_work")
decode/dump_logprobs.py:31:HOME = Path.home()
decode/dump_logprobs.py:98:    ap.add_argument("--head", default=str(HOME / "gemma_stt/runs/full_160ms/head_F0-0-0-4c.pt"))
decode/dump_logprobs.py:99:    ap.add_argument("--feats", default=str(HOME / "gemma_stt/feats/full"), help="dir with one cache per split")
decode/dump_logprobs.py:101:    ap.add_argument("--out", default=str(HOME / "gemma_stt/decode/logprobs"))
decode/report.py:4:  python decode/report.py --split dev-clean                # sweep_* runs in ~/gemma_stt/decode/results
decode/report.py:5:  python decode/report.py --split test-clean --prefix sweep_ --out ~/gemma_stt/decode/results/report_test-clean
decode/report.py:19:RESULTS = Path.home() / "gemma_stt/decode/results"
decode/requirements-ctc-beam-decoder.txt:3:#   ~/.venv-gemma-stt/bin/pip install --no-deps --require-hashes -r decode/requirements-ctc-beam-decoder.txt
decode/requirements-pyctcdecode.txt:5:#   ~/.venv-pyctcdecode/bin/pip install --require-hashes -r ~/emb_gemma2/decode/requirements-pyctcdecode.txt
decode/requirements-pyctcdecode.txt:12:#   extracted at ~/gemma_stt/tools/src/kenlm-4cb443e60b7bf2c0ddf3c745378f76cb59e254e5;
decode/requirements-pyctcdecode.txt:14:#   cd ~/gemma_stt/tools/src/kenlm-4cb443e60b7bf2c0ddf3c745378f76cb59e254e5 && \
decode/run_decode.py:38:HOME = Path.home()
decode/run_decode.py:39:LOGPROBS = HOME / "gemma_stt/decode/logprobs"
decode/run_decode.py:40:RESULTS = HOME / "gemma_stt/decode/results"
decode/run_decode.py:41:LM_DIR = HOME / "gemma_stt/decode/lm"
decode/stream_study.py:40:HOME = Path.home()
decode/stream_study.py:41:LOGPROBS = HOME / "gemma_stt/decode/logprobs"
decode/stream_study.py:42:SP6 = HOME / "gemma_stt/decode/lm/sp256_6gram.bin"
decode/stream_study.py:43:RESULTS = HOME / "gemma_stt/decode/results/streaming"
decode/stream_study.py:44:V1_HYPS = HOME / "gemma_stt/decode/results/sweep_dev-clean_cbd_sp6_beam32_units-piece/hyps.jsonl"
decode/test_dec_cbd.py:3:    cd ~/emb_gemma2 && OMP_NUM_THREADS=1 ~/.venv-gemma-stt/bin/python -m unittest decode/test_dec_cbd.py -v
decode/test_dec_cbd.py:18:SPM = Path(os.environ.get("SPM_MODEL", "~/gemma_stt/feats/full/spm_unigram256.model")).expanduser()
decode/test_dec_pyctc.py:3:Run: ~/.venv-pyctcdecode/bin/python ~/emb_gemma2/decode/test_dec_pyctc.py
decode/test_dec_pyctc.py:18:SPM = os.path.expanduser("~/gemma_stt/feats/full/spm_unigram256.model")
decode/test_harness.py:3:  cd ~/emb_gemma2 && python -m unittest decode.test_harness -v
decode/test_harness.py:27:SPM = Path.home() / "gemma_stt/feats/full/spm_unigram256.model"
decode/tune.py:25:RESULTS = Path.home() / "gemma_stt/decode/results"
decode/tune.py:26:LOGPROBS = Path.home() / "gemma_stt/decode/logprobs"
decode/tune.py:71:PYTHONS = {"cbd": Path.home() / ".venv-gemma-stt/bin/python",
decode/tune.py:72:           "pyctc": Path.home() / ".venv-pyctcdecode/bin/python",
decode/tune.py:73:           "subword": Path.home() / ".venv-pyctcdecode/bin/python"}
demo/README.md:22:P=~/.venv-gemma-stt/bin/python
demo/README.md:85:Folder: `~/gemma_stt/embed/demo/folder/<spk>/<chap>/<id>.flac` holds **symlinks** into
demo/README.md:86:`~/gemma_stt/data/LibriSpeech/dev-clean` (no copies). There is one utterance from each of 40 chapters:
demo/README.md:91:The ids and reference sentences are in `~/gemma_stt/embed/demo/folder_refs.json`. Outputs are in
demo/README.md:92:`~/gemma_stt/embed/demo/` (`transcribe_out/`, `index.npz`, `search_examples.txt`). Logs are in
demo/README.md:93:`~/gemma_stt/logs/EmbedDemo_*.log`.
demo/README.md:113:3 file(s) -> ~/gemma_stt/embed/demo/transcribe_out/<name>.{txt,npy,json}
demo/README.md:119:$ $P demo/embed_and_transcribe.py index ~/gemma_stt/embed/demo/folder --db ~/gemma_stt/embed/demo/index.npz
demo/README.md:120:~/gemma_stt/embed/demo/folder: 40 audio file(s): 40 to embed, 0 unchanged, 0 removed from index      (stderr)
demo/README.md:124:index   ~/gemma_stt/embed/demo/index.npz
demo/README.md:125:folder  ~/gemma_stt/embed/demo/folder
demo/README.md:152:      "a woman pricked her finger on a spindle" --db ~/gemma_stt/embed/demo/index.npz -k 5
demo/README.md:153:index: ~/gemma_stt/embed/demo/index.npz  (40 files, 768-d, query prefix 'task: search result | query: ')
demo/README.md:196:- Sanity check (`~/gemma_stt/logs/EmbedDemo_refquery_check.log`): each file's LibriSpeech reference
demo/embed_and_transcribe.py:11:Run with V1's environment:  ~/.venv-gemma-stt/bin/python demo/embed_and_transcribe.py ...
demo/embed_and_transcribe.py:174:    home = str(Path.home())
diag.py:10:  python diag.py --feats ~/gemma_stt/feats/probe
download_librispeech.sh:5:# Usage:  bash download_librispeech.sh ~/gemma_stt/data train-clean-100 dev-clean
download_librispeech_hf.py:11:  python download_librispeech_hf.py ~/gemma_stt/data train-other-500 dev-other test-clean test-other
embed/check_reference.py:18:Writes ~/gemma_stt/embed/check_ids.json, check_reference.json, check_reference_clips.jsonl.
embed/check_reference.py:35:OUT = Path.home() / "gemma_stt/embed"
embed/check_reference.py:36:V1_HYPS = Path.home() / "gemma_stt/decode/results/sweep_dev-clean_cbd_sp6_beam32_units-piece/hyps.jsonl"
embed/cost.py:4:  ~/.venv-gemma-stt/bin/python embed/cost.py [--n 100] [--warmup 5] [--out ~/gemma_stt/embed/cost.json]
embed/cost.py:94:    ap.add_argument("--out", default=str(Path.home() / "gemma_stt/embed/cost.json"))
embed/onepass.py:4:  ~/.venv-gemma-stt/bin/python embed/onepass.py AUDIO [AUDIO ...]        # prints text + embedding head
embed/onepass.py:13:    .encode(audio)` returns (see ~/gemma_stt/embed/REPORT_core.md for the code references).
embed/retrieval.py:3:python embed/retrieval.py  -> ~/gemma_stt/embed/retrieval/ + ~/gemma_stt/embed/REPORT_retrieval.md
embed/retrieval.py:12:G = Path.home() / "gemma_stt"
embed/retrieval.py:132:          "Reproduce: `~/.venv-gemma-stt/bin/python embed/retrieval.py`."]
eval_head.py:6:  python eval_head.py --head ~/gemma_stt/runs/full_160ms/head_F0-0-0-4c.pt --feats ~/gemma_stt/feats/full
head.py:18:  python head.py --feats ~/gemma_stt/feats/probe --lookahead 0 4
head.py:19:  python head.py --feats ~/gemma_stt/feats/probe --lookahead 0 --epochs 20 --int8 --out DIR
head.py:20:  python head.py --feats ~/gemma_stt/feats/probe --block-future 0 0 0 4 --causal-conv --epochs 30
head.py:21:  python head.py --feats ~/gemma_stt/feats/full --stream     # training cache larger than RAM
probe.py:10:  python probe.py --feats ~/gemma_stt/feats/probe --runs L6 L9 L11 L11:int8
probe.py:11:  python probe.py --feats ~/gemma_stt/feats/probe --runs L11:stack2 L11:stack4 L11:stack6
probe.py:12:  python probe.py --feats ~/gemma_stt/feats/probe --runs L11:cat
setup.sh:5:VENV="${VENV:-$HOME/.venv-gemma-stt}"
test_cache_plan.py:13:TRAIN_CLEAN_100 = Path("~/gemma_stt/data/LibriSpeech/train-clean-100").expanduser()
train_tokenizer.py:8:  python train_tokenizer.py --feats ~/gemma_stt/feats/full --compare ~/gemma_stt/feats/probe/spm_unigram256.model
watch.py:5:  python3 watch.py                                   # follows the newest log in ~/gemma_stt/logs
watch.py:6:  python3 watch.py ~/gemma_stt/bench                 # follows the newest log in that directory
watch.py:7:  python3 watch.py ~/gemma_stt/logs/head_R4_fast.log --interval 5
watch.py:191:    ap.add_argument("log", nargs="?", default="~/gemma_stt/logs",
watch.py:192:                    help="log file, or a directory to follow its newest *.log (default ~/gemma_stt/logs)")
```

</details>

**Data-root defaults in files the patch does not cover** (newer work: the embedding one-pass, streaming and live runners). They should read `GEMMA_STT_HOME` the same way; re-run the grep once they are final:

| file:line | code |
|---|---|
| `coreml/live.py:34` | `SESSIONS = Path.home() / "gemma_stt/coreml/live_sessions"` |
| `coreml/live_latency.py:80` | `V1_HYPS = {s: Path.home() / f"gemma_stt/decode/results/{p}_{s}_cbd_sp6_beam32_units-piece/hyps.jsonl"` |
| `coreml/stream_latency.py:45` | `V1_HYPS = {"dev-clean": Path.home() / "gemma_stt/decode/results/sweep_dev-clean_cbd_sp6_beam32_units-piece/hyps.jsonl",` |
| `coreml/stream_latency.py:46` | `"dev-other": Path.home() / "gemma_stt/decode/results/apply_dev-other_cbd_sp6_beam32_units-piece/hyps.jsonl"}` |
| `embed/check_reference.py:35` | `OUT = Path.home() / "gemma_stt/embed"` |
| `embed/check_reference.py:36` | `V1_HYPS = Path.home() / "gemma_stt/decode/results/sweep_dev-clean_cbd_sp6_beam32_units-piece/hyps.jsonl"` |
| `embed/cost.py:94` | `ap.add_argument("--out", default=str(Path.home() / "gemma_stt/embed/cost.json"))` |
| `embed/retrieval.py:12` | `G = Path.home() / "gemma_stt"` |

`coreml/live.py:716` and `demo/embed_and_transcribe.py:174` (`home = str(Path.home())`) only shorten printed paths; keep them.

**Embedded paths in V1 artifacts** (found by the release script's scan):
- `spm_unigram256.model` (released): SentencePiece stores its `model_prefix`,
  `/Users/<you>/gemma_stt/feats/full/spm_unigram256`, in the trainer spec.
- `head_F0-0-0-4c.pt` (not released; the head is inside the Core ML package): contains the same path.
- Both only reveal the user name. Changing them changes the sha256 locked in `V1.md` and checked
  by `coreml/common.check_v1_files`. Either ship the tokenizer as it is (recommended: V1 stays
  byte-identical) or accept a new hash.

**Proposed fix: one configurable data root, `GEMMA_STT_HOME`** (default `~/gemma_stt`).
- Every data path becomes `STT_HOME / "<relative path>"`.
- `coreml/common.py` also finds the repo from its own location instead of `~/emb_gemma2`. The
  repo can then be cloned anywhere.
- With the variable unset, every path is unchanged, so V1 and the test results stay valid.
- The patch is generated against the current tree; `git apply --check` passes, and the patched
  files compile.
- Not applied. To apply: save the block below as `gemma_stt_home.patch`, then
  `git apply gemma_stt_home.patch`.

```diff
diff --git a/cache_feats.py b/cache_feats.py
--- a/cache_feats.py
+++ b/cache_feats.py
@@ -299,7 +299,7 @@
 
 def main():
     ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
-    ap.add_argument("--data-root", default="~/gemma_stt/data")
+    ap.add_argument("--data-root", default=os.path.join(os.environ.get("GEMMA_STT_HOME", "~/gemma_stt"), "data"))
     ap.add_argument("--splits", nargs="+", required=True)
     ap.add_argument("--hours", type=float, default=None, help="random subset of this many hours")
     ap.add_argument("--taps", nargs="+", default=["L11"])
diff --git a/coreml/common.py b/coreml/common.py
--- a/coreml/common.py
+++ b/coreml/common.py
@@ -9,25 +9,26 @@
 """
 import hashlib
 import json
+import os
 import random
 import sys
 from pathlib import Path
 
 import numpy as np
 
-HOME = Path.home()
-REPO = HOME / "emb_gemma2"
+STT_HOME = Path(os.environ.get("GEMMA_STT_HOME", "~/gemma_stt")).expanduser()   # data, models, results (outside git)
+REPO = Path(__file__).resolve().parents[1]
 sys.path.insert(0, str(REPO))
 sys.path.insert(0, str(REPO / "decode"))
 
 MODEL_ID = "google/embeddinggemma-2"
 MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
-HEAD = HOME / "gemma_stt/runs/full_160ms/head_F0-0-0-4c.pt"
-SPM = HOME / "gemma_stt/feats/full/spm_unigram256.model"
-LM = HOME / "gemma_stt/decode/lm/sp256_6gram.bin"
-DATA = HOME / "gemma_stt/data/LibriSpeech"
-LOGPROBS = HOME / "gemma_stt/decode/logprobs"
-OUT = HOME / "gemma_stt/coreml"
+HEAD = STT_HOME / "runs/full_160ms/head_F0-0-0-4c.pt"
+SPM = STT_HOME / "feats/full/spm_unigram256.model"
+LM = STT_HOME / "decode/lm/sp256_6gram.bin"
+DATA = STT_HOME / "data/LibriSpeech"
+LOGPROBS = STT_HOME / "decode/logprobs"
+OUT = STT_HOME / "coreml"
 V1_SHA256 = {HEAD: "748cd9e5b27861ef19fe48adba458de5c19914b9f1a048dd15d57e632c385911",
              SPM: "50002ddf56a79f19c6598240e684cf065a4a94b1fc88c5aff25b05962cccfe05",
              LM: "381c408f36fb5ea3cbece7eba408deb2b2320f0fe4679fd45d6bd394395fb371"}
diff --git a/coreml/tests/test_release.py b/coreml/tests/test_release.py
--- a/coreml/tests/test_release.py
+++ b/coreml/tests/test_release.py
@@ -3,16 +3,15 @@
 import json
 import re
 import time
-from pathlib import Path
 
 import pytest
 
-from common import OUT, V1_WER, load_audio, load_list, norm_text
+from common import OUT, STT_HOME, V1_WER, load_audio, load_list, norm_text
 from conftest import PRECISION, RUN_ID, UNITS, VARIANT
 from wer import corpus_wer
 
 pytestmark = pytest.mark.release
-PHONON2 = Path.home() / "gemma_stt/phonon2/results"
+PHONON2 = STT_HOME / "phonon2/results"
 PUNCT = re.compile(r"[^a-z0-9' ]+")
 
 
diff --git a/decode/build_subword_lm.py b/decode/build_subword_lm.py
--- a/decode/build_subword_lm.py
+++ b/decode/build_subword_lm.py
@@ -21,8 +21,8 @@
 
 import sentencepiece as spm
 
-HOME = Path.home()
-KENLM_BIN = HOME / "gemma_stt/tools/kenlm/build/bin"
+STT_HOME = Path(os.environ.get("GEMMA_STT_HOME", "~/gemma_stt")).expanduser()
+KENLM_BIN = STT_HOME / "tools/kenlm/build/bin"
 
 
 def norm_text(t):  # identical to probe.norm_text
@@ -72,10 +72,10 @@
 
 def main():
     ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
-    ap.add_argument("--corpus", type=Path, default=HOME / "gemma_stt/decode/lm/src/librispeech-lm-norm.txt.gz")
-    ap.add_argument("--spm", type=Path, default=HOME / "gemma_stt/feats/full/spm_unigram256.model")
-    ap.add_argument("--out", type=Path, default=HOME / "gemma_stt/decode/lm/sp256_6gram.bin")
-    ap.add_argument("--work", type=Path, default=HOME / "gemma_stt/decode/lm/sp256_work")
+    ap.add_argument("--corpus", type=Path, default=STT_HOME / "decode/lm/src/librispeech-lm-norm.txt.gz")
+    ap.add_argument("--spm", type=Path, default=STT_HOME / "feats/full/spm_unigram256.model")
+    ap.add_argument("--out", type=Path, default=STT_HOME / "decode/lm/sp256_6gram.bin")
+    ap.add_argument("--work", type=Path, default=STT_HOME / "decode/lm/sp256_work")
     ap.add_argument("--max-lines", type=int, default=None, help="only the first N corpus lines (testing)")
     ap.add_argument("--order", type=int, default=6)
     ap.add_argument("--prune", default="0 0 1 1 2 2",
diff --git a/decode/dump_logprobs.py b/decode/dump_logprobs.py
--- a/decode/dump_logprobs.py
+++ b/decode/dump_logprobs.py
@@ -28,7 +28,7 @@
 from featstore import read_meta  # noqa: E402
 from probe import decode, edit_distance, greedy_collapse, load_split, pick_device  # noqa: E402
 
-HOME = Path.home()
+STT_HOME = Path(os.environ.get("GEMMA_STT_HOME", "~/gemma_stt")).expanduser()
 
 
 def dump_split(model, ck, sp, feats_dir, split, out_dir, device, max_frames):
@@ -95,10 +95,10 @@
 
 def main():
     ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
-    ap.add_argument("--head", default=str(HOME / "gemma_stt/runs/full_160ms/head_F0-0-0-4c.pt"))
-    ap.add_argument("--feats", default=str(HOME / "gemma_stt/feats/full"), help="dir with one cache per split")
+    ap.add_argument("--head", default=str(STT_HOME / "runs/full_160ms/head_F0-0-0-4c.pt"))
+    ap.add_argument("--feats", default=str(STT_HOME / "feats/full"), help="dir with one cache per split")
     ap.add_argument("--splits", nargs="+", default=list(SPLITS))
-    ap.add_argument("--out", default=str(HOME / "gemma_stt/decode/logprobs"))
+    ap.add_argument("--out", default=str(STT_HOME / "decode/logprobs"))
     ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
     ap.add_argument("--max-frames", type=int, default=12000, help="batches hold 2x this many frames (as eval)")
     ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
diff --git a/decode/report.py b/decode/report.py
--- a/decode/report.py
+++ b/decode/report.py
@@ -14,9 +14,11 @@
 import argparse
 import csv
 import json
+import os
 from pathlib import Path
 
-RESULTS = Path.home() / "gemma_stt/decode/results"
+STT_HOME = Path(os.environ.get("GEMMA_STT_HOME", "~/gemma_stt")).expanduser()
+RESULTS = STT_HOME / "decode/results"
 KNEE_WER_POINTS = 0.05
 RULE = (f"Knee = the fastest run (highest x real time per core) whose WER is within {KNEE_WER_POINTS} WER "
         f"points (absolute percentage points) of the best WER: per decoder + LM (+ options), and overall.")
diff --git a/decode/run_decode.py b/decode/run_decode.py
--- a/decode/run_decode.py
+++ b/decode/run_decode.py
@@ -35,10 +35,10 @@
 sys.path.insert(0, str(Path(__file__).resolve().parent))
 from wer import utt_errors  # noqa: E402
 
-HOME = Path.home()
-LOGPROBS = HOME / "gemma_stt/decode/logprobs"
-RESULTS = HOME / "gemma_stt/decode/results"
-LM_DIR = HOME / "gemma_stt/decode/lm"
+STT_HOME = Path(os.environ.get("GEMMA_STT_HOME", "~/gemma_stt")).expanduser()
+LOGPROBS = STT_HOME / "decode/logprobs"
+RESULTS = STT_HOME / "decode/results"
+LM_DIR = STT_HOME / "decode/lm"
 LMS = {"none": None,
        "3g": LM_DIR / "3-gram.pruned.1e-7.lower.bin",
        "4g": LM_DIR / "4-gram.lower.bin",
diff --git a/decode/stream_study.py b/decode/stream_study.py
--- a/decode/stream_study.py
+++ b/decode/stream_study.py
@@ -23,6 +23,7 @@
 """
 import argparse
 import json
+import os
 import sys
 import time
 from pathlib import Path
@@ -37,11 +38,11 @@
 from streaming import StreamingBeam, StreamingGreedy, StreamingSubword, ctc_align_ends  # noqa: E402
 from wer import corpus_wer, norm_text  # noqa: E402
 
-HOME = Path.home()
-LOGPROBS = HOME / "gemma_stt/decode/logprobs"
-SP6 = HOME / "gemma_stt/decode/lm/sp256_6gram.bin"
-RESULTS = HOME / "gemma_stt/decode/results/streaming"
-V1_HYPS = HOME / "gemma_stt/decode/results/sweep_dev-clean_cbd_sp6_beam32_units-piece/hyps.jsonl"
+STT_HOME = Path(os.environ.get("GEMMA_STT_HOME", "~/gemma_stt")).expanduser()
+LOGPROBS = STT_HOME / "decode/logprobs"
+SP6 = STT_HOME / "decode/lm/sp256_6gram.bin"
+RESULTS = STT_HOME / "decode/results/streaming"
+V1_HYPS = STT_HOME / "decode/results/sweep_dev-clean_cbd_sp6_beam32_units-piece/hyps.jsonl"
 FRAME_S = 0.04
 
 
diff --git a/decode/test_dec_cbd.py b/decode/test_dec_cbd.py
--- a/decode/test_dec_cbd.py
+++ b/decode/test_dec_cbd.py
@@ -15,7 +15,7 @@
 sys.path.insert(0, str(Path(__file__).resolve().parent))
 from dec_cbd import make_decoder, norm_text  # noqa: E402
 
-SPM = Path(os.environ.get("SPM_MODEL", "~/gemma_stt/feats/full/spm_unigram256.model")).expanduser()
+SPM = Path(os.environ.get("SPM_MODEL", os.path.join(os.environ.get("GEMMA_STT_HOME", "~/gemma_stt"), "feats/full/spm_unigram256.model"))).expanduser()
 BLANK = 256
 
 
diff --git a/decode/test_dec_pyctc.py b/decode/test_dec_pyctc.py
--- a/decode/test_dec_pyctc.py
+++ b/decode/test_dec_pyctc.py
@@ -15,7 +15,7 @@
 sys.path.insert(0, str(Path(__file__).resolve().parent))
 import dec_pyctc  # noqa: E402
 
-SPM = os.path.expanduser("~/gemma_stt/feats/full/spm_unigram256.model")
+SPM = os.path.expanduser(os.path.join(os.environ.get("GEMMA_STT_HOME", "~/gemma_stt"), "feats/full/spm_unigram256.model"))
 SP = spm.SentencePieceProcessor(model_file=SPM)
 VOCAB = [SP.id_to_piece(i) for i in range(SP.get_piece_size())]
 BLANK = len(VOCAB)
diff --git a/decode/test_harness.py b/decode/test_harness.py
--- a/decode/test_harness.py
+++ b/decode/test_harness.py
@@ -9,6 +9,7 @@
 import itertools
 import json
 import math
+import os
 import subprocess
 import sys
 import tempfile
@@ -24,7 +25,7 @@
 import dec_subword  # noqa: E402
 from wer import corpus_wer, edit_distance, norm_text  # noqa: E402
 
-SPM = Path.home() / "gemma_stt/feats/full/spm_unigram256.model"
+SPM = Path(os.environ.get("GEMMA_STT_HOME", "~/gemma_stt")).expanduser() / "feats/full/spm_unigram256.model"
 V, BLANK, C = 256, 256, 257
 HAVE_TORCH = importlib.util.find_spec("torch") is not None
 HAVE_KENLM = importlib.util.find_spec("kenlm") is not None
diff --git a/decode/tune.py b/decode/tune.py
--- a/decode/tune.py
+++ b/decode/tune.py
@@ -12,6 +12,7 @@
 """
 import argparse
 import json
+import os
 import subprocess
 import sys
 import time
@@ -22,8 +23,9 @@
 sys.path.insert(0, str(HERE))
 from run_decode import fix_break  # noqa: E402
 
-RESULTS = Path.home() / "gemma_stt/decode/results"
-LOGPROBS = Path.home() / "gemma_stt/decode/logprobs"
+STT_HOME = Path(os.environ.get("GEMMA_STT_HOME", "~/gemma_stt")).expanduser()
+RESULTS = STT_HOME / "decode/results"
+LOGPROBS = STT_HOME / "decode/logprobs"
 HELD_OUT = ("dev-other", "test-clean", "test-other")
 ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
 BETAS = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
diff --git a/test_cache_plan.py b/test_cache_plan.py
--- a/test_cache_plan.py
+++ b/test_cache_plan.py
@@ -4,13 +4,14 @@
 
   python -m unittest test_cache_plan.py
 """
+import os
 import random
 import unittest
 from pathlib import Path
 
 from cache_feats import make_plan, plan_shards, shard_layout
 
-TRAIN_CLEAN_100 = Path("~/gemma_stt/data/LibriSpeech/train-clean-100").expanduser()
+TRAIN_CLEAN_100 = Path(os.environ.get("GEMMA_STT_HOME", "~/gemma_stt")).expanduser() / "data/LibriSpeech/train-clean-100"
 
 
 def synthetic(n, seed=0):
diff --git a/watch.py b/watch.py
--- a/watch.py
+++ b/watch.py
@@ -12,6 +12,7 @@
 Stdlib only; Ctrl-C to quit.
 """
 import argparse
+import os
 import re
 import statistics
 import subprocess
@@ -188,8 +189,8 @@
 
 def main():
     ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
-    ap.add_argument("log", nargs="?", default="~/gemma_stt/logs",
-                    help="log file, or a directory to follow its newest *.log (default ~/gemma_stt/logs)")
+    ap.add_argument("log", nargs="?", default=os.path.join(os.environ.get("GEMMA_STT_HOME", "~/gemma_stt"), "logs"),
+                    help="log file, or a directory to follow its newest *.log (default $GEMMA_STT_HOME/logs, ~/gemma_stt/logs if unset)")
     ap.add_argument("--interval", type=float, default=5.0, help="refresh seconds")
     ap.add_argument("--epochs", type=int, default=None, help="total epochs, if the run header doesn't say")
     ap.add_argument("--rows", type=int, default=10, help="epoch rows to show")
```

After applying, document it in `README.md`, for example:
`export GEMMA_STT_HOME=/Volumes/data/gemma_stt   # default ~/gemma_stt`.

### A.4 Suggested `docs/` folder

Reports from the data root that explain V1 (all small text, plus one 168 KB PNG):

| Source (`~/gemma_stt/…`) | Into the repo | Notes |
|---|---|---|
| `V1.md` | `docs/V1.md` | the V1 lock: components + sha256, settings, WERs, LM build |
| `coreml/PHASE1.md` | `docs/coreml/PHASE1.md` | export, ANE placement, fp16 analysis |
| `coreml/VERSIONS.md`, `coreml/requirements-coreml.freeze.txt` | `docs/coreml/` | Core ML venv versions (VERSIONS.md cites the freeze file) |
| `coreml/summary.md` | `docs/coreml/TESTS.md` | T1–T11 results (regenerate with `make summary` first) |
| `phonon2/results/SUMMARY.md` | `docs/phonon2.md` | Phonon-2 baseline |
| `decode/results/report_dev-clean.md`, `.png` | `docs/decode/` | decoder / LM sweep. **Line 3 contains `/Users/<you>/...`; change it to `~/gemma_stt/...`** |
| `decode/results/apply_knees.md` | `docs/decode/` | dev-clean knees on the other splits |
| `decode/results/streaming/dev-clean_chunk160ms.md` | `docs/decode/streaming_dev-clean_chunk160ms.md` | streaming study |

```bash
cd ~/emb_gemma2 && mkdir -p docs/coreml docs/decode
cp ~/gemma_stt/V1.md docs/
cp ~/gemma_stt/coreml/{PHASE1.md,VERSIONS.md,requirements-coreml.freeze.txt} docs/coreml/
cp ~/gemma_stt/coreml/summary.md docs/coreml/TESTS.md
cp ~/gemma_stt/phonon2/results/SUMMARY.md docs/phonon2.md
cp ~/gemma_stt/decode/results/{report_dev-clean.md,report_dev-clean.png,apply_knees.md} docs/decode/
cp ~/gemma_stt/decode/results/streaming/dev-clean_chunk160ms.md docs/decode/streaming_dev-clean_chunk160ms.md
git grep -n -I '/Users/' docs/      # must print nothing before committing
```

- **Paths inside the reports:** `V1.md`, `PHASE1.md` and the Phonon-2 summary refer to `~/gemma_stt/...`
  paths (15 / 3 / 3 mentions). They are fine as a description of the data-root layout.
- **Do not copy:** `coreml/models/export_log.jsonl` (21 `/Users/<you>` paths), `coreml/results.jsonl`
  (1 such path), logs, and the `runs/*.sh` scripts (one of them reads the webhook).
- **Also worth copying once final:** `~/gemma_stt/coreml/models/release/fp16/release_check.json` (the release package's
  verification against its source windows) and the embedding / streaming reports.

### A.5 Git identity and authorship

All 9 commits have author and committer `gemma-stt <gemma-stt@localhost>`. No `user.name` /
`user.email` is configured (`git config --get user.name` prints nothing).

```bash
cd ~/emb_gemma2
git config user.name  "Your Name"
git config user.email "you@example.com"      # or <id>+<user>@users.noreply.github.com to keep your email private
```

**Optional: rewrite the existing commits to your identity before the first push.**
- Rewriting changes every commit hash. `V1.md` (`8f661e0…`), `coreml/results.jsonl` and
  `summary.md` (`2a29bc34…`) cite the old hashes, so keep the old → new map.
- Run it only with a clean working tree, after the pending files are committed.

```bash
cd ~/emb_gemma2
test -z "$(git status --porcelain)" || echo "commit or stash first"
git log --format='%H %s' master > ~/gemma_stt/commit_map_old.txt
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --env-filter '
  if [ "$GIT_AUTHOR_EMAIL" = "gemma-stt@localhost" ]; then
    GIT_AUTHOR_NAME="Your Name"; GIT_AUTHOR_EMAIL="you@example.com"; fi
  if [ "$GIT_COMMITTER_EMAIL" = "gemma-stt@localhost" ]; then
    GIT_COMMITTER_NAME="Your Name"; GIT_COMMITTER_EMAIL="you@example.com"; fi
  export GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL GIT_COMMITTER_NAME GIT_COMMITTER_EMAIL
' --tag-name-filter cat -- --all
# old -> new map (history is linear, same order); keep it next to the reports
paste -d' ' <(git log --format=%H refs/original/refs/heads/master) <(git log --format='%H %s' master) > ~/gemma_stt/commit_map.txt
# the annotated tag v1 keeps its old tagger; re-create it with your identity on the rewritten commit
git tag -f -a v1 -m "$(git tag -l --format='%(contents)' v1)" v1^{commit}
# drop filter-branch's backup refs once you are happy
git for-each-ref --format='%(refname)' refs/original | xargs -n1 git update-ref -d
git log --all --format='%an <%ae> | %cn <%ce>' | sort | uniq -c
git for-each-ref refs/tags --format='%(refname:short) %(taggername) %(taggeremail)'
```

**Co-authored-by trailers.** Confirm that there are none (there must be none):

```bash
git log --all --format=%B | grep -ci 'co-authored-by'
```

Current output: `0` (no commit message contains the trailer). Run the check again after
any rewrite and before pushing.

### A.6 Push

```bash
cd ~/emb_gemma2
# optional: git branch -m master main
gh repo create YOUR_GITHUB_USER/emb_gemma2 --private --source . --remote origin   # or create it on github.com, then:
# git remote add origin git@github.com:YOUR_GITHUB_USER/emb_gemma2.git
git push -u origin HEAD --tags
```

Make the repo public once the README, LICENSE and the A.2 checks are done. Then pass its URL to
`make_hf_release.py --github-url`.

---

## B. Hugging Face (model repo)

The release is **V1 only, as one Core ML package**:
- `v1.mlpackage`: fp16, multifunction (`win_5s`, `win_10s` default, `win_20s`, `win_35s`) with
  shared weights, built from the fp16-safe variant by `coreml/export_release.py`. It needs macOS 15 /
  iOS 18 and runs under `CPU_AND_NE` (recommended), `CPU_AND_GPU` and `CPU_ONLY`.
- The tokenizer, the subword 6-gram LM, `config.json` (pinned decoder settings, windows → functions,
  front end, minimum OS), `LICENSE`, `NOTICE`, `README.md` (model card) and `SHA256SUMS`.
- The per-window packages and the fp32 build are internal references and are not released.

### B.1 Build the folder

Prerequisites (the script lists every missing one; `--write` refuses while any is missing):
1. **The package:** `~/gemma_stt/coreml/models/release/fp16/v1.mlpackage`, from
   `~/.venv-coreml/bin/python coreml/export_release.py`. The script checks its functions, default
   function, shapes and metadata (V1 hashes, decoder settings).
2. **Its own test results** in `~/gemma_stt/coreml/results.jsonl`: rows with `variant == "release"`
   whose `model_weight_sha256` is this package's `weight.bin`.
   - **T9** (dev WER) and **T10** (speed / WER / RSS / load under `CPU_AND_NE`, `CPU_AND_GPU` and
     `CPU_ONLY`) are required and must have passed:
     `cd ~/emb_gemma2 && V1_VARIANT=release V1_PRECISION=fp16 make test-full` (T10 needs an idle Mac).
   - **T11** (test sets next to V1 and Phonon-2: `make test-release` with the same variables) and T6
     (Neural Engine share, `make test-fast`) are added to the card when present.
   - The card shows no other numbers; there is nothing to fill in by hand.
3. `--hf-user`, `--author` and `--github-url` set (no placeholders).

```bash
cd ~/emb_gemma2
PY=~/.venv-coreml/bin/python
ARGS=(--hf-user YOUR_HF_USER --author "Your Name" --github-url https://github.com/YOUR_GITHUB_USER/emb_gemma2)
# 1. dry run (default): layout, sizes, sha256, secret scan, missing prerequisites; writes nothing
$PY release/make_hf_release.py "${ARGS[@]}"
# preview the generated README.md / config.json / NOTICE
$PY release/make_hf_release.py "${ARGS[@]}" --no-hash --preview | less
# 2. write it (default ~/gemma_stt/release/hf/<repo-name>/; refuses to overwrite)
$PY release/make_hf_release.py "${ARGS[@]}" --write
cd ~/gemma_stt/release/hf/gemma-stt-v1-coreml && shasum -a 256 -c SHA256SUMS
```

- **Another license:** `--license <hf-id> --license-file <file>`; see the decisions at the top.
- **`transcribe.py` in the code repo** checks the sha256 of the PyTorch head as well
  (`coreml/common.check_v1_files`). The head is not part of the HF release, so the repo's runner
  needs your local `~/gemma_stt/runs/full_160ms/head_F0-0-0-4c.pt`. The card's standalone usage
  example needs only the HF files.

### B.2 Expected size

Measured by the dry run on 2026-10-14 against the package built that day (a rebuild can change it):

| Path | Size |
|---|---|
| `v1.mlpackage/` (`weight.bin` 646.9 MB, `model.mlmodel` 5.7 MB, `Manifest.json`) | 652.6 MB |
| `sp256_6gram.bin` | 761.7 MB |
| `spm_unigram256.model` | 243.6 KB |
| `LICENSE`, `config.json`, `NOTICE`, `README.md`, `SHA256SUMS` | ~32 KB |
| **total** | **1.4 GB in 10 files** |

- The four functions share one `weight.bin`. That is 646.9 MB, against 2.5 GB for four separate
  window packages.
- The largest single file is the 762 MB LM, well under the Hub's per-file limit.

### B.3 Upload

```bash
PY=~/.venv-coreml/bin   # huggingface_hub 2.2.0 with the `hf` CLI and hf-xet 1.7.0
$PY/hf auth login                                   # a token with write access
$PY/hf repos create YOUR_HF_USER/gemma-stt-v1-coreml --private
$PY/hf upload YOUR_HF_USER/gemma-stt-v1-coreml ~/gemma_stt/release/hf/gemma-stt-v1-coreml . \
    --commit-message "V1: Core ML package (fp16, multifunction), tokenizer, subword 6-gram LM"
# after checking the page renders (model card, files, license), make it public in Settings
```

- **Resuming:** `hf upload` splits big folders into several commits. If it is interrupted, re-run
  the same command to resume.
- **Older CLI:** `huggingface-cli upload YOUR_HF_USER/gemma-stt-v1-coreml <folder> .` does the same.
- **Updating later:** re-run `make_hf_release.py --write --out <new dir>`, then `hf upload` again.
  Unchanged files are deduplicated.

### B.4 LFS / Xet notes

- **Xet storage:** new Hub repos store large files with Xet. `hf upload` (with `hf-xet` installed,
  as in `~/.venv-coreml`) chunks and deduplicates them, so you need neither `git` nor `git-lfs`.
  `git-lfs` is not installed on this Mac.
- **The `.mlpackage` directory:** `v1.mlpackage` is a plain directory: `Manifest.json` plus
  `Data/com.apple.CoreML/{model.mlmodel,weights/weight.bin}`. It uploads as ordinary files.
  Keep the structure: Core ML loads the directory, and `snapshot_download` recreates it.
- **If you push with git:** large files must be tracked by LFS.
  - `git lfs install`, then `git clone https://huggingface.co/YOUR_HF_USER/gemma-stt-v1-coreml`.
  - Check that the repo's `.gitattributes` has `*.bin`, `*.mlmodel` and `*.model`. The Hub's default
    `.gitattributes` covers these; add any missing pattern with `git lfs track "<pattern>"`.
  - Copy the folder in, then `git add . && git commit && git push`.
  - `weight.bin` and `sp256_6gram.bin` go through LFS; `Manifest.json` and the text files stay in git.
- **Skipped and not needed:** `.DS_Store` files inside the package are skipped by the release script.
  `config.json` has no `model_type`, so `transformers` does not try to load the repo as a model.
  The feature extractor is rebuilt from the copy of `preprocessor_config.json` embedded in
  `config.json`; that output matches `coreml/common.mel` exactly.
