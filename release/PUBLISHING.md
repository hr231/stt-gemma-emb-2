# Publishing V1: GitHub (code) and Hugging Face (model)

Nothing here has been run against GitHub or Hugging Face. Each step is a command for you to run
when you are ready. The check outputs quoted below were produced on 2026-10-14 at commit `d6af85b`
(`git status`: untracked files only).

**Decisions that are yours:**
- **Repo names:** GitHub `emb_gemma2` or another name; Hugging Face `<user>/gemma-stt-v1-coreml`
  (`--repo-name`).
- **Code license:** there is no `LICENSE` file yet. Common choices are Apache-2.0 and MIT. This is
  separate from the model weights, which fall under the Gemma Terms of Use.
- **Model license:** the card says `license: gemma`.
  - Check that this still matches the license shown on
    [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) before publishing.
  - Optionally gate the HF repo (Settings → "Access requests"), as Google does for Gemma weights.
- **Git identity:** whether to rewrite the 6 local commits.
  - They are authored `gemma-stt <gemma-stt@localhost>`.
  - The annotated tag `v1` has tagger `Harshit Brahmbhatt <harshit@Harshits-MacBook-Pro.local>`, which includes a local hostname.
  - See A.5.
- **What to ship on HF:** fp16 only (3.2 GB), and optionally the fp32 build (+4.9 GB) and the
  PyTorch head (+53 MB).
- **Tokenizer path:** the tokenizer embeds a local path (see A.3).

---

## A. GitHub (code repo `~/emb_gemma2`)

### A.1 What is in the repo

| Path | Contents |
|---|---|
| `README.md`, `setup.sh`, `download_librispeech*.{sh,py}` | setup, LibriSpeech download (OpenSLR layout) |
| `cache_feats.py`, `featstore.py` | mel + Gemma audio tower features, int8 shard store |
| `probe.py`, `diag.py`, `head.py`, `eval_head.py`, `train_tokenizer.py`, `watch.py`, `test_*.py` | layer probe, CTC head training / eval, tokenizer, log watcher, tests |
| `decode/` | decoding study: `dec_*.py` decoders, `run_decode.py`, `sweep.py`, `tune.py`, `report.py`, `streaming.py`, `stream_study.py`, `build_subword_lm.py`, pinned `requirements-*.txt` |
| `coreml/` | Core ML export (`common.py`, `wrapper.py`, `export.py`, `palettize.py`), runtime (`cmlrun.py`, `transcribe.py`), evaluation (`eval_fast.py`, `placement.py`, `bench.py`, `summary.py`), `tests/` (T1–T11) |
| `Makefile` | `make test-fast / test-full / test-release / bench / summary` |
| `release/` | this file, `make_hf_release.py`, `hf_model_card_template.md` |

Untracked at the time of writing (the main session commits them): `coreml/pytest.ini`, `coreml/LIVE.md`, `coreml/live.py`, `coreml/livecore.py`, `coreml/palettize.py`, `demo/`, `embed/`, `release/`.
Tracked files: 54, 408K in total, so the repo itself is tiny.

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

<details><summary>Full grep output (135 lines)</summary>

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
coreml/LIVE.md:9:Everything is in `~/.venv-coreml`. The app needs one extra package, `sounddevice` (its macOS wheel
coreml/LIVE.md:11:Versions added to the env are listed in `~/gemma_stt/coreml/VERSIONS_ADDED.md`.
coreml/LIVE.md:14:~/.venv-coreml/bin/pip install sounddevice==0.5.6
coreml/LIVE.md:15:~/.venv-coreml/bin/python -c "import sounddevice, kenlm; print(sounddevice.__version__)"
coreml/LIVE.md:22:From the repo root (`~/emb_gemma2`):
coreml/LIVE.md:26:~/.venv-coreml/bin/python coreml/live.py
coreml/LIVE.md:29:~/.venv-coreml/bin/python coreml/live.py --mode A --device "MacBook Pro Microphone"
coreml/LIVE.md:30:~/.venv-coreml/bin/python coreml/live.py --list-devices
coreml/LIVE.md:33:~/.venv-coreml/bin/python coreml/live.py --mode C --replay ~/gemma_stt/coreml/live_sessions/dev-clean_6345-93306_0021-0025.flac
coreml/LIVE.md:36:~/.venv-coreml/bin/python coreml/live.py --replay FILE --fast --exit-at-end --session ~/gemma_stt/coreml/live_sessions/try1
coreml/LIVE.md:69:| `c` | save the transcript to `~/gemma_stt/coreml/live_sessions/transcript-YYYYMMDD-HHMMSS.txt` (or the `--session` folder) and copy it to the clipboard |
coreml/LIVE.md:149:| Mode C ends with `engine stopped: ModuleNotFoundError: No module named 'kenlm'` | The LM fusion decoder needs `kenlm` in `~/.venv-coreml`; mode A works without it. |
coreml/LIVE.md:175:  script -q ~/gemma_stt/coreml/live_sessions/take1.ansi \
coreml/LIVE.md:176:      ~/.venv-coreml/bin/python coreml/live.py --mode C --replay FILE --exit-at-end \
coreml/LIVE.md:177:      --session ~/gemma_stt/coreml/live_sessions/take1
coreml/LIVE.md:185:Sample sessions are in `~/gemma_stt/coreml/live_sessions/`:
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
coreml/fp16_layers.py:5:  ~/.venv-coreml/bin/python coreml/fp16_layers.py --variant faithful [--n 12]
coreml/live.py:5:  ~/.venv-coreml/bin/python coreml/live.py [--mode A|C] [--device NAME|INDEX]
coreml/live.py:6:  ~/.venv-coreml/bin/python coreml/live.py --mode C --replay FILE [--fast] [--exit-at-end]
coreml/live.py:34:SESSIONS = Path.home() / "gemma_stt/coreml/live_sessions"
coreml/live.py:713:    home = str(Path.home())
coreml/make_reference.py:2:"""Fixed test lists and the PyTorch fp32 CPU reference outputs for FAST (run in ~/.venv-gemma-stt).
coreml/make_reference.py:4:  ~/.venv-gemma-stt/bin/python coreml/make_reference.py
coreml/palettize.py:4:  ~/.venv-coreml/bin/python coreml/palettize.py --builds g8 g16 pt [--windows 5s 10s 20s 35s] [--device mps]
coreml/placement.py:4:  ~/.venv-coreml/bin/python coreml/placement.py --variant faithful --precision fp16 [--windows 5s ...]
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
demo/embed_and_transcribe.py:11:Run with V1's environment:  ~/.venv-gemma-stt/bin/python demo/embed_and_transcribe.py ...
demo/embed_and_transcribe.py:174:    home = str(Path.home())
diag.py:10:  python diag.py --feats ~/gemma_stt/feats/probe
download_librispeech.sh:5:# Usage:  bash download_librispeech.sh ~/gemma_stt/data train-clean-100 dev-clean
download_librispeech_hf.py:11:  python download_librispeech_hf.py ~/gemma_stt/data train-other-500 dev-other test-clean test-other
embed/check_reference.py:18:Writes ~/gemma_stt/embed/check_ids.json, check_reference.json, check_reference_clips.jsonl.
embed/check_reference.py:35:OUT = Path.home() / "gemma_stt/embed"
embed/check_reference.py:36:V1_HYPS = Path.home() / "gemma_stt/decode/results/sweep_dev-clean_cbd_sp6_beam32_units-piece/hyps.jsonl"
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

**In progress when this was written** (uncommitted work of other sessions: `coreml/LIVE.md`, `coreml/live.py`, `coreml/livecore.py`, `coreml/palettize.py`, `demo/`, `embed/`; not covered by the patch, so re-run the grep after they land):

- `coreml/LIVE.md:9`: `Everything is in `~/.venv-coreml`. The app needs one extra package, `sounddevice` (its macOS wheel`
- `coreml/LIVE.md:11`: `Versions added to the env are listed in `~/gemma_stt/coreml/VERSIONS_ADDED.md`.`
- `coreml/LIVE.md:14`: `~/.venv-coreml/bin/pip install sounddevice==0.5.6`
- `coreml/LIVE.md:15`: `~/.venv-coreml/bin/python -c "import sounddevice, kenlm; print(sounddevice.__version__)"`
- `coreml/LIVE.md:22`: `From the repo root (`~/emb_gemma2`):`
- `coreml/LIVE.md:26`: `~/.venv-coreml/bin/python coreml/live.py`
- `coreml/LIVE.md:29`: `~/.venv-coreml/bin/python coreml/live.py --mode A --device "MacBook Pro Microphone"`
- `coreml/LIVE.md:30`: `~/.venv-coreml/bin/python coreml/live.py --list-devices`
- `coreml/LIVE.md:33`: `~/.venv-coreml/bin/python coreml/live.py --mode C --replay ~/gemma_stt/coreml/live_sessions/dev-clean_6345-93306_0021-0025.flac`
- `coreml/LIVE.md:36`: `~/.venv-coreml/bin/python coreml/live.py --replay FILE --fast --exit-at-end --session ~/gemma_stt/coreml/live_sessions/try1`
- `coreml/LIVE.md:69`: `| `c` | save the transcript to `~/gemma_stt/coreml/live_sessions/transcript-YYYYMMDD-HHMMSS.txt` (or the `--session` folder) and copy it to the clipboard |`
- `coreml/LIVE.md:149`: `| Mode C ends with `engine stopped: ModuleNotFoundError: No module named 'kenlm'` | The LM fusion decoder needs `kenlm` in `~/.venv-coreml`; mode A works without it. |`
- `coreml/LIVE.md:175`: `script -q ~/gemma_stt/coreml/live_sessions/take1.ansi \`
- `coreml/LIVE.md:176`: `~/.venv-coreml/bin/python coreml/live.py --mode C --replay FILE --exit-at-end \`
- `coreml/LIVE.md:177`: `--session ~/gemma_stt/coreml/live_sessions/take1`
- `coreml/LIVE.md:185`: `Sample sessions are in `~/gemma_stt/coreml/live_sessions/`:`
- `coreml/live.py:5`: `~/.venv-coreml/bin/python coreml/live.py [--mode A|C] [--device NAME|INDEX]`
- `coreml/live.py:6`: `~/.venv-coreml/bin/python coreml/live.py --mode C --replay FILE [--fast] [--exit-at-end]`
- `coreml/live.py:34`: `SESSIONS = Path.home() / "gemma_stt/coreml/live_sessions"`
- `coreml/live.py:713`: `home = str(Path.home())`
- `coreml/palettize.py:4`: `~/.venv-coreml/bin/python coreml/palettize.py --builds g8 g16 pt [--windows 5s 10s 20s 35s] [--device mps]`
- `demo/embed_and_transcribe.py:11`: `Run with V1's environment:  ~/.venv-gemma-stt/bin/python demo/embed_and_transcribe.py ...`
- `demo/embed_and_transcribe.py:174`: `home = str(Path.home())`
- `embed/check_reference.py:18`: `Writes ~/gemma_stt/embed/check_ids.json, check_reference.json, check_reference_clips.jsonl.`
- `embed/check_reference.py:35`: `OUT = Path.home() / "gemma_stt/embed"`
- `embed/check_reference.py:36`: `V1_HYPS = Path.home() / "gemma_stt/decode/results/sweep_dev-clean_cbd_sp6_beam32_units-piece/hyps.jsonl"`
- `embed/onepass.py:4`: `~/.venv-gemma-stt/bin/python embed/onepass.py AUDIO [AUDIO ...]        # prints text + embedding head`
- `embed/onepass.py:13`: `.encode(audio)` returns (see ~/gemma_stt/embed/REPORT_core.md for the code references).`
- `embed/retrieval.py:3`: `python embed/retrieval.py  -> ~/gemma_stt/embed/retrieval/ + ~/gemma_stt/embed/REPORT_retrieval.md`
- `embed/retrieval.py:12`: `G = Path.home() / "gemma_stt"`
- `embed/retrieval.py:132`: `"Reproduce: `~/.venv-gemma-stt/bin/python embed/retrieval.py`."]`

The data-root lines among them (`Path.home() / "gemma_stt"...`) should use `GEMMA_STT_HOME` the same way.

**Embedded paths in V1 artifacts** (found by the release script's scan):
- `spm_unigram256.model`: SentencePiece stores its `model_prefix`,
  `/Users/<you>/gemma_stt/feats/full/spm_unigram256`, in the trainer spec.
- `head_F0-0-0-4c.pt`: contains the same path, from the checkpoint's tokenizer reference.
- Both only reveal the user name. Changing them changes the sha256 locked in `V1.md` and checked
  by `coreml/common.check_v1_files`. Either ship them as they are (recommended: V1 stays
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
- **Not copied yet:** add Phase 4 (palettization) and the embedding/retrieval reports once they exist.

### A.5 Git identity and authorship

All 6 commits have author and committer `gemma-stt <gemma-stt@localhost>`. No `user.name` /
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

### B.1 Build the folder

```bash
cd ~/emb_gemma2
PY=~/.venv-coreml/bin/python
# 1. dry run (default): layout, sizes, sha256, secret scan; writes nothing
$PY release/make_hf_release.py --hf-user YOUR_HF_USER --author "Your Name" \
    --github-url https://github.com/YOUR_GITHUB_USER/emb_gemma2 [--include-pytorch-head] [--include-fp32]
# preview the generated README.md / config.json / NOTICE / lm/README.md
$PY release/make_hf_release.py --no-hash --preview --hf-user YOUR_HF_USER | less
# 2. write it (default ~/gemma_stt/release/hf/<repo-name>/; refuses to overwrite)
$PY release/make_hf_release.py --write --hf-user YOUR_HF_USER --author "Your Name" \
    --github-url https://github.com/YOUR_GITHUB_USER/emb_gemma2 [--include-pytorch-head] [--include-fp32]
cd ~/gemma_stt/release/hf/gemma-stt-v1-coreml && shasum -a 256 -c SHA256SUMS
```

- **Other builds:** `--build <variant>/<precision>` selects another build, for example a Phase 4
  palettized `faithful/fp16_pal6_g16`.
- **Results in the README:** the card takes the T9 / T10 / T11 tables from
  `~/gemma_stt/coreml/results.jsonl` for exactly that build. Run `make test-full` (and
  `make test-release`) for it first, with `V1_VARIANT` / `V1_PRECISION` set. Otherwise the card
  says the result is missing.
- **`transcribe.py` and the head:** the repo's `coreml/transcribe.py` checks the PyTorch head's
  sha256, so users of the GitHub code need the head. `--include-pytorch-head` (53 MB) makes the HF
  repo self-sufficient for that.

### B.2 Expected size

Measured by the dry run on 2026-10-14:

| Contents | Size | Files |
|---|---|---|
| `faithful/fp16` build (4 windows) | 2.5 GB (609–635 MB `weight.bin` each) | 12 |
| `lm/sp256_6gram.bin` | 761.7 MB | 1 |
| tokenizer, configs, README, NOTICE, SHA256SUMS | ~0.27 MB | 7 |
| **default total** | **3.2 GB** | 20 |
| `--include-pytorch-head` | +53.2 MB | +1 |
| `--include-fp32` (`coreml_fp32/`) | +4.9 GB (~1.2 GB per window) | +12 |
| everything | 8.2 GB | 32 |

- The largest single file is the 762 MB LM, well under the Hub's per-file limit.
- A 6-bit palettized build should shrink the 4 windows by roughly 6/16 [INFERENCE]. Run the dry run
  with its `--build` to get the real figure.

### B.3 Upload

```bash
PY=~/.venv-coreml/bin   # huggingface_hub 2.2.0 with the `hf` CLI and hf-xet 1.7.0
$PY/hf auth login                                   # a token with write access
$PY/hf repos create YOUR_HF_USER/gemma-stt-v1-coreml --private
$PY/hf upload YOUR_HF_USER/gemma-stt-v1-coreml ~/gemma_stt/release/hf/gemma-stt-v1-coreml . \
    --commit-message "V1 Core ML (faithful/fp16), tokenizer, subword 6-gram LM"
# after checking the page renders (model card, files, license), make it public in Settings (or keep it gated)
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
- **`.mlpackage` directories:** a `.mlpackage` is a plain directory: `Manifest.json` plus
  `Data/com.apple.CoreML/{model.mlmodel,weights/weight.bin}`. It uploads as ordinary files.
  Keep the directory structure: Core ML loads the directory, and `snapshot_download` recreates it.
- **If you push with git:** large files must be tracked by LFS.
  - `git lfs install`, then `git clone https://huggingface.co/YOUR_HF_USER/gemma-stt-v1-coreml`.
  - Check that the repo's `.gitattributes` has `*.bin`, `*.mlmodel`, `*.model` and `*.pt`. The
    Hub's default `.gitattributes` covers these; add any missing pattern with `git lfs track "<pattern>"`.
  - Copy the folder in, then `git add . && git commit && git push`.
  - `weight.bin` (~0.6 GB, or 1.2 GB for fp32) and `sp256_6gram.bin` go through LFS; `Manifest.json`
    and the text files stay in git.
- **Not uploaded:** `.DS_Store` files inside packages are skipped by the release script.
  `config.json` has no `model_type`, so `transformers` does not try to load this repo as a
  transformers model. `AutoFeatureExtractor.from_pretrained(<folder>)` works: it reads
  `preprocessor_config.json`, and its output matches `coreml/common.mel` exactly.
