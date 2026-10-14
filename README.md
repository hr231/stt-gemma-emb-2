# EmbeddingGemma 2 → STT, all on the Mac

Frozen Gemma 4 / EmbeddingGemma 2 audio tower (25 fps) → cached features → linear CTC head.
Everything runs from this folder, inside the project venv.

| File | What it does |
|---|---|
| `setup.sh` | Python 3.12 venv at `~/.venv-gemma-stt` (separate from vllm-metal) |
| `download_librispeech.sh` | LibriSpeech in the OpenSLR layout, resumable |
| `download_librispeech_hf.py` | Same files from the Hugging Face Hub (much faster than openslr.org), same `.done_<split>` markers |
| `cache_feats.py` | Parallel decode + mel on CPU workers, audio tower on MPS (bf16), resumable shards, optional int8 |
| `probe.py` | SentencePiece-256 + one linear CTC head per run, greedy dev WER per epoch |
| `diag.py` | Linear-head diagnostics: train-subset WER, dev examples + S/D/I counts, audio-tower lookahead test |
| `head.py` | Small nonlinear CTC head (4 local-attention/conv/FFN blocks, R future frames per block) on cached features |
| `eval_head.py` | Score a saved head on each cached split separately (greedy, corpus-level WER) |
| `train_tokenizer.py` | SentencePiece unigram-256 on every transcript of a cached train split (same settings as `probe.py`) |
| `watch.py` | Terminal dashboard for a training log: progress, ETA, best/last WER, sparklines, power, process |
| `featstore.py` | Shared shard format / loaders |

## 1. Setup (once)

```bash
bash setup.sh
source ~/.venv-gemma-stt/bin/activate
```

## 2. Data for the probe

If the research folder already has `LibriSpeech/train-clean-100/<spk>/<chapter>/*.flac`,
pass its parent as `--data-root` below and skip the download. Otherwise:

```bash
bash download_librispeech.sh ~/gemma_stt/data train-clean-100 dev-clean
# or, much faster (parquet from huggingface.co, converted to the same OpenSLR layout):
python download_librispeech_hf.py ~/gemma_stt/data train-clean-100 dev-clean
```

openslr.org gave ~4 MB/s per connection here (its US mirror fails the TLS name check); the Hub
copy `openslr/librispeech_asr` gave ~90 MB/s and is byte-identical (dev-clean: every `.flac` and
`trans.txt` matches). A stale saved HF token is ignored; the dataset is public.

## 3. Cache the probe set (~15 min, ~14 GB)

```bash
caffeinate -dims python cache_feats.py --splits train-clean-100 --hours 20 \
    --taps L6 L9 L11 --out ~/gemma_stt/feats/probe/train --verify
caffeinate -dims python cache_feats.py --splits dev-clean \
    --taps L6 L9 L11 --out ~/gemma_stt/feats/probe/dev
```

- `--verify` first compares MPS bf16 against a CPU fp32 reference on one batch. Expect cosine
  mean > 0.99 per tap; it stops if not (then retry with `--dtype fp32`).
- The progress line's `x real time` is the measured Mac speed: it replaces the ~150× estimate.
- No `WARNING: ... frames (expected ...)` lines should appear (that would mean truncation).
- `L11` = the 12th, last conformer layer (PyTorch counts from 0).
- Shards hold a random sample of the chosen clips (`--seed`), sorted by duration inside each
  shard, so every shard covers the full length range (needed by `head.py --stream`; caches
  written before this layout are duration-banded). `--plan-only` prints the shard durations and
  padding overhead without loading the model; `python -m unittest test_cache_plan.py` checks it.

## 4. Probe (~minutes per run)

```bash
python probe.py --feats ~/gemma_stt/feats/probe --runs L6 L9 L11 L11:int8
```

Expect ~100% WER and ~99% blank for the first epoch or two (CTC's "all blank" phase), then a drop.
Decision point on the best run's dev-clean WER:

- under 10% → linear path; next try `--runs L11:stack2 L11:stack4 L11:stack6` (lookahead)
- 10–20% → go to a small nonlinear causal head
- over 20% → plan on unfreezing top encoder layers

`L11:int8` vs `L11`: if the gap is under ~0.2 WER points, cache the full set with `--int8`.
`L11:cat` joins L9 and L11 per frame (2048-dim input).

## 4b. Diagnostics and nonlinear head (cached features only)

```bash
caffeinate -dims python diag.py --feats ~/gemma_stt/feats/probe              # uses probe_results/head_L11.pt
caffeinate -dims python head.py --feats ~/gemma_stt/feats/probe --lookahead 0 4
```

`head.py` applies R per block, so its total lookahead is 4 × (R + min(R, 2)) frames of 40 ms
(R=4 → 960 ms). Best heads and a results JSON go to `--out` (default `<feats>/head_results`).
Options: `--block-future F1 F2 F3 F4` (attention future frames per block, one run),
`--causal-conv` (every depthwise conv strictly causal), `--int8` (int8 round-trip on train and
dev features), `--stream` (read training shards from disk one at a time with the next one
prefetched, for caches larger than RAM). The total lookahead is printed at start-up from the
actual masks and conv padding; `python -m unittest test_head.py` checks it is the exact future
receptive field, and `--expect-lookahead N` stops before loading data unless it is N frames.
After every epoch `head.py` writes `ckpt_<run>_ep<N>.pt` (model, optimizer, scheduler, every RNG
state, best WER, history, feature stats); `--resume` continues from the latest one.
`--plateau-patience N` prints a `[plateau]` line once dev WER has not improved for N epochs;
`--stop-patience M` ends the run after M such epochs (the best head is saved on every improvement).

## 5. Full run (overnight)

```bash
bash download_librispeech.sh ~/gemma_stt/data train-clean-360 train-other-500 dev-other test-clean test-other
caffeinate -dims python cache_feats.py --splits train-clean-100 train-clean-360 train-other-500 \
    --taps L11 --int8 --out ~/gemma_stt/feats/full/train
for s in dev-clean dev-other test-clean test-other; do
  caffeinate -dims python cache_feats.py --splits $s --taps L11 --int8 --out ~/gemma_stt/feats/full/$s
done
ln -sfn dev-clean ~/gemma_stt/feats/full/dev        # head.py evaluates on <feats>/dev
```

Use the tap the probe picked instead of `L11` if another won. Disk: ~89 GB int8 (~177 GB bf16)
plus ~60 GB of raw LibriSpeech. If the run is interrupted, rerun the same command: finished
shards are skipped.

Then the tokenizer on all 281,241 training transcripts, the 960 h head (run-4a architecture,
160 ms lookahead, streaming, 15 epochs, resumable: rerun to continue) and per-split scoring:

```bash
python train_tokenizer.py --feats ~/gemma_stt/feats/full --compare ~/gemma_stt/feats/probe/spm_unigram256.model
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25 caffeinate -dims python head.py --feats ~/gemma_stt/feats/full --stream \
    --block-future 0 0 0 4 --causal-conv --expect-lookahead 4 --epochs 15 --warmup 1000 \
    --plateau-patience 3 --stop-patience 5 \
    --out ~/gemma_stt/runs/full_160ms --resume 2>&1 | tee -a ~/gemma_stt/logs/full_160ms.log
python eval_head.py --head ~/gemma_stt/runs/full_160ms/head_F0-0-0-4c.pt --feats ~/gemma_stt/feats/full
```
