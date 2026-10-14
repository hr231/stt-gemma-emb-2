---
license: gemma
base_model: google/embeddinggemma-2
pipeline_tag: automatic-speech-recognition
library_name: coreml
language:
- en
tags:
- coreml
- ane
- ctc
- apple-silicon
- speech-recognition
datasets:
- openslr/librispeech_asr
---

# {{MODEL_NAME}}: English speech recognition on the Apple Neural Engine (Core ML)

This is a streaming-capable English speech recognizer for Apple silicon. It reuses the audio tower of
[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) up to layer 11 and adds a
small causal CTC head trained on LibriSpeech 960 h. The encoder and head are exported as one Core ML
program per fixed input window. A KenLM subword 6-gram is used for beam-search decoding.

- **Build in this repo:** `{{BUILD}}` ({{BUILD_DESC}}), minimum deployment target {{MIN_OS}}.
- **Code, training and evaluation:** {{GITHUB_URL}} (tag `v1`).

## Model description

```
16 kHz audio
  -> log-mel features [T_mel, 128], 100 frames/s (Gemma4AudioFeatureExtractor, preprocessor_config.json)
  -> Gemma audio tower, layers 0-11 (google/embeddinggemma-2 @ {{MODEL_REVISION_SHORT}}), 25 frames/s (40 ms)
  -> per-frame int8 round trip (absmax / 127), as the head saw during training
  -> standardize (training mean / std, 1,024 dims)
  -> 13.3M-parameter causal CTC head: 256 SentencePiece pieces + blank (id 256), 160 ms lookahead
  -> ctc-beam-decoder 0.2.0 + subword 6-gram LM (beam 32, alpha 0.8, beta 1.0)
  -> lowercase text
```

- **Inside each Core ML package:** everything from the mel input to the CTC log-probabilities.
  Feature extraction runs on the CPU (NumPy). The beam search and LM run on the CPU as well.
- **Inputs:** `mel` float32 `[1, T_mel, 128]`, zero-padded to the window; `valid_len` int32 `[1]`,
  the number of valid mel frames.
- **Outputs:** `log_probs` `[1, T_mel/4, 257]` and `l11` (the layer-11 features) `[1, T_mel/4, 1024]`.
  Frames past `ceil(valid_len / 4)` are padding.
- **Windows:** 5 / 10 / 20 / 35 s = 500 / 1000 / 2000 / 3500 mel frames. Pick the smallest window
  that fits. Longer audio is cut at pauses, with 2 s of left context per segment
  (`coreml/transcribe.py` in the code repo).
- **Latency:** the encoder looks 0 ms ahead; the head looks 160 ms (4 frames) ahead.
- **Neural Engine placement:** ≥ 99.6 % of the estimated cost runs on the Neural Engine
  (`CPU_AND_NE`), measured for the faithful fp16 build. Only integer mask ops run on the CPU.
- **Conversion changes**, all exact in fp32:
  - dense banded attention instead of HF's blocked attention (the Neural Engine compiler rejects the blocked form);
  - a numerically stable log-softmax;
  - static shapes.
- `config.json` holds every input, output, window, feature and decoder setting in machine-readable form.

## Intended use

- **Intended:** offline or chunked transcription of English speech on Macs with Apple silicon
  ({{MIN_OS}} or later), for example local dictation, note-taking and research on on-device ASR.
- **Out of scope:** other languages, speaker identification, and high-stakes uses (medical,
  legal, safety-critical) without your own evaluation.
- **Prohibited uses:** anything the
  [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) forbids.

## Files

{{FILES_TABLE}}

Total: **{{TOTAL_SIZE}}**. `SHA256SUMS` lists every file; check them with `shasum -a 256 -c SHA256SUMS`.

## Usage

Requirements (versions used for the results below): Python 3.12, `coremltools==9.0`,
`transformers==5.19.0`, `numpy==2.5.4`, `sentencepiece==0.2.2`, `soundfile==0.14.0`,
`ctc-beam-decoder==0.2.0`, `huggingface_hub`.

```python
import json
import numpy as np
import soundfile as sf
import coremltools as ct
import sentencepiece as spm
from ctc_beam_decoder import BeamSearchDecoder
from huggingface_hub import snapshot_download
from transformers import AutoFeatureExtractor

root = snapshot_download("{{REPO_ID}}")
cfg = json.load(open(f"{root}/config.json"))

# 1. Mel features (CPU), exactly as in training.
fe = AutoFeatureExtractor.from_pretrained(root)          # preprocessor_config.json
wav, sr = sf.read("speech.flac", dtype="float32")         # 16 kHz mono
assert sr == 16000 and wav.ndim == 1
f = fe([wav], sampling_rate=16000, padding="longest", truncation=False, return_tensors="np")
n = int(f["input_features_mask"][0].sum())
mel = np.asarray(f["input_features"][0, :n], dtype=np.float32)

# 2. Encoder + CTC head (Core ML) in the smallest window that fits (<= 35 s here).
win = min((w for w in cfg["windows"] if w["mel_frames"] >= n), key=lambda w: w["mel_frames"])
model = ct.models.MLModel(f"{root}/{win['file']}", compute_units=ct.ComputeUnit.CPU_AND_NE)
x = np.zeros((1, win["mel_frames"], 128), np.float32)
x[0, :n] = mel
out = model.predict({"mel": x, "valid_len": np.array([n], np.int32)})
log_probs = np.asarray(out["log_probs"], np.float32)[0, : -(-n // 4)]   # [frames, 257]

# 3. Beam search + subword 6-gram: every piece is an LM "word" t<id>.
d = cfg["decoder"]
sp = spm.SentencePieceProcessor(model_file=f"{root}/{cfg['vocab']['tokenizer']}")
pieces = [f"t{i}" for i in range(sp.get_piece_size())]
dec = BeamSearchDecoder([f"\u2581{p}" for p in pieces], kenlm_model_path=f"{root}/{d['lm']}",
                        unigrams=pieces, alpha=d["alpha"], beta=d["beta"],
                        unk_score_offset=d["unk_score_offset"], score_boundary=d["lm_score_boundary"])
best = dec.decode_beams(log_probs, beam_width=d["beam"], beam_prune_logp=d["beam_prune_logp"],
                        token_min_logp=d["token_min_logp"], prune_history=d["prune_history"])[0][0]
print(" ".join(sp.decode([int(w[1:]) for w in best.split()]).lower().split()))
```

**Compile once.** `ct.models.MLModel("….mlpackage")` compiles the package to a new temporary
`.mlmodelc` on every load. The Neural Engine compile is then repeated each time (about 45 s for
all four windows). Keep the compiled model and load that instead:

```python
import os, shutil
cache = os.path.expanduser(f"~/Library/Caches/{{REPO_NAME}}/v1_{win['name']}.mlmodelc")
if not os.path.isdir(cache):
    shutil.copytree(model.get_compiled_model_path(), cache)
model = ct.models.CompiledMLModel(cache, compute_units=ct.ComputeUnit.CPU_AND_NE)
```

**Long-running processes: keep the input arrays alive.** Core ML releases a prediction's input
arrays later, on its own queue. If Python frees them first, a process that keeps predicting can
crash (SIGSEGV in `MLE5ExecutionStream`).
- Allocate one `mel` / `valid_len` pair per window and keep it for the model's lifetime.
- Overwrite those arrays in place for each call; serialize calls with a lock if you use threads.
- Copy the outputs (`np.array(out["log_probs"], np.float32)`) before the next call.
- `coreml/cmlrun.py` in the code repo does exactly this.

**With the code repo** ({{GITHUB_URL}}): `coreml/transcribe.py` adds pause-based segmentation
for audio longer than 35 s and runs the same decoder. It reads its files from the data root
(`~/gemma_stt`, or `$GEMMA_STT_HOME` once the root is configurable). It also checks the sha256 of
the PyTorch head, the tokenizer and the LM. Link this download into that layout:

```bash
export GEMMA_STT_HOME=~/gemma_stt  ROOT=/path/to/{{REPO_NAME}}
mkdir -p $GEMMA_STT_HOME/coreml/models/{{VARIANT}} $GEMMA_STT_HOME/feats/full $GEMMA_STT_HOME/decode/lm $GEMMA_STT_HOME/runs/full_160ms
ln -s $ROOT/coreml $GEMMA_STT_HOME/coreml/models/{{VARIANT}}/{{PRECISION}}
ln -s $ROOT/tokenizer/spm_unigram256.model $GEMMA_STT_HOME/feats/full/
ln -s $ROOT/lm/sp256_6gram.bin $GEMMA_STT_HOME/decode/lm/
{{HEAD_LINK}}
python coreml/transcribe.py speech.flac --variant {{VARIANT}} --precision {{PRECISION}}
```

## Results

All WERs are corpus-level (word edits / reference words) on LibriSpeech, after lowercasing and
collapsing whitespace. Decoding settings were tuned on dev-clean only. The test sets were decoded
once.

**V1, PyTorch reference** (bf16 encoder features on MPS, CTC head, same decoder and LM):

{{V1_WER_TABLE}}

**This Core ML build, end to end** (audio -> mel -> Core ML -> decoder), test T9:

{{T9_TABLE}}

**Compared with Phonon-2** (FluidAudio `FluidInference/phonon-2-coreml`, also Core ML on the Neural
Engine). Same references and the same corpus scorer. Punctuation except apostrophes is mapped to
spaces; this is a no-op on this model's output.

{{PHONON2_TABLE}}

{{SPEED_SECTION}}

## Limitations

- **Domain:**
  - English read speech (LibriSpeech audiobooks) only.
  - No punctuation, casing or numerals: the output is lowercase words.
  - The LM was trained only on the LibriSpeech LM corpus (public-domain books).
  - Expect higher error rates on conversational, accented, noisy or far-field audio, and on
    vocabulary outside that domain.
  - dev-other and test-other WERs (~10.5 %) show the gap on harder audio.
- **fp16 vs fp32:**
  - The fp16 packages run on the Neural Engine. Their WER matches the fp32 PyTorch reference
    within 0.1 points (T9).
  - Per-frame features are not bit-identical. The layer-11 cosine is 0.99996 on average, with
    rare frames down to ~0.93. As a result, ~5 % of utterances differ in greedy output, at equal
    WER.
  - The investigation found that this is the model's sensitivity to tiny upstream fp16 errors,
    not a failing fp16 op.
  - The fp32 packages (if included) reproduce the reference exactly, but they run on the CPU/GPU,
    not the Neural Engine.
- **Per-frame sensitivity:**
  - The head relies on the int8 round trip. Nine feature dimensions have a training std of
    ~1e-5; they are exactly zero after int8 and explode without it.
  - Feeding un-quantized layer-11 features to the head gives 100 % WER. The packages include the
    round trip, so do not remove it when re-exporting.
- **Fixed windows:** the inputs are padded to 5/10/20/35 s, so a 6 s clip costs as much as a 10 s one.
- **Streaming:** the head is causal with 160 ms lookahead, but these packages are offline windows.
  Chunked streaming with kept decoder state is studied in the code repo (`decode/streaming.py`).
- **Bias:** inherited from LibriSpeech (a limited set of speakers and accents, audiobook style) and
  from the base model. Not evaluated for fairness across demographic groups.

## License and attribution

- **License:** the Core ML packages contain weights derived from `google/embeddinggemma-2`.
  - This repository is distributed under, and subject to, the
    [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
  - By using these files you agree to those terms and to the
    [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
  - The packages are a modified version of the base model; see `NOTICE` for what was changed.
- **Base model:** [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2), revision
  `{{MODEL_REVISION}}`. `preprocessor_config.json` is copied from it unchanged.
- **CTC head and tokenizer:** trained by {{AUTHOR}} on the LibriSpeech ASR corpus
  ([OpenSLR SLR12](https://www.openslr.org/12/), CC BY 4.0; V. Panayotov, G. Chen, D. Povey,
  S. Khudanpur, "LibriSpeech: an ASR corpus based on public domain audio books", ICASSP 2015).
- **Language model:** built with KenLM from the LibriSpeech LM corpus normalized text
  ([OpenSLR SLR11](https://www.openslr.org/11/), public domain). See `lm/README.md`.
- **Decoder (not included):** [ctc-beam-decoder](https://pypi.org/project/ctc-beam-decoder/) 0.2.0,
  Apache License 2.0.
- The full notice is in `NOTICE`.
