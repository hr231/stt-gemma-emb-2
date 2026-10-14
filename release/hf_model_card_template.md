---
license: {{LICENSE_ID}}
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

# {{MODEL_NAME}}: English speech recognition on Apple silicon (Core ML)

An English speech recognizer for Apple silicon in **one Core ML package**. It reuses the audio tower
of [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) up to layer 11 and adds
a small causal CTC head trained on LibriSpeech 960 h. A KenLM subword 6-gram is used for beam-search
decoding.

- **Package:** `v1.mlpackage`, fp16. It has one function per input window (`win_5s`, `win_10s`,
  `win_20s`, `win_35s`; default `win_10s`), and the functions share one set of weights.
- **Requirements:** {{MIN_OS}} or later.
- **Compute units:** `CPU_AND_NE` (Neural Engine) is recommended. `CPU_AND_GPU` and `CPU_ONLY` are
  supported and tested (see Results).
- **Code, training and evaluation:** {{GITHUB_URL}} (tag `v1`).

## Model description

```
16 kHz audio
  -> log-mel features [T_mel, 128], 100 frames/s (Gemma4AudioFeatureExtractor; settings in config.json)
  -> Gemma audio tower, layers 0-11 (google/embeddinggemma-2 @ {{MODEL_REVISION_SHORT}}), 25 frames/s (40 ms)
  -> per-frame int8 round trip (absmax / 127), as the head saw during training
  -> standardize (training mean / std, 1,024 dims)
  -> 13.3M-parameter causal CTC head: 256 SentencePiece pieces + blank (id 256), 160 ms lookahead
  -> ctc-beam-decoder 0.2.0 + subword 6-gram LM (beam 32, alpha 0.8, beta 1.0)
  -> lowercase text
```

- **Inside the package:** everything from the mel input to the CTC log-probabilities. Feature
  extraction, beam search and the LM run on the CPU, outside Core ML.
- **Inputs** (every function): `mel` float32 `[1, T_mel, 128]`, zero-padded to the function's window;
  `valid_len` int32 `[1]`, the number of valid mel frames.
- **Outputs:** `log_probs` `[1, T_mel/4, 257]` (natural log) and `l11` (the layer-11 features)
  `[1, T_mel/4, 1024]`. Frames past `ceil(valid_len / 4)` are padding.
- **Windows:** `win_5s` / `win_10s` / `win_20s` / `win_35s` take 500 / 1000 / 2000 / 3500 mel frames.
  Use the smallest that fits. Longer audio is cut at pauses, with 2 s of left context per segment
  (`coreml/transcribe.py` in the code repo).
- **Latency:** the encoder looks 0 ms ahead; the head looks 160 ms (4 frames) ahead.
{{PLACEMENT}}
- **Conversion changes**, all exact in fp32:
  - dense banded attention instead of HF's blocked attention (the Neural Engine compiler rejects the
    blocked form);
  - RMSNorm inputs scaled by powers of two, so fp16 norms on the CPU and GPU do not overflow;
  - a numerically stable log-softmax;
  - static shapes.
- `config.json` holds every window, input, output, front-end and decoder setting in machine-readable
  form.

## Intended use

- **Intended:** offline or chunked transcription of English speech on Apple silicon ({{MIN_OS}} or
  later), for example local dictation, note-taking and research on on-device ASR.
- **Out of scope:** other languages, speaker identification, and high-stakes uses (medical,
  legal, safety-critical) without your own evaluation.
- **Prohibited uses:** the upstream model card states that deployments must adhere to the
  [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

## Files

{{FILES_TABLE}}

Total: **{{TOTAL_SIZE}}**. Check the download with `shasum -a 256 -c SHA256SUMS`.

## Usage

Requirements (versions used for the results below): Python 3.12, `coremltools==9.0`,
`transformers==5.19.0`, `numpy==2.5.4`, `sentencepiece==0.2.2`, `soundfile==0.14.0`,
`ctc-beam-decoder==0.2.0`, `huggingface_hub`.

```python
import json, os
import numpy as np
import soundfile as sf
import coremltools as ct
import sentencepiece as spm
import transformers
from ctc_beam_decoder import BeamSearchDecoder
from huggingface_hub import snapshot_download

root = snapshot_download("{{REPO_ID}}")
cfg = json.load(open(f"{root}/config.json"))

# 1. Mel features (CPU), exactly as in training.
fe_cfg = cfg["feature_extractor"]
fe = getattr(transformers, fe_cfg["class"]).from_dict(dict(fe_cfg["preprocessor_config"]))
wav, sr = sf.read("speech.flac", dtype="float32")         # 16 kHz mono
assert sr == 16000 and wav.ndim == 1
f = fe([wav], sampling_rate=16000, padding="longest", truncation=False, return_tensors="np")
n = int(f["input_features_mask"][0].sum())
mel = np.asarray(f["input_features"][0, :n], dtype=np.float32)

# 2. Compile the package once and keep the compiled model (see "Compile once" below).
compiled = os.path.expanduser("~/Library/Caches/{{REPO_NAME}}/v1.mlmodelc")
if not os.path.isdir(compiled):
    os.makedirs(os.path.dirname(compiled), exist_ok=True)
    ct.utils.compile_model(f"{root}/{cfg['package']['file']}", compiled)

# 3. Encoder + CTC head in the smallest window that fits (<= 35 s here).
win = min((w for w in cfg["windows"] if w["mel_frames"] >= n), key=lambda w: w["mel_frames"])
model = ct.models.CompiledMLModel(compiled, compute_units=ct.ComputeUnit.CPU_AND_NE,   # or CPU_AND_GPU / CPU_ONLY
                                  function_name=win["function"])
x = np.zeros((1, win["mel_frames"], 128), np.float32)
x[0, :n] = mel
out = model.predict({"mel": x, "valid_len": np.array([n], np.int32)})
log_probs = np.array(out["log_probs"], np.float32)[0, : -(-n // 4)]   # [frames, 257]

# 4. Beam search + subword 6-gram: every piece is an LM "word" t<id>.
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

**Compile once.** `ct.models.MLModel("v1.mlpackage")` compiles the package to a new temporary
`.mlmodelc` on every load, so the device compile is repeated every time. Compile it once, as in
step 2, and load the compiled model with `CompiledMLModel`. One `.mlmodelc` holds all four
functions; pick one with `function_name=`.

**Long-running processes: keep the input arrays alive.** Core ML releases a prediction's input
arrays later, on its own queue. If Python frees them first, a process that keeps predicting can
crash (SIGSEGV in `MLE5ExecutionStream`).
- Allocate one `mel` / `valid_len` pair per window and keep it for the model's lifetime.
- Overwrite those arrays in place for each call; serialize calls with a lock if you use threads.
- Copy the outputs (`np.array(out["log_probs"], np.float32)`) before the next call.
- `coreml/cmlrun.py` in the code repo does exactly this.

**Full pipeline:** the code repo ({{GITHUB_URL}}) has it. `coreml/transcribe.py` adds pause-based
segmentation for audio longer than 35 s. `coreml/cmlrun.py` (`V1CoreML("release", "fp16", units)`)
handles the compiled-model cache and the input buffers.

## Results

All numbers below come from this package's own test runs in the code repo (`coreml/tests`). WERs are
corpus-level (word edits / reference words) on LibriSpeech, after lowercasing and collapsing
whitespace. Decoding settings were tuned on dev-clean only; the test sets were decoded once. The V1
PyTorch reference is the same model run in PyTorch (bf16 encoder features on MPS), with the same
decoder and LM.

**Development sets** (audio → mel → Core ML → decoder), test T9:

{{T9_TABLE}}

**Speed and accuracy per compute unit**, test T10:

{{T10_TABLE}}

**Test sets, next to Phonon-2**, test T11:

{{T11_TABLE}}

## Language model

`sp256_6gram.bin` is a KenLM binary (trie, 8-bit quantized) over the 256 pieces of
`spm_unigram256.model`. Every piece id `i` is the LM word `t<i>`; decode with piece units, as in the
usage example.
- **Training text:** all 40,418,260 lines of `librispeech-lm-norm.txt.gz`
  ([OpenSLR SLR11](https://www.openslr.org/11/)). Each line was lowercased, whitespace-collapsed and
  encoded with the tokenizer: 1,894,497,415 tokens, 0 `<unk>`. No LibriSpeech transcripts were added.
- **Toolkit:** [KenLM](https://github.com/kpu/kenlm) commit `4cb443e60b7bf2c0ddf3c745378f76cb59e254e5`,
  `MAX_ORDER=6`.
- **N-grams kept (orders 1 to 6):** {{LM_NGRAMS}}
- `--discount_fallback` is needed: the 1-gram Kneser-Ney discounts are undefined with only 256 symbols.

Build command (code repo, `decode/build_subword_lm.py`):

```bash
{{LM_BUILD}}
```

## Limitations

- **Domain:**
  - English read speech (LibriSpeech audiobooks) only.
  - No punctuation, casing or numerals: the output is lowercase words.
  - The LM was trained only on the LibriSpeech LM corpus (books).
  - Expect higher error rates on conversational, accented, noisy or far-field audio, and on
    vocabulary outside that domain. The dev-other and test-other results show the gap on harder audio.
- **fp16:** the package computes in fp16. Its per-frame features are not bit-identical to the fp32
  PyTorch reference. Rare frames deviate more: the model is sensitive to tiny upstream fp16 errors.
  As a result, a few percent of utterances differ in greedy output. The WER tables above measure the
  end-to-end effect for each compute unit.
- **Per-frame sensitivity:**
  - The head relies on the int8 round trip. Nine feature dimensions have a training std of ~1e-5;
    they are exactly zero after int8 and explode without it.
  - Feeding un-quantized layer-11 features to the head gives 100 % WER. The package includes the
    round trip, so do not remove it when re-exporting.
- **Fixed windows:** inputs are padded to 5 / 10 / 20 / 35 s, so a 6 s clip costs as much as a 10 s one.
- **Streaming:** the head is causal with 160 ms lookahead, but the functions process fixed windows.
  The code repo studies chunked streaming with kept decoder state.
- **Bias:** inherited from LibriSpeech (a limited set of speakers and accents, audiobook style) and
  from the base model. Not evaluated for fairness across demographic groups.

## License and attribution

- **This release:** {{LICENSE_TEXT}}. `NOTICE` lists the attribution and the modifications.
- **Base model:** [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) by Google
  DeepMind, revision `{{MODEL_REVISION}}`, licensed under the
  [Apache License 2.0](https://ai.google.dev/gemma/docs/gemma_4_license). The package contains a
  modified version of its audio tower (layers 0-11). `config.json` embeds its
  `preprocessor_config.json` unchanged. The upstream model card states that deployments must adhere to
  the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
- **CTC head and tokenizer:** trained by {{AUTHOR}} on the LibriSpeech ASR corpus,
  [OpenSLR SLR12](https://www.openslr.org/12/) (license stated there: CC BY 4.0). V. Panayotov,
  G. Chen, D. Povey, S. Khudanpur, "LibriSpeech: an ASR corpus based on public domain audio books",
  ICASSP 2015.
- **Language model:** built from `librispeech-lm-norm.txt.gz`, [OpenSLR SLR11](https://www.openslr.org/11/)
  (license stated there: Public domain).
- **Decoder (not included):** [ctc-beam-decoder](https://pypi.org/project/ctc-beam-decoder/) 0.2.0,
  Apache License 2.0.
