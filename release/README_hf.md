---
language:
- en
license: apache-2.0
library_name: coreml
pipeline_tag: automatic-speech-recognition
base_model: google/embeddinggemma-2
datasets:
- openslr/librispeech_asr
tags:
- speech-recognition
- streaming
- ctc
- coreml
- apple-neural-engine
- kenlm
- embeddinggemma
model-index:
- name: stt-gemma_embedding-2
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: LibriSpeech (clean)
      type: openslr/librispeech_asr
      config: clean
      split: test
      args:
        language: en
    metrics:
    - type: wer
      value: 4.18
      name: Test WER (Core ML fp16, beam search + subword 6-gram LM)
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: LibriSpeech (other)
      type: openslr/librispeech_asr
      config: other
      split: test
      args:
        language: en
    metrics:
    - type: wer
      value: 10.71
      name: Test WER (Core ML fp16, beam search + subword 6-gram LM)
---

# stt-gemma_embedding-2

**English speech recognition for the Apple Neural Engine, built by training a 13M-parameter CTC head on top of the frozen audio encoder of Google's [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2).**

stt-gemma_embedding-2 keeps EmbeddingGemma 2's 305M-parameter audio encoder completely frozen and trains only a small causal head that turns the encoder's frames into text. The encoder never looks at future audio and the head looks ahead by just 160 ms, so the model can transcribe while someone is speaking: in streaming mode, words appear a median of **207 ms** after they are spoken. Exported to Core ML in fp16, it runs on the Neural Engine. Offline, the full pipeline (encoder, head, and beam search with a language model) transcribes at **233× real time** on a MacBook Pro with an M4 Pro; in streaming mode, each 160 ms step takes 5.6 ms of compute.

| | test-clean | test-other | Streaming latency (median / p90) | Offline speed (M4 Pro) |
|---|---|---|---|---|
| **stt-gemma_embedding-2, beam search + subword 6-gram LM** | **4.18%** | **10.71%** | 207 / 287 ms | **233× real time** |

Word error rate on LibriSpeech, corpus-level, of the released Core ML package. Latency is from the end of a spoken word to its final text on screen, measured on dev-clean. See [Evaluation protocol](#evaluation-protocol) for how every number was measured.

## Why this model

Many fast on-device speech recognizers either work on fixed windows of audio, which rules out true streaming, or train a large encoder from scratch. This model takes a different route: it reuses an audio encoder that Google has already trained at scale, freezes it, and trains only what is needed to read text out of it. The encoder is EmbeddingGemma 2's audio encoder (a Gemma 4 audio tower, `gemma4_audio`), which also makes it possible to produce a transcript and a search embedding from a single encoder pass.

The trainable part is small (13.3M parameters), it was trained on a single laptop, and it runs on the Neural Engine, leaving the GPU free for other work.

## Results

### Accuracy

WER in percent. Decoder settings were tuned on dev-clean only; the test sets were each evaluated once with the final settings.

| Pipeline | dev-clean | dev-other | test-clean | test-other |
|---|---|---|---|---|
| Reference (PyTorch), greedy, no LM | 5.92 | 15.47 | 6.30 | 15.61 |
| Reference (PyTorch), beam 32 + subword 6-gram LM | 3.73 | 10.47 | 4.14 | 10.70 |
| **This release (Core ML fp16), beam 32 + subword 6-gram LM** | **3.74** | **10.50** | **4.18** | **10.71** |

The released Core ML package reproduces the reference beam + LM results to within 0.04 points (dev-clean +0.011, dev-other +0.026, test-clean +0.038, test-other +0.009). It gives the same dev-clean WER under all three Core ML compute settings: 3.737% on the Neural Engine, 3.746% on the GPU, 3.731% on the CPU only.

### Streaming

The streaming build processes 160 ms of audio per step and keeps the encoder's and head's state between steps; the decoder re-decodes the open segment after every step. Its feature extraction is bitwise identical to the offline pipeline, and 100% of its operations run on the Neural Engine. The streaming build is not part of this repository yet; it lives in the code repository.

| Split | Latency median / p90 | WER, streaming | WER, reference | Transcripts identical to reference | Revised words |
|---|---|---|---|---|---|
| dev-clean | 207 / 287 ms | 3.731% | 3.726% | 2,617 / 2,703 | 4.6% |
| dev-other | 207 / 288 ms | 10.479% | 10.470% | 2,561 / 2,864 | 10.2% |

Latency is measured from the end of a word in the audio to the moment its final text is displayed, and includes the head's 160 ms lookahead, the wait for each 160 ms chunk, and compute. Every decoded word is shown as soon as it appears; "revised words" counts displayed words that are later replaced by a different word as more audio arrives (a word growing while it is spelled out does not count). Each 160 ms step takes 5.62 ms of compute (median; p99 6.03 ms), about 3.5% of real time.

### Speed

Offline transcription of this package, measured on a MacBook Pro (M4 Pro, 24 GB) on dev-clean, single stream, after a 20-file warm-up, with model loading excluded.

| Component | Hardware | Real-time factor |
|---|---|---|
| End to end (16 kHz audio in memory → text) | Neural Engine + CPU | 233.3× |
| Encoder + head (Core ML, fp16) | Neural Engine | 253.9× |
| Beam search + language model | CPU | 8,452× |

| Compute units | End to end | Peak memory | Load, first / later |
|---|---|---|---|
| CPU_AND_NE (recommended) | 233.3× | 3.3 GB | 54.6 s / 3.3 s |
| CPU_AND_GPU | 194.2× | 4.7 GB | 11.3 s / 3.5 s |
| CPU_ONLY | 111.7× | 5.2 GB | 7.5 s / 2.7 s |

The first load on the Neural Engine includes the device compile; later loads reuse the compiled copy.

### Comparison with Phonon-2

For context, we ran [FluidInference/phonon-2-coreml](https://huggingface.co/FluidInference/phonon-2-coreml), a strong Core ML speech recognizer for Apple silicon, on the same Mac, scored it with the same scorer, and timed it with the same method.

| Model | test-clean | test-other | Offline speed (M4 Pro) |
|---|---|---|---|
| Phonon-2 (FluidAudio, Neural Engine) | **2.47%** | **4.66%** | 187× (test-clean) |
| stt-gemma_embedding-2 (beam + LM, Neural Engine) | 4.18% | 10.71% | **233×** (dev-clean) |

Phonon-2 is clearly more accurate, especially on harder audio. This table shows the trade-off rather than claiming better accuracy: stt-gemma_embedding-2 streams with about 0.2 s of latency and is somewhat faster offline, at a higher error rate. Phonon-2 was run with FluidAudio at commit 90c1b07e; because it outputs punctuation and casing, punctuation other than apostrophes was mapped to spaces before scoring.

## How it works

```
16 kHz mono audio
  → log-mel features (128 bins, 100 frames per second)
  → EmbeddingGemma 2 audio encoder, frozen
      12 layers, 305M parameters, causal (no right context)
  → output of the 12th layer: 1024 numbers per frame, 25 frames per second
  → causal CTC head, trained
      4 blocks, 13.3M parameters, 160 ms lookahead
  → scores for 256 SentencePiece tokens + a CTC blank, every 40 ms
  → CTC beam search with a subword 6-gram language model
  → text
```

The head first rounds each encoder frame to int8 (matching how the training features were cached), standardizes it with statistics from the training set, and projects it from 1024 to 512 dimensions. Each of its four blocks then applies masked self-attention (8 heads, up to 64 past frames, or 2.56 s), a causal depthwise convolution (kernel size 5) and a 2048-wide feed-forward layer, each with a residual connection. Only the last block's attention may see future frames, 4 of them (160 ms), so the head's total lookahead is exactly 160 ms; this was checked when the head was trained and again on the Core ML package. The encoder itself has no lookahead: changing the audio after any point never changes its output before that point.

The output is CTC over a 256-piece SentencePiece unigram vocabulary. A beam search ([ctc-beam-decoder](https://pypi.org/project/ctc-beam-decoder/), a C++ CTC prefix beam search that reproduces pyctcdecode's behaviour) combines the head's scores with a KenLM 6-gram over the same SentencePiece pieces.

The Core ML package is the offline build: one package with shared weights and four functions for input windows of 5, 10, 20 and 35 s (`win_5s` … `win_35s`); longer audio is split at pauses.

## Usage

Requirements: a Mac with Apple silicon and macOS 15 or later, Python 3.12, `coremltools==9.0`, `transformers==5.19.0`, `numpy==2.5.4`, `sentencepiece==0.2.2`, `soundfile==0.14.0`, `ctc-beam-decoder==0.2.0`, `huggingface_hub`.

```python
import json, os
import numpy as np
import soundfile as sf
import coremltools as ct
import sentencepiece as spm
import transformers
from ctc_beam_decoder import BeamSearchDecoder
from huggingface_hub import snapshot_download

root = snapshot_download("harshit2312/stt-gemma_embedding-2")
cfg = json.load(open(f"{root}/config.json"))

# 1. Mel features (CPU), exactly as in training.
fe_cfg = cfg["feature_extractor"]
fe = getattr(transformers, fe_cfg["class"]).from_dict(dict(fe_cfg["preprocessor_config"]))
wav, sr = sf.read("speech.flac", dtype="float32")         # 16 kHz mono
assert sr == 16000 and wav.ndim == 1
f = fe([wav], sampling_rate=16000, padding="longest", truncation=False, return_tensors="np")
n = int(f["input_features_mask"][0].sum())
mel = np.asarray(f["input_features"][0, :n], dtype=np.float32)

# 2. Compile the package once and keep the compiled model.
compiled = os.path.expanduser("~/Library/Caches/stt-gemma_embedding-2/v1.mlmodelc")
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

**Compile once.** `ct.models.MLModel("v1.mlpackage")` compiles the package to a new temporary `.mlmodelc` on every load. Compile it once, as in step 2, and load the compiled model with `CompiledMLModel`; one `.mlmodelc` holds all four functions, picked with `function_name=`.

**Long-running processes: keep the input arrays alive.** Core ML releases a prediction's input arrays later, on its own queue; if Python frees them first, a process that keeps predicting can crash. Allocate one `mel` / `valid_len` pair per window, keep it for the model's lifetime, overwrite it in place for each call (float32 / int32; fp16 inputs are cached by Core ML), serialize calls with a lock if you use threads, and copy the outputs before the next call.

## Files

| File | Contents |
|---|---|
| `v1.mlpackage` | Core ML package: frozen encoder + trained head, fp16, functions `win_5s`, `win_10s`, `win_20s`, `win_35s` with shared weights |
| `spm_unigram256.model` | SentencePiece unigram tokenizer, 256 pieces (blank = id 256) |
| `sp256_6gram.bin` | KenLM 6-gram over the tokenizer's pieces, 8-bit quantized trie (762 MB) |
| `config.json` | Windows → functions, inputs / outputs, feature extractor settings, decoder settings (beam 32, α = 0.8, β = 1, pruning), minimum OS |
| `LICENSE`, `NOTICE` | Apache License 2.0; upstream attribution, modifications and data sources |
| `SHA256SUMS` | SHA-256 of every file (`shasum -a 256 -c SHA256SUMS`) |

## Training

**Data.** LibriSpeech 960 h (train-clean-100, train-clean-360 and train-other-500; 281,241 utterances), with lowercased transcripts.

**Features.** The frozen encoder was run once over all training audio (in bf16 on the GPU), and the output of its 12th layer was cached on disk as per-frame int8 (about 91 GB for all splits); the 961 h of training audio took 4.1 hours, at 234× real time.

**Head.** 15 epochs on the cached features with CTC loss, AdamW (learning rate 1e-3, weight decay 1e-4), 1,000 warmup steps followed by cosine decay, dropout 0.1, and batches of up to 12,000 frames. Training took about 11 hours on one MacBook Pro M4 Pro (24 GB). The released head is from epoch 15, the best on dev-clean (5.92% greedy WER).

**Language model.** A KenLM 6-gram over SentencePiece pieces, trained only on the OpenSLR `librispeech-lm-norm` corpus (no dev or test transcripts; checked by searching the corpus for dev and test sentences) and quantized to 8 bits.

**Decoder settings.** Beam width 32, LM weight α = 0.8 and word bonus β = 1, tuned on dev-clean only.

## Evaluation protocol

WER is corpus-level: total word edits divided by total reference words. Before scoring, text is lowercased and whitespace is collapsed (for Phonon-2, punctuation other than apostrophes was also mapped to spaces). Offline speed is the audio duration divided by the processing time, with 16 kHz samples already in memory, model loading excluded, a 20-file warm-up, and a single stream. Streaming latency runs each utterance through the streaming build 160 ms at a time and measures, for every word, the time from the word's end in the audio (from the CTC alignment of the final transcript) to the moment the displayed text up to that word stops changing.

## Limitations

This release is English only and was trained only on read audiobook speech. Expect noticeably higher error rates on conversational, accented, noisy or far-field audio; its test-other error rate is about 2.3 times Phonon-2's, so the gap to stronger models widens on harder audio. In streaming mode, harder audio also means more on-screen corrections: about 10% of displayed words are revised on dev-other, against 4.6% on dev-clean. Names, numbers and rare words are weak points, since both the tokenizer and the language model come from book text, and the output is lowercase without punctuation. The language model needs about 0.8 GB of memory, so the default setup targets desktop Macs rather than phones. The model should not be used for high-stakes transcription (legal, medical or similar) without human review.

## License and attribution

This model includes, and depends on, the audio encoder of [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2), redistributed under that model's Apache 2.0 license; deployments must also follow the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). The trained head, tokenizer, language model and code are released under the license declared above. Training data: [LibriSpeech](https://www.openslr.org/12) (CC BY 4.0). Language-model text: the LibriSpeech LM corpus ([OpenSLR 11](https://www.openslr.org/11), public domain). The decoder, ctc-beam-decoder (Apache-2.0), links KenLM (LGPL) dynamically.

This project is independent and is not affiliated with or endorsed by Google or FluidInference.

## Citation

```bibtex
@misc{stt_gemma_embedding_2_2026,
  title  = {stt-gemma\_embedding-2: streaming CTC speech recognition on a frozen EmbeddingGemma 2 audio encoder},
  author = {Harshit B.},
  year   = {2026},
  url    = {https://huggingface.co/harshit2312/stt-gemma_embedding-2}
}

@inproceedings{panayotov2015librispeech,
  title     = {Librispeech: an {ASR} corpus based on public domain audio books},
  author    = {Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle = {2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {5206--5210},
  year      = {2015}
}

@inproceedings{graves2006ctc,
  title     = {Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks},
  author    = {Graves, Alex and Fern{\'a}ndez, Santiago and Gomez, Faustino and Schmidhuber, J{\"u}rgen},
  booktitle = {Proceedings of the 23rd International Conference on Machine Learning},
  pages     = {369--376},
  year      = {2006}
}

@inproceedings{heafield2011kenlm,
  title     = {{KenLM}: Faster and smaller language model queries},
  author    = {Heafield, Kenneth},
  booktitle = {Proceedings of the Sixth Workshop on Statistical Machine Translation},
  pages     = {187--197},
  year      = {2011}
}
```
