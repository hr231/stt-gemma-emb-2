# demo: transcribe, embed and search audio with one encoder pass

`demo/embed_and_transcribe.py` is a command-line tool built on `embed/onepass.py` (`OnePass`).
For each audio file the EmbeddingGemma 2 audio tower runs **once**:

- `audio_tower.layers[11]` goes through the V1 transcript path: int8 round trip, standardize, the 13M-param
  causal CTC head, then ctc-beam-decoder with the subword 6-gram LM (beam 32, alpha 0.8, beta 1.0).
- the tower's final output goes through EmbeddingGemma 2's own audio embedding path: embed_audio,
  language_model, mean pooling, projection, L2 normalization. The result is a 768-d vector.
  Audio gets no prompt prefix.

Text queries are embedded with the documented SearchQuery prefix `task: search result | query: `
(model card, https://huggingface.co/google/embeddinggemma-2). The stored vectors are exactly what the
official API returns: L2-normalized in bf16, so their norm is 1 ± about 1e-3. `search` renormalizes
index and query vectors in fp32 before taking dot products, so the score is exact cosine similarity.
The search compares the text query with the **audio**
embeddings. Transcripts are only stored and displayed; they play no part in ranking.

Use V1's environment. Commands run from the repo root:

```
P=~/.venv-gemma-stt/bin/python
$P demo/embed_and_transcribe.py transcribe FILE [FILE ...] --out DIR
$P demo/embed_and_transcribe.py index FOLDER --db PATH [--force]
$P demo/embed_and_transcribe.py search "text query" ["another query" ...] --db PATH [-k 5]
```

Common options: `--device mps|cpu` (default `mps`) and `--dtype bfloat16|float32` (default `bfloat16`;
float16 is unsafe for this model). `search` uses the device and dtype stored in the index.
Progress and model-load messages go to stderr. Results go to stdout.

## Commands

**transcribe**: takes any file soundfile can read (WAV, FLAC, OGG/Opus, MP3, AIFF, CAF, ...).
Multi-channel audio is averaged to mono. Audio that is not at 16 kHz is resampled with a band-limited
Hann-windowed sinc filter (the same kernel as torchaudio's default resampler, written in torch
because torchaudio is not installed). Each file `<name>` produces three outputs in `--out`:

| file | content |
| --- | --- |
| `<name>.txt` | transcript (V1, lower case) |
| `<name>.npy` | float32 [768] embedding, as OnePass returns it (bf16-normalized, norm 1 ± ~1e-3; renormalize in fp32 for exact cosine) |
| `<name>.json` | file, transcript, word count, embedding dim and norm, duration, input sample rate and channels, resampled/downmixed flags, timings (audio load, one-pass total, and OnePass's own split: mel+tower, head, decode, embedding wait), real-time factor, model metadata (model id + revision, pooling, normalization, prompts, dtype/device, V1 head/LM names and sha256, decoder settings) |

If two inputs have the same name, the later ones get `-2`, `-3`, ... suffixes.

**index**: walks FOLDER recursively and follows symlinks. Each audio file gets one pass, and the
results go into a single `.npz` file. The write is atomic, with a checkpoint every 25 new files.
The file holds these arrays:

| array | dtype / shape |
| --- | --- |
| `paths` | str [N], absolute paths as found (symlinks not resolved) |
| `vectors` | float32 [N, 768] |
| `transcripts` | str [N] |
| `durations` | float64 [N] seconds |
| `mtimes_ns`, `sizes` | int64 [N], stat of the file (of the symlink target) |
| `meta` | JSON string: index format, created/updated, roots, n_files, OnePass meta (model id, revision, prompts, pooling, dtype, device, max_embed_seconds) and the V1 transcriber settings |

Load it with `np.load(db)` (no pickle needed).

Re-indexing is incremental:
- a file whose mtime and size are unchanged is skipped;
- new or changed files are embedded;
- indexed files under FOLDER that no longer exist are removed.

Several folders can share one db. If the model id, revision or dtype differ from the ones the
index was built with, the index is rebuilt. `--force` re-embeds every file. The model is only
loaded when something needs to be embedded.

**search**: embeds one or more queries in a single model load and prints the top-k files with
cosine score, duration, path relative to the indexed folder, and a transcript snippet. When the
query occurs verbatim in a transcript, the snippet is centred on the match. Long cells are
truncated so every row fits on one line (120 columns when not writing to a terminal).

Limits:
- The embedding covers the first 30 s of each file, the same as the official feature extractor
  (`OnePass(max_embed_seconds=30)`). The tool prints a note for longer files.
- The transcript always covers the whole file.
- The V1 head was trained on LibriSpeech utterances (at most about 35 s), so expect worse
  transcripts on much longer recordings.

## Test: 40 dev-clean files

Folder: `~/gemma_stt/embed/demo/folder/<spk>/<chap>/<id>.flac` holds **symlinks** into
`~/gemma_stt/data/LibriSpeech/dev-clean` (no copies). There is one utterance from each of 40 chapters:
- chapters shuffled with `random.Random(0)`;
- for each chapter, one utterance drawn from those lasting 6-20 s;
- 6.8 min of audio in total.

The ids and reference sentences are in `~/gemma_stt/embed/demo/folder_refs.json`. Outputs are in
`~/gemma_stt/embed/demo/` (`transcribe_out/`, `index.npz`, `search_examples.txt`). Logs are in
`~/gemma_stt/logs/EmbedDemo_*.log`.

The timings shown below are ordinary wall-clock numbers from runs on a shared machine. They are not
benchmark results. The first file of each run also includes warm-up.

### transcribe

The third input is `2035-147960-0005` resampled to 44.1 kHz and converted to stereo (WAV PCM16).
It exercises the downmix and resample path. It gives the same transcript as the original, and its
embedding has cosine 0.983 with the original's embedding.

```
$ $P demo/embed_and_transcribe.py transcribe folder/2035/147960/2035-147960-0005.flac \
      folder/1919/142785/1919-142785-0005.flac resample_test/snake_44k_stereo.wav --out transcribe_out
file              dur s  time s  transcript
----------------  -----  ------  ---------------------------------------------------------------------------------------
2035-147960-0005    6.6    2.08  i whirled round and there on one of those dry gravel beds was the biggest snake i ha...
1919-142785-0005   18.9    0.64  mode parent slace a cucumbers as for the table sprinkle well withth salt and let the...
snake_44k_stereo    6.6    0.25  i whirled round and there on one of those dry gravel beds was the biggest snake i ha...

3 file(s) -> ~/gemma_stt/embed/demo/transcribe_out/<name>.{txt,npy,json}
```

### index (first run, then incremental)

```
$ $P demo/embed_and_transcribe.py index ~/gemma_stt/embed/demo/folder --db ~/gemma_stt/embed/demo/index.npz
~/gemma_stt/embed/demo/folder: 40 audio file(s): 40 to embed, 0 unchanged, 0 removed from index      (stderr)
[ 1/40] 1272/135031/1272-135031-0000.flac   10.9 s audio   4.34 s  because you are as sleeping in soul of conqueri...
...
[40/40] 8842/304647/8842-304647-0009.flac   12.0 s audio   0.33 s  organ of rot not reason is the lord who from th...
index   ~/gemma_stt/embed/demo/index.npz
folder  ~/gemma_stt/embed/demo/folder
files   40 indexed (6.8 min of audio); this run: 40 embedded, 0 unchanged, 0 removed
time    17.5 s for 406.5 s of audio (model load excluded)

$ (same command again: nothing changed, the model is not loaded)
files   40 indexed (6.8 min of audio); this run: 0 embedded, 40 unchanged, 0 removed

$ (after adding a symlink to 2035-147960-0006.flac)
[1/1] 2035/147960/2035-147960-0006.flac    3.3 s audio   1.51 s  i know i am just awful jem i was so scared
files   41 indexed (6.8 min of audio); this run: 1 embedded, 40 unchanged, 0 removed

$ (after deleting that symlink again)
files   40 indexed (6.8 min of audio); this run: 0 embedded, 40 unchanged, 1 removed
```

### search

The example queries are:
- a phrase from one transcript;
- a paraphrase whose words do not occur in the target transcript ("they felt my pulse they looked at
  my tongue they listened at my chest ...");
- a topic word (the folder holds three recipe clips);
- a second paraphrase.

```
$ $P demo/embed_and_transcribe.py search "the biggest snake i had ever seen" \
      "the doctors examined my body to check whether i was healthy" "cooking" \
      "a woman pricked her finger on a spindle" --db ~/gemma_stt/embed/demo/index.npz -k 5
index: ~/gemma_stt/embed/demo/index.npz  (40 files, 768-d, query prefix 'task: search result | query: ')

query: "the biggest snake i had ever seen"
#  score  dur s  file                               transcript
-  -----  -----  ---------------------------------  --------------------------------------------------------------------
1  0.773    6.6  2035/147960/2035-147960-0005.flac  ...gravel beds was the biggest snake i had ever seen
2  0.569   13.3  1272/141231/1272-141231-0002.flac  the cut on his chest still dripping blood the ache of his over st...
3  0.564    6.1  2803/154328/2803-154328-0005.flac  god who reads our hearts knows that we had a noble end in view
4  0.562    9.6  5694/64038/5694-64038-0004.flac    we saw the united states flag flying for the ramparts and thought...
5  0.561   12.1  2412/153954/2412-153954-0018.flac  they felt my pulse they looked at my tongue they listened at my c...

query: "the doctors examined my body to check whether i was healthy"
#  score  dur s  file                               transcript
-  -----  -----  ---------------------------------  --------------------------------------------------------------------
1  0.706   12.1  2412/153954/2412-153954-0018.flac  they felt my pulse they looked at my tongue they listened at my c...
2  0.686   10.4  84/121123/84-121123-0024.flac      the district doctor approached with the indifference of a man acc...
3  0.662   12.0  8842/304647/8842-304647-0009.flac  organ of rot not reason is the lord who from the body politic dot...
4  0.650    8.7  2412/153947/2412-153947-0009.flac  i also wrote about this time the substance of what ultimately bec...
5  0.627    6.1  2803/154328/2803-154328-0005.flac  god who reads our hearts knows that we had a noble end in view

query: "cooking"
#  score  dur s  file                               transcript
-  -----  -----  ---------------------------------  --------------------------------------------------------------------
1  0.674    9.1  652/129742/652-129742-0006.flac    cauliflower mayonnaise take cold boiled cauliflower break into br...
2  0.664   16.7  2078/142845/2078-142845-0031.flac  cut as many nice even slices as may be required rather more than...
3  0.630   18.9  1919/142785/1919-142785-0005.flac  mode parent slace a cucumbers as for the table sprinkle well with...
4  0.612   12.1  2412/153954/2412-153954-0018.flac  they felt my pulse they looked at my tongue they listened at my c...
5  0.590   13.2  174/168635/174-168635-0003.flac    he suffered all the pangs of a mother and he knew not what it mea...

query: "a woman pricked her finger on a spindle"
#  score  dur s  file                               transcript
-  -----  -----  ---------------------------------  --------------------------------------------------------------------
1  0.784    9.7  6319/64726/6319-64726-0005.flac    she had no sooner taken up the spindle then being hasty and carel...
2  0.656   10.9  1272/135031/1272-135031-0000.flac  because you are as sleeping in soul of conquering the lovely rose...
3  0.617   12.1  2412/153954/2412-153954-0018.flac  they felt my pulse they looked at my tongue they listened at my c...
4  0.614   11.3  3536/23268/3536-23268-0016.flac    miss woodley obediently sat down and though her thoughts and hear...
5  0.608   12.0  8842/304647/8842-304647-0009.flac  organ of rot not reason is the lord who from the body politic dot...
```

Results:
- The phrase query and both paraphrases put the correct clip first.
- "cooking" ranks the three recipe clips 1-3. One of them has a poor V1 transcript
  ("mode parent slace a cucumbers"), which does not hurt it because ranking uses the audio embedding.
- Sanity check (`~/gemma_stt/logs/EmbedDemo_refquery_check.log`): each file's LibriSpeech reference
  sentence used as a query retrieves its own file at rank 1 for **40/40** files.
