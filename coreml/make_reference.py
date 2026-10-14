#!/usr/bin/env python3
"""Fixed test lists and the PyTorch fp32 CPU reference outputs for FAST (run in ~/.venv-gemma-stt).

  ~/.venv-gemma-stt/bin/python coreml/make_reference.py

Writes OUT/lists/{fast,full,release}.json (seed 0; refused if they exist with other content) and
OUT/reference/fast/: mel.npy [sum T_mel, 128], l11.npy [sum T, 1024], logprobs.npy (int8 round trip,
= V1) and logprobs_noint8.npy [sum T, 257], all float32, plus index.json (per utterance: id, offsets,
lengths, reference, greedy and beam+LM transcripts). Checks every L11 length against V1's cache and
compares WER with V1's cached-feature (bf16 MPS) results on the same utterances.
"""
import json
import time

import numpy as np
import torch

from common import (LOGPROBS, OUT, beam_decoder, check_v1_files, greedy_text, head_logprobs,
                    int8_roundtrip, load_audio, load_head_cpu, load_tower_fp32, make_lists, mel, norm_text,
                    tower_l11)
from wer import corpus_wer

RESULTS = OUT.parent / "decode/results"


def main():
    check_v1_files()
    torch.set_num_threads(8)
    fast = make_lists()["fast"]
    cache_frames = {u["id"]: u["frames"] for u in json.loads((LOGPROBS / "dev-clean.json").read_text())["utts"]}
    base = load_tower_fp32()
    head, ck = load_head_cpu()
    dec = beam_decoder()
    mels, l11s, lps, lps0, rows = [], [], [], [], []
    moff = off = 0
    t0 = time.time()
    for i, u in enumerate(fast):
        m, nv = mel(load_audio(u["audio"]))
        x = tower_l11(base, m, nv)
        if len(x) != cache_frames[u["id"]]:
            raise SystemExit(f"{u['id']}: {len(x)} L11 frames, V1 cache has {cache_frames[u['id']]}")
        lp = head_logprobs(head, ck, int8_roundtrip(x))
        lp0 = head_logprobs(head, ck, x)
        rows.append({**u, "mel_off": moff, "mel_frames": len(m), "mel_valid": nv, "off": off, "frames": len(x),
                     "greedy": greedy_text(lp), "beam": norm_text(dec.decode(lp)),
                     "greedy_noint8": greedy_text(lp0)})
        mels.append(m)
        l11s.append(x)
        lps.append(lp)
        lps0.append(lp0)
        moff += len(m)
        off += len(x)
        if (i + 1) % 50 == 0:
            print(f"[ref] {i + 1}/{len(fast)} utterances, {time.time() - t0:.0f} s", flush=True)
    d = OUT / "reference" / "fast"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "mel.npy", np.concatenate(mels))
    np.save(d / "l11.npy", np.concatenate(l11s))
    np.save(d / "logprobs.npy", np.concatenate(lps))
    np.save(d / "logprobs_noint8.npy", np.concatenate(lps0))
    (d / "index.json").write_text(json.dumps({"utts": rows, "torch": torch.__version__,
                                              "seconds": round(time.time() - t0, 1)}, indent=0))

    refs = [r["ref"] for r in rows]
    v1g = {r["id"]: r["hyp"] for r in map(json.loads, (RESULTS / "dev-clean_greedy_none_a0_b0_beam16/hyps.jsonl")
                                           .read_text().splitlines())}
    v1b = {r["id"]: r["hyp"] for r in map(json.loads, (RESULTS / "sweep_dev-clean_cbd_sp6_beam32_units-piece/hyps.jsonl")
                                           .read_text().splitlines())}
    for name, hyps, v1 in (("greedy", [r["greedy"] for r in rows], v1g), ("beam+LM", [r["beam"] for r in rows], v1b)):
        w, e, n = corpus_wer(refs, hyps)
        wv, ev, _ = corpus_wer(refs, [v1[r["id"]] for r in rows])
        same = sum(h == v1[r["id"]] for h, r in zip(hyps, rows))
        print(f"[ref] FAST {name}: reference (fp32 CPU) {100 * w:.3f}% ({e}/{n}) | V1 cached bf16 MPS features "
              f"{100 * wv:.3f}% ({ev}) | identical transcripts {same}/{len(rows)}")
    w0, e0, _ = corpus_wer(refs, [r["greedy_noint8"] for r in rows])
    print(f"[ref] FAST greedy without the int8 round trip: {100 * w0:.3f}% ({e0})")
    print(f"[ref] {len(rows)} utterances, {off} L11 frames, {moff} mel frames, {time.time() - t0:.0f} s -> {d}")


if __name__ == "__main__":
    main()
