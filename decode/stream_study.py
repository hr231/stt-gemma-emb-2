#!/usr/bin/env python3
"""Streaming decoding study on saved log-probs (offline, dev-clean): every utterance is fed chunk by
chunk (default 4 frames = 160 ms) through a decoder that keeps its search state between chunks.

  ~/.venv-pyctcdecode/bin/python decode/stream_study.py [--split dev-clean] [--chunk 4] [--limit N]

Decoders (settings fixed on dev-clean): beam = V1 (pyctcdecode algorithm = ctc-beam-decoder, piece
units, subword 6-gram, beam 32, alpha 0.8, beta 1, prune -10 / -5); fusion = greedy + subword LM
fusion (dec_subword, beam 1, alpha 0.5, beta 1.5); greedy = argmax, no LM.

Per decoder:
  final WER of the streamed result vs decoding each utterance at once (identical outputs expected;
  for beam also vs V1's own ctc-beam-decoder outputs);
  changed = share of shown words that later change: after chunk k the display D_k is compared with the
  next display; words past their common prefix were shown and then changed (a later word revision
  invalidates everything after it, as on screen);
  latency = per final word, (time of the first chunk after which the display's prefix up to and
  including that word never changes again) - (the word's end: exclusive end frame of its last token in
  a Viterbi CTC alignment of the final hypothesis, x 40 ms). The acoustic model's own 160 ms lookahead
  is not included.
Display policies: all = every decoded word, including one still being spelled; hold = all but the
last word until the utterance ends.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dec_pyctc import PyctcDecoder  # noqa: E402
from dec_subword import SubwordBeamDecoder  # noqa: E402
from streaming import StreamingBeam, StreamingGreedy, StreamingSubword, ctc_align_ends  # noqa: E402
from wer import corpus_wer, norm_text  # noqa: E402

HOME = Path.home()
LOGPROBS = HOME / "gemma_stt/decode/logprobs"
SP6 = HOME / "gemma_stt/decode/lm/sp256_6gram.bin"
RESULTS = HOME / "gemma_stt/decode/results/streaming"
V1_HYPS = HOME / "gemma_stt/decode/results/sweep_dev-clean_cbd_sp6_beam32_units-piece/hyps.jsonl"
FRAME_S = 0.04


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def word_ends(ids, tok_ends, pieces, sp):
    """Words of a token sequence (grouped at pieces starting with U+2581) and the end frame of each;
    None if the grouping disagrees with sp.decode (e.g. <unk> inside a word)."""
    groups = []
    for i, e in zip(ids, tok_ends):
        if not groups or pieces[i].startswith("\u2581"):
            groups.append([[i], e])
        else:
            groups[-1][0].append(i)
            groups[-1][1] = e
    words, ends = [], []
    for g, e in groups:
        w = sp.decode(g).lower().strip()
        if w:
            words.append(w)
            ends.append(e)
    return (words, ends) if words == sp.decode(ids).lower().split() else None


def policy_stats(displays, final, T, chunk, ends):
    """(shown, changed, revised, latencies in s, words settled only at the end) for one utterance.
    revised = changed minus the changes where only the last shown word grew (a word still being
    spelled: 'tw' -> 'twelve')."""
    K = len(displays)
    seq = displays + [final]
    shown = sum(len(seq[k]) - lcp(seq[k - 1] if k else [], seq[k]) for k in range(K + 1))
    changed = revised = 0
    for k in range(K):
        a, b = seq[k], seq[k + 1]
        n = lcp(a, b)
        changed += len(a) - n
        grew = n == len(a) - 1 and n < len(b) and b[n] != a[n] and b[n].startswith(a[n])
        revised += 0 if grew else len(a) - n
    times = [min(chunk * (k + 1), T) * FRAME_S for k in range(K)] + [T * FRAME_S]
    last_bad = [-1] * len(final)                  # last step whose display differs up to word i
    for k in range(K):
        n = lcp(displays[k], final)
        for i in range(n, len(final)):
            last_bad[i] = k
    lat, at_end = [], 0
    if ends is not None:
        for i, e in enumerate(ends):
            k = last_bad[i] + 1                   # first step from which word i never changes
            at_end += k == K
            lat.append(times[k] - e * FRAME_S)
    return shown, changed, revised, lat, at_end


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--chunk", type=int, default=4, help="frames per chunk (40 ms each)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.split.startswith("test"):
        raise SystemExit("streaming study runs on dev splits only")

    idx = json.loads((LOGPROBS / f"{args.split}.json").read_text())
    arr = np.load(LOGPROBS / f"{args.split}.npy", mmap_mode="r")
    vocab, blank, spm_model = idx["vocab"], idx["blank"], idx["spm_model"]
    sp = spm.SentencePieceProcessor(model_file=spm_model)
    pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    utts = idx["utts"][: args.limit] if args.limit else idx["utts"]
    v1 = {r["id"]: r["hyp"] for r in map(json.loads, V1_HYPS.read_text().splitlines())}

    beam_w = PyctcDecoder(vocab, blank, spm_model, lm=SP6, alpha=0.8, beta=1.0, beam=32, units="piece",
                          fix_break=False)
    fusion = SubwordBeamDecoder(blank, spm_model, lm=SP6, alpha=0.5, beta=1.5, beam=1)
    greedy = StreamingGreedy(blank)
    decoders = {
        "beam (V1: beam 32, alpha 0.8, beta 1)": (StreamingBeam(beam_w), beam_w.decode),
        "greedy + LM fusion (beam 1, alpha 0.5, beta 1.5)": (StreamingSubword(fusion), fusion.decode),
        "greedy, no LM": (greedy, None),
    }

    def words(ids):
        return sp.decode(ids).lower().split()

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, (s, full_decode) in decoders.items():
        t0 = time.time()
        refs, finals, fulls, same_full, same_v1 = [], [], [], 0, 0
        acc = {p: {"shown": 0, "changed": 0, "revised": 0, "lat": [], "at_end": 0} for p in ("all", "hold")}
        bad_groups, hyp_words = 0, 0
        for u in utts:
            lp = np.asarray(arr[u["offset"]:u["offset"] + u["frames"]], dtype=np.float32)
            T = len(lp)
            s.reset()
            disp = []
            for a in range(0, T, args.chunk):
                s.accept(lp[a:a + args.chunk])
                disp.append(words(s.partial_ids()))
            fids = s.final_ids()
            final = words(fids)
            text = " ".join(final)
            hyp_words += len(final)
            if full_decode is None:
                g = StreamingGreedy(blank)
                g.accept(lp)
                full = " ".join(words(g.final_ids()))
            else:
                full = norm_text(full_decode(lp))
            refs.append(norm_text(u["ref"]))
            finals.append(text)
            fulls.append(full)
            same_full += text == full
            same_v1 += text == v1.get(u["id"])
            ends = ctc_align_ends(lp, fids, blank)
            we = None if ends is None else word_ends(fids, ends, pieces, sp)
            bad_groups += we is None
            wend = None if we is None else we[1]
            for pol, dd in (("all", disp), ("hold", [d[:-1] for d in disp])):
                sh, ch, rv, lat, at_end = policy_stats(dd, final, T, args.chunk, wend)
                a_ = acc[pol]
                a_["shown"] += sh
                a_["changed"] += ch
                a_["revised"] += rv
                a_["lat"] += lat
                a_["at_end"] += at_end
        wer_s, e_s, w_s = corpus_wer(refs, finals)
        wer_f, e_f, _ = corpus_wer(refs, fulls)
        res = {"utts": len(utts), "chunk_ms": args.chunk * 40, "final_wer": wer_s, "final_edits": e_s, "words": w_s,
               "full_wer": wer_f, "full_edits": e_f, "identical_to_full": same_full,
               "identical_to_v1_ctc_beam_decoder": same_v1, "utts_without_word_alignment": bad_groups,
               "hyp_words": hyp_words, "seconds": round(time.time() - t0, 1)}
        for pol, a_ in acc.items():
            lat = np.asarray(a_["lat"]) * 1000
            res[pol] = {"shown": a_["shown"], "changed": a_["changed"], "changed_share": a_["changed"] / a_["shown"],
                        "revised": a_["revised"], "revised_share": a_["revised"] / a_["shown"],
                        "latency_words": len(lat), "latency_ms_median": float(np.median(lat)),
                        "latency_ms_p90": float(np.percentile(lat, 90)), "latency_ms_mean": float(lat.mean()),
                        "settled_only_at_end": a_["at_end"], "negative_latency_words": int((lat < 0).sum())}
        summary[name] = res
        print(f"[{name}] {json.dumps(res)}", flush=True)

    tag = f"{args.split}_chunk{args.chunk * 40}ms" + (f"_n{args.limit}" if args.limit else "")
    (RESULTS / f"{tag}.json").write_text(json.dumps(summary, indent=1))
    lines = [f"# Streaming decoding, {args.split}, {args.chunk * 40} ms chunks, state kept between chunks", "",
             "| decoder | display | final WER % (streamed) | full-utterance WER % | identical to full | "
             "shown words later changed | of which revised (not just the spelled word growing) | "
             "latency median ms | latency p90 ms | settled only at end |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in summary.items():
        for pol in ("all", "hold"):
            p = r[pol]
            lines.append(f"| {name} | {pol} | {100 * r['final_wer']:.3f} | {100 * r['full_wer']:.3f} | "
                         f"{r['identical_to_full']}/{r['utts']} | {100 * p['changed_share']:.2f}% "
                         f"({p['changed']}/{p['shown']}) | {100 * p['revised_share']:.2f}% ({p['revised']}) | "
                         f"{p['latency_ms_median']:.0f} | {p['latency_ms_p90']:.0f} | "
                         f"{p['settled_only_at_end']}/{p['latency_words']} |")
    (RESULTS / f"{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
