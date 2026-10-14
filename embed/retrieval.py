"""Text<->audio retrieval on 500 seeded dev-clean clips using EmbedCore's OnePass.

python embed/retrieval.py  -> ~/gemma_stt/embed/retrieval/ + ~/gemma_stt/embed/REPORT_retrieval.md
"""
import json, random, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "embed"), str(REPO / "coreml"), str(REPO / "decode")]
G = Path.home() / "gemma_stt"
OUT = G / "embed" / "retrieval"
REPORT = G / "embed" / "REPORT_retrieval.md"
LIBRI = G / "data" / "LibriSpeech"
CACHED_HYPS = G / "decode/results/sweep_dev-clean_cbd_sp6_beam32_units-piece/hyps.jsonl"
N, SEED = 500, 0


def flac(uid):
    spk, chap, _ = uid.split("-")
    return LIBRI / "dev-clean" / spk / chap / f"{uid}.flac"


def metrics(sim):  # sim[q, d]; correct doc for query i is i
    order = np.argsort(-sim, axis=1)
    rank = np.argmax(order == np.arange(len(sim))[:, None], axis=1)  # 0-based
    r = {f"R@{k}": float(np.mean(rank < k)) for k in (1, 5, 10)}
    r["MRR"] = float(np.mean(1.0 / (rank + 1)))
    return r, rank, order


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    utts = json.load(open(G / "decode/logprobs/dev-clean.json"))["utts"]
    utts = random.Random(SEED).sample(utts, N)
    ids = [u["id"] for u in utts]
    refs = [u["ref"] for u in utts]
    (OUT / "ids.txt").write_text("\n".join(ids) + "\n")

    import soundfile as sf
    from common import norm_text
    from onepass import OnePass
    import onepass as op_mod

    t0 = time.time()
    op = OnePass()
    t_load = time.time() - t0
    hyps, aemb, t_clip, dur = [], [], [], 0.0
    for i, uid in enumerate(ids):
        wav, sr = sf.read(flac(uid), dtype="float32")
        assert sr == 16000
        dur += len(wav) / sr
        t = time.time()
        r = op.transcribe_and_embed(wav)
        t_clip.append(time.time() - t)
        hyps.append(r["text"])
        aemb.append(np.asarray(r["embedding"], dtype=np.float32).reshape(-1))
        if i % 50 == 0:
            print(i, f"{t_clip[-1]:.2f}s", r["text"][:60], flush=True)
    A = np.stack(aemb)
    t = time.time()
    Qa = np.asarray(op.embed_queries(refs), dtype=np.float32)
    Qb = np.asarray(op.embed_queries(hyps), dtype=np.float32)
    t_q = time.time() - t

    def nrm(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    A, Qa, Qb = nrm(A), nrm(Qa), nrm(Qb)
    np.save(OUT / "audio_emb.npy", A); np.save(OUT / "q_ref.npy", Qa); np.save(OUT / "q_hyp.npy", Qb)
    with open(OUT / "onepass_hyps.jsonl", "w") as f:
        for uid, h in zip(ids, hyps):
            f.write(json.dumps({"id": uid, "hyp": h}) + "\n")

    # agreement with V1 cached hyps
    cached = {}
    for line in open(CACHED_HYPS):
        d = json.loads(line)
        cached[d.get("id") or d.get("utt")] = d.get("hyp") or d.get("text")
    same = [norm_text(h) == norm_text(cached.get(u, "")) for u, h in zip(ids, hyps)]
    diffs = [(u, cached.get(u), h) for u, h, s in zip(ids, hyps, same) if not s]
    from wer import corpus_wer
    try:
        w_one = corpus_wer([norm_text(r) for r in refs], [norm_text(h) for h in hyps])
        w_cache = corpus_wer([norm_text(r) for r in refs], [norm_text(cached.get(u, "")) for u in ids])
    except Exception as e:  # signature mismatch: report but continue
        w_one = w_cache = f"n/a ({e})"

    rows = {}
    ra, rank_a, ord_a = metrics(Qa @ A.T); rows["text->audio, (a) reference"] = ra
    rb, rank_b, ord_b = metrics(Qb @ A.T); rows["text->audio, (b) V1 one-pass hyp"] = rb
    rows["audio->text, (a) reference"] = metrics(A @ Qa.T)[0]
    rows["audio->text, (b) V1 one-pass hyp"] = metrics(A @ Qb.T)[0]
    json.dump({"rows": rows, "rank_ref": rank_a.tolist(), "rank_hyp": rank_b.tolist()},
              open(OUT / "metrics.json", "w"), indent=1)

    L = ["# Text -> audio retrieval, EmbeddingGemma 2 one-pass (500 dev-clean clips)", "",
         f"Clips: {N} sampled with `random.Random({SEED}).sample(utts, {N})` from dev-clean.json "
         f"(ids: `{OUT}/ids.txt`). Index = one-pass audio embeddings; cosine similarity.", "",
         "| direction / query set | R@1 | R@5 | R@10 | MRR |", "|---|---|---|---|---|"]
    for k, v in rows.items():
        L.append(f"| {k} | {v['R@1']:.3f} | {v['R@5']:.3f} | {v['R@10']:.3f} | {v['MRR']:.3f} |")
    L += ["", "## One-pass transcripts vs V1 cached beam+LM hyps", "",
          f"Exact agreement (norm_text): {sum(same)}/{N}. WER one-pass {w_one}, cached {w_cache}.", ""]
    for u, c, h in diffs[:10]:
        L += [f"- `{u}` cached: {c}", f"  one-pass: {h}"]
    L += ["", "## Failures (query set (a): correct clip not in top 10)", ""]
    fails = [i for i in np.argsort(-rank_a) if rank_a[i] >= 10][:5]
    S = Qa @ A.T
    off = S.copy(); np.fill_diagonal(off, -np.inf)
    margin = np.diag(S) - off.max(1)  # correct-clip cosine minus best wrong-clip cosine
    L.append(f"Cosine of correct pair: mean {np.diag(S).mean():.3f}, min {np.diag(S).min():.3f}; "
             f"best wrong clip mean {off.max(1).mean():.3f}; margin min {margin.min():.3f}.")
    L.append("")
    if not fails:
        L.append("None: every reference query found its clip in the top 10. "
                 "Instead, the 5 queries with the smallest margin over the best wrong clip:")
        fails = list(np.argsort(margin)[:5])
    for i in fails:
        L.append(f"- `{ids[i]}` rank {rank_a[i] + 1}; query: {refs[i]}")
        for j in ord_a[i][:3]:
            L.append(f"  - top: `{ids[j]}` {refs[j]}")
    L += ["", "## Runtime (not slot-timed; machine may be shared)", "",
          f"Model load {t_load:.1f}s; audio {dur:.0f}s total; one-pass per clip mean "
          f"{np.mean(t_clip):.3f}s, total {sum(t_clip):.1f}s (RTF {sum(t_clip) / dur:.3f}); "
          f"query embedding of {2 * N} texts {t_q:.1f}s.", "",
          "## Settings", "",
          f"OnePass from `embed/onepass.py`; query prefix: "
          f"`{getattr(op_mod, 'QUERY_PREFIX', getattr(op, 'query_prefix', 'see onepass.py'))}`; "
          f"meta: `{json.dumps(getattr(op, 'meta', {}), default=str)}`; audio gets no prefix; "
          f"V1 head + beam32/LM decoder inside OnePass (bf16 MPS).",
          "Reproduce: `~/.venv-gemma-stt/bin/python embed/retrieval.py`."]
    REPORT.write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
