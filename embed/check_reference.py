#!/usr/bin/env python3
"""Check embed/onepass.py against the official EmbeddingGemma 2 API and against V1.

  ~/.venv-embed/bin/python embed/check_reference.py [--n 50] [--seed 0] [--no-fp32]

(~/.venv-embed = V1's torch/transformers via a .pth file + sentence-transformers 6.1.0.)
Clips: --n seeded dev-clean utterances (ids saved) plus every dev-clean clip longer than 30 s
(the official feature extractor truncates those; reported separately).
Per clip:
  * embedding: cosine(OnePass, SentenceTransformer bf16 on MPS) -- pass if >= 0.999 for every clip;
    also vs SentenceTransformer fp32 on CPU (bf16 noise floor), and embed_audio() == one-pass vector;
  * official token frame: processor input_ids == <bos><|audio>  N x <|audio|>  <audio|><eos> with N
    the number of tower frames OnePass pools;
  * layers[11]: the one-pass run's tap vs a separate tower run (cache_feats.run_taps style);
  * transcript: one-pass beam+LM text vs V1's cached dev-clean beam+LM hyps (identical count, WER),
    transcribe() == one-pass text, log-probs vs V1's saved dev-clean log-probs.
Text: embed_queries / embed_documents vs SentenceTransformer.encode(prompt_name=SearchQuery/Document).
Writes ~/gemma_stt/embed/check_ids.json, check_reference.json, check_reference_clips.jsonl.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onepass import DOCUMENT_PREFIX, QUERY_PREFIX, OnePass  # noqa: E402
import common  # noqa: E402  (onepass put coreml/ on sys.path)
from common import LOGPROBS, MODEL_ID, MODEL_REVISION, SR, audio_path, greedy_text, norm_text  # noqa: E402
from wer import corpus_wer  # noqa: E402

OUT = Path.home() / "gemma_stt/embed"
V1_HYPS = Path.home() / "gemma_stt/decode/results/sweep_dev-clean_cbd_sp6_beam32_units-piece/hyps.jsonl"
PASS_COS = 0.999
QUERIES = ["what did mister quilter write about", "a story about a ship at sea", "the price of bread",
           "who was the king", "letters from a friend", "how to cook eggs"]


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def stats(v):
    v = np.asarray(v, np.float64)
    return {"n": int(len(v)), "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())} if len(v) else {}


def load_st(device, dtype):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device=device, model_kwargs={"dtype": dtype})


def st_audio(st, wav):
    return st.encode([{"array": wav, "sampling_rate": SR}], batch_size=1, convert_to_numpy=True)[0]


def tower_l11(op, wav):
    """A separate tower run, as cache_feats.run_taps: hook layers[11], audio_tower(mel, mask)."""
    m, n_valid = common.mel(wav)
    store = {}
    h = op.tower.layers[11].register_forward_hook(
        lambda mod, inp, out: store.__setitem__("x", out[0] if isinstance(out, tuple) else out))
    try:
        x = torch.from_numpy(m)[None].to(op.device, op.dtype)
        mask = (torch.arange(x.shape[1]) < n_valid)[None].to(op.device)
        with torch.no_grad():
            out = op.tower(x, attention_mask=mask)
    finally:
        h.remove()
    L = int(out.attention_mask.sum())
    return store["x"][0, :L].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-long", action="store_true", help="skip the dev-clean clips longer than 30 s")
    ap.add_argument("--no-fp32", action="store_true", help="skip the CPU fp32 official reference")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import AutoProcessor

    index = json.loads((LOGPROBS / "dev-clean.json").read_text())
    utts = index["utts"]
    rng = np.random.default_rng(args.seed)
    pick = sorted(int(i) for i in rng.choice(len(utts), size=args.n, replace=False))
    sample = [dict(utts[i], set="sample") for i in pick]
    long = [] if args.no_long else [dict(u, set="long") for u in utts if u["dur"] > 30.0]
    (OUT / "check_ids.json").write_text(json.dumps(
        {"split": "dev-clean", "seed": args.seed, "n": args.n, "rule": "np.random.default_rng(seed).choice("
         "len(utts), n, replace=False) over decode/logprobs/dev-clean.json utts, sorted",
         "ids": [u["id"] for u in sample], "long_ids": [u["id"] for u in long]}, indent=1))
    v1_hyp = {r["id"]: r["hyp"] for r in map(json.loads, V1_HYPS.read_text().splitlines())}
    v1_lp = np.load(LOGPROBS / "dev-clean.npy", mmap_mode="r")

    t0 = time.time()
    op = OnePass()
    print(f"[load] OnePass {op.load_s:.1f} s", flush=True)
    st = load_st("mps", torch.bfloat16)
    proc = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    st_info = {"modules": [type(m).__module__ + "." + type(m).__name__ for m in st],
               "max_seq_length": st.max_seq_length, "prompts_used": {k: st.prompts.get(k) for k in
                                                                       ("SearchQuery", "Document", "query", "document")},
               "default_prompt_name": st.default_prompt_name, "similarity_fn_name": st.similarity_fn_name,
               "pooling": st[1].get_config_dict(), "model_dtype": str(next(st[0].model.parameters()).dtype)}
    assert st.prompts["SearchQuery"] == QUERY_PREFIX and st.prompts["Document"] == DOCUMENT_PREFIX, st.prompts
    print(f"[load] official SentenceTransformer bf16/mps {time.time() - t0:.1f} s total; {st_info}", flush=True)

    # capture the one-pass run's own layers[11] output and tower mask (first tower run of the call;
    # clips over 30 s add a short tail run for the embedding)
    cap = {"l11": [], "mask": []}
    op.tower.layers[11].register_forward_hook(lambda mod, inp, out: cap["l11"].append(out))
    op.tower.register_forward_hook(lambda mod, inp, out: cap["mask"].append(out.attention_mask))

    rows = []
    for k, u in enumerate(sample + long):
        wav = common.load_audio(audio_path("dev-clean", u["id"]))
        cap["l11"].clear()
        cap["mask"].clear()
        r = op.transcribe_and_embed(wav)
        n_tower = int(cap["mask"][0].sum())
        l11_one = cap["l11"][0][0, :n_tower].float().cpu().numpy()
        l11_sep = tower_l11(op, wav)
        e_api = op.embed_audio(wav)
        text_only = op.transcribe(wav)
        e_off = st_audio(st, wav)
        ids = proc(audio=[wav], padding=True, return_tensors="np")["input_ids"][0].tolist()
        n_audio_tok = ids.count(op.model.config.audio_token_id)
        frame_ok = ids[:2] == op._frame_ids[0][0].tolist() and ids[-2:] == op._frame_ids[1][0].tolist() \
            and len(ids) == n_audio_tok + 4
        n_pool = n_tower if len(wav) <= op.max_embed_samples else op._tokens(op._valid_mel(op.max_embed_samples))
        lp_v1 = np.asarray(v1_lp[u["offset"]:u["offset"] + u["frames"]], np.float32)
        lp = r["logprobs"]
        row = {"id": u["id"], "set": u["set"], "dur": u["dur"], "cos_official_bf16": cos(r["embedding"], e_off),
               "norm": float(np.linalg.norm(r["embedding"])), "norm_official": float(np.linalg.norm(e_off)),
               "embed_audio_equal": bool(np.array_equal(e_api, r["embedding"])),
               "equal_official": bool(np.array_equal(e_off, r["embedding"])),
               "official_audio_tokens": n_audio_tok, "onepass_pooled_tokens": n_pool, "tower_frames": n_tower,
               "tower_frames_formula": op._tokens(common.mel(wav)[1]), "frame_ids_ok": bool(frame_ok),
               "l11_max_abs_diff": float(np.abs(l11_one - l11_sep).max()),
               "l11_equal": bool(np.array_equal(l11_one, l11_sep)),
               "ref": norm_text(u["ref"]), "hyp": r["text"], "v1_hyp": v1_hyp[u["id"]],
               "text_equal_v1": r["text"] == v1_hyp[u["id"]], "transcribe_equal": text_only == r["text"],
               "lp_frames": int(len(lp)), "v1_lp_frames": int(len(lp_v1)),
               "lp_max_abs_diff_v1": float(np.abs(lp - lp_v1).max()) if lp.shape == lp_v1.shape else None,
               "greedy_equal_v1": greedy_text(lp) == greedy_text(lp_v1),
               "timings": r["timings"], "_emb": r["embedding"], "_wav": wav}
        rows.append(row)
        print(f"[{k + 1}/{len(sample) + len(long)}] {u['id']} {u['dur']:.1f}s cos {row['cos_official_bf16']:.6f} "
              f"tok {n_pool}/{n_audio_tok} l11 {'==' if row['l11_equal'] else row['l11_max_abs_diff']} "
              f"v1 {'==' if row['text_equal_v1'] else '!='}", flush=True)

    # text path
    docs = [norm_text(u["ref"]) for u in sample[:10]]
    q_one, q_off = op.embed_queries(QUERIES), st.encode(QUERIES, prompt_name="SearchQuery", convert_to_numpy=True)
    d_one, d_off = op.embed_documents(docs), st.encode(docs, prompt_name="Document", convert_to_numpy=True)
    text_cos = {"queries": stats([cos(a, b) for a, b in zip(q_one, q_off)]),
                "documents": stats([cos(a, b) for a, b in zip(d_one, d_off)])}

    fp32 = None
    if not args.no_fp32:
        del st
        torch.mps.empty_cache()
        st32 = load_st("cpu", torch.float32)
        for row in rows:
            e32 = st_audio(st32, row["_wav"])
            row["cos_official_fp32_cpu"] = cos(row["_emb"], e32)
        fp32 = stats([r["cos_official_fp32_cpu"] for r in rows if r["set"] == "sample"])
        q32 = st32.encode(QUERIES, prompt_name="SearchQuery", convert_to_numpy=True)
        text_cos["queries_vs_fp32_cpu"] = stats([cos(a, b) for a, b in zip(q_one, q32)])

    for row in rows:
        row.pop("_wav")
        row.pop("_emb")
    smp = [r for r in rows if r["set"] == "sample"]
    lng = [r for r in rows if r["set"] == "long"]
    wer1, e1, w1 = corpus_wer([r["ref"] for r in smp], [r["hyp"] for r in smp])
    wer0, e0, w0 = corpus_wer([r["ref"] for r in smp], [r["v1_hyp"] for r in smp])
    summary = {
        "clips": len(smp), "seed": args.seed, "official": "SentenceTransformer bf16 on mps, batch_size=1",
        "st_info": st_info,
        "cos_official_bf16": stats([r["cos_official_bf16"] for r in smp]),
        "pass_cos_ge_0.999_all": all(r["cos_official_bf16"] >= PASS_COS for r in smp),
        "cos_official_fp32_cpu": fp32,
        "long_clips": {"n": len(lng), "cos_official_bf16": stats([r["cos_official_bf16"] for r in lng]),
                       "pass": all(r["cos_official_bf16"] >= PASS_COS for r in lng),
                       "cos_official_fp32_cpu": stats([r["cos_official_fp32_cpu"] for r in lng])
                       if lng and not args.no_fp32 else None},
        "token_count_match": all(r["official_audio_tokens"] == r["onepass_pooled_tokens"] for r in rows),
        "frame_ids_ok": all(r["frame_ids_ok"] for r in rows),
        "tower_frames_formula_ok": all(r["tower_frames"] == r["tower_frames_formula"] for r in rows),
        "embed_audio_equal": all(r["embed_audio_equal"] for r in rows),
        "bitwise_equal_official": {"sample": sum(r["equal_official"] for r in smp),
                                   "long": sum(r["equal_official"] for r in lng)},
        "embedding_norm": stats([r["norm"] for r in rows]),
        "embedding_norm_official": stats([r["norm_official"] for r in rows]),
        "l11_equal_all": all(r["l11_equal"] for r in rows),
        "l11_max_abs_diff": max(r["l11_max_abs_diff"] for r in rows),
        "text_identical_v1": sum(r["text_equal_v1"] for r in smp),
        "transcribe_equal_all": all(r["transcribe_equal"] for r in rows),
        "wer_onepass": wer1, "wer_v1_cached": wer0, "edits_onepass": e1, "edits_v1": e0, "words": w1,
        "long_text_identical_v1": sum(r["text_equal_v1"] for r in lng),
        "lp_frames_match": sum(r["lp_frames"] == r["v1_lp_frames"] for r in rows),
        "lp_max_abs_diff_v1": stats([r["lp_max_abs_diff_v1"] for r in smp if r["lp_max_abs_diff_v1"] is not None]),
        "greedy_identical_v1": sum(r["greedy_equal_v1"] for r in smp),
        "text_cos": text_cos, "wall_s": time.time() - t0,
    }
    (OUT / "check_reference_clips.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (OUT / "check_reference.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
