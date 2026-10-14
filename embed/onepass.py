#!/usr/bin/env python3
"""V1 transcript + EmbeddingGemma 2 audio embedding from ONE audio-tower pass (PyTorch, MPS, bf16).

  ~/.venv-gemma-stt/bin/python embed/onepass.py AUDIO [AUDIO ...]        # prints text + embedding head

One clip = one run of `audio_tower`:
  * a forward hook on audio_tower.layers[11] (the tower's LAST conformer layer; it returns a bare
    tensor) gives V1's tap -> int8 round trip (featstore) -> (x - mean) / std -> V1 head (CPU fp32)
    -> log_softmax fp32 -> ctc-beam-decoder + subword 6-gram LM with V1_DECODER;
  * the tower's own output (after output_proj, 1536-d) -> embed_audio -> language_model (with the
    <bos><|audio> ... <audio|><eos> token frame the official processor builds) -> mean pooling over
    all tokens -> L2 normalize: the 768-d embedding `SentenceTransformer("google/embeddinggemma-2")
    .encode(audio)` returns (see ~/gemma_stt/embed/REPORT_core.md for the code references).

Audio gets no prompt. Text queries get QUERY_PREFIX, documents DOCUMENT_PREFIX (model card prompts).
Like the official feature extractor (truncation=True, max_length=480000), the embedding only sees
the first 30 s of longer audio (max_embed_seconds=None embeds everything); the transcript always
uses all of it. For such audio the last embedded tower frame is recomputed from a 256-frame tail
window (the only extra tower work; see _embed_audio_tokens). Embeddings are float32 numpy,
L2-normalized in bf16 exactly like the official API (norm 1 +- ~0.4%); the official similarity is
cosine (it renormalizes), so renormalize in fp32 before using a dot product as cosine.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coreml"))
import common  # noqa: E402  (also puts the repo and repo/decode on sys.path)
from common import MODEL_ID, MODEL_REVISION, SR  # noqa: E402

QUERY_PREFIX = "task: search result | query: "     # prompt_name "SearchQuery" / "query"
DOCUMENT_PREFIX = "title: none | text: "            # prompt_name "Document" / "document"
EMBEDDING_DIM = 768
OFFICIAL_AUDIO_MAX_SAMPLES = 480_000                # Gemma4AudioFeatureExtractor.__call__ default (30 s)
TAIL_TOKENS = 256                                   # recompute window for the truncated last frame


class _StopTower(Exception):
    """Raised from the layers[11] hook to skip output_proj when only the transcript is wanted."""


class OnePass:
    def __init__(self, device="mps", dtype=torch.bfloat16, load_language_model=True,
                 max_embed_seconds=30.0, check_v1=True):
        """load_language_model=False keeps only the audio tower (transcript path; for cost.py)."""
        from transformers import AutoModel, AutoTokenizer
        if check_v1:
            common.check_v1_files()
        t0 = time.time()
        self.device = torch.device(device)
        self.dtype = dtype
        model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=dtype)
        drop = ["vision_tower", "embed_vision"]
        if not load_language_model:
            drop += ["language_model", "embed_audio"]
        for name in drop:
            if getattr(model, name, None) is not None:
                delattr(model, name)
        self.model = model.to(self.device).eval()
        self.tower = self.model.audio_tower
        self.has_lm = load_language_model
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION) if load_language_model else None
        self.head, self.ck = common.load_head_cpu()
        self.decoder = common.beam_decoder()
        self.fe = common.feature_extractor()
        self.max_embed_samples = None if max_embed_seconds is None else int(round(max_embed_seconds * SR))
        if self.max_embed_samples is not None and self.max_embed_samples > OFFICIAL_AUDIO_MAX_SAMPLES:
            raise ValueError("max_embed_seconds above 30 s no longer matches the official API; use None")
        if load_language_model:
            cfg = self.model.config
            self._frame_ids = torch.tensor([[self.tok.bos_token_id, cfg.boa_token_id]]), \
                torch.tensor([[cfg.eoa_token_index, self.tok.eos_token_id]])
        self.meta = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "embedding_dim": EMBEDDING_DIM,
                     "pooling": "mean", "normalized": True, "similarity": "cosine",
                     "query_prefix": QUERY_PREFIX, "document_prefix": DOCUMENT_PREFIX, "audio_prefix": None,
                     "dtype": str(dtype).replace("torch.", ""), "device": str(self.device),
                     "max_embed_seconds": max_embed_seconds}
        self.load_s = time.time() - t0

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _wav(wav):
        if isinstance(wav, (str, Path)):
            return common.load_audio(wav)
        w = np.asarray(wav, dtype=np.float32)
        if w.ndim != 1:
            raise ValueError(f"expected a 1-D 16 kHz mono waveform, got shape {w.shape}")
        return w

    def _valid_mel(self, n_samples):
        """Valid mel frames for n_samples (processing_embedding_gemma2._compute_audio_num_tokens)."""
        fl, hop = self.fe.frame_length, self.fe.hop_length
        return max((n_samples + fl // 2 - (fl + 1)) // hop + 1, 0)

    @staticmethod
    def _tokens(n_valid_mel):
        """Tower output frames: two stride-2 subsamplings of the mask (mask[:, ::2] twice)."""
        return -(-(-(-n_valid_mel // 2)) // 2)

    def _tower(self, wav, stop_at_l11=False):
        """Mel (V1's exact call, truncation=False) and one tower run. Returns (l11 [1, T_pad, 1024]
        model dtype, tower output [1, T_pad, 1536] or None, n_valid_mel, mel float32 [T_mel, 128])."""
        m, n_valid = common.mel(wav)
        x = torch.from_numpy(m)[None].to(self.device)          # fp32, the tower casts (as the processor)
        mask = (torch.arange(m.shape[0]) < n_valid)[None].to(self.device)
        store = {}

        def hook(mod, inp, out):
            store["l11"] = out[0] if isinstance(out, tuple) else out
            if stop_at_l11:
                raise _StopTower

        h = self.tower.layers[11].register_forward_hook(hook)
        try:
            with torch.no_grad():
                out = self.tower(x, attention_mask=mask)
            last = out.last_hidden_state
        except _StopTower:
            last = None
        finally:
            h.remove()
        return store["l11"], last, n_valid, m

    def _logprobs(self, l11, n_valid):
        """V1 transcript path on the tap: int8 round trip, standardize, head, log_softmax fp32 (CPU)."""
        L = self._tokens(n_valid)
        x = l11[0, :L].float().cpu().numpy()
        return common.head_logprobs(self.head, self.ck, common.int8_roundtrip(x))

    @torch.no_grad()
    def _embed_audio_tokens(self, tower_out, m, n_samples, n_valid):
        """Tower output -> embed_audio -> LM -> mean pool -> normalize (async on the device)."""
        L = self._tokens(n_valid)
        if self.max_embed_samples is not None and n_samples > self.max_embed_samples:
            # The official features stop at max_embed_samples (mel frames there == our first frames).
            # The tower is causal (attention context right = 0, causal LightConv1d), so the truncated
            # run's frames equal our first L frames except the last one, whose 3x3 stride-2
            # subsampling convs see padding instead of the next audio. That frame is recomputed from a
            # TAIL_TOKENS window of the truncated mel (start on the 4-mel grid; the receptive field is
            # 12 layers x (12 attention + 4 conv) = 192 frames, so the window start has no effect).
            v = self._valid_mel(self.max_embed_samples)
            L = self._tokens(v)
            s = max(L - TAIL_TOKENS, 0)
            x = torch.from_numpy(m[4 * s:v])[None].to(self.device)
            tail = self.tower(x, attention_mask=torch.ones(x.shape[:2], dtype=torch.bool,
                                                            device=self.device)).last_hidden_state
            tower_out = torch.cat([tower_out[:, :L - 1], tail[:, L - 1 - s:L - s]], dim=1)
        soft = self.model.embed_audio(tower_out[:, :L])                     # [1, L, 512]
        emb_tokens = self.model.language_model.get_input_embeddings()
        pre, post = (emb_tokens(t.to(self.device)) for t in self._frame_ids)
        inputs = torch.cat([pre, soft.to(pre.dtype), post], dim=1)          # == masked_scatter result
        attn = torch.ones(inputs.shape[:2], dtype=torch.long, device=self.device)
        tok = self.model.language_model(inputs_embeds=inputs, attention_mask=attn,
                                        return_dict=True).last_hidden_state
        return self._pool(tok, attn)

    @staticmethod
    def _pool(tok, attn):
        """sentence_transformers Pooling(mean, include_prompt=True) + Normalize, in the model dtype."""
        mask = attn.unsqueeze(-1).expand_as(tok).to(tok.dtype)
        mean = (tok * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return F.normalize(mean, p=2, dim=-1)

    def _require_lm(self):
        if not self.has_lm:
            raise RuntimeError("OnePass(load_language_model=False) only transcribes")

    # ------------------------------------------------------------------ public API
    def transcribe_and_embed(self, wav):
        """{'text', 'embedding' float32 [768] (L2-normalized in bf16 like the official API: norm 1 +- ~0.4%),
        'logprobs' float32 [T, 257], 'timings' (s)}."""
        self._require_lm()
        t = {}
        t0 = time.perf_counter()
        wav = self._wav(wav)
        l11, last, n_valid, m = self._tower(wav)
        l11_cpu = l11[0, :self._tokens(n_valid)].float().cpu()               # waits for the tower only
        t1 = time.perf_counter()
        emb = self._embed_audio_tokens(last, m, len(wav), n_valid)          # queued on the GPU
        t2 = time.perf_counter()
        lp = common.head_logprobs(self.head, self.ck, common.int8_roundtrip(l11_cpu.numpy()))
        t3 = time.perf_counter()
        text = common.norm_text(self.decoder.decode(lp))
        t4 = time.perf_counter()
        emb = emb[0].float().cpu().numpy()
        t5 = time.perf_counter()
        t.update(audio_s=len(wav) / SR, mel_tower_s=t1 - t0, embed_enqueue_s=t2 - t1, head_s=t3 - t2,
                 decode_s=t4 - t3, embed_wait_s=t5 - t4, total_s=t5 - t0)
        return {"text": text, "embedding": emb, "logprobs": lp, "timings": t}

    def transcribe(self, wav):
        """Transcript path only: the tower stops at layers[11] (its last layer; output_proj skipped)."""
        wav = self._wav(wav)
        l11, _, n_valid, _ = self._tower(wav, stop_at_l11=True)
        return common.norm_text(self.decoder.decode(self._logprobs(l11, n_valid)))

    def embed_audio(self, wav):
        """float32 [768] normalized as transcribe_and_embed; no prompt (audio takes no prefix)."""
        self._require_lm()
        wav = self._wav(wav)
        _, last, n_valid, m = self._tower(wav)
        return self._embed_audio_tokens(last, m, len(wav), n_valid)[0].float().cpu().numpy()

    @torch.no_grad()
    def _embed_texts(self, texts, prefix, batch_size=32):
        self._require_lm()
        texts = [prefix + t for t in ([texts] if isinstance(texts, str) else list(texts))]
        order = np.argsort([-len(s) for s in texts], kind="stable")          # as SentenceTransformer.encode
        out = np.zeros((len(texts), EMBEDDING_DIM), np.float32)
        for i in range(0, len(texts), batch_size):
            idx = order[i:i + batch_size]
            enc = self.tok([texts[j] for j in idx], padding=True, truncation="longest_first",
                           return_tensors="pt").to(self.device)
            tok = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
            out[idx] = self._pool(tok, enc["attention_mask"]).float().cpu().numpy()
        return out

    def embed_queries(self, texts, prompt=QUERY_PREFIX, batch_size=32):
        """float32 [N, 768], unit norm; prompt defaults to the SearchQuery prefix."""
        return self._embed_texts(texts, prompt, batch_size)

    def embed_documents(self, texts, prompt=DOCUMENT_PREFIX, batch_size=32):
        """float32 [N, 768], unit norm; 'title: none | text: ' (pass prompt='' for preformatted titles)."""
        return self._embed_texts(texts, prompt, batch_size)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    op = OnePass(device=args.device)
    print(f"[load] {op.load_s:.1f} s", flush=True)
    for path in args.audio:
        r = op.transcribe_and_embed(path)
        print(f"{path}\n  text: {r['text']}\n  embedding[:6]: {np.round(r['embedding'][:6], 4).tolist()} "
              f"norm {np.linalg.norm(r['embedding']):.4f}\n  timings: "
              f"{ {k: round(v, 4) for k, v in r['timings'].items()} }", flush=True)


if __name__ == "__main__":
    main()
