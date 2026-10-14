"""Shared pieces of the Core ML export of V1: paths, V1 settings, seeded test lists, mel features,
the PyTorch fp32 CPU reference, and V1's decoders. Imported by the reference, export, runner and
tests; works in ~/.venv-gemma-stt (reference, V1's torch) and ~/.venv-coreml (conversion, tests).

Pipeline (V1): 16 kHz audio -> Gemma4AudioFeatureExtractor mel [T_mel, 128] (100 frames/s,
truncation=False) -> audio_tower up to layers[11] (25 frames/s) -> per-frame int8 round trip
(featstore.quantize_rows / dequantize_rows) -> (x - mean) / std, zero beyond the valid length ->
V1 head -> log-probs [T, 257] -> ctc-beam-decoder (subword 6-gram, beam 32, alpha 0.8, beta 1).
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

HOME = Path.home()
REPO = HOME / "emb_gemma2"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "decode"))

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
HEAD = HOME / "gemma_stt/runs/full_160ms/head_F0-0-0-4c.pt"
SPM = HOME / "gemma_stt/feats/full/spm_unigram256.model"
LM = HOME / "gemma_stt/decode/lm/sp256_6gram.bin"
DATA = HOME / "gemma_stt/data/LibriSpeech"
LOGPROBS = HOME / "gemma_stt/decode/logprobs"
OUT = HOME / "gemma_stt/coreml"
V1_SHA256 = {HEAD: "748cd9e5b27861ef19fe48adba458de5c19914b9f1a048dd15d57e632c385911",
             SPM: "50002ddf56a79f19c6598240e684cf065a4a94b1fc88c5aff25b05962cccfe05",
             LM: "381c408f36fb5ea3cbece7eba408deb2b2320f0fe4679fd45d6bd394395fb371"}
V1_DECODER = dict(alpha=0.8, beta=1.0, beam=32, units="piece", beam_prune_logp=-10.0, token_min_logp=-5.0)
V1_WER = {"dev-clean": 0.037259659571339286, "dev-other": 0.10469498312004397,
          "test-clean": 0.041388466219567866, "test-other": 0.10700570467875361}   # V1.md, beam+LM
SR = 16000
MEL_FPS = 100
SUBSAMPLE = 4
WINDOWS = {"5s": 500, "10s": 1000, "20s": 2000, "35s": 3500}        # mel frames per fixed window
LIST_SEED = 0
FAST_N = 200


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def check_v1_files():
    """V1 files must be byte-identical to V1.md."""
    for p, want in V1_SHA256.items():
        got = sha256(p)
        if got != want:
            raise SystemExit(f"{p}: sha256 {got} != V1 {want}")


def norm_text(t):  # probe.norm_text
    return " ".join(t.lower().split())


# ---------------------------------------------------------------- test lists
def _split_utts(split):
    idx = json.loads((LOGPROBS / f"{split}.json").read_text())
    return [{"id": u["id"], "split": split, "dur": u["dur"], "ref": norm_text(u["ref"]),
             "audio": str(audio_path(split, u["id"]))} for u in idx["utts"]]


def audio_path(split, uid):
    spk, chap, _ = uid.split("-")
    return DATA / split / spk / chap / f"{uid}.flac"


def make_lists():
    """FAST = 200 dev-clean utterances (seed 0); FULL = all dev-clean + dev-other;
    RELEASE = all test-clean + test-other. Written once to OUT/lists/*.json."""
    dev = _split_utts("dev-clean")
    fast = sorted(random.Random(LIST_SEED).sample(dev, FAST_N), key=lambda u: u["id"])
    lists = {"fast": fast, "full": dev + _split_utts("dev-other"),
             "release": _split_utts("test-clean") + _split_utts("test-other")}
    d = OUT / "lists"
    d.mkdir(parents=True, exist_ok=True)
    for name, utts in lists.items():
        p = d / f"{name}.json"
        if p.exists():
            if json.loads(p.read_text()) != utts:
                raise SystemExit(f"{p} exists with different content; lists are fixed")
            continue
        p.write_text(json.dumps(utts, indent=0))
    return lists


def load_list(name):
    return json.loads((OUT / "lists" / f"{name}.json").read_text())


# ---------------------------------------------------------------- features
_FE = None


def feature_extractor():
    global _FE
    if _FE is None:
        from transformers import AutoFeatureExtractor
        _FE = AutoFeatureExtractor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    return _FE


def load_audio(path):
    import soundfile as sf
    w, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SR or w.ndim != 1:
        raise ValueError(f"{path}: expected 16 kHz mono, got {sr} Hz {w.shape}")
    return w


def mel(wav):
    """(mel float32 [T_mel, 128], valid mel frames) exactly as cache_feats.py computes them."""
    f = feature_extractor()([wav], sampling_rate=SR, padding="longest", truncation=False, return_tensors="np")
    m = np.asarray(f["input_features"][0], dtype=np.float32)
    return m, int(np.asarray(f["input_features_mask"][0]).sum())


# ---------------------------------------------------------------- reference (PyTorch fp32, CPU)
def load_tower_fp32():
    import torch
    from transformers import AutoModel
    base = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32)
    for name in ("vision_tower", "language_model", "embed_vision"):
        if hasattr(base, name):
            delattr(base, name)
    return base.eval()


def load_head_cpu():
    from eval_head import load_head
    import torch
    model, ck = load_head(HEAD, torch.device("cpu"))
    return model.float().eval(), ck


def tower_l11(base, mel_np, n_valid):
    """L11 frames [T, 1024] fp32 for one utterance (batch 1, no padding), as cache_feats.run_taps."""
    import torch
    store = {}
    h = base.audio_tower.layers[11].register_forward_hook(
        lambda mod, inp, out: store.__setitem__("x", out[0] if isinstance(out, tuple) else out))
    try:
        x = torch.from_numpy(mel_np)[None]
        m = (torch.arange(x.shape[1]) < n_valid)[None]
        with torch.no_grad():
            out = base.audio_tower(x, attention_mask=m)
    finally:
        h.remove()
    L = int(out.attention_mask.sum())
    return store["x"][0, :L].float().numpy()


def int8_roundtrip(x):
    """featstore.quantize_rows -> dequantize_rows, dequantized in fp32."""
    import torch
    from featstore import dequantize_rows, quantize_rows
    q, s = quantize_rows(torch.from_numpy(np.ascontiguousarray(x)))
    return dequantize_rows(q, s, torch.float32).numpy()


def head_logprobs(head, ck, x):
    """(x - mean) / std -> head -> log_softmax, fp32; x [T, 1024] (one utterance, no padding)."""
    import torch
    with torch.no_grad():
        z = (torch.from_numpy(np.ascontiguousarray(x)) - ck["mean"].float()) / ck["std"].float()
        logits = head(z[None], torch.tensor([len(x)]))
        return logits.float().log_softmax(-1)[0].numpy()


# ---------------------------------------------------------------- decoders (V1)
_SP = None


def sp():
    global _SP
    if _SP is None:
        import sentencepiece as spm
        _SP = spm.SentencePieceProcessor(model_file=str(SPM))
    return _SP


def greedy_text(lp, blank=256):
    ids, prev = [], -1
    for k in np.asarray(lp).argmax(1).tolist():
        if k != prev and k != blank:
            ids.append(k)
        prev = k
    return norm_text(sp().decode(ids))


def beam_decoder():
    from dec_cbd import make_decoder
    vocab = [sp().id_to_piece(i) for i in range(sp().get_piece_size())]
    return make_decoder(vocab, len(vocab), str(SPM), lm=str(LM), **V1_DECODER)
