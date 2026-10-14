"""pyctcdecode wrapper (reference beam-search decoder) for the SP-256 CTC head.

Interface: make_decoder(vocab, blank, spm_model, lm=None, alpha, beta, beam, **opts) -> obj with
.decode(logp[T, 257]) -> lowercase words joined by single spaces.

Notes on pyctcdecode 0.5.0 behaviour (checked in its source):
- Labels starting with U+2581 make the alphabet BPE; "<unk>" is mapped to "▁⁇▁" (forced word break on
  both sides), which matches sentencepiece's decode of the unk id (" ⁇ ") after whitespace norm.
- Blank must be "" (we append it at index 256 = our blank).
- Input may be logits or log-probs: it re-applies log_softmax (idempotent on log-probs) and clips
  at log(1e-15). Per frame only tokens with logp >= token_min_logp (plus the argmax) are expanded;
  beams worse than best + beam_prune_logp are dropped.
- Beam merge key includes the last emitted label, so beam_width=1 without LM is greedy
  (argmax -> collapse -> drop blank), up to argmax ties -- except for a pyctcdecode bug fixed here
  (opt fix_break=True, default): `force_next_break` (set by labels ending in U+2581, i.e. our bare
  "▁" piece and "▁⁇▁") is ONE variable shared by all candidate expansions of all beams, so a bare
  "▁" that is merely a candidate (logp >= token_min_logp) splits the next word of another beam
  ("jom" -> "jo m") and a real break can be lost. The fix derives the break per beam: the
  beam's open word part is "" (after bare "▁" or at start) or "⁇" (after unk). fix_break=False
  runs stock pyctcdecode. units="piece" is unaffected (no label ends in U+2581).
- With an LM, word-level OOV/partial-word penalties (unk_score_offset) are NOT scaled by alpha, so
  alpha=0 with an LM is not identical to no LM. A unigram list is needed for .bin LMs
  (pyctcdecode auto-reads them only from .arpa); pass opts unigrams=<list | path, one word per
  line>, else for "<stem>.bin" the sidecar "<stem>.unigrams.txt" is used if it exists.

units="word": labels = SP pieces as-is; LM is a word LM (e.g. OpenSLR 3/4-gram lowercase).
units="piece": labels = "▁t<id>" so each piece is an LM "word" (sp256_6gram, words t0..t255); the
  hypothesis "t3 t17 ..." is mapped back to ids and decoded with sentencepiece.
"""
import inspect
import os
import textwrap

import numpy as np
import pyctcdecode.decoder as _pd
import sentencepiece as spm
from pyctcdecode import build_ctcdecoder
from pyctcdecode.alphabet import UNK_TOKEN

_STOCK_BREAK = "elif self._is_bpe and (char[:1] == BPE_TOKEN or force_next_break):"
_FIXED_BREAK = "elif self._is_bpe and (char[:1] == BPE_TOKEN or word_part in _BREAK_PARTS):"


def _fixed_decoder_class():
    src = textwrap.dedent(inspect.getsource(_pd.BeamSearchDecoderCTC._decode_logits))
    if src.count(_STOCK_BREAK) != 1:
        raise RuntimeError("pyctcdecode _decode_logits changed; re-check the break-flag fix")
    ns = dict(vars(_pd), _BREAK_PARTS=("", UNK_TOKEN))
    exec(compile(src.replace(_STOCK_BREAK, _FIXED_BREAK), _pd.__file__, "exec"), ns)
    return type("BeamSearchDecoderCTCFixedBreak", (_pd.BeamSearchDecoderCTC,),
                {"_decode_logits": ns["_decode_logits"]})


_FixedBreak = _fixed_decoder_class()


def _load_unigrams(unigrams):
    if unigrams is None or not isinstance(unigrams, str):
        return unigrams
    with open(unigrams, encoding="utf-8") as f:
        return [w for w in (line.strip() for line in f) if w]


class PyctcDecoder:
    def __init__(self, vocab, blank, spm_model, lm=None, alpha=0.0, beta=0.0, beam=16,
                 units="word", unigrams=None, beam_prune_logp=-10.0, token_min_logp=-5.0,
                 unk_score_offset=-10.0, lm_score_boundary=True, prune_history=True,
                 fix_break=True):
        if blank != len(vocab):
            raise ValueError(f"blank must be the last index ({len(vocab)}), got {blank}")
        if units not in ("word", "piece"):
            raise ValueError(f"units must be 'word' or 'piece', got {units!r}")
        self.units = units
        self.sp = spm.SentencePieceProcessor(model_file=str(spm_model))
        if units == "word":
            labels = list(vocab) + [""]
            # KenLM .bin carries no readable unigram list: use the "<stem>.unigrams.txt" sidecar
            if unigrams is None and lm is not None and str(lm).endswith(".bin"):
                side = str(lm)[: -len(".bin")] + ".unigrams.txt"
                if os.path.exists(side):
                    unigrams = side
        else:
            labels = [f"\u2581t{i}" for i in range(len(vocab))] + [""]
            if unigrams is None and lm is not None:
                unigrams = [f"t{i}" for i in range(len(vocab))]
        self.dec = build_ctcdecoder(
            labels,
            kenlm_model_path=None if lm is None else str(lm),
            unigrams=_load_unigrams(unigrams),
            alpha=float(alpha),
            beta=float(beta),
            unk_score_offset=float(unk_score_offset),
            lm_score_boundary=bool(lm_score_boundary),
        )
        if fix_break:
            if any(UNK_TOKEN in p for p in vocab):
                raise ValueError(f"vocab contains {UNK_TOKEN!r}; the break fix would misfire")
            self.dec.__class__ = _FixedBreak
        self.beam = int(beam)
        self.beam_prune_logp = float(beam_prune_logp)
        self.token_min_logp = float(token_min_logp)
        self.prune_history = bool(prune_history)

    def decode(self, logp):
        beams = self.dec.decode_beams(
            np.asarray(logp, dtype=np.float32),
            beam_width=self.beam,
            beam_prune_logp=self.beam_prune_logp,
            token_min_logp=self.token_min_logp,
            prune_history=self.prune_history,
        )
        text = beams[0][0]
        if self.units == "piece":
            text = self.sp.decode([int(w[1:]) for w in text.split()])
        return " ".join(text.lower().split())


def make_decoder(vocab, blank, spm_model, lm=None, alpha=0.0, beta=0.0, beam=16, **opts):
    return PyctcDecoder(vocab, blank, spm_model, lm=lm, alpha=alpha, beta=beta, beam=beam, **opts)
