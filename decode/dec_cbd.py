"""ctc-beam-decoder 0.2.0 (C++ clone of pyctcdecode, ctypes binding) behind the shared decoder interface.

    dec = make_decoder(vocab, blank, spm_model, lm=None, alpha=0.0, beta=0.0, beam=16, **opts)
    text = dec.decode(logp)        # logp: [T, 257] log-probs -> lowercase words joined by single spaces

vocab is the 256 SentencePiece pieces (index 0 "<unk>"); the blank must be the last column (index 256):
ctc-beam-decoder, like pyctcdecode, appends the blank itself when no label looks like one.

units="word" (default): labels are the pieces as-is ("\u2581" starts a word), so a word-level KenLM
  (OpenSLR 3-gram / 4-gram) scores whole words. "<unk>" is mapped by the library to "\u2581\u2047\u2581",
  the same " \u2047 " sentencepiece emits for it.
units="piece": every piece becomes its own word "\u2581t<id>", so the subword LM (sp256_6gram, words
  t0..t255) scores one piece per word and beta is a per-piece bonus; the winning "t3 t17 ..." is
  mapped back to ids and sentencepiece-decoded. With no LM and beam 1 this is exactly greedy.

Word units inherit pyctcdecode 0.5.0's force_next_break quirk, reproduced on purpose by the library:
one flag shared by all beams and candidates, set by any candidate label ending in "\u2581" (the bare
"\u2581" piece, "<unk>") and consumed by whichever expansion comes next, so even beam 1 can split or
join words where greedy does not. It cannot be avoided through labels; use units="piece" for the
exact "beam 1, no LM" reproduction, and pyctcdecode without its break fix for parity.

Opts (pyctcdecode names and defaults): unigrams (list, or path to one word per line; default for a
non-ARPA word LM is the sidecar "<lm minus .bin>.unigrams.txt", for piece units t0..t255 — without
it every word prefix is charged unk_score_offset), beam_prune_logp, token_min_logp,
unk_score_offset, lm_score_boundary, prune_history (True = pyctcdecode's decode()), hotwords,
hotword_weight.
"""

from pathlib import Path

import numpy as np
import sentencepiece as spm
from ctc_beam_decoder import BeamSearchDecoder

PIECE_PREFIX = "t"


def norm_text(t):
    return " ".join(t.lower().split())


def _read_unigrams(src):
    if isinstance(src, (str, Path)):
        return [w for w in Path(src).expanduser().read_text(encoding="utf-8").split() if w]
    return list(src)


class CbdDecoder:
    def __init__(self, vocab, blank, spm_model, lm=None, alpha=0.0, beta=0.0, beam=16, units="word",
                 unigrams=None, beam_prune_logp=-10.0, token_min_logp=-5.0, unk_score_offset=-10.0,
                 lm_score_boundary=True, prune_history=True, hotwords=None, hotword_weight=10.0):
        vocab = list(vocab)
        if blank != len(vocab):
            raise ValueError(f"blank must be the last column ({len(vocab)}), got {blank}")
        if units not in ("word", "piece"):
            raise ValueError(f"units must be 'word' or 'piece', got {units!r}")
        self.units = units
        self.sp = spm.SentencePieceProcessor(model_file=str(Path(spm_model).expanduser()))
        if units == "word":
            labels = vocab
        else:
            labels = [f"\u2581{PIECE_PREFIX}{i}" for i in range(len(vocab))]
        if lm is not None:
            lm = Path(lm).expanduser()
            if unigrams is None:
                if units == "piece":
                    unigrams = [f"{PIECE_PREFIX}{i}" for i in range(len(vocab))]
                elif lm.suffix != ".arpa":   # a plain .arpa: the library reads its unigrams itself
                    side = lm.with_suffix(".unigrams.txt")
                    if not side.exists():
                        raise FileNotFoundError(f"{lm} needs unigrams: pass unigrams= or create {side}")
                    unigrams = side
            if unigrams is not None:
                unigrams = _read_unigrams(unigrams)
        self.dec = BeamSearchDecoder(labels, kenlm_model_path=lm, unigrams=unigrams, alpha=alpha, beta=beta,
                                     unk_score_offset=unk_score_offset, score_boundary=lm_score_boundary)
        if self.dec.blank_id != blank or self.dec.n_columns != len(vocab) + 1:
            raise RuntimeError(f"library put blank at {self.dec.blank_id} of {self.dec.n_columns} columns")
        self.kw = dict(beam_width=int(beam), beam_prune_logp=beam_prune_logp, token_min_logp=token_min_logp,
                       prune_history=prune_history, hotwords=list(hotwords) if hotwords else None,
                       hotword_weight=hotword_weight)

    def decode(self, logp):
        beams = self.dec.decode_beams(np.asarray(logp, dtype=np.float32), **self.kw)
        text = beams[0][0] if beams else ""
        if self.units == "piece":
            text = self.sp.decode([int(w[len(PIECE_PREFIX):]) for w in text.split()])
        return norm_text(text)


def make_decoder(vocab, blank, spm_model, lm=None, alpha=0.0, beta=0.0, beam=16, **opts):
    return CbdDecoder(vocab, blank, spm_model, lm=lm, alpha=alpha, beta=beta, beam=beam, **opts)
