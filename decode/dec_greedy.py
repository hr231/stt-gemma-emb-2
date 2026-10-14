"""Greedy CTC decoder with the shared decoder interface.

argmax per frame -> collapse repeats -> drop blank -> SentencePiece decode -> lowercase, single
spaces. Identical to probe.greedy_collapse + sp.decode + probe.norm_text; beam, LM, alpha and beta
are accepted for the common interface and ignored.
"""
import numpy as np
import sentencepiece as spm


def collapse(frame_ids, blank):
    """probe.greedy_collapse on a 1-D integer array: token ids (list of int)."""
    ids = np.asarray(frame_ids)
    if ids.size == 0:
        return []
    keep = ids != blank
    keep[1:] &= ids[1:] != ids[:-1]
    return ids[keep].tolist()


class GreedyDecoder:
    def __init__(self, blank, spm_model):
        self.blank = blank
        self.sp = spm.SentencePieceProcessor(model_file=str(spm_model))

    def decode(self, logp):
        ids = collapse(np.asarray(logp).argmax(-1), self.blank)
        return " ".join(self.sp.decode(ids).lower().split())


def make_decoder(vocab, blank, spm_model, lm=None, alpha=0.0, beta=0.0, beam=16, **opts):
    if opts:
        raise ValueError(f"greedy decoder takes no options, got {sorted(opts)}")
    return GreedyDecoder(blank, spm_model)
