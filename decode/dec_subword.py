"""CTC prefix beam search over SentencePiece tokens with shallow fusion of a token n-gram LM.

The LM is a KenLM model whose "words" are the piece ids written t0..t255 (sp256_6gram.bin). Every
hypothesis is a token prefix; its score is

    acoustic(prefix) + alpha * ln P_LM(prefix tokens [+ </s> at the end]) + beta * len(prefix)

acoustic = the CTC score of the prefix: the best alignment (merge="max", default) or the sum over
all alignments (merge="sum", the classic prefix beam search), kept separately for alignments ending
in blank and in the last token. alpha = LM weight on natural-log probabilities (KenLM's log10 is
converted), beta = bonus per emitted token. With merge="max", beam 1 and alpha 0 reproduce greedy
decoding exactly (ties included: equal scores go to the lower label, blank = highest, like argmax).

Per-frame pruning (options):
  token_min_logp  only tokens with log-prob >= this value are extended (default -5.0; the frame's
                  best token is always kept, blank is always scored)
  topk            at most this many tokens per frame, best first (0 = no cap; default 0)
  beam_prune_logp drop hypotheses scoring more than this below the frame's best (default -10.0)
  merge           "max" or "sum" (default "max")
  eos             add alpha * ln P(</s> | prefix) before the final ranking (default True)

lm: None, a KenLM path (str / Path), or any object with start() -> state,
score(state, token_id) -> (ln prob, next state) and final(state) -> ln P(</s>).
"""
import heapq
import math

import numpy as np
import sentencepiece as spm

NEG_INF = float("-inf")
LN10 = math.log(10.0)


class KenLMTokenLM:
    """KenLM model over the words t0..tN (natural-log scores)."""

    def __init__(self, path):
        import kenlm
        self._kenlm = kenlm
        self.model = kenlm.Model(str(path))
        self.words = [f"t{i}" for i in range(1024)]

    def start(self):
        s = self._kenlm.State()
        self.model.BeginSentenceWrite(s)
        return s

    def score(self, state, tok):
        out = self._kenlm.State()
        return self.model.BaseScore(state, self.words[tok], out) * LN10, out

    def final(self, state):
        return self.model.BaseScore(state, "</s>", self._kenlm.State()) * LN10


class Node:
    """One token prefix in the prefix tree. fused = alpha * LM + beta * length so far."""
    __slots__ = ("parent", "tok", "children", "state", "fused")

    def __init__(self, parent, tok, state, fused):
        self.parent, self.tok, self.state, self.fused = parent, tok, state, fused
        self.children = {}

    def tokens(self):
        out, n = [], self
        while n.parent is not None:
            out.append(n.tok)
            n = n.parent
        return out[::-1]


def _logaddexp(a, b):
    if a < b:
        a, b = b, a
    if b == NEG_INF:
        return a
    return a + math.log1p(math.exp(b - a))


class SubwordBeamDecoder:
    def __init__(self, blank, spm_model, lm=None, alpha=0.0, beta=0.0, beam=16,
                 token_min_logp=-5.0, topk=0, beam_prune_logp=-10.0, merge="max", eos=True):
        if merge not in ("max", "sum"):
            raise ValueError(f"merge must be 'max' or 'sum', got {merge!r}")
        if beam < 1:
            raise ValueError("beam must be >= 1")
        self.blank = blank
        self.sp = spm.SentencePieceProcessor(model_file=str(spm_model))
        self.lm = KenLMTokenLM(lm) if isinstance(lm, (str, bytes)) or hasattr(lm, "__fspath__") else lm
        self.alpha, self.beta, self.beam = float(alpha), float(beta), int(beam)
        self.token_min_logp, self.topk = float(token_min_logp), int(topk)
        self.beam_prune_logp = float(beam_prune_logp)
        self.merge = merge
        self.eos = bool(eos)
        self.use_lm = self.lm is not None and self.alpha != 0.0

    def _child(self, node, tok):
        ch = node.children.get(tok)
        if ch is None:
            fused = node.fused + self.beta
            state = None
            if self.use_lm:
                lp, state = self.lm.score(node.state, tok)
                fused += self.alpha * lp
            ch = Node(node, tok, state, fused)
            node.children[tok] = ch
        return ch

    def _frame_tokens(self, logp):
        """Per frame: list of (token, log-prob) to extend, ascending token id."""
        lt = logp[:, :self.blank]
        mask = lt >= self.token_min_logp
        best = lt.argmax(1)
        mask[np.arange(len(lt)), best] = True
        if self.topk > 0:
            counts = mask.sum(1)
            for t in np.flatnonzero(counts > self.topk):
                keep = np.argsort(-lt[t], kind="stable")[:self.topk]
                mask[t] = False
                mask[t, keep] = True
        rows, cols = np.nonzero(mask)
        vals = lt[rows, cols].tolist()
        cols = cols.tolist()
        bounds = np.searchsorted(rows, np.arange(len(lt) + 1)).tolist()
        return [list(zip(cols[a:b], vals[a:b])) for a, b in zip(bounds, bounds[1:])]

    def decode(self, logp):
        return " ".join(self.sp.decode(self.decode_ids(logp)).lower().split())

    def decode_ids(self, logp):
        """Best token id sequence for log-probs [T, C]."""
        logp = np.asarray(logp, dtype=np.float32)
        if len(logp) == 0:
            return []
        comb = max if self.merge == "max" else _logaddexp
        root = Node(None, -1, self.lm.start() if self.use_lm else None, 0.0)
        beams = [(root, 0.0, NEG_INF)]               # (node, score ending in blank, in last token)
        blank = self.blank
        frame_toks = self._frame_tokens(logp)
        rows = logp.tolist()
        for t, toks in enumerate(frame_toks):
            row = rows[t]
            lb = row[blank]
            cand = {}                                # node -> [vb, vnb, label of the best vnb path]

            def add_nb(node, s, lab):
                e = cand.get(node)
                if e is None:
                    cand[node] = [NEG_INF, s, lab]
                elif self.merge == "max":
                    if s > e[1] or (s == e[1] and lab < e[2]):
                        e[1], e[2] = s, lab
                else:
                    e[1] = _logaddexp(e[1], s)

            for node, vb, vnb in beams:
                v = comb(vb, vnb)
                last = node.tok
                e = cand.get(node)                   # stay: blank, or repeat of the last token
                if e is None:
                    cand[node] = [v + lb, NEG_INF, blank]
                else:
                    e[0] = comb(e[0], v + lb)
                if last >= 0 and vnb > NEG_INF:
                    add_nb(node, vnb + row[last], last)
                for c, lpc in toks:
                    if c == last:                    # a repeated token needs a blank in between
                        if vb > NEG_INF:
                            add_nb(self._child(node, c), vb + lpc, c)
                    else:
                        add_nb(self._child(node, c), v + lpc, c)

            scored = []
            for node, (vb, vnb, lab) in cand.items():
                tot = comb(vb, vnb) + node.fused
                scored.append((-tot, lab if vnb >= vb else blank, len(scored), node, vb, vnb))
            top = heapq.nsmallest(self.beam, scored)   # == sorted(scored)[:beam]
            floor = -top[0][0] + self.beam_prune_logp
            beams = [(s[3], s[4], s[5]) for s in top if -s[0] >= floor]

        if self.use_lm and self.eos:
            final = [(comb(vb, vnb) + node.fused + self.alpha * self.lm.final(node.state), i, node)
                     for i, (node, vb, vnb) in enumerate(beams)]
            node = max(final, key=lambda x: (x[0], -x[1]))[2]
        else:
            node = beams[0][0]
        return node.tokens()


def make_decoder(vocab, blank, spm_model, lm=None, alpha=0.0, beta=0.0, beam=16, **opts):
    return SubwordBeamDecoder(blank, spm_model, lm=lm, alpha=alpha, beta=beta, beam=beam, **opts)
