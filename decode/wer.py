"""Word error rate exactly like probe.py (pure Python: no torch or numpy needed).

Corpus WER = sum of word edit distances / sum of reference words.
"""


def norm_text(t):
    """probe.norm_text: lowercase, single spaces."""
    return " ".join(t.lower().split())


def edit_distance(ref, hyp):
    """Levenshtein distance over two word lists (copy of probe.edit_distance)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def utt_errors(ref, hyp):
    """(word edits, reference words) for one utterance; both strings, split on whitespace."""
    r = ref.split()
    return edit_distance(r, hyp.split()), len(r)


def corpus_wer(refs, hyps):
    """(wer, edits, words) over parallel lists of reference and hypothesis strings."""
    edits = words = 0
    for ref, hyp in zip(refs, hyps, strict=True):
        e, n = utt_errors(ref, hyp)
        edits += e
        words += n
    return edits / max(words, 1), edits, words
