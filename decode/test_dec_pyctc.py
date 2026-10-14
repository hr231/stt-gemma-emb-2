"""CPU-only test of dec_pyctc on synthetic log-probs.

Run: ~/.venv-pyctcdecode/bin/python ~/emb_gemma2/decode/test_dec_pyctc.py
(also collectable by pytest if installed).
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import sentencepiece as spm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dec_pyctc  # noqa: E402

SPM = os.path.expanduser("~/gemma_stt/feats/full/spm_unigram256.model")
SP = spm.SentencePieceProcessor(model_file=SPM)
VOCAB = [SP.id_to_piece(i) for i in range(SP.get_piece_size())]
BLANK = len(VOCAB)


def greedy_ref(logp):
    # == probe.greedy_collapse + sp.decode + probe.norm_text
    out, prev = [], None
    for t in logp.argmax(-1).tolist():
        if t != BLANK and t != prev:
            out.append(t)
        prev = t
    return " ".join(SP.decode(out).lower().split())


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def synth_logp(rng, T, p_unk=0.02, noise=1.5, boost=(1.0, 8.0), unk_logit=None):
    """CTC-like log-probs: runs of blanks and pieces (incl. repeats, '<unk>', bare '▁').
    Small boosts give non-peaky frames with many candidates above token_min_logp."""
    path = []
    while len(path) < T:
        r = rng.random()
        if r < 0.45:
            tok = BLANK
        elif r < 0.50:
            tok = path[-1] if path else BLANK  # repeated label, no blank in between
        elif r < 0.50 + p_unk:
            tok = 0  # <unk>
        elif r < 0.53 + p_unk:
            tok = VOCAB.index("\u2581")
        else:
            tok = int(rng.integers(1, BLANK))
        path += [tok] * int(rng.integers(1, 4))
    path = np.array(path[:T])
    logits = rng.normal(0.0, noise, size=(T, BLANK + 1))
    if unk_logit is not None:
        logits[:, 0] = unk_logit
    logits[np.arange(T), path] += rng.uniform(*boost, size=T)
    return log_softmax(logits).astype(np.float32)


def test_beam1_no_lm_equals_greedy():
    rng = np.random.default_rng(0)
    sets = {
        # noisy frames: many candidates >= token_min_logp per frame, incl. bare "▁" and <unk>
        "noisy+unk": dict(p_unk=0.05),
        # noisy frames, <unk> never a candidate (as in a model that never emits it)
        "noisy": dict(p_unk=0.0, unk_logit=-30.0),
        # <unk> emitted, peaky frames (only the argmax survives token_min_logp=-5)
        "peaky+unk": dict(p_unk=0.05, noise=1.0, boost=(16.0, 22.0)),
    }
    for units in ("word", "piece"):
        dec = dec_pyctc.make_decoder(VOCAB, BLANK, SPM, beam=1, units=units)
        for name, kw in sets.items():
            n_unk = 0
            for _ in range(300):
                lp = synth_logp(rng, int(rng.integers(1, 120)), **kw)
                ref = greedy_ref(lp)
                hyp = dec.decode(lp)
                assert hyp == ref, (units, name, ref, hyp)
                n_unk += "\u2047" in ref
            assert (n_unk > 0) == (name != "noisy")
            print(f"beam1/noLM == greedy: 300/300 synthetic utts (units={units}, set={name}, "
                  f"{n_unk} with unk)")

    # Stock pyctcdecode (fix_break=False): shared force_next_break flag -> word units diverge
    # from greedy on noisy frames, but only in word spacing.
    stock = dec_pyctc.make_decoder(VOCAB, BLANK, SPM, beam=1, units="word", fix_break=False)
    n_mis = 0
    for _ in range(300):
        lp = synth_logp(rng, int(rng.integers(1, 120)), p_unk=0.0, unk_logit=-30.0)
        ref, hyp = greedy_ref(lp), stock.decode(lp)
        if hyp != ref:
            n_mis += 1
            assert ref.replace(" ", "") == hyp.replace(" ", "")
    assert n_mis > 0
    print(f"stock pyctcdecode (fix_break=False), set=noisy: {300 - n_mis}/300 == greedy "
          f"(rest differ only in word spacing)")

    # float16 input (as stored in the dump) and the clip/renorm path
    lp = synth_logp(rng, 80, p_unk=0.0, unk_logit=-30.0)
    assert dec_pyctc.make_decoder(VOCAB, BLANK, SPM, beam=1).decode(lp.astype(np.float16)) == \
        greedy_ref(lp.astype(np.float16).astype(np.float32))


def _write_arpa(path, words, logps):
    """Tiny bigram ARPA (KenLM needs order >= 2); bigrams only around words[0]."""
    w0 = words[0]
    lines = ["\\data\\", f"ngram 1={len(words) + 3}", "ngram 2=3", "", "\\1-grams:",
             "-99\t<s>\t-0.3", "-1.0\t</s>", "-5.0\t<unk>"]
    lines += [f"{lp}\t{w}\t-0.3" for w, lp in zip(words, logps)]
    lines += ["", "\\2-grams:", f"-0.1\t<s> {w0}", f"-0.1\t{w0} {w0}", f"-0.2\t{w0} </s>",
              "", "\\end\\", ""]
    Path(path).write_text("\n".join(lines))


def _frame(ids_probs, T_blank=2):
    """Frames: blank*T_blank, one frame with the given (id, prob) mix, blank*T_blank."""
    rows = []
    eps = 1e-4
    for _ in range(T_blank):
        p = np.full(BLANK + 1, eps); p[BLANK] = 1.0; rows.append(p)
    p = np.full(BLANK + 1, eps)
    for i, pr in ids_probs:
        p[i] = pr
    rows.append(p)
    for _ in range(T_blank):
        p = np.full(BLANK + 1, eps); p[BLANK] = 1.0; rows.append(p)
    rows = np.array(rows)
    return np.log(rows / rows.sum(-1, keepdims=True)).astype(np.float32)


def test_tiny_arpa_lm():
    the, a = VOCAB.index("\u2581the"), VOCAB.index("\u2581a")
    # acoustics slightly prefer "a"; LM strongly prefers "the"
    lp = np.concatenate([_frame([(a, 0.55), (the, 0.45)]), _frame([(the, 0.9)])])
    with tempfile.TemporaryDirectory() as d:
        arpa = os.path.join(d, "tiny.arpa")
        _write_arpa(arpa, ["the", "a", "and", "of"], [-0.3, -3.0, -1.0, -1.0])
        uni = os.path.join(d, "tiny.unigrams.txt")
        Path(uni).write_text("the\na\nand\nof\n")
        no_lm = dec_pyctc.make_decoder(VOCAB, BLANK, SPM, beam=8).decode(lp)
        with_lm = dec_pyctc.make_decoder(VOCAB, BLANK, SPM, lm=arpa, alpha=1.0, beta=0.0,
                                         beam=8, unigrams=uni).decode(lp)
        print(f"tiny word ARPA: noLM={no_lm!r} withLM={with_lm!r}")
        assert no_lm == "a the" and with_lm == "the the"

        # piece-unit LM (words t0..t255): LM prefers t<the> -> same flip
        parpa = os.path.join(d, "tiny_piece.arpa")
        words = [f"t{i}" for i in range(BLANK)]
        _write_arpa(parpa, words, [-0.3 if i == the else -3.0 for i in range(BLANK)])
        rng = np.random.default_rng(1)
        pdec = dec_pyctc.make_decoder(VOCAB, BLANK, SPM, lm=parpa, alpha=1.0, beta=0.0,
                                      beam=8, units="piece")
        p_lm = pdec.decode(lp)
        p_rand = pdec.decode(synth_logp(rng, 100))
        print(f"tiny piece ARPA: withLM={p_lm!r}; random utt -> {p_rand!r}")
        assert p_lm == "the the" and isinstance(p_rand, str)


if __name__ == "__main__":
    from importlib.metadata import version
    import kenlm
    print("numpy", np.__version__, "| pyctcdecode", version("pyctcdecode"), "| kenlm",
          version("kenlm"), kenlm.__file__)
    test_beam1_no_lm_equals_greedy()
    test_tiny_arpa_lm()
    print("ALL OK")
