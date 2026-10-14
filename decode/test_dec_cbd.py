"""Tiny CPU test of decode/dec_cbd.py on synthetic log-probs (no model, no real data).

    cd ~/emb_gemma2 && OMP_NUM_THREADS=1 ~/.venv-gemma-stt/bin/python -m unittest decode/test_dec_cbd.py -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dec_cbd import make_decoder, norm_text  # noqa: E402

SPM = Path(os.environ.get("SPM_MODEL", "~/gemma_stt/feats/full/spm_unigram256.model")).expanduser()
BLANK = 256


def log_softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def greedy_ref(logp, sp):
    """argmax -> collapse repeats -> drop blank -> sp.decode -> norm (probe.greedy_collapse path)."""
    out, prev = [], None
    for t in logp.argmax(-1).tolist():
        if t != BLANK and t != prev:
            out.append(t)
        prev = t
    return norm_text(sp.decode(out))


def synth(rng, ids, peak, noise):
    """CTC-like frames: each id held 1-3 frames, blanks (0-2) between, a blank forced between repeats."""
    frames = [BLANK] * int(rng.integers(0, 3))
    for k, t in enumerate(ids):
        if k and ids[k - 1] == t:
            frames.append(BLANK)
        frames += [t] * int(rng.integers(1, 4))
        frames += [BLANK] * int(rng.integers(0, 3))
    x = rng.normal(0.0, noise, (len(frames), BLANK + 1))
    x[np.arange(len(frames)), frames] += peak
    return log_softmax(x).astype(np.float32)


class TestDecCbd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sp = spm.SentencePieceProcessor(model_file=str(SPM))
        cls.vocab = [cls.sp.id_to_piece(i) for i in range(cls.sp.get_piece_size())]
        assert len(cls.vocab) == 256 and cls.vocab[0] == "<unk>"
        rng = np.random.default_rng(0)
        texts = ["the cat sat on the mat", "a moment of silence", "she sells sea shells",
                 "hello world", "zzz qqq xx"]
        utts = [cls.sp.encode(t) for t in texts]
        utts += [[0, 5, 0, 6], [cls.vocab.index("\u2581"), 1, 3, 3], [3, 3, 3], []]  # <unk>, lone "\u2581", repeats, empty
        utts += [list(rng.integers(0, 256, int(rng.integers(5, 40)))) for _ in range(12)]
        cls.logps = []
        for k, ids in enumerate(utts):
            # confident, then noisy (near ties between the top tokens), then flat (argmax below token_min_logp)
            for peak, noise in ((8.0, 1.0), (2.0, 1.0), (0.5, 0.3)):
                cls.logps.append(synth(np.random.default_rng(1000 + k), [int(t) for t in ids], peak, noise))
        cls.refs = [greedy_ref(lp, cls.sp) for lp in cls.logps]

    def test_beam1_piece_units_equals_greedy(self):
        # the exact-reproduction path: no label ends in "\u2581", so pyctcdecode's force_next_break never fires
        dec = make_decoder(self.vocab, BLANK, SPM, beam=1, units="piece")
        for i, (lp, ref) in enumerate(zip(self.logps, self.refs)):
            self.assertEqual(dec.decode(lp), ref, f"utt {i}")
        x16 = self.logps[0].astype(np.float16)   # store dtype
        self.assertEqual(dec.decode(x16), greedy_ref(x16.astype(np.float32), self.sp))
        print(f"\n  piece units, beam 1 == greedy on {len(self.logps)}/{len(self.logps)} utts;"
              f" e.g. {self.refs[0]!r} / {self.refs[15]!r}")

    def test_beam1_word_units_pyctcdecode_quirk(self):
        """Word units reproduce pyctcdecode 0.5.0's shared force_next_break flag: any candidate label ending
        in "\u2581" (bare "\u2581" id 13, "<unk>" -> "\u2581\u2047\u2581" id 0) with logp >= token_min_logp
        sets it and whichever expansion comes next consumes it, so beam 1 can split or join words that
        greedy does not. Exact when only the argmax is a candidate, or when ids 0 and 13 never reach
        token_min_logp."""
        exact = make_decoder(self.vocab, BLANK, SPM, beam=1, token_min_logp=0.0)
        dflt = make_decoder(self.vocab, BLANK, SPM, beam=1)
        clean, bad_trig = 0, 0
        for i, (lp, ref) in enumerate(zip(self.logps, self.refs)):
            self.assertEqual(exact.decode(lp), ref, f"token_min_logp=0 utt {i}")
            if (lp[:, [0, 13]] >= -5.0).any():
                bad_trig += dflt.decode(lp) != ref
            else:
                clean += 1
                self.assertEqual(dflt.decode(lp), ref, f"untriggered utt {i}")
        self.assertGreater(clean, 0)
        self.assertGreater(bad_trig, 0)   # quirk still present (a release that fixes it should fail here)
        print(f"\n  word units, beam 1: == greedy on all {len(self.logps)} with token_min_logp=0 and on"
              f" {clean}/{clean} untriggered utts; differs on {bad_trig}/{len(self.logps) - clean} triggered")

    def test_beam16_runs(self):
        dec = make_decoder(self.vocab, BLANK, SPM, beam=16)
        hyps = [dec.decode(lp) for lp in self.logps]
        same = sum(h == r for h, r in zip(hyps, self.refs))
        self.assertTrue(all(isinstance(h, str) and h == norm_text(h) for h in hyps))
        self.assertEqual(hyps[0], self.refs[0])   # confident input: beam agrees with greedy
        print(f"\n  beam 16: {same}/{len(hyps)} equal to greedy")

    def test_word_lm_arpa_and_sidecar(self):
        words = ["the", "cat", "sat", "on", "mat"]
        arpa = "\\data\\\nngram 1=8\nngram 2=5\n\n\\1-grams:\n-99\t<s>\t-0.3\n-1.0\t</s>\n-3.0\t<unk>\t0\n" \
            + "".join(f"-0.7\t{w}\t-0.3\n" for w in words) \
            + "\n\\2-grams:\n-0.2\t<s> the\n-0.2\tthe cat\n-0.2\tcat sat\n-0.2\tsat on\n-0.2\ton the\n\n\\end\\\n"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tiny.arpa"
            p.write_text(arpa)
            no_lm = make_decoder(self.vocab, BLANK, SPM, beam=16)
            lm = make_decoder(self.vocab, BLANK, SPM, lm=str(p), alpha=2.0, beta=0.5, beam=16)
            noisy = self.logps[1::3]

            def share(dec):
                hyp = " ".join(dec.decode(x) for x in noisy).split()
                return sum(w in words for w in hyp) / max(len(hyp), 1)
            known = share(no_lm), share(lm)
            self.assertGreater(known[1], known[0])
            self.assertEqual(lm.decode(self.logps[0]), "the cat sat on the mat")
            # a non-ARPA path without unigrams must find the sidecar "<lm minus suffix>.unigrams.txt"
            q = Path(d) / "tiny.lm"
            q.write_text(arpa)
            with self.assertRaises(FileNotFoundError):
                make_decoder(self.vocab, BLANK, SPM, lm=str(q), alpha=2.0)
            (Path(d) / "tiny.unigrams.txt").write_text("\n".join(words) + "\n")
            side = make_decoder(self.vocab, BLANK, SPM, lm=str(q), alpha=2.0, beta=0.5, beam=16)
            self.assertEqual([side.decode(x) for x in noisy], [lm.decode(x) for x in noisy])
        print(f"\n  word LM: share of output words known to the tiny ARPA {known[0]:.2f} -> {known[1]:.2f}")

    def test_piece_lm_6gram(self):
        # 6-gram over piece words t<id> (KENLM_MAX_ORDER=6 in the bundled libkenlm)
        ids = self.sp.encode("the cat sat on the mat")
        chain = ["<s>"] + [f"t{i}" for i in ids] + ["</s>"]
        grams = {n: sorted({" ".join(chain[k:k + n]) for k in range(len(chain) - n + 1)}) for n in range(1, 7)}
        grams[1] = sorted(set(grams[1]) | {"<unk>"} | {f"t{i}" for i in range(256)})
        out = ["\\data\\"] + [f"ngram {n}={len(g)}" for n, g in grams.items()] + [""]
        for n, g in grams.items():
            out.append(f"\\{n}-grams:")
            for e in g:
                p = -99 if e == "<s>" else (-0.1 if n > 1 else -2.5)
                out.append(f"{p}\t{e}" + ("\t-0.5" if n < 6 and not e.endswith("</s>") else ""))
            out.append("")
        out.append("\\end\\")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sp6.arpa"
            p.write_text("\n".join(out) + "\n")
            dec = make_decoder(self.vocab, BLANK, SPM, lm=str(p), alpha=0.5, beta=0.0, beam=16, units="piece")
            self.assertEqual(dec.decode(self.logps[0]), "the cat sat on the mat")
            self.assertTrue(all(h == norm_text(h) for h in map(dec.decode, self.logps)))
        print(f"\n  piece 6-gram LM loaded ({len(grams[6])} 6-grams) and decoded {len(self.logps)} utts")

    def test_rejects_wrong_blank(self):
        with self.assertRaises(ValueError):
            make_decoder(self.vocab, 0, SPM)


if __name__ == "__main__":
    unittest.main()
