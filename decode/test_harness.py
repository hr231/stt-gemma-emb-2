"""Tests for the decoding harness on synthetic log-probs.

  cd ~/emb_gemma2 && python -m unittest decode.test_harness -v

Runs in either venv: the probe.py comparison needs torch (training venv), the KenLM test needs the
kenlm module (pyctcdecode venv); each skips when its module is missing.
"""
import importlib.util
import itertools
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dec_greedy  # noqa: E402
import dec_subword  # noqa: E402
from wer import corpus_wer, edit_distance, norm_text  # noqa: E402

SPM = Path.home() / "gemma_stt/feats/full/spm_unigram256.model"
V, BLANK, C = 256, 256, 257
HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_KENLM = importlib.util.find_spec("kenlm") is not None


def random_logp(rng, T, peaky=True, ties=False):
    """[T, 257] float32 log_softmax of random logits; peaky = mostly blank like a CTC head."""
    z = rng.normal(0, 1.5, (T, C))
    if peaky:
        z[:, BLANK] += rng.choice([0.0, 6.0], size=T, p=[0.4, 0.6])
        hot = rng.integers(0, V, size=T)
        z[np.arange(T), hot] += rng.uniform(0, 6, size=T)
    lp = z - np.logaddexp.reduce(z, axis=1, keepdims=True)
    if ties:                           # coarse grid: many equal values, incl. at the frame maximum
        lp = np.round(lp * 2) / 2
    return lp.astype(np.float32)


def frames_for(ids, rng):
    """Peaky log-probs whose greedy decoding is exactly ids."""
    path = []
    for j, t in enumerate(ids):
        if j and ids[j - 1] == t or rng.random() < 0.5:
            path.append(BLANK)
        path += [t] * int(rng.integers(1, 3))
    path += [BLANK] * int(rng.integers(0, 3))
    z = rng.normal(0, 1, (len(path), C))
    z[np.arange(len(path)), path] += 12
    return (z - np.logaddexp.reduce(z, axis=1, keepdims=True)).astype(np.float32)


def write_store(d, split, utts_lp, refs, vocab):
    arr = np.concatenate(utts_lp).astype(np.float16)
    np.save(d / f"{split}.npy", arr)
    utts, off = [], 0
    for k, (lp, ref) in enumerate(zip(utts_lp, refs)):
        utts.append({"id": f"u{k}", "ref": ref, "offset": off, "frames": len(lp), "dur": len(lp) * 0.04})
        off += len(lp)
    (d / f"{split}.json").write_text(json.dumps(
        {"split": split, "head": "synthetic", "blank": BLANK, "vocab": vocab, "spm_model": str(SPM),
         "frame_ms": 40, "utts": utts, "greedy_wer": None, "greedy_wer_fp16": None}))
    return arr


class MockLM:
    """Bigram-free LM: fixed ln P(token) per id, ln P(</s>) = 0."""

    def __init__(self, table):
        self.table = table

    def start(self):
        return ()

    def score(self, state, tok):
        return self.table[tok], state + (tok,)

    def final(self, state):
        return 0.0


@unittest.skipUnless(SPM.exists(), f"tokenizer {SPM} missing")
class TestHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sentencepiece as spm
        cls.sp = spm.SentencePieceProcessor(model_file=str(SPM))
        cls.vocab = [cls.sp.id_to_piece(i) for i in range(V)]

    def test_wer(self):
        self.assertEqual(edit_distance("a b c".split(), "a x c d".split()), 2)
        self.assertEqual(corpus_wer(["a b c", "d e"], ["a x c d", "d e"]), (2 / 5, 2, 5))
        self.assertEqual(norm_text("  Hello   WORLD "), "hello world")

    @unittest.skipUnless(HAVE_TORCH, "probe.py needs torch")
    def test_greedy_matches_probe(self):
        sys.path.insert(0, str(HERE.parent))
        from probe import greedy_collapse
        from probe import norm_text as probe_norm
        dec = dec_greedy.make_decoder(self.vocab, BLANK, str(SPM))
        rng = np.random.default_rng(0)
        for k in range(200):
            lp = random_logp(rng, int(rng.integers(0, 120)), peaky=k % 2 == 0)
            ref = probe_norm(self.sp.decode(greedy_collapse(lp.argmax(-1).tolist(), BLANK)))
            self.assertEqual(dec.decode(lp), ref)
            self.assertEqual(dec_greedy.collapse(lp.argmax(-1), BLANK),
                             greedy_collapse(lp.argmax(-1).tolist(), BLANK))

    def test_subword_beam1_alpha0_is_greedy(self):
        g = dec_greedy.make_decoder(self.vocab, BLANK, str(SPM))
        rng = np.random.default_rng(1)
        variants = [{}, {"token_min_logp": -1.0}, {"topk": 3}, {"beam_prune_logp": -2.0}]
        n = 0
        for k in range(240):
            lp = random_logp(rng, int(rng.integers(1, 100)), peaky=k % 3 != 0, ties=k % 2 == 0)
            opts = variants[k % len(variants)]
            for lm in (None, MockLM(rng.normal(-5, 2, V).tolist())):   # alpha 0: the LM is unused
                s = dec_subword.make_decoder(self.vocab, BLANK, str(SPM), lm=lm, alpha=0.0, beta=0.0,
                                             beam=1, **opts)
                ids = s.decode_ids(lp)
                self.assertEqual(ids, dec_greedy.collapse(lp.argmax(-1), BLANK), (k, opts))
                self.assertEqual(s.decode(lp), g.decode(lp))
                n += 1
        self.assertEqual(n, 480)

    def test_subword_exact_search(self):
        """No pruning, huge beam: the best prefix equals brute force over all alignments
        (max: best single path; sum: best total prefix probability)."""
        rng = np.random.default_rng(2)
        toks = [5, 9, 17]                                  # tiny alphabet: 3 tokens + blank
        labels = toks + [BLANK]
        for trial in range(30):
            T = int(rng.integers(1, 7))
            lp = np.full((T, C), -1e4, np.float32)
            z = rng.normal(0, 1.5, (T, 4))
            lp[:, labels] = z - np.logaddexp.reduce(z, axis=1, keepdims=True)
            best = {"max": {}, "sum": {}}
            for path in itertools.product(labels, repeat=T):
                pre = tuple(dec_greedy.collapse(np.array(path), BLANK))
                s = float(sum(lp[t, c] for t, c in enumerate(path)))
                best["max"][pre] = max(best["max"].get(pre, -math.inf), s)
                best["sum"][pre] = np.logaddexp(best["sum"].get(pre, -math.inf), s)
            for merge in ("max", "sum"):
                d = dec_subword.make_decoder(self.vocab, BLANK, str(SPM), beam=10_000, token_min_logp=-100.0,
                                             beam_prune_logp=-1e9, merge=merge)
                want = max(best[merge].items(), key=lambda kv: kv[1])
                got = tuple(d.decode_ids(lp))
                self.assertAlmostEqual(best[merge][got], want[1], places=4, msg=(trial, merge))

    def test_subword_lm_fusion(self):
        """A mock LM that strongly prefers token 7 over token 3 flips an ambiguous frame;
        beta > 0 favours longer outputs."""
        lp = np.full((3, C), -20.0, np.float32)
        lp[:, BLANK] = math.log(0.5)
        lp[1, 3], lp[1, 7] = math.log(0.3), math.log(0.2)
        table = [-10.0] * V
        table[7] = -0.1
        d0 = dec_subword.make_decoder(self.vocab, BLANK, str(SPM), beam=8)
        d1 = dec_subword.make_decoder(self.vocab, BLANK, str(SPM), lm=MockLM(table), alpha=1.0, beam=8)
        self.assertEqual(d0.decode_ids(lp), [])           # blank wins every frame
        self.assertEqual(d1.decode_ids(lp), [])           # LM cost alone does not add a token
        db = dec_subword.make_decoder(self.vocab, BLANK, str(SPM), lm=MockLM(table), alpha=1.0, beta=12.0, beam=8)
        self.assertEqual(db.decode_ids(lp), [7])          # token bonus + LM: the LM-preferred token
        dn = dec_subword.make_decoder(self.vocab, BLANK, str(SPM), beta=12.0, beam=8)
        self.assertEqual(dn.decode_ids(lp), [3])          # same bonus without LM: the acoustic best

    @unittest.skipUnless(HAVE_KENLM, "kenlm module missing (use ~/.venv-pyctcdecode)")
    def test_subword_kenlm(self):
        uni = {f"t{i}": -2.0 - 0.01 * i for i in range(V)}
        uni.update({"<s>": -99.0, "</s>": -1.0, "<unk>": -5.0})
        lines = ["\\data\\", f"ngram 1={len(uni)}", "ngram 2=1", "", "\\1-grams:"]
        lines += [f"{p}\t{w}\t0" for w, p in uni.items()]
        lines += ["", "\\2-grams:", "-0.5\tt3 t5", "", "\\end\\", ""]
        with tempfile.TemporaryDirectory() as td:
            arpa = Path(td) / "tiny.arpa"
            arpa.write_text("\n".join(lines))
            lm = dec_subword.KenLMTokenLM(str(arpa))
            s, st = lm.score(lm.start(), 3)
            self.assertAlmostEqual(s, -2.03 * math.log(10), places=4)
            s, st = lm.score(st, 5)
            self.assertAlmostEqual(s, -0.5 * math.log(10), places=4)
            self.assertAlmostEqual(lm.final(st), -1.0 * math.log(10), places=4)
            lp = np.full((3, C), -20.0, np.float32)    # t3, then t9 (0.45) vs t5 (0.40) vs blank
            lp[0, 3], lp[1, BLANK] = 0.0, 0.0
            lp[2, [9, 5, BLANK]] = np.log([0.45, 0.40, 0.15])
            no_lm = dec_subword.make_decoder(self.vocab, BLANK, str(SPM), beam=8)
            fused = dec_subword.make_decoder(self.vocab, BLANK, str(SPM), lm=str(arpa), alpha=1.0, beta=1.0,
                                             beam=8)
            self.assertEqual(no_lm.decode_ids(lp), [3, 9])
            self.assertEqual(fused.decode_ids(lp), [3, 5])         # the bigram t3 t5 wins
            rng = np.random.default_rng(3)
            lp = random_logp(rng, 60)
            self.assertIsInstance(fused.decode(lp), str)
            g = dec_subword.make_decoder(self.vocab, BLANK, str(SPM), lm=str(arpa), alpha=0.0, beam=1)
            self.assertEqual(g.decode_ids(lp), dec_greedy.collapse(lp.argmax(-1), BLANK))

    def test_run_decode_end_to_end(self):
        rng = np.random.default_rng(4)
        texts = ["the cat sat on the mat", "hello world", "a quiet evening by the sea", "", "once more"]
        lps = [frames_for(self.sp.encode(t), rng) for t in texts]
        refs = list(texts)
        refs[1] = "hello there world"                     # one insertion needed -> 1 edit
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "lp").mkdir()
            write_store(td / "lp", "synth", lps, refs, self.vocab)
            for dec, extra in (("greedy", []), ("subword", ["--beam", "4"])):
                out = subprocess.run(
                    [sys.executable, str(HERE / "run_decode.py"), "--decoder", dec, "--split", "synth",
                     "--logprobs-dir", str(td / "lp"), "--results-dir", str(td / "res"), "--tag", dec] + extra,
                    capture_output=True, text=True, check=True).stdout
                self.assertIn("WER", out)
                m = json.loads((td / "res" / dec / "metrics.json").read_text())
                hyps = [json.loads(x) for x in (td / "res" / dec / "hyps.jsonl").read_text().splitlines()]
                self.assertEqual([h["hyp"] for h in hyps], texts)
                words = sum(len(r.split()) for r in refs)
                self.assertEqual((m["edits"], m["words"], m["n_utts"]), (1, words, 5))
                self.assertAlmostEqual(m["wer"], 1 / words)
                self.assertAlmostEqual(m["audio_s"], sum(len(x) for x in lps) * 0.04, places=6)
                self.assertGreater(m["peak_rss_bytes"], 1 << 20)
                for k in ("xrt_per_core", "decode_cpu_s", "lm_load_s"):
                    self.assertIn(k, m)
            out = subprocess.run(
                [sys.executable, str(HERE / "run_decode.py"), "--decoder", "greedy", "--split", "synth",
                 "--logprobs-dir", str(td / "lp"), "--results-dir", str(td / "res"), "--limit", "2"],
                capture_output=True, text=True, check=True).stdout
            self.assertIn("synth_greedy_none_a0_b0_beam16_n2 | 2 utts", out)


if __name__ == "__main__":
    unittest.main()
