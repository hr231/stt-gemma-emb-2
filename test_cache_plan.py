"""Shard plan of cache_feats.py: shards are seeded random samples sorted by duration inside, every
shard spans the full length range, batches never cross shards, and the same seed gives the same
plan (needed for resume). No model is loaded.

  python -m unittest test_cache_plan.py
"""
import random
import unittest
from pathlib import Path

from cache_feats import make_plan, plan_shards, shard_layout

TRAIN_CLEAN_100 = Path("~/gemma_stt/data/LibriSpeech/train-clean-100").expanduser()


def synthetic(n, seed=0):
    rng = random.Random(seed)
    return [{"id": f"u{i}", "dur": rng.uniform(1.0, 25.0)} for i in range(n)]


def assert_full_range_shards(test, utts, sizes, n_shards):
    durs = sorted(u["dur"] for u in utts)
    p5, p95 = durs[len(durs) // 20], durs[-(len(durs) // 20)]
    start = 0
    for k, n in enumerate(sizes[:n_shards]):
        d = [u["dur"] for u in utts[start:start + n]]
        test.assertEqual(d, sorted(d), f"shard {k} not sorted by duration")
        test.assertLessEqual(d[0], p5, f"shard {k} misses the short end")
        test.assertGreaterEqual(d[-1], p95, f"shard {k} misses the long end")
        start += n


class ShardPlan(unittest.TestCase):
    def test_layout_is_a_seeded_partition_of_full_range_shards(self):
        rows = synthetic(2000)
        utts, sizes = shard_layout(rows, 200, seed=0)
        self.assertEqual(sizes, [200] * 10)
        self.assertEqual(sorted(u["id"] for u in utts), sorted(r["id"] for r in rows))
        self.assertEqual([u["id"] for u in shard_layout(rows, 200, seed=0)[0]], [u["id"] for u in utts])
        self.assertNotEqual([u["id"] for u in shard_layout(rows, 200, seed=1)[0]], [u["id"] for u in utts])
        assert_full_range_shards(self, utts, sizes, len(sizes))

    def test_batches_stay_inside_their_shard(self):
        utts, sizes = shard_layout(synthetic(2050), 200, seed=0)          # last shard is partial
        batches, shards = plan_shards(utts, sizes, batch_seconds=240.0, max_batch=32)
        start, order = 0, []
        for n, shard in zip(sizes, shards):
            members = sorted(j for bi in shard for j in batches[bi])
            self.assertEqual(members, list(range(start, start + n)))
            for bi in shard:
                b = batches[bi]
                self.assertLessEqual(len(b), 32)
                self.assertTrue(len(b) == 1 or max(utts[j]["dur"] for j in b) * len(b) <= 240.0)
            order += shard
            start += n
        self.assertEqual(order, list(range(len(batches))))               # shards own consecutive batches

    @unittest.skipUnless(TRAIN_CLEAN_100.is_dir(), "needs LibriSpeech train-clean-100")
    def test_train_clean_100(self):
        root = TRAIN_CLEAN_100.parent.parent
        utts, sizes = make_plan(root, ["train-clean-100"], None, 0, 1000)
        again, _ = make_plan(root, ["train-clean-100"], None, 0, 1000)
        self.assertEqual([u["id"] for u in again], [u["id"] for u in utts])  # same seed, same plan
        assert_full_range_shards(self, utts, sizes, 5)


if __name__ == "__main__":
    unittest.main()
