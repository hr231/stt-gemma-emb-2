"""Lookahead budget of head.py's CausalHead: the printed lookahead (from masks + conv padding) is
exactly the receptive field into the future.

  python -m unittest test_head.py
"""
import unittest

import torch

from head import CausalHead

# (attention future frames per block, strictly causal convs, expected total lookahead in frames)
CASES = [
    ((0, 0, 0, 0), False, 0),     # R=0
    ((4, 4, 4, 4), False, 24),    # R=4: 4 x (4 attention + 2 conv)
    ((0, 0, 0, 4), True, 4),      # 160 ms budget: last block only
    ((0, 0, 4, 4), True, 8),      # 320 ms budget: last two blocks
]


class LookaheadBudget(unittest.TestCase):
    def test_output_at_t_sees_exactly_budget_future_frames(self):
        T, t = 120, 50
        for futures, causal_conv, budget in CASES:
            with self.subTest(futures=futures, causal_conv=causal_conv):
                torch.manual_seed(0)
                model = CausalHead(1024, 257, future=futures, causal_conv=causal_conv).double().eval()
                self.assertEqual(model.lookahead, budget)
                x = torch.randn(1, T, 1024, dtype=torch.float64)
                L = torch.tensor([T])
                with torch.no_grad():
                    ref = model(x, L)[0, t]
                    beyond = x.clone()
                    beyond[:, t + budget + 1:] += torch.randn(1, T - t - budget - 1, 1024, dtype=torch.float64)
                    self.assertTrue(torch.equal(model(beyond, L)[0, t], ref),
                                    f"frames after t + {budget} changed the output at t")
                    edge = x.clone()
                    edge[:, t + budget] += torch.randn(1024, dtype=torch.float64)
                    self.assertFalse(torch.equal(model(edge, L)[0, t], ref),
                                     f"frame t + {budget} did not change the output at t")


if __name__ == "__main__":
    unittest.main()
