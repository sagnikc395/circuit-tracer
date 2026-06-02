import unittest

import torch

from ioi_repro.metrics import logit_diff, logit_diff_recovery


class MetricsTest(unittest.TestCase):
    def test_logit_diff_uses_final_position(self):
        logits = torch.zeros((2, 3, 5))
        logits[0, -1, 1] = 3.0
        logits[0, -1, 2] = 1.0
        logits[1, -1, 3] = -1.0
        logits[1, -1, 4] = -4.0

        diff = logit_diff(
            logits,
            torch.tensor([1, 3]),
            torch.tensor([2, 4]),
            mean=False,
        )

        self.assertTrue(torch.equal(diff, torch.tensor([2.0, 3.0])))
        self.assertEqual(float(logit_diff(logits, torch.tensor([1, 3]), torch.tensor([2, 4]))), 2.5)

    def test_logit_diff_recovery_normalizes_against_baselines(self):
        recovery = logit_diff_recovery(
            torch.tensor(3.0),
            clean_diff=torch.tensor(5.0),
            corrupted_diff=torch.tensor(1.0),
        )

        self.assertAlmostEqual(float(recovery), 0.5)


if __name__ == "__main__":
    unittest.main()
