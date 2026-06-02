import unittest

from ioi_repro.dataset import build_ioi_dataset


class DatasetTest(unittest.TestCase):
    def test_builds_paired_clean_and_corrupted_prompts(self):
        examples = build_ioi_dataset(8, seed=1)

        self.assertEqual(len(examples), 8)
        for example in examples:
            self.assertNotEqual(example.io_name, example.s_name)
            self.assertNotEqual(example.clean_prompt, example.corrupted_prompt)
            self.assertTrue(example.clean_prompt.endswith(" to"))
            self.assertTrue(example.corrupted_prompt.endswith(" to"))
            self.assertIn(example.io_name, example.clean_prompt)
            self.assertIn(example.s_name, example.clean_prompt)

    def test_seed_is_deterministic(self):
        first = build_ioi_dataset(4, seed=123)
        second = build_ioi_dataset(4, seed=123)

        self.assertEqual(first, second)

    def test_can_select_single_template_id(self):
        examples = build_ioi_dataset(4, seed=1, template_id="abba_store")

        self.assertEqual({example.template_id for example in examples}, {"abba_store"})


if __name__ == "__main__":
    unittest.main()
