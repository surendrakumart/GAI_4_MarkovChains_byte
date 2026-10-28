import random
import unittest

import generate


class MarkovGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokens = generate.tokenize(generate.CORPUS_PATH.read_text(encoding="utf-8"))
        cls.generator = generate.MarkovTextGenerator(cls.tokens, order=3)

    def test_all_prompts_exist_in_corpus(self):
        seeds = generate.read_seeds(generate.PROMPTS_PATH)
        self.assertEqual(len(seeds), 20)
        for seed in seeds:
            self.assertIn(seed, self.generator.positions)

    def test_generation_starts_with_seed(self):
        output = self.generator.generate("mira", token_limit=30, rng=random.Random(123))
        self.assertTrue(output.lower().startswith("mira"))
        self.assertGreater(len(output.split()), 5)

    def test_generation_is_reproducible_with_same_rng_seed(self):
        first = self.generator.generate("the", token_limit=30, rng=random.Random(99))
        second = self.generator.generate("the", token_limit=30, rng=random.Random(99))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
  
