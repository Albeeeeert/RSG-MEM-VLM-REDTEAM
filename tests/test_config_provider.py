import unittest
from pathlib import Path

from rsg_mem.config import load_config
from rsg_mem.dataio import load_samples, load_seed_file
from rsg_mem.providers.openai_compatible import extract_json


class ConfigAndProviderTests(unittest.TestCase):
    def test_repository_examples_load(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "mock.yaml")
        seeds = load_seed_file(root / "examples" / "seeds.yaml")
        samples = load_samples(root / "examples" / "samples.jsonl")
        self.assertTrue(config.run_dir.is_absolute())
        self.assertTrue(config.agent.enabled)
        self.assertAlmostEqual(config.agent.advisor_weight, 0.35)
        self.assertIsNone(config.prompt_dir)
        self.assertEqual(len(seeds.strategies), 4)
        self.assertEqual(len(samples), 2)

    def test_json_extraction_accepts_fenced_provider_output(self):
        self.assertEqual(extract_json('```json\n{"valid": true}\n```'), {"valid": True})
        self.assertEqual(extract_json('prefix {"score": 0.7} suffix')["score"], 0.7)


if __name__ == "__main__":
    unittest.main()
