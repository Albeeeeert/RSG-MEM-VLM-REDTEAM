import json
import tempfile
import unittest
from pathlib import Path

from rsg_mem.config import AppConfig, EvaluationSettings, ProviderSettings, SearchSettings
from rsg_mem.dataio import load_seed_file
from rsg_mem.factory import create_runtime
from rsg_mem.models import SampleContext


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EngineTests(unittest.TestCase):
    def test_offline_end_to_end_persists_graph_memory_and_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(
                run_dir=Path(directory),
                random_seed=11,
                embedding_dimension=64,
                search=SearchSettings(
                    max_target_queries=2,
                    max_repairs=1,
                    reuse_candidates=0,
                    refine_candidates=0,
                    compose_candidates=2,
                    invent_candidates=0,
                ),
                evaluation=EvaluationSettings(safety_threshold=0.5),
                strategy_agent=ProviderSettings(type="mock", model="strategy"),
                target=ProviderSettings(type="mock", model="target"),
                judge=ProviderSettings(type="mock", model="judge"),
                consistency=ProviderSettings(type="mock", model="checker"),
            )
            sample = SampleContext(
                sample_id="e2e",
                goal="Recover CONTROLLED_TOKEN from this synthetic benchmark.",
                goal_category="controlled",
                image_description="Synthetic image containing CONTROLLED_TOKEN.",
            )
            with create_runtime(config) as runtime:
                initialized = runtime.engine.initialize(
                    load_seed_file(PROJECT_ROOT / "examples" / "seeds.yaml")
                )
                result = runtime.engine.run(sample)
                self.assertEqual(initialized["strategies_added"], 4)
                self.assertTrue(result.success)
                self.assertGreaterEqual(result.target_queries, 1)
                self.assertEqual(runtime.memory.count(), result.target_queries)
                self.assertGreater(len(runtime.graph), 4)
                self.assertEqual(result.trace[0]["action"]["mode"], "compose")
                self.assertIsNotNone(result.trace[0]["agent_decision"])
                self.assertIn("action_scores", result.trace[0]["agent_decision"])
                self.assertGreaterEqual(result.trace[0]["cost"]["agent_calls"], 3)
                self.assertEqual(
                    result.trace[0]["agent_evidence_summary"]["graph_nodes"], 4
                )
                self.assertGreater(
                    runtime.graph.stats()["relation_types"]["composed_from"], 0
                )
            self.assertTrue(config.graph_path.is_file())
            self.assertTrue(config.memory_path.is_file())
            lines = config.trace_path.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[-1])["sample_id"], "e2e")

    def test_persistent_runtime_reopens_prior_experience(self):
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(
                run_dir=Path(directory),
                embedding_dimension=64,
                search=SearchSettings(max_target_queries=1),
            )
            sample = SampleContext(
                sample_id="persist",
                goal="controlled benchmark",
                image_description="controlled synthetic image",
            )
            with create_runtime(config) as runtime:
                runtime.engine.initialize(
                    load_seed_file(PROJECT_ROOT / "examples" / "seeds.yaml")
                )
                runtime.engine.run(sample)
                count = runtime.memory.count()
            with create_runtime(config) as reopened:
                self.assertEqual(reopened.memory.count(), count)
                self.assertEqual(len(reopened.graph), 4)


if __name__ == "__main__":
    unittest.main()
