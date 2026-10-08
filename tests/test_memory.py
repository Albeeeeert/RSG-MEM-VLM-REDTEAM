import random
import tempfile
import unittest
from pathlib import Path

from rsg_mem.embeddings import HashingEmbedder
from rsg_mem.memory import ExperienceMemory
from rsg_mem.models import (
    ActionMode,
    CostRecord,
    EvaluationResult,
    ExperienceRecord,
    SampleContext,
    TargetResponse,
    TestCase,
)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.memory = ExperienceMemory(Path(self.temp.name) / "memory.sqlite3")
        self.embedder = HashingEmbedder(64)
        self.context = SampleContext(
            sample_id="s1",
            goal="find synthetic marker",
            goal_category="grounding",
            image_description="blue synthetic card",
            target_model="mock",
        )

    def tearDown(self):
        self.memory.close()
        self.temp.cleanup()

    def _experience(self):
        goal = self.embedder.encode_text(self.context.goal)
        image = self.embedder.encode_image(None, self.context.image_description)
        cross = self.embedder.encode_cross_modal(
            self.context.goal, self.context.image_description
        )
        return ExperienceRecord(
            context=self.context,
            strategy_ids=("a",),
            action_mode=ActionMode.REUSE,
            test_case=TestCase(
                prompt="controlled test",
                strategy_id="a",
                component_strategy_ids=("a",),
                action_mode=ActionMode.REUSE,
            ),
            response=TargetResponse(text="controlled response", model="mock"),
            evaluation=EvaluationResult(0.8, 0.9, 0.8, 0.9, 0.85, True),
            cost=CostRecord(target_queries=1),
            goal_embedding=goal,
            image_embedding=image,
            cross_modal_embedding=cross,
        )

    def test_retrieval_and_round_trip(self):
        experience = self._experience()
        self.memory.add_experience(experience)
        retrieved = self.memory.retrieve(
            self.context,
            goal_embedding=experience.goal_embedding,
            image_embedding=experience.image_embedding,
            cross_modal_embedding=experience.cross_modal_embedding,
            top_k=1,
        )
        self.assertEqual(retrieved[0][0].experience_id, experience.experience_id)
        self.assertGreater(retrieved[0][1], 0.5)
        restored = self.memory.get_experience(experience.experience_id)
        self.assertEqual(restored.response.text, "controlled response")

    def test_contextual_and_pair_posteriors(self):
        key = self.context.context_key()
        self.memory.update_strategy_posterior(
            "a", key, success=1.0, progress=0.8, reward=0.9
        )
        success, progress = self.memory.strategy_means("a", key)
        self.assertGreater(success, 0.5)
        self.assertGreater(progress, 0.5)
        sampled = self.memory.sample_strategy_posterior("a", key, random.Random(2))
        self.assertEqual(len(sampled), 2)
        self.memory.update_pair_posterior("b", "a", key, positive=True, delta=0.2)
        posterior = self.memory.pair_posterior("a", "b", key)
        self.assertGreater(posterior["alpha"], posterior["beta"])
        self.assertAlmostEqual(posterior["mean_delta"], 0.2)


if __name__ == "__main__":
    unittest.main()
