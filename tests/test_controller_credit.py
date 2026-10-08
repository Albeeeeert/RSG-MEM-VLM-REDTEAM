import tempfile
import unittest
from pathlib import Path

from rsg_mem.config import LifecycleSettings, SearchSettings
from rsg_mem.controller import GraphController
from rsg_mem.credit import CreditAssigner
from rsg_mem.embeddings import HashingEmbedder
from rsg_mem.graph import RelationalStrategyGraph
from rsg_mem.lifecycle import LifecycleManager
from rsg_mem.memory import ExperienceMemory
from rsg_mem.models import (
    ActionMode,
    CostRecord,
    EvaluationResult,
    ExperienceRecord,
    LifecycleState,
    RelationType,
    SampleContext,
    StrategySpec,
    TargetResponse,
    TestCase,
)


class ControllerAndCreditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.graph = RelationalStrategyGraph()
        self.memory = ExperienceMemory(Path(self.temp.name) / "memory.sqlite3")
        self.embedder = HashingEmbedder(64)
        for identifier, name in (("a", "visual grounding"), ("b", "context framing")):
            spec = StrategySpec(name=name, description=name, mechanism=name)
            self.graph.add_strategy(
                spec,
                self.embedder.encode_text(spec.text()),
                strategy_id=identifier,
                lifecycle_state=LifecycleState.ACTIVE,
            )
        self.graph.add_relation("a", "b", RelationType.COMPLEMENTARY_TO, weight=0.9)
        self.context = SampleContext(
            sample_id="sample",
            goal="visual grounding benchmark",
            goal_category="grounding",
            image_description="synthetic card",
            target_model="mock",
        )

    def tearDown(self):
        self.memory.close()
        self.temp.cleanup()

    def test_controller_exposes_all_four_graph_actions(self):
        settings = SearchSettings(
            reuse_candidates=2,
            refine_candidates=2,
            compose_candidates=2,
            invent_candidates=1,
        )
        controller = GraphController(
            self.graph, self.memory, self.embedder, settings, random_seed=3
        )
        vectors = controller.encode_context(self.context)
        candidates = controller.generate_candidates(
            self.context, vectors, [], stagnation=settings.stagnation_limit
        )
        modes = {candidate.mode for candidate in candidates}
        self.assertEqual(
            modes,
            {ActionMode.REUSE, ActionMode.REFINE, ActionMode.COMPOSE, ActionMode.INVENT},
        )
        composition = next(item for item in candidates if item.mode == ActionMode.COMPOSE)
        self.assertEqual(set(composition.strategy_ids), {"a", "b"})
        self.assertGreater(composition.component_scores["synergy"], 0.0)
        self.assertIn(controller.select(candidates), candidates)

    def test_composition_credit_updates_pair_and_lifecycle(self):
        for parent_id in ("a", "b"):
            parent_case = TestCase(
                prompt=f"controlled {parent_id}",
                strategy_id=parent_id,
                component_strategy_ids=(parent_id,),
                action_mode=ActionMode.REUSE,
            )
            self.memory.add_experience(
                ExperienceRecord(
                    context=self.context,
                    strategy_ids=(parent_id,),
                    action_mode=ActionMode.REUSE,
                    test_case=parent_case,
                    response=TargetResponse(text="controlled", model="mock"),
                    evaluation=EvaluationResult(0.5, 0.7, 0.6, 0.8, 0.6, False),
                    cost=CostRecord(target_queries=1),
                    goal_embedding=self.embedder.encode_text(self.context.goal),
                    image_embedding=self.embedder.encode_image(
                        None, self.context.image_description
                    ),
                    cross_modal_embedding=self.embedder.encode_cross_modal(
                        self.context.goal, self.context.image_description
                    ),
                )
            )
        composition_spec = StrategySpec(
            name="composition", description="combined", mechanism="joint"
        )
        child = self.graph.add_composition(
            composition_spec,
            ("a", "b"),
            self.embedder.encode_text(composition_spec.text()),
        )
        test_case = TestCase(
            prompt="controlled composition",
            strategy_id=child.strategy_id,
            component_strategy_ids=("a", "b"),
            action_mode=ActionMode.COMPOSE,
        )
        experience = ExperienceRecord(
            context=self.context,
            strategy_ids=(child.strategy_id, "a", "b"),
            action_mode=ActionMode.COMPOSE,
            test_case=test_case,
            response=TargetResponse(text="controlled", model="mock"),
            evaluation=EvaluationResult(0.9, 0.9, 0.9, 0.9, 0.9, True),
            cost=CostRecord(target_queries=1),
            goal_embedding=self.embedder.encode_text(self.context.goal),
            image_embedding=self.embedder.encode_image(None, self.context.image_description),
            cross_modal_embedding=self.embedder.encode_cross_modal(
                self.context.goal, self.context.image_description
            ),
        )
        self.memory.add_experience(experience)
        diagnostics = CreditAssigner(self.graph, self.memory).assign(
            experience,
            executed_strategy_id=child.strategy_id,
            parent_ids=("a", "b"),
        )
        self.assertEqual(diagnostics["pair_update_status"], "updated")
        self.assertEqual(len(diagnostics["pair_updates"]), 1)
        self.assertGreater(
            self.memory.pair_posterior("a", "b", self.context.context_key())["observations"],
            0,
        )
        manager = LifecycleManager(
            self.graph,
            LifecycleSettings(promotion_min_evidence=1, promotion_success_mean=0.5),
        )
        transitions = manager.update()
        self.assertEqual(child.lifecycle_state, LifecycleState.ACTIVE)
        self.assertTrue(any(item["strategy_id"] == child.strategy_id for item in transitions))


if __name__ == "__main__":
    unittest.main()
