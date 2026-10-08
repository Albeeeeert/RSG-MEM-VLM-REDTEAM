import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from rsg_mem.cli import main
from rsg_mem.config import ProviderSettings
from rsg_mem.controller import GraphController
from rsg_mem.models import ActionMode, CandidateAction, SampleContext
from rsg_mem.prompting import PROMPT_NAMES, PromptLibrary
from rsg_mem.providers.mock import RuleBasedStrategyAgent
from rsg_mem.providers.openai_compatible import OpenAICompatibleStrategyAgent


class FakeChatClient:
    def __init__(self, response: dict):
        self.response = response
        self.messages = None

    def chat(self, messages, **kwargs):
        self.messages = messages
        return json.dumps(self.response), {"provider": "fake"}, 0.01


class PromptingAndAgentTests(unittest.TestCase):
    def test_builtin_prompt_manifest_is_complete_and_versioned(self):
        library = PromptLibrary()
        manifest = library.manifest()
        self.assertEqual(set(manifest), set(PROMPT_NAMES))
        self.assertIn("graph-aware planning agent", library.get("action_advisor"))
        self.assertIn("retrieved success and failure memories", library.get("strategy_designer"))
        self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest.values()))

    def test_single_prompt_override_falls_back_for_other_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            override = Path(directory)
            (override / "action_advisor.md").write_text(
                "experiment-specific advisor", encoding="utf-8"
            )
            library = PromptLibrary(override)
            self.assertEqual(library.get("action_advisor"), "experiment-specific advisor")
            self.assertIn("abstract", library.get("strategy_designer"))

    def test_cli_prints_named_prompt(self):
        output = StringIO()
        with redirect_stdout(output):
            status = main(["prompts", "--show", "action_advisor"])
        self.assertEqual(status, 0)
        self.assertIn("graph-aware planning agent", output.getvalue())

    def test_mock_advisor_and_controller_fusion_use_structured_scores(self):
        low = CandidateAction(mode=ActionMode.REUSE, action_id="low", utility=0.2)
        high = CandidateAction(mode=ActionMode.INVENT, action_id="high", utility=0.8)
        evidence = {"retrieved_memories": [{"experience_id": "exp_1"}]}
        decision = RuleBasedStrategyAgent().advise(
            [low, high], SampleContext(sample_id="s", goal="controlled"), evidence
        )
        self.assertEqual(decision.selected_action_id, "high")
        self.assertEqual(decision.memory_references, ("exp_1",))
        selected = GraphController.select(
            [low, high], {"low": 1.0, "high": 0.0}, advisor_weight=1.0
        )
        self.assertEqual(selected.action_id, "low")
        self.assertEqual(low.component_scores["agent_advice"], 1.0)

    def test_openai_advisor_receives_candidates_and_memory_evidence(self):
        settings = ProviderSettings(
            type="openai_compatible", base_url="http://127.0.0.1:9/v1", model="fake"
        )
        agent = OpenAICompatibleStrategyAgent(settings)
        agent.client.close()
        fake = FakeChatClient(
            {
                "selected_action_id": "a2",
                "ranked_actions": [
                    {"action_id": "a2", "score": 0.9, "reason": "memory match"},
                    {"action_id": "a1", "score": 0.2, "reason": "prior failure"},
                ],
                "rationale": "use retrieved evidence",
                "memory_references": ["exp_1", "not_supplied"],
            }
        )
        agent.client = fake
        candidates = [
            CandidateAction(mode=ActionMode.REUSE, action_id="a1"),
            CandidateAction(mode=ActionMode.REFINE, action_id="a2"),
        ]
        evidence = {"retrieved_memories": [{"experience_id": "exp_1", "success": True}]}
        decision = agent.advise(
            candidates,
            SampleContext(sample_id="sample", goal="controlled objective"),
            evidence,
        )
        self.assertEqual(decision.selected_action_id, "a2")
        self.assertEqual(decision.memory_references, ("exp_1",))
        self.assertIn("graph-aware planning agent", fake.messages[0]["content"])
        payload = json.loads(fake.messages[1]["content"])
        self.assertEqual(payload["evidence"], evidence)
        self.assertEqual({item["action_id"] for item in payload["candidates"]}, {"a1", "a2"})


if __name__ == "__main__":
    unittest.main()
