import tempfile
import unittest
from pathlib import Path

from rsg_mem.embeddings import HashingEmbedder
from rsg_mem.graph import RelationalStrategyGraph
from rsg_mem.models import LifecycleState, RelationType, StrategySpec


def spec(name: str) -> StrategySpec:
    return StrategySpec(name=name, description=f"{name} description", mechanism=f"{name} mechanism")


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.embedder = HashingEmbedder(64)
        self.graph = RelationalStrategyGraph()
        for identifier, name in (("a", "alpha"), ("b", "beta")):
            value = spec(name)
            self.graph.add_strategy(
                value,
                self.embedder.encode_text(value.text()),
                strategy_id=identifier,
                lifecycle_state=LifecycleState.ACTIVE,
            )

    def test_typed_composition_and_persistence(self):
        combined = spec("alpha beta composition")
        child = self.graph.add_composition(
            combined, ("a", "b"), self.embedder.encode_text(combined.text())
        )
        self.assertEqual(child.lineage, ("a", "b"))
        self.assertIsNotNone(
            self.graph.get_relation("a", child.strategy_id, RelationType.COMPOSED_FROM)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            self.graph.save(path)
            restored = RelationalStrategyGraph.load(path)
            self.assertEqual(restored.stats(), self.graph.stats())
            self.assertEqual(restored.get_strategy(child.strategy_id).spec.name, combined.name)

    def test_conflict_blocks_composition_and_local_subgraph(self):
        self.graph.add_relation("a", "b", RelationType.CONFLICTS_WITH, weight=0.9)
        self.assertFalse(self.graph.composition_allowed("a", "b", 0.65))
        self.assertEqual(set(self.graph.local_strategy_ids(["a"], hops=1)), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
