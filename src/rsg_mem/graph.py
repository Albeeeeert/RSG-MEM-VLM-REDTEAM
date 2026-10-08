"""Persistent typed relational strategy graph."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Iterable

import networkx as nx

from .embeddings import cosine
from .models import (
    LifecycleState,
    RelationType,
    StrategyNode,
    StrategyRelation,
    StrategySpec,
    make_id,
    utc_now,
)


SYMMETRIC_RELATIONS = {
    RelationType.SIMILAR_TO,
    RelationType.COMPLEMENTARY_TO,
    RelationType.CONFLICTS_WITH,
}


class RelationalStrategyGraph:
    """A typed graph whose nodes are abstract strategies, not prompt instances."""

    schema_version = 1

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()

    def __len__(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def relation_count(self) -> int:
        return self._graph.number_of_edges()

    def has_strategy(self, strategy_id: str) -> bool:
        return self._graph.has_node(strategy_id)

    def add_strategy(
        self,
        spec: StrategySpec,
        embedding: list[float],
        *,
        strategy_id: str | None = None,
        lifecycle_state: LifecycleState = LifecycleState.CANDIDATE,
        lineage: Iterable[str] = (),
    ) -> StrategyNode:
        identifier = strategy_id or make_id("strategy")
        if self.has_strategy(identifier):
            raise ValueError(f"strategy already exists: {identifier}")
        node = StrategyNode(
            strategy_id=identifier,
            spec=spec,
            embedding=list(embedding),
            lifecycle_state=lifecycle_state,
            lineage=tuple(lineage),
        )
        self._graph.add_node(identifier, node=node)
        return node

    def get_strategy(self, strategy_id: str) -> StrategyNode:
        try:
            return self._graph.nodes[strategy_id]["node"]
        except KeyError as exc:
            raise KeyError(f"unknown strategy: {strategy_id}") from exc

    def strategies(
        self, states: Iterable[LifecycleState] | None = None
    ) -> list[StrategyNode]:
        allowed = set(states) if states is not None else None
        nodes = [data["node"] for _, data in self._graph.nodes(data=True)]
        if allowed is not None:
            nodes = [node for node in nodes if node.lifecycle_state in allowed]
        return sorted(nodes, key=lambda node: node.strategy_id)

    def set_state(self, strategy_id: str, state: LifecycleState) -> None:
        node = self.get_strategy(strategy_id)
        node.lifecycle_state = state
        node.updated_at = utc_now()

    def touch_strategy(
        self,
        strategy_id: str,
        *,
        direct: bool,
        success_mean: float,
        progress_mean: float,
    ) -> None:
        node = self.get_strategy(strategy_id)
        node.usage_count += 1
        if direct:
            node.direct_evidence += 1
        node.cached_success_mean = float(success_mean)
        node.cached_progress_mean = float(progress_mean)
        node.updated_at = utc_now()

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        *,
        weight: float = 0.5,
        metadata: dict | None = None,
    ) -> StrategyRelation:
        if not self.has_strategy(source_id) or not self.has_strategy(target_id):
            raise KeyError("both relation endpoints must already exist")
        key = relation_type.value
        if self._graph.has_edge(source_id, target_id, key=key):
            relation = self._graph[source_id][target_id][key]["relation"]
            relation.weight = float(weight)
            if metadata:
                relation.metadata.update(metadata)
            return relation
        relation = StrategyRelation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=float(weight),
            metadata=dict(metadata or {}),
        )
        self._graph.add_edge(source_id, target_id, key=key, relation=relation)
        return relation

    def get_relation(
        self, source_id: str, target_id: str, relation_type: RelationType
    ) -> StrategyRelation | None:
        key = relation_type.value
        if self._graph.has_edge(source_id, target_id, key=key):
            return self._graph[source_id][target_id][key]["relation"]
        if relation_type in SYMMETRIC_RELATIONS and self._graph.has_edge(
            target_id, source_id, key=key
        ):
            return self._graph[target_id][source_id][key]["relation"]
        return None

    def relation_weight(
        self, source_id: str, target_id: str, relation_type: RelationType
    ) -> float:
        relation = self.get_relation(source_id, target_id, relation_type)
        return relation.weight if relation else 0.0

    def record_relation_outcome(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        *,
        positive: bool,
        delta: float = 0.0,
    ) -> StrategyRelation:
        relation = self.get_relation(source_id, target_id, relation_type)
        if relation is None:
            relation = self.add_relation(source_id, target_id, relation_type)
        old_count = relation.evidence_count
        relation.evidence_count += 1
        relation.positive_count += int(positive)
        relation.negative_count += int(not positive)
        relation.mean_delta = (
            relation.mean_delta * old_count + float(delta)
        ) / relation.evidence_count
        relation.weight = (1.0 + relation.positive_count) / (
            2.0 + relation.evidence_count
        )
        return relation

    def add_refinement(
        self,
        spec: StrategySpec,
        parent_id: str,
        embedding: list[float],
    ) -> StrategyNode:
        child = self.add_strategy(spec, embedding, lineage=(parent_id,))
        self.add_relation(parent_id, child.strategy_id, RelationType.REFINED_FROM, weight=1.0)
        return child

    def add_composition(
        self,
        spec: StrategySpec,
        parent_ids: Iterable[str],
        embedding: list[float],
    ) -> StrategyNode:
        parents = tuple(dict.fromkeys(parent_ids))
        if len(parents) < 2:
            raise ValueError("a composition requires at least two distinct parent strategies")
        child = self.add_strategy(spec, embedding, lineage=parents)
        for parent_id in parents:
            self.add_relation(
                parent_id,
                child.strategy_id,
                RelationType.COMPOSED_FROM,
                weight=1.0,
            )
        return child

    def find_similar(
        self,
        embedding: Iterable[float],
        *,
        threshold: float = 0.92,
        states: Iterable[LifecycleState] | None = None,
    ) -> list[tuple[StrategyNode, float]]:
        scored = [
            (node, cosine(embedding, node.embedding)) for node in self.strategies(states=states)
        ]
        return sorted(
            [(node, score) for node, score in scored if score >= threshold],
            key=lambda item: item[1],
            reverse=True,
        )

    def local_strategy_ids(
        self,
        seed_ids: Iterable[str],
        *,
        hops: int = 1,
        limit: int = 40,
        include_retired: bool = False,
    ) -> list[str]:
        seeds = [identifier for identifier in seed_ids if self.has_strategy(identifier)]
        if not seeds:
            seeds = [
                node.strategy_id
                for node in self.strategies(
                    states=(LifecycleState.ACTIVE, LifecycleState.CANDIDATE)
                )
            ]
        undirected = self._graph.to_undirected()
        seen: set[str] = set()
        queue: deque[tuple[str, int]] = deque((identifier, 0) for identifier in seeds)
        result: list[str] = []
        while queue and len(result) < limit:
            identifier, depth = queue.popleft()
            if identifier in seen:
                continue
            seen.add(identifier)
            node = self.get_strategy(identifier)
            allowed = node.lifecycle_state != LifecycleState.ELIMINATED and (
                include_retired or node.lifecycle_state != LifecycleState.RETIRED
            )
            if allowed:
                result.append(identifier)
            if depth < hops:
                queue.extend((neighbor, depth + 1) for neighbor in undirected.neighbors(identifier))
        return result

    def composition_allowed(self, left_id: str, right_id: str, conflict_limit: float) -> bool:
        if left_id == right_id:
            return False
        left = self.get_strategy(left_id)
        right = self.get_strategy(right_id)
        if LifecycleState.ELIMINATED in (left.lifecycle_state, right.lifecycle_state):
            return False
        conflict = self.relation_weight(left_id, right_id, RelationType.CONFLICTS_WITH)
        return conflict < conflict_limit

    def merge_duplicate(self, duplicate_id: str, canonical_id: str) -> None:
        if duplicate_id == canonical_id:
            return
        duplicate = self.get_strategy(duplicate_id)
        canonical = self.get_strategy(canonical_id)
        duplicate.lifecycle_state = LifecycleState.ELIMINATED
        duplicate.updated_at = utc_now()
        canonical.lineage = tuple(dict.fromkeys(canonical.lineage + (duplicate_id,)))
        self.add_relation(duplicate_id, canonical_id, RelationType.SUBSUMES, weight=1.0)

    def relation_records(self) -> list[StrategyRelation]:
        return [data["relation"] for *_, data in self._graph.edges(data=True, keys=True)]

    def stats(self) -> dict[str, object]:
        state_counts = {
            state.value: sum(node.lifecycle_state == state for node in self.strategies())
            for state in LifecycleState
        }
        relation_counts = {
            relation_type.value: sum(
                relation.relation_type == relation_type for relation in self.relation_records()
            )
            for relation_type in RelationType
        }
        return {
            "strategies": len(self),
            "relations": self.relation_count,
            "states": state_counts,
            "relation_types": relation_counts,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "nodes": [node.to_dict() for node in self.strategies()],
            "relations": [relation.to_dict() for relation in self.relation_records()],
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(destination)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "RelationalStrategyGraph":
        graph = cls()
        for raw_node in data.get("nodes", []):
            node = StrategyNode.from_dict(raw_node)
            graph._graph.add_node(node.strategy_id, node=node)
        for raw_relation in data.get("relations", []):
            relation = StrategyRelation.from_dict(raw_relation)
            graph._graph.add_edge(
                relation.source_id,
                relation.target_id,
                key=relation.relation_type.value,
                relation=relation,
            )
        return graph

    @classmethod
    def load(cls, path: str | Path) -> "RelationalStrategyGraph":
        source = Path(path)
        if not source.exists():
            return cls()
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))

    def make_stable_seed_id(self, name: str) -> str:
        slug = "".join(character.lower() if character.isalnum() else "_" for character in name)
        slug = "_".join(part for part in slug.split("_") if part)
        identifier = f"seed_{slug[:48]}" or make_id("seed")
        suffix = 2
        candidate = identifier
        while self.has_strategy(candidate):
            candidate = f"{identifier}_{suffix}"
            suffix += 1
        return candidate
