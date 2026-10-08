"""Credit assignment for executed nodes, ancestors, and strategy pairs."""

from __future__ import annotations

import itertools
from typing import Any

from .graph import RelationalStrategyGraph
from .memory import ExperienceMemory
from .models import ActionMode, ExperienceRecord, RelationType


class CreditAssigner:
    def __init__(self, graph: RelationalStrategyGraph, memory: ExperienceMemory) -> None:
        self.graph = graph
        self.memory = memory

    def assign(
        self,
        experience: ExperienceRecord,
        *,
        executed_strategy_id: str,
        parent_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        evaluation = experience.evaluation
        context_key = experience.context.context_key()
        direct_ids = {executed_strategy_id}
        all_ids = tuple(dict.fromkeys((executed_strategy_id, *parent_ids)))
        updates: dict[str, float] = {}
        for identifier in all_ids:
            if not self.graph.has_strategy(identifier):
                continue
            weight = 1.0 if identifier in direct_ids else 0.35
            self.memory.update_strategy_posterior(
                identifier,
                context_key,
                success=float(evaluation.success),
                progress=evaluation.progress,
                reward=evaluation.reward,
                weight=weight,
            )
            success_mean, progress_mean = self.memory.strategy_means(identifier, "*")
            self.graph.touch_strategy(
                identifier,
                direct=identifier in direct_ids,
                success_mean=success_mean,
                progress_mean=progress_mean,
            )
            updates[identifier] = weight

        pair_diagnostics: list[dict[str, Any]] = []
        pair_status = "not_applicable"
        if experience.action_mode == ActionMode.COMPOSE and len(parent_ids) >= 2:
            baseline = self.memory.best_parent_reward(parent_ids, experience.context)
            if baseline is None:
                pair_status = "deferred_missing_parent_control"
            else:
                pair_status = "updated"
                delta = evaluation.reward - baseline
                positive = delta > 0.02
                for left, right in itertools.combinations(parent_ids, 2):
                    self.memory.update_pair_posterior(
                        left,
                        right,
                        context_key,
                        positive=positive,
                        delta=delta,
                    )
                    self.graph.record_relation_outcome(
                        left,
                        right,
                        RelationType.COMPLEMENTARY_TO,
                        positive=positive,
                        delta=delta,
                    )
                    if delta < -0.08:
                        self.graph.record_relation_outcome(
                            left,
                            right,
                            RelationType.CONFLICTS_WITH,
                            positive=True,
                            delta=-delta,
                        )
                    pair_diagnostics.append(
                        {
                            "left": left,
                            "right": right,
                            "parent_baseline": baseline,
                            "delta": delta,
                            "positive": positive,
                        }
                    )
        return {
            "node_weights": updates,
            "pair_update_status": pair_status,
            "pair_updates": pair_diagnostics,
        }
