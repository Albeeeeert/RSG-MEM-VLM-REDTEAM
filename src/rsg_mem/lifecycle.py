"""Evidence-driven strategy lifecycle management."""

from __future__ import annotations

from .config import LifecycleSettings
from .graph import RelationalStrategyGraph
from .models import LifecycleState


class LifecycleManager:
    def __init__(self, graph: RelationalStrategyGraph, settings: LifecycleSettings) -> None:
        self.graph = graph
        self.settings = settings

    def update(self) -> list[dict[str, str]]:
        transitions: list[dict[str, str]] = []
        for node in self.graph.strategies():
            before = node.lifecycle_state
            if before == LifecycleState.ELIMINATED:
                continue
            if (
                before == LifecycleState.CANDIDATE
                and node.direct_evidence >= self.settings.promotion_min_evidence
                and node.cached_success_mean >= self.settings.promotion_success_mean
            ):
                self.graph.set_state(node.strategy_id, LifecycleState.ACTIVE)
            elif (
                before == LifecycleState.RETIRED
                and node.cached_success_mean >= self.settings.reactivation_success_mean
            ):
                self.graph.set_state(node.strategy_id, LifecycleState.ACTIVE)
            elif (
                node.usage_count >= self.settings.elimination_min_uses
                and node.cached_success_mean < self.settings.retirement_success_mean / 2.0
            ):
                self.graph.set_state(node.strategy_id, LifecycleState.ELIMINATED)
            elif (
                before == LifecycleState.ACTIVE
                and node.usage_count >= self.settings.retirement_min_uses
                and node.cached_success_mean < self.settings.retirement_success_mean
            ):
                self.graph.set_state(node.strategy_id, LifecycleState.RETIRED)
            after = self.graph.get_strategy(node.strategy_id).lifecycle_state
            if after != before:
                transitions.append(
                    {"strategy_id": node.strategy_id, "from": before.value, "to": after.value}
                )

        resident = [
            node
            for node in self.graph.strategies()
            if node.lifecycle_state != LifecycleState.ELIMINATED
        ]
        overflow = len(resident) - self.settings.max_resident_strategies
        if overflow > 0:
            removable = sorted(
                resident,
                key=lambda node: (
                    node.lifecycle_state == LifecycleState.ACTIVE,
                    node.cached_success_mean,
                    node.direct_evidence,
                    node.updated_at,
                ),
            )
            for node in removable[:overflow]:
                before = node.lifecycle_state
                self.graph.set_state(node.strategy_id, LifecycleState.ELIMINATED)
                transitions.append(
                    {
                        "strategy_id": node.strategy_id,
                        "from": before.value,
                        "to": LifecycleState.ELIMINATED.value,
                    }
                )
        return transitions
