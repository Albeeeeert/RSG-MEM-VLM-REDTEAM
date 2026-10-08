"""End-to-end RSG-Mem exploration engine."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from .actuator import MultimodalActuator
from .config import AppConfig
from .controller import GraphController
from .credit import CreditAssigner
from .dataio import SeedBundle, append_jsonl
from .embeddings import Embedder
from .graph import RelationalStrategyGraph
from .lifecycle import LifecycleManager
from .memory import ExperienceMemory
from .models import (
    ActionMode,
    CandidateAction,
    CostRecord,
    ExperienceRecord,
    FailureType,
    LifecycleState,
    RelationType,
    RunResult,
    SampleContext,
    StrategyNode,
)
from .providers.base import ConsistencyChecker, Judge, StrategyAgent, TargetClient


class RSGMemEngine:
    """Coordinates retrieval, graph actions, execution, evaluation, and learning."""

    def __init__(
        self,
        *,
        config: AppConfig,
        graph: RelationalStrategyGraph,
        memory: ExperienceMemory,
        embedder: Embedder,
        controller: GraphController,
        strategy_agent: StrategyAgent,
        actuator: MultimodalActuator,
        target: TargetClient,
        judge: Judge,
        consistency_checker: ConsistencyChecker,
        credit_assigner: CreditAssigner,
        lifecycle_manager: LifecycleManager,
    ) -> None:
        self.config = config
        self.graph = graph
        self.memory = memory
        self.embedder = embedder
        self.controller = controller
        self.strategy_agent = strategy_agent
        self.actuator = actuator
        self.target = target
        self.judge = judge
        self.consistency_checker = consistency_checker
        self.credit_assigner = credit_assigner
        self.lifecycle_manager = lifecycle_manager
        self.config.run_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self, bundle: SeedBundle) -> dict[str, int]:
        added = 0
        for seed in bundle.strategies:
            if self.graph.has_strategy(seed.strategy_id):
                continue
            self.graph.add_strategy(
                seed.spec,
                self.embedder.encode_text(seed.spec.text()),
                strategy_id=seed.strategy_id,
                lifecycle_state=LifecycleState.ACTIVE,
            )
            added += 1
        relations = 0
        for relation in bundle.relations:
            if not (
                self.graph.has_strategy(relation.source_id)
                and self.graph.has_strategy(relation.target_id)
            ):
                raise KeyError(
                    f"seed relation references an unknown strategy: "
                    f"{relation.source_id} -> {relation.target_id}"
                )
            existed = self.graph.get_relation(
                relation.source_id, relation.target_id, relation.relation_type
            )
            self.graph.add_relation(
                relation.source_id,
                relation.target_id,
                relation.relation_type,
                weight=relation.weight,
                metadata=relation.metadata,
            )
            relations += int(existed is None)
        self.graph.save(self.config.graph_path)
        return {"strategies_added": added, "relations_added": relations}

    def _connect_existing(
        self, action: CandidateAction, canonical: StrategyNode
    ) -> None:
        if action.mode == ActionMode.REFINE and action.strategy_ids:
            self.graph.add_relation(
                action.strategy_ids[0],
                canonical.strategy_id,
                RelationType.REFINED_FROM,
                weight=0.75,
                metadata={"deduplicated": True},
            )
        elif action.mode == ActionMode.COMPOSE:
            for parent_id in action.strategy_ids:
                self.graph.add_relation(
                    parent_id,
                    canonical.strategy_id,
                    RelationType.COMPOSED_FROM,
                    weight=0.75,
                    metadata={"deduplicated": True},
                )

    def _materialize_strategy(
        self,
        action: CandidateAction,
        context: SampleContext,
        failure: FailureType | None,
        evidence: dict[str, Any],
    ) -> tuple[StrategyNode, bool, float | None]:
        parents = [self.graph.get_strategy(identifier) for identifier in action.strategy_ids]
        if action.mode == ActionMode.REUSE:
            if len(parents) != 1:
                raise ValueError("REUSE requires exactly one existing strategy")
            return parents[0], False, None

        spec = self.strategy_agent.design(action, context, parents, failure, evidence)
        embedding = self.embedder.encode_text(spec.text())
        similar = self.graph.find_similar(
            embedding,
            threshold=self.config.search.duplicate_threshold,
            states=(
                LifecycleState.ACTIVE,
                LifecycleState.CANDIDATE,
                LifecycleState.RETIRED,
            ),
        )
        if similar:
            canonical, score = similar[0]
            self._connect_existing(action, canonical)
            return canonical, False, score

        if action.mode == ActionMode.REFINE:
            node = self.graph.add_refinement(spec, action.strategy_ids[0], embedding)
        elif action.mode == ActionMode.COMPOSE:
            node = self.graph.add_composition(spec, action.strategy_ids, embedding)
        else:
            node = self.graph.add_strategy(spec, embedding)

        for related, score in self.graph.find_similar(
            embedding,
            threshold=max(0.72, self.config.search.duplicate_threshold - 0.18),
        )[:4]:
            if related.strategy_id != node.strategy_id:
                self.graph.add_relation(
                    node.strategy_id,
                    related.strategy_id,
                    RelationType.SIMILAR_TO,
                    weight=score,
                )
        return node, True, None

    def _trace(self, payload: dict[str, Any]) -> None:
        append_jsonl(self.config.trace_path, [payload])

    def _build_agent_evidence(
        self,
        context: SampleContext,
        candidates: list[CandidateAction],
        retrieved: list[tuple[ExperienceRecord, float]],
        trace: list[dict[str, Any]],
        failure: FailureType | None,
        stagnation: int,
    ) -> dict[str, Any]:
        node_ids = list(
            dict.fromkeys(
                identifier
                for candidate in candidates
                for identifier in candidate.strategy_ids
                if self.graph.has_strategy(identifier)
            )
        )[: self.config.agent.graph_nodes]
        node_set = set(node_ids)
        graph_nodes = []
        for identifier in node_ids:
            node = self.graph.get_strategy(identifier)
            success_mean, progress_mean = self.memory.strategy_means(
                identifier, context.context_key()
            )
            graph_nodes.append(
                {
                    "strategy_id": identifier,
                    "spec": node.spec.to_dict(),
                    "lifecycle_state": node.lifecycle_state.value,
                    "usage_count": node.usage_count,
                    "success_mean": success_mean,
                    "progress_mean": progress_mean,
                }
            )
        graph_relations = [
            relation.to_dict()
            for relation in self.graph.relation_records()
            if relation.source_id in node_set and relation.target_id in node_set
        ]
        memories = []
        for experience, retrieval_score in retrieved[: self.config.agent.memory_items]:
            memories.append(
                {
                    "experience_id": experience.experience_id,
                    "sample_id": experience.context.sample_id,
                    "goal_category": experience.context.goal_category,
                    "target_model": experience.context.target_model,
                    "failure_signature": experience.context.refusal_signature,
                    "strategy_ids": list(experience.strategy_ids),
                    "action_mode": experience.action_mode.value,
                    "success": experience.evaluation.success,
                    "reward": experience.evaluation.reward,
                    "progress": experience.evaluation.progress,
                    "failure_type": (
                        experience.evaluation.failure_type.value
                        if experience.evaluation.failure_type
                        else None
                    ),
                    "evaluation_rationale": experience.evaluation.rationale,
                    "target_feedback_excerpt": experience.response.text[:500],
                    "retrieval_score": retrieval_score,
                }
            )
        recent = []
        recent_source = (
            trace[-self.config.agent.recent_attempts :]
            if self.config.agent.recent_attempts
            else []
        )
        for item in recent_source:
            evaluation = item.get("evaluation", {})
            response = item.get("response", {})
            recent.append(
                {
                    "action": item.get("action"),
                    "selected_strategy_id": item.get("selected_strategy_id"),
                    "status": item.get("status"),
                    "success": evaluation.get("success"),
                    "reward": evaluation.get("reward"),
                    "failure_type": evaluation.get("failure_type"),
                    "target_feedback_excerpt": str(response.get("text", ""))[:500],
                }
            )
        return {
            "failure_signature": failure.value if failure else None,
            "stagnation_rounds": stagnation,
            "remaining_target_queries": context.remaining_budget,
            "graph_nodes": graph_nodes,
            "graph_relations": graph_relations,
            "retrieved_memories": memories,
            "recent_attempts": recent,
        }

    def run(self, context: SampleContext) -> RunResult:
        if context.target_model in {"", "unknown"}:
            context.target_model = self.target.model_name
        context.remaining_budget = self.config.search.max_target_queries
        vectors = self.controller.encode_context(context)
        attempted: set[tuple[str, tuple[str, ...]]] = set()
        trace: list[dict[str, Any]] = []
        best_reward = 0.0
        best_test_case = None
        best_response = None
        success = False
        target_queries = 0
        action_attempts = 0
        stagnation = 0
        failure: FailureType | None = None
        max_actions = self.config.search.max_target_queries * (self.config.search.max_repairs + 2)

        while (
            target_queries < self.config.search.max_target_queries
            and action_attempts < max_actions
        ):
            action_attempts += 1
            retrieved = self.controller.retrieve(context, vectors)
            candidates = self.controller.generate_candidates(
                context,
                vectors,
                retrieved,
                failure=failure,
                attempted=attempted,
                stagnation=stagnation,
            )
            evidence = self._build_agent_evidence(
                context, candidates, retrieved, trace, failure, stagnation
            )
            agent_decision = None
            advisor_calls = 0
            if self.config.agent.enabled:
                agent_decision = self.strategy_agent.advise(candidates, context, evidence)
                advisor_calls = 1
                action = self.controller.select(
                    candidates,
                    advisor_scores=agent_decision.action_scores,
                    advisor_weight=self.config.agent.advisor_weight,
                )
            else:
                action = self.controller.select(candidates)
            evidence["selected_action"] = action.to_dict()
            action_key = (action.mode.value, tuple(sorted(action.strategy_ids)))
            attempted.add(action_key)
            started = time.monotonic()
            strategy, created, duplicate_score = self._materialize_strategy(
                action, context, failure, evidence
            )
            agent_calls = advisor_calls + int(action.mode != ActionMode.REUSE)
            test_case = self.strategy_agent.realize(
                strategy, action, context, action_attempts, evidence
            )
            agent_calls += 1
            test_case = self.actuator.materialize(test_case)
            check = self.consistency_checker.check(context, strategy, test_case)
            repairs = 0
            while not check.valid and repairs < self.config.search.max_repairs:
                repairs += 1
                test_case = self.strategy_agent.repair(
                    test_case, check, strategy, context, evidence
                )
                agent_calls += 1
                test_case = self.actuator.materialize(test_case)
                check = self.consistency_checker.check(context, strategy, test_case)

            base_trace: dict[str, Any] = {
                "sample_id": context.sample_id,
                "action_attempt": action_attempts,
                "action": action.to_dict(),
                "agent_decision": (
                    agent_decision.to_dict() if agent_decision is not None else None
                ),
                "agent_evidence_summary": {
                    "graph_nodes": len(evidence["graph_nodes"]),
                    "graph_relations": len(evidence["graph_relations"]),
                    "retrieved_memories": [
                        item["experience_id"] for item in evidence["retrieved_memories"]
                    ],
                    "recent_attempts": len(evidence["recent_attempts"]),
                },
                "selected_strategy_id": strategy.strategy_id,
                "strategy_created": created,
                "duplicate_similarity": duplicate_score,
                "repairs": repairs,
                "consistency": asdict(check),
                "retrieved_experiences": [
                    {"experience_id": item.experience_id, "score": score}
                    for item, score in retrieved
                ],
            }
            if not check.valid:
                failure = FailureType.STRATEGY_DRIFT
                base_trace["status"] = "prequery_rejected"
                trace.append(base_trace)
                self._trace(base_trace)
                stagnation += 1
                continue

            response = self.target.generate(test_case, context)
            target_queries += 1
            context.remaining_budget = self.config.search.max_target_queries - target_queries
            evaluation = self.judge.evaluate(
                context, test_case, response, self.config.evaluation
            )
            image_calls = int(bool(test_case.metadata.get("executed_image_operations")))
            cost = CostRecord(
                target_queries=1,
                agent_calls=agent_calls,
                judge_calls=1,
                image_calls=image_calls,
                wall_time_seconds=time.monotonic() - started,
            )
            recorded_ids = tuple(
                dict.fromkeys((strategy.strategy_id, *action.strategy_ids))
            )
            experience = ExperienceRecord(
                context=SampleContext.from_dict(context.to_dict()),
                strategy_ids=recorded_ids,
                action_mode=action.mode,
                test_case=test_case,
                response=response,
                evaluation=evaluation,
                cost=cost,
                goal_embedding=vectors.goal,
                image_embedding=vectors.image,
                cross_modal_embedding=vectors.cross_modal,
            )
            self.memory.add_experience(experience)
            credit = self.credit_assigner.assign(
                experience,
                executed_strategy_id=strategy.strategy_id,
                parent_ids=action.strategy_ids,
            )
            transitions = self.lifecycle_manager.update()
            self.graph.save(self.config.graph_path)

            improved = evaluation.reward > best_reward + 1e-9
            if improved:
                best_reward = evaluation.reward
                best_test_case = test_case
                best_response = response
                stagnation = 0
            else:
                stagnation += 1
            success = success or evaluation.success
            failure = evaluation.failure_type
            context.refusal_signature = failure.value if failure else ""
            context.history = (*context.history, response.text[:500])
            vectors = self.controller.encode_context(context)

            base_trace.update(
                {
                    "status": "evaluated",
                    "target_query": target_queries,
                    "test_case": test_case.to_dict(),
                    "response": response.to_dict(),
                    "evaluation": evaluation.to_dict(),
                    "cost": cost.to_dict(),
                    "credit": credit,
                    "lifecycle_transitions": transitions,
                    "experience_id": experience.experience_id,
                }
            )
            trace.append(base_trace)
            self._trace(base_trace)
            if evaluation.success and not self.config.search.continue_after_success:
                break

        self.graph.save(self.config.graph_path)
        return RunResult(
            sample_id=context.sample_id,
            success=success,
            attempts=action_attempts,
            target_queries=target_queries,
            best_reward=best_reward,
            best_test_case=best_test_case,
            best_response=best_response,
            trace=trace,
        )
