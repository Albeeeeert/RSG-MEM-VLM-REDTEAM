"""Graph-guided candidate generation and contextual Thompson selection."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Iterable

from .config import SearchSettings
from .embeddings import Embedder, cosine, mean_vectors
from .graph import RelationalStrategyGraph
from .memory import ExperienceMemory
from .models import (
    ActionMode,
    CandidateAction,
    ExperienceRecord,
    FailureType,
    LifecycleState,
    RelationType,
    SampleContext,
)


@dataclass(slots=True)
class ContextVectors:
    goal: list[float]
    image: list[float]
    cross_modal: list[float]


class GraphController:
    """Builds a local relational action space instead of traversing a strict tree."""

    def __init__(
        self,
        graph: RelationalStrategyGraph,
        memory: ExperienceMemory,
        embedder: Embedder,
        settings: SearchSettings,
        *,
        random_seed: int = 7,
    ) -> None:
        self.graph = graph
        self.memory = memory
        self.embedder = embedder
        self.settings = settings
        self.rng = random.Random(random_seed)

    def encode_context(self, context: SampleContext) -> ContextVectors:
        contextual_text = context.search_text()
        return ContextVectors(
            goal=self.embedder.encode_text(contextual_text),
            image=self.embedder.encode_image(context.source_image, context.image_description),
            cross_modal=self.embedder.encode_cross_modal(
                contextual_text, context.image_description or context.ocr_text
            ),
        )

    def retrieve(
        self, context: SampleContext, vectors: ContextVectors
    ) -> list[tuple[ExperienceRecord, float]]:
        return self.memory.retrieve(
            context,
            goal_embedding=vectors.goal,
            image_embedding=vectors.image,
            cross_modal_embedding=vectors.cross_modal,
            top_k=self.settings.retrieval_top_k,
        )

    def _local_nodes(
        self,
        retrieved: list[tuple[ExperienceRecord, float]],
        vectors: ContextVectors,
    ) -> list[str]:
        seeds: list[str] = []
        for experience, _ in retrieved:
            seeds.extend(identifier for identifier in experience.strategy_ids if self.graph.has_strategy(identifier))
        identifiers = self.graph.local_strategy_ids(
            seeds,
            hops=self.settings.local_graph_hops,
            limit=self.settings.local_graph_limit,
        )
        if not identifiers:
            identifiers = [
                node.strategy_id
                for node in self.graph.strategies(
                    states=(LifecycleState.ACTIVE, LifecycleState.CANDIDATE)
                )
            ]
        query = mean_vectors((vectors.goal, vectors.cross_modal))
        identifiers.sort(
            key=lambda identifier: cosine(query, self.graph.get_strategy(identifier).embedding),
            reverse=True,
        )
        return identifiers

    def generate_candidates(
        self,
        context: SampleContext,
        vectors: ContextVectors,
        retrieved: list[tuple[ExperienceRecord, float]],
        *,
        failure: FailureType | None = None,
        attempted: set[tuple[str, tuple[str, ...]]] | None = None,
        stagnation: int = 0,
    ) -> list[CandidateAction]:
        attempted = attempted or set()
        local_ids = self._local_nodes(retrieved, vectors)
        candidates: list[CandidateAction] = []

        for identifier in local_ids[: self.settings.reuse_candidates]:
            candidates.append(
                CandidateAction(
                    mode=ActionMode.REUSE,
                    strategy_ids=(identifier,),
                    rationale="reuse a context-relevant graph node",
                    estimated_cost=1.0,
                )
            )

        if local_ids:
            refinement_count = self.settings.refine_candidates if failure else min(
                1, self.settings.refine_candidates
            )
            for identifier in local_ids[:refinement_count]:
                candidates.append(
                    CandidateAction(
                        mode=ActionMode.REFINE,
                        strategy_ids=(identifier,),
                        rationale=f"repair observed failure: {(failure or FailureType.NO_PROGRESS).value}",
                        estimated_cost=1.35,
                    )
                )

        pair_pool = local_ids[: min(10, len(local_ids))]
        pairs = list(itertools.combinations(pair_pool, 2))
        pairs.sort(
            key=lambda pair: (
                self.graph.relation_weight(*pair, RelationType.COMPLEMENTARY_TO)
                - self.graph.relation_weight(*pair, RelationType.CONFLICTS_WITH),
                self.memory.pair_posterior(*pair, context.context_key())["mean_delta"],
            ),
            reverse=True,
        )
        for left, right in pairs:
            if len([candidate for candidate in candidates if candidate.mode == ActionMode.COMPOSE]) >= self.settings.compose_candidates:
                break
            if self.graph.composition_allowed(left, right, self.settings.conflict_limit):
                candidates.append(
                    CandidateAction(
                        mode=ActionMode.COMPOSE,
                        strategy_ids=(left, right),
                        rationale="compose compatible nodes in the local relation graph",
                        estimated_cost=1.65,
                    )
                )

        invent_count = self.settings.invent_candidates
        if not local_ids:
            invent_count = max(1, invent_count)
        if stagnation >= self.settings.stagnation_limit:
            invent_count = max(2, invent_count)
        for _ in range(invent_count):
            candidates.append(
                CandidateAction(
                    mode=ActionMode.INVENT,
                    rationale="fill a coverage gap outside the current local graph",
                    estimated_cost=1.8,
                )
            )

        query = mean_vectors((vectors.goal, vectors.cross_modal))
        for candidate in candidates:
            self._score(
                candidate,
                context,
                query,
                retrieved,
                attempted,
                stagnation,
                failure,
            )
        candidates.sort(key=lambda candidate: (candidate.utility, candidate.action_id), reverse=True)
        return candidates

    def _score(
        self,
        candidate: CandidateAction,
        context: SampleContext,
        query_embedding: Iterable[float],
        retrieved: list[tuple[ExperienceRecord, float]],
        attempted: set[tuple[str, tuple[str, ...]]],
        stagnation: int,
        failure: FailureType | None,
    ) -> None:
        identifiers = candidate.strategy_ids
        context_key = context.context_key()
        samples = [
            self.memory.sample_strategy_posterior(identifier, context_key, self.rng)
            for identifier in identifiers
        ]
        success = sum(item[0] for item in samples) / len(samples) if samples else 0.5
        progress = sum(item[1] for item in samples) / len(samples) if samples else 0.5
        similarity = (
            sum(
                max(0.0, cosine(query_embedding, self.graph.get_strategy(identifier).embedding))
                for identifier in identifiers
            )
            / len(identifiers)
            if identifiers
            else 0.25
        )
        synergy = 0.5
        if len(identifiers) >= 2:
            pair_samples = [
                self.memory.sample_pair_posterior(left, right, context_key, self.rng)
                for left, right in itertools.combinations(identifiers, 2)
            ]
            relational = [
                self.graph.relation_weight(left, right, RelationType.COMPLEMENTARY_TO)
                for left, right in itertools.combinations(identifiers, 2)
            ]
            synergy = 0.65 * (sum(pair_samples) / len(pair_samples)) + 0.35 * (
                sum(relational) / len(relational)
            )

        if len(identifiers) >= 2:
            pair_similarity = [
                max(
                    0.0,
                    cosine(
                        self.graph.get_strategy(left).embedding,
                        self.graph.get_strategy(right).embedding,
                    ),
                )
                for left, right in itertools.combinations(identifiers, 2)
            ]
            novelty = 1.0 - sum(pair_similarity) / len(pair_similarity)
        elif candidate.mode == ActionMode.INVENT:
            novelty = 1.0
        elif candidate.mode == ActionMode.REFINE:
            novelty = 0.45
        else:
            novelty = 0.2

        retrieved_ids = {
            identifier
            for experience, _ in retrieved
            for identifier in experience.strategy_ids
        }
        coverage = 1.0 if not identifiers else sum(
            identifier not in retrieved_ids for identifier in identifiers
        ) / len(identifiers)
        key = (candidate.mode.value, tuple(sorted(identifiers)))
        redundancy = 1.0 if key in attempted else 0.0
        weights = self.settings.weights
        utility = (
            weights["success"] * success
            + weights["progress"] * progress
            + weights["similarity"] * similarity
            + weights["synergy"] * synergy
            + weights["novelty"] * novelty
            + weights["coverage"] * coverage
            - weights["cost"] * candidate.estimated_cost
            - weights["redundancy"] * redundancy
        )
        mode_prior = 0.0
        if failure in {
            FailureType.DIRECT_REFUSAL,
            FailureType.SAFETY_REFRAME,
            FailureType.INSUFFICIENT_INFORMATION,
        } and candidate.mode == ActionMode.REFINE:
            mode_prior += 0.25
        if failure in {
            FailureType.DIRECT_REFUSAL,
            FailureType.CROSS_MODAL_FAILURE,
        } and candidate.mode == ActionMode.COMPOSE:
            mode_prior += 0.18
        if not retrieved and candidate.mode == ActionMode.INVENT:
            mode_prior += 0.12
        if context.remaining_budget <= 2 and candidate.mode == ActionMode.REUSE:
            mode_prior += 0.10
        if stagnation >= self.settings.stagnation_limit and candidate.mode == ActionMode.INVENT:
            mode_prior += 0.45
        utility += mode_prior
        candidate.component_scores = {
            "success": success,
            "progress": progress,
            "similarity": similarity,
            "synergy": synergy,
            "novelty": novelty,
            "coverage": coverage,
            "cost": candidate.estimated_cost,
            "redundancy": redundancy,
            "mode_prior": mode_prior,
        }
        candidate.utility = utility

    @staticmethod
    def select(candidates: list[CandidateAction]) -> CandidateAction:
        if not candidates:
            raise RuntimeError("controller generated no candidate actions")
        return max(candidates, key=lambda candidate: (candidate.utility, candidate.action_id))
