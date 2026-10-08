"""Abstract provider contracts."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from ..config import EvaluationSettings
from ..models import (
    AgentDecision,
    CandidateAction,
    ConsistencyResult,
    EvaluationResult,
    FailureType,
    SampleContext,
    StrategyNode,
    StrategySpec,
    TargetResponse,
    TestCase,
)


class StrategyAgent(Protocol):
    def advise(
        self,
        candidates: Sequence[CandidateAction],
        context: SampleContext,
        evidence: dict[str, Any],
    ) -> AgentDecision: ...

    def design(
        self,
        action: CandidateAction,
        context: SampleContext,
        parents: Sequence[StrategyNode],
        failure: FailureType | None,
        evidence: dict[str, Any],
    ) -> StrategySpec: ...

    def realize(
        self,
        strategy: StrategyNode,
        action: CandidateAction,
        context: SampleContext,
        attempt_index: int,
        evidence: dict[str, Any],
    ) -> TestCase: ...

    def repair(
        self,
        test_case: TestCase,
        check: ConsistencyResult,
        strategy: StrategyNode,
        context: SampleContext,
        evidence: dict[str, Any],
    ) -> TestCase: ...


class TargetClient(Protocol):
    model_name: str

    def generate(self, test_case: TestCase, context: SampleContext) -> TargetResponse: ...


class Judge(Protocol):
    def evaluate(
        self,
        context: SampleContext,
        test_case: TestCase,
        response: TargetResponse,
        settings: EvaluationSettings,
    ) -> EvaluationResult: ...


class ConsistencyChecker(Protocol):
    def check(
        self,
        context: SampleContext,
        strategy: StrategyNode,
        test_case: TestCase,
    ) -> ConsistencyResult: ...
