"""Abstract provider contracts."""

from __future__ import annotations

from typing import Protocol, Sequence

from ..config import EvaluationSettings
from ..models import (
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
    def design(
        self,
        action: CandidateAction,
        context: SampleContext,
        parents: Sequence[StrategyNode],
        failure: FailureType | None,
    ) -> StrategySpec: ...

    def realize(
        self,
        strategy: StrategyNode,
        action: CandidateAction,
        context: SampleContext,
        attempt_index: int,
    ) -> TestCase: ...

    def repair(
        self,
        test_case: TestCase,
        check: ConsistencyResult,
        strategy: StrategyNode,
        context: SampleContext,
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
