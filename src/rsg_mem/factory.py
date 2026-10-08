"""Runtime construction from validated configuration."""

from __future__ import annotations

from dataclasses import dataclass

from .actuator import MultimodalActuator
from .config import AppConfig, ProviderSettings
from .controller import GraphController
from .credit import CreditAssigner
from .embeddings import HashingEmbedder
from .engine import RSGMemEngine
from .graph import RelationalStrategyGraph
from .lifecycle import LifecycleManager
from .memory import ExperienceMemory
from .providers.mock import (
    MockConsistencyChecker,
    MockJudge,
    MockTargetClient,
    RuleBasedStrategyAgent,
)
from .providers.openai_compatible import (
    OpenAICompatibleConsistencyChecker,
    OpenAICompatibleJudge,
    OpenAICompatibleStrategyAgent,
    OpenAICompatibleTarget,
)


@dataclass(slots=True)
class Runtime:
    config: AppConfig
    graph: RelationalStrategyGraph
    memory: ExperienceMemory
    engine: RSGMemEngine

    def close(self) -> None:
        for component in (
            self.engine.strategy_agent,
            self.engine.target,
            self.engine.judge,
            self.engine.consistency_checker,
        ):
            client = getattr(component, "client", None)
            close = getattr(client, "close", None)
            if callable(close):
                close()
        self.memory.close()

    def __enter__(self) -> "Runtime":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _strategy_agent(settings: ProviderSettings):
    if settings.type == "mock":
        return RuleBasedStrategyAgent()
    return OpenAICompatibleStrategyAgent(settings)


def _target(settings: ProviderSettings):
    if settings.type == "mock":
        return MockTargetClient(model_name=settings.model)
    return OpenAICompatibleTarget(settings)


def _judge(settings: ProviderSettings):
    if settings.type == "mock":
        return MockJudge()
    return OpenAICompatibleJudge(settings)


def _consistency(settings: ProviderSettings):
    if settings.type == "mock":
        return MockConsistencyChecker()
    return OpenAICompatibleConsistencyChecker(settings)


def create_runtime(config: AppConfig) -> Runtime:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    graph = RelationalStrategyGraph.load(config.graph_path)
    memory = ExperienceMemory(config.memory_path)
    embedder = HashingEmbedder(config.embedding_dimension)
    controller = GraphController(
        graph,
        memory,
        embedder,
        config.search,
        random_seed=config.random_seed,
    )
    engine = RSGMemEngine(
        config=config,
        graph=graph,
        memory=memory,
        embedder=embedder,
        controller=controller,
        strategy_agent=_strategy_agent(config.strategy_agent),
        actuator=MultimodalActuator(config.artifacts_dir),
        target=_target(config.target),
        judge=_judge(config.judge),
        consistency_checker=_consistency(config.consistency),
        credit_assigner=CreditAssigner(graph, memory),
        lifecycle_manager=LifecycleManager(graph, config.lifecycle),
    )
    return Runtime(config=config, graph=graph, memory=memory, engine=engine)
