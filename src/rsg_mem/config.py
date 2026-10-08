"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class SearchSettings:
    max_target_queries: int = 8
    max_repairs: int = 2
    retrieval_top_k: int = 8
    local_graph_hops: int = 1
    local_graph_limit: int = 40
    reuse_candidates: int = 4
    refine_candidates: int = 2
    compose_candidates: int = 4
    invent_candidates: int = 1
    conflict_limit: float = 0.65
    duplicate_threshold: float = 0.96
    stagnation_limit: int = 3
    continue_after_success: bool = False
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "success": 1.40,
            "progress": 0.55,
            "similarity": 0.75,
            "synergy": 0.65,
            "novelty": 0.30,
            "coverage": 0.25,
            "cost": 0.20,
            "redundancy": 0.30,
        }
    )


@dataclass(slots=True)
class LifecycleSettings:
    promotion_min_evidence: int = 2
    promotion_success_mean: float = 0.55
    retirement_min_uses: int = 6
    retirement_success_mean: float = 0.25
    reactivation_success_mean: float = 0.60
    elimination_min_uses: int = 12
    max_resident_strategies: int = 250


@dataclass(slots=True)
class EvaluationSettings:
    safety_threshold: float = 0.70
    relevance_threshold: float = 0.65
    informativeness_threshold: float = 0.60
    consistency_threshold: float = 0.70


@dataclass(slots=True)
class AgentSettings:
    enabled: bool = True
    advisor_weight: float = 0.35
    memory_items: int = 5
    recent_attempts: int = 4
    graph_nodes: int = 12


@dataclass(slots=True)
class ProviderSettings:
    type: str = "mock"
    base_url: str = ""
    model: str = "mock"
    api_key_env: str = ""
    timeout_seconds: float = 120.0
    temperature: float = 0.2
    max_tokens: int = 1200
    extra_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ProviderSettings":
        return cls(**dict(data or {}))


@dataclass(slots=True)
class AppConfig:
    run_dir: Path = Path("runs/default")
    random_seed: int = 7
    embedding_dimension: int = 256
    search: SearchSettings = field(default_factory=SearchSettings)
    lifecycle: LifecycleSettings = field(default_factory=LifecycleSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    prompt_dir: Path | None = None
    strategy_agent: ProviderSettings = field(default_factory=ProviderSettings)
    target: ProviderSettings = field(default_factory=ProviderSettings)
    judge: ProviderSettings = field(default_factory=ProviderSettings)
    consistency: ProviderSettings = field(default_factory=ProviderSettings)

    @property
    def graph_path(self) -> Path:
        return self.run_dir / "graph.json"

    @property
    def memory_path(self) -> Path:
        return self.run_dir / "memory.sqlite3"

    @property
    def trace_path(self) -> Path:
        return self.run_dir / "traces.jsonl"

    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / "artifacts"


def _resolve_run_dir(value: str | Path, config_path: Path | None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or config_path is None:
        return path.resolve()
    return (config_path.parent / path).resolve()


def _resolve_optional_dir(value: str | Path | None, config_path: Path) -> Path | None:
    if not value:
        return None
    return _resolve_run_dir(value, config_path)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    providers = raw.get("providers", {})
    config = AppConfig(
        run_dir=_resolve_run_dir(raw.get("run_dir", "../runs/default"), config_path),
        random_seed=int(raw.get("random_seed", 7)),
        embedding_dimension=int(raw.get("embedding_dimension", 256)),
        search=SearchSettings(**raw.get("search", {})),
        lifecycle=LifecycleSettings(**raw.get("lifecycle", {})),
        evaluation=EvaluationSettings(**raw.get("evaluation", {})),
        agent=AgentSettings(**raw.get("agent", {})),
        prompt_dir=_resolve_optional_dir(raw.get("prompt_dir"), config_path),
        strategy_agent=ProviderSettings.from_dict(providers.get("strategy_agent")),
        target=ProviderSettings.from_dict(providers.get("target")),
        judge=ProviderSettings.from_dict(providers.get("judge")),
        consistency=ProviderSettings.from_dict(
            providers.get("consistency", providers.get("judge"))
        ),
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    if config.embedding_dimension < 16:
        raise ValueError("embedding_dimension must be >= 16")
    if config.search.max_target_queries <= 0:
        raise ValueError("max_target_queries must be positive")
    if config.search.max_repairs < 0:
        raise ValueError("max_repairs cannot be negative")
    if not 0.0 <= config.search.conflict_limit <= 1.0:
        raise ValueError("conflict_limit must be in [0, 1]")
    if not 0.0 <= config.search.duplicate_threshold <= 1.0:
        raise ValueError("duplicate_threshold must be in [0, 1]")
    if not 0.0 <= config.agent.advisor_weight <= 1.0:
        raise ValueError("agent.advisor_weight must be in [0, 1]")
    if min(
        config.agent.memory_items,
        config.agent.recent_attempts,
        config.agent.graph_nodes,
    ) < 0:
        raise ValueError("agent evidence limits cannot be negative")
    for name, provider in (
        ("strategy_agent", config.strategy_agent),
        ("target", config.target),
        ("judge", config.judge),
        ("consistency", config.consistency),
    ):
        if provider.type not in {"mock", "openai_compatible"}:
            raise ValueError(f"unsupported provider type for {name}: {provider.type}")
        if provider.type == "openai_compatible" and not provider.base_url:
            raise ValueError(f"{name}.base_url is required for openai_compatible")
