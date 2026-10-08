"""Input/output helpers for samples, seed strategies, and run summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from .models import RelationType, SampleContext, StrategySpec


@dataclass(slots=True)
class SeedStrategy:
    strategy_id: str
    spec: StrategySpec


@dataclass(slots=True)
class SeedRelation:
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SeedBundle:
    strategies: list[SeedStrategy] = field(default_factory=list)
    relations: list[SeedRelation] = field(default_factory=list)


def load_seed_file(path: str | Path) -> SeedBundle:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    strategy_rows = raw if isinstance(raw, list) else raw.get("strategies", [])
    relation_rows = [] if isinstance(raw, list) else raw.get("relations", [])
    bundle = SeedBundle()
    for index, row in enumerate(strategy_rows):
        if not isinstance(row, dict):
            raise ValueError(f"seed strategy {index} must be an object")
        spec_data = dict(row)
        identifier = str(spec_data.pop("id", spec_data.pop("strategy_id", ""))).strip()
        if not identifier:
            identifier = f"seed_{index + 1}"
        spec_data.pop("lifecycle_state", None)
        bundle.strategies.append(SeedStrategy(identifier, StrategySpec.from_dict(spec_data)))
    for index, row in enumerate(relation_rows):
        if not isinstance(row, dict):
            raise ValueError(f"seed relation {index} must be an object")
        bundle.relations.append(
            SeedRelation(
                source_id=str(row["source"]),
                target_id=str(row["target"]),
                relation_type=RelationType(row["type"]),
                weight=float(row.get("weight", 0.5)),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return bundle


def load_samples(path: str | Path) -> list[SampleContext]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        parsed = json.loads(source.read_text(encoding="utf-8"))
        rows = parsed if isinstance(parsed, list) else parsed.get("samples", [])
    return [SampleContext.from_dict(row) for row in rows]


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
