"""SQLite-backed episodic memory and posterior ledgers."""

from __future__ import annotations

import json
import random
import sqlite3
import threading
from pathlib import Path
from typing import Iterable

from .embeddings import cosine
from .models import ExperienceRecord, SampleContext, utc_now


class ExperienceMemory:
    """Persistent source of truth for attempts, utility, and interaction evidence."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._create_schema()

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS experiences (
                    experience_id TEXT PRIMARY KEY,
                    sample_id TEXT NOT NULL,
                    goal_category TEXT NOT NULL,
                    target_model TEXT NOT NULL,
                    target_version TEXT NOT NULL,
                    refusal_signature TEXT NOT NULL,
                    strategy_ids_json TEXT NOT NULL,
                    action_mode TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    reward REAL NOT NULL,
                    progress REAL NOT NULL,
                    goal_embedding_json TEXT NOT NULL,
                    image_embedding_json TEXT NOT NULL,
                    cross_modal_embedding_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_exp_context
                    ON experiences(goal_category, target_model, success);
                CREATE INDEX IF NOT EXISTS idx_exp_created
                    ON experiences(created_at DESC);

                CREATE TABLE IF NOT EXISTS strategy_posteriors (
                    strategy_id TEXT NOT NULL,
                    context_key TEXT NOT NULL,
                    success_alpha REAL NOT NULL DEFAULT 1.0,
                    success_beta REAL NOT NULL DEFAULT 1.0,
                    progress_alpha REAL NOT NULL DEFAULT 1.0,
                    progress_beta REAL NOT NULL DEFAULT 1.0,
                    uses REAL NOT NULL DEFAULT 0.0,
                    reward_sum REAL NOT NULL DEFAULT 0.0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(strategy_id, context_key)
                );

                CREATE TABLE IF NOT EXISTS pair_posteriors (
                    left_id TEXT NOT NULL,
                    right_id TEXT NOT NULL,
                    context_key TEXT NOT NULL,
                    synergy_alpha REAL NOT NULL DEFAULT 1.0,
                    synergy_beta REAL NOT NULL DEFAULT 1.0,
                    observations REAL NOT NULL DEFAULT 0.0,
                    delta_sum REAL NOT NULL DEFAULT 0.0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(left_id, right_id, context_key)
                );
                """
            )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "ExperienceMemory":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def add_experience(self, experience: ExperienceRecord) -> None:
        payload = json.dumps(experience.to_dict(), ensure_ascii=False)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO experiences(
                    experience_id, sample_id, goal_category, target_model,
                    target_version, refusal_signature, strategy_ids_json,
                    action_mode, success, reward, progress, goal_embedding_json,
                    image_embedding_json, cross_modal_embedding_json,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experience.experience_id,
                    experience.context.sample_id,
                    experience.context.goal_category,
                    experience.context.target_model,
                    experience.context.target_version,
                    experience.context.refusal_signature,
                    json.dumps(list(experience.strategy_ids)),
                    experience.action_mode.value,
                    int(experience.evaluation.success),
                    experience.evaluation.reward,
                    experience.evaluation.progress,
                    json.dumps(experience.goal_embedding),
                    json.dumps(experience.image_embedding),
                    json.dumps(experience.cross_modal_embedding),
                    payload,
                    experience.created_at,
                ),
            )

    def get_experience(self, experience_id: str) -> ExperienceRecord | None:
        row = self._connection.execute(
            "SELECT payload_json FROM experiences WHERE experience_id = ?", (experience_id,)
        ).fetchone()
        return ExperienceRecord.from_dict(json.loads(row[0])) if row else None

    def recent(self, limit: int = 100) -> list[ExperienceRecord]:
        rows = self._connection.execute(
            "SELECT payload_json FROM experiences ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [ExperienceRecord.from_dict(json.loads(row[0])) for row in rows]

    def retrieve(
        self,
        context: SampleContext,
        *,
        goal_embedding: Iterable[float],
        image_embedding: Iterable[float],
        cross_modal_embedding: Iterable[float],
        top_k: int = 8,
        scan_limit: int = 2000,
        weights: dict[str, float] | None = None,
    ) -> list[tuple[ExperienceRecord, float]]:
        default_weights = {
            "goal": 0.28,
            "image": 0.18,
            "cross_modal": 0.18,
            "failure": 0.10,
            "target": 0.10,
            "utility": 0.16,
        }
        default_weights.update(weights or {})
        rows = self._connection.execute(
            "SELECT payload_json FROM experiences ORDER BY created_at DESC LIMIT ?",
            (scan_limit,),
        ).fetchall()
        scored: list[tuple[ExperienceRecord, float]] = []
        for row in rows:
            experience = ExperienceRecord.from_dict(json.loads(row[0]))
            target_match = float(experience.context.target_model == context.target_model)
            failure_match = float(
                bool(context.refusal_signature)
                and experience.context.refusal_signature == context.refusal_signature
            )
            utility = 0.65 * experience.evaluation.reward + 0.35 * float(
                experience.evaluation.success
            )
            score = (
                default_weights["goal"]
                * max(0.0, cosine(goal_embedding, experience.goal_embedding))
                + default_weights["image"]
                * max(0.0, cosine(image_embedding, experience.image_embedding))
                + default_weights["cross_modal"]
                * max(0.0, cosine(cross_modal_embedding, experience.cross_modal_embedding))
                + default_weights["failure"] * failure_match
                + default_weights["target"] * target_match
                + default_weights["utility"] * utility
            )
            scored.append((experience, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _strategy_row(self, strategy_id: str, context_key: str) -> sqlite3.Row | None:
        return self._connection.execute(
            """
            SELECT * FROM strategy_posteriors
            WHERE strategy_id = ? AND context_key = ?
            """,
            (strategy_id, context_key),
        ).fetchone()

    def _ensure_strategy_row(self, strategy_id: str, context_key: str) -> None:
        self._connection.execute(
            """
            INSERT OR IGNORE INTO strategy_posteriors(
                strategy_id, context_key, updated_at
            ) VALUES (?, ?, ?)
            """,
            (strategy_id, context_key, utc_now()),
        )

    def update_strategy_posterior(
        self,
        strategy_id: str,
        context_key: str,
        *,
        success: float,
        progress: float,
        reward: float,
        weight: float = 1.0,
    ) -> None:
        success_value = max(0.0, min(1.0, float(success)))
        progress_value = max(0.0, min(1.0, float(progress)))
        reward_value = max(0.0, min(1.0, float(reward)))
        keys = (context_key,) if context_key == "*" else (context_key, "*")
        with self._lock, self._connection:
            for key in keys:
                self._ensure_strategy_row(strategy_id, key)
                self._connection.execute(
                    """
                    UPDATE strategy_posteriors SET
                        success_alpha = success_alpha + ?,
                        success_beta = success_beta + ?,
                        progress_alpha = progress_alpha + ?,
                        progress_beta = progress_beta + ?,
                        uses = uses + ?,
                        reward_sum = reward_sum + ?,
                        updated_at = ?
                    WHERE strategy_id = ? AND context_key = ?
                    """,
                    (
                        weight * success_value,
                        weight * (1.0 - success_value),
                        weight * progress_value,
                        weight * (1.0 - progress_value),
                        weight,
                        weight * reward_value,
                        utc_now(),
                        strategy_id,
                        key,
                    ),
                )

    def strategy_posterior(self, strategy_id: str, context_key: str) -> dict[str, float]:
        row = self._strategy_row(strategy_id, context_key) or self._strategy_row(strategy_id, "*")
        if row is None:
            return {
                "success_alpha": 1.0,
                "success_beta": 1.0,
                "progress_alpha": 1.0,
                "progress_beta": 1.0,
                "uses": 0.0,
                "reward_mean": 0.5,
            }
        uses = float(row["uses"])
        return {
            "success_alpha": float(row["success_alpha"]),
            "success_beta": float(row["success_beta"]),
            "progress_alpha": float(row["progress_alpha"]),
            "progress_beta": float(row["progress_beta"]),
            "uses": uses,
            "reward_mean": float(row["reward_sum"]) / uses if uses else 0.5,
        }

    def sample_strategy_posterior(
        self, strategy_id: str, context_key: str, rng: random.Random
    ) -> tuple[float, float]:
        posterior = self.strategy_posterior(strategy_id, context_key)
        success = rng.betavariate(posterior["success_alpha"], posterior["success_beta"])
        progress = rng.betavariate(posterior["progress_alpha"], posterior["progress_beta"])
        return success, progress

    def strategy_means(self, strategy_id: str, context_key: str = "*") -> tuple[float, float]:
        posterior = self.strategy_posterior(strategy_id, context_key)
        success = posterior["success_alpha"] / (
            posterior["success_alpha"] + posterior["success_beta"]
        )
        progress = posterior["progress_alpha"] / (
            posterior["progress_alpha"] + posterior["progress_beta"]
        )
        return success, progress

    @staticmethod
    def _ordered_pair(left_id: str, right_id: str) -> tuple[str, str]:
        return tuple(sorted((left_id, right_id)))

    def _pair_row(self, left_id: str, right_id: str, context_key: str) -> sqlite3.Row | None:
        left, right = self._ordered_pair(left_id, right_id)
        return self._connection.execute(
            """
            SELECT * FROM pair_posteriors
            WHERE left_id = ? AND right_id = ? AND context_key = ?
            """,
            (left, right, context_key),
        ).fetchone()

    def update_pair_posterior(
        self,
        left_id: str,
        right_id: str,
        context_key: str,
        *,
        positive: bool,
        delta: float,
        weight: float = 1.0,
    ) -> None:
        left, right = self._ordered_pair(left_id, right_id)
        keys = (context_key,) if context_key == "*" else (context_key, "*")
        with self._lock, self._connection:
            for key in keys:
                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO pair_posteriors(
                        left_id, right_id, context_key, updated_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (left, right, key, utc_now()),
                )
                self._connection.execute(
                    """
                    UPDATE pair_posteriors SET
                        synergy_alpha = synergy_alpha + ?,
                        synergy_beta = synergy_beta + ?,
                        observations = observations + ?,
                        delta_sum = delta_sum + ?,
                        updated_at = ?
                    WHERE left_id = ? AND right_id = ? AND context_key = ?
                    """,
                    (
                        weight if positive else 0.0,
                        0.0 if positive else weight,
                        weight,
                        weight * float(delta),
                        utc_now(),
                        left,
                        right,
                        key,
                    ),
                )

    def pair_posterior(self, left_id: str, right_id: str, context_key: str) -> dict[str, float]:
        row = self._pair_row(left_id, right_id, context_key) or self._pair_row(
            left_id, right_id, "*"
        )
        if row is None:
            return {"alpha": 1.0, "beta": 1.0, "observations": 0.0, "mean_delta": 0.0}
        observations = float(row["observations"])
        return {
            "alpha": float(row["synergy_alpha"]),
            "beta": float(row["synergy_beta"]),
            "observations": observations,
            "mean_delta": float(row["delta_sum"]) / observations if observations else 0.0,
        }

    def sample_pair_posterior(
        self, left_id: str, right_id: str, context_key: str, rng: random.Random
    ) -> float:
        posterior = self.pair_posterior(left_id, right_id, context_key)
        return rng.betavariate(posterior["alpha"], posterior["beta"])

    def best_parent_reward(
        self, parent_ids: Iterable[str], context: SampleContext
    ) -> float | None:
        parents = set(parent_ids)
        if not parents:
            return None
        rows = self._connection.execute(
            """
            SELECT strategy_ids_json, reward FROM experiences
            WHERE goal_category = ? AND target_model = ?
            ORDER BY created_at DESC LIMIT 500
            """,
            (context.goal_category, context.target_model),
        ).fetchall()
        rewards: list[float] = []
        for row in rows:
            used = set(json.loads(row["strategy_ids_json"]))
            if len(used & parents) == 1:
                rewards.append(float(row["reward"]))
        return max(rewards) if rewards else None

    def count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM experiences").fetchone()[0])

    def stats(self) -> dict[str, int]:
        success = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM experiences WHERE success = 1"
            ).fetchone()[0]
        )
        strategy_rows = int(
            self._connection.execute("SELECT COUNT(*) FROM strategy_posteriors").fetchone()[0]
        )
        pair_rows = int(
            self._connection.execute("SELECT COUNT(*) FROM pair_posteriors").fetchone()[0]
        )
        return {
            "experiences": self.count(),
            "successful_experiences": success,
            "strategy_posterior_rows": strategy_rows,
            "pair_posterior_rows": pair_rows,
        }
