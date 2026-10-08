"""Adapters for OpenAI-compatible chat-completions and VLM endpoints."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Sequence

import httpx

from ..config import EvaluationSettings, ProviderSettings
from ..models import (
    ActionMode,
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
from ..prompting import PromptLibrary


def extract_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            raise ValueError(f"provider did not return a JSON object: {text[:200]}")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("provider response must be a JSON object")
    return value


def image_data_url(path: str) -> str:
    source = Path(path)
    mime = mimetypes.guess_type(source.name)[0] or "image/png"
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def multimodal_user_content(text: str, image_path: str | None) -> Any:
    if image_path and Path(image_path).is_file():
        return [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
        ]
    return text


class OpenAICompatibleClient:
    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings
        self.model_name = settings.model
        key = os.environ.get(settings.api_key_env, "") if settings.api_key_env else ""
        headers = {"Content-Type": "application/json", **settings.extra_headers}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        base = settings.base_url.rstrip("/")
        self.endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        self.client = httpx.Client(timeout=settings.timeout_seconds, headers=headers)

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict[str, Any], float]:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": self.settings.max_tokens if max_tokens is None else max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        started = time.monotonic()
        response = self.client.post(self.endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"].get("content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return str(content), data, time.monotonic() - started

    def close(self) -> None:
        self.client.close()


class OpenAICompatibleStrategyAgent:
    def __init__(
        self, settings: ProviderSettings, prompts: PromptLibrary | None = None
    ) -> None:
        self.client = OpenAICompatibleClient(settings)
        self.prompts = prompts or PromptLibrary()

    def advise(
        self,
        candidates: Sequence[CandidateAction],
        context: SampleContext,
        evidence: dict[str, Any],
    ) -> AgentDecision:
        candidate_payload = [candidate.to_dict() for candidate in candidates]
        content, raw, _ = self.client.chat(
            [
                {"role": "system", "content": self.prompts.get("action_advisor")},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "context": context.to_dict(),
                            "candidates": candidate_payload,
                            "evidence": evidence,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        data = extract_json(content)
        valid_ids = {candidate.action_id for candidate in candidates}
        scores: dict[str, float] = {}
        for item in data.get("ranked_actions", []):
            if not isinstance(item, dict):
                continue
            action_id = str(item.get("action_id", ""))
            if action_id in valid_ids:
                scores[action_id] = max(0.0, min(1.0, float(item.get("score", 0.0))))
        selected = str(data.get("selected_action_id", ""))
        if selected not in valid_ids:
            selected = max(
                candidates,
                key=lambda candidate: (scores.get(candidate.action_id, 0.0), candidate.utility),
            ).action_id
        scores.setdefault(selected, 1.0)
        valid_memories = {
            str(item.get("experience_id"))
            for item in evidence.get("retrieved_memories", [])
            if item.get("experience_id")
        }
        references = tuple(
            str(identifier)
            for identifier in data.get("memory_references", [])
            if str(identifier) in valid_memories
        )
        return AgentDecision(
            selected_action_id=selected,
            action_scores=scores,
            rationale=str(data.get("rationale", "")),
            memory_references=references,
            raw=raw,
        )

    def design(
        self,
        action: CandidateAction,
        context: SampleContext,
        parents: Sequence[StrategyNode],
        failure: FailureType | None,
        evidence: dict[str, Any],
    ) -> StrategySpec:
        parent_payload = [parent.to_dict() for parent in parents]
        schema = {
            "name": "short strategy name",
            "description": "sample-independent strategy description",
            "mechanism": "why the strategy should expose a model weakness",
            "modality_signature": ["text", "image", "cross_modal"],
            "preconditions": ["condition"],
            "execution_plan": ["ordered abstract step"],
            "tool_requirements": ["optional tool name"],
        }
        content, _, _ = self.client.chat(
            [
                {
                    "role": "system",
                    "content": self.prompts.get("strategy_designer"),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "operation": action.mode.value,
                            "goal_category": context.goal_category,
                            "image_description": context.image_description,
                            "failure_signature": (failure.value if failure else None),
                            "parents": parent_payload,
                            "evidence": evidence,
                            "required_schema": schema,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        return StrategySpec.from_dict(extract_json(content))

    def realize(
        self,
        strategy: StrategyNode,
        action: CandidateAction,
        context: SampleContext,
        attempt_index: int,
        evidence: dict[str, Any],
    ) -> TestCase:
        request_text = json.dumps(
            {
                "objective": context.goal,
                "image_description": context.image_description,
                "strategy": strategy.to_dict(),
                "attempt": attempt_index,
                "evidence": evidence,
            },
            ensure_ascii=False,
        )
        content, _, _ = self.client.chat(
            [
                {
                    "role": "system",
                    "content": self.prompts.get("testcase_realizer"),
                },
                {
                    "role": "user",
                    "content": multimodal_user_content(request_text, context.source_image),
                },
            ],
            json_mode=True,
        )
        data = extract_json(content)
        return TestCase(
            prompt=str(data.get("text_prompt", "")),
            image_path=context.source_image,
            strategy_id=strategy.strategy_id,
            component_strategy_ids=action.strategy_ids,
            action_mode=action.mode,
            metadata={
                "attempt_index": attempt_index,
                "image_operations": list(data.get("image_operations", [])),
                "actuator_notes": data.get("notes", ""),
                "controller_scores": action.component_scores,
            },
        )

    def repair(
        self,
        test_case: TestCase,
        check: ConsistencyResult,
        strategy: StrategyNode,
        context: SampleContext,
        evidence: dict[str, Any],
    ) -> TestCase:
        request_text = json.dumps(
            {
                "objective": context.goal,
                "strategy": strategy.to_dict(),
                "test_case": test_case.to_dict(),
                "checker_feedback": check.rationale,
                "evidence": evidence,
            },
            ensure_ascii=False,
        )
        content, _, _ = self.client.chat(
            [
                {
                    "role": "system",
                    "content": self.prompts.get("testcase_repair"),
                },
                {
                    "role": "user",
                    "content": multimodal_user_content(request_text, test_case.image_path),
                },
            ],
            json_mode=True,
        )
        data = extract_json(content)
        repaired = TestCase.from_dict(test_case.to_dict())
        repaired.prompt = str(data.get("text_prompt", repaired.prompt))
        repaired.metadata["image_operations"] = list(
            data.get("image_operations", repaired.metadata.get("image_operations", []))
        )
        repaired.metadata["repair_count"] = int(repaired.metadata.get("repair_count", 0)) + 1
        return repaired


class OpenAICompatibleTarget:
    def __init__(self, settings: ProviderSettings) -> None:
        self.client = OpenAICompatibleClient(settings)
        self.model_name = settings.model

    def generate(self, test_case: TestCase, context: SampleContext) -> TargetResponse:
        user_content = multimodal_user_content(test_case.prompt, test_case.image_path)
        content, raw, latency = self.client.chat(
            [{"role": "user", "content": user_content}], json_mode=False
        )
        return TargetResponse(
            text=content,
            model=self.model_name,
            latency_seconds=latency,
            raw=raw,
        )


class OpenAICompatibleJudge:
    def __init__(
        self, settings: ProviderSettings, prompts: PromptLibrary | None = None
    ) -> None:
        self.client = OpenAICompatibleClient(settings)
        self.prompts = prompts or PromptLibrary()

    def evaluate(
        self,
        context: SampleContext,
        test_case: TestCase,
        response: TargetResponse,
        settings: EvaluationSettings,
    ) -> EvaluationResult:
        request_text = json.dumps(
            {
                "objective": context.goal,
                "test_case": test_case.to_dict(),
                "target_response": response.text,
            },
            ensure_ascii=False,
        )
        content, raw, _ = self.client.chat(
            [
                {
                    "role": "system",
                    "content": self.prompts.get("response_judge"),
                },
                {
                    "role": "user",
                    "content": multimodal_user_content(request_text, test_case.image_path),
                },
            ],
            json_mode=True,
            temperature=0.0,
        )
        data = extract_json(content)
        safety = float(data.get("safety_violation", 0.0))
        relevance = float(data.get("relevance", 0.0))
        information = float(data.get("informativeness", 0.0))
        consistency = float(data.get("strategy_consistency", 0.0))
        success = (
            safety >= settings.safety_threshold
            and relevance >= settings.relevance_threshold
            and information >= settings.informativeness_threshold
            and consistency >= settings.consistency_threshold
        )
        failure_name = data.get("failure_type")
        try:
            failure = FailureType(failure_name) if failure_name else FailureType.UNKNOWN
        except ValueError:
            failure = FailureType.UNKNOWN
        return EvaluationResult(
            safety_violation=safety,
            relevance=relevance,
            informativeness=information,
            strategy_consistency=consistency,
            progress=float(data.get("progress", 0.0)),
            success=success,
            failure_type=None if success else failure,
            rationale=str(data.get("rationale", "")),
            raw=raw,
        )


class OpenAICompatibleConsistencyChecker:
    def __init__(
        self, settings: ProviderSettings, prompts: PromptLibrary | None = None
    ) -> None:
        self.client = OpenAICompatibleClient(settings)
        self.prompts = prompts or PromptLibrary()

    def check(
        self,
        context: SampleContext,
        strategy: StrategyNode,
        test_case: TestCase,
    ) -> ConsistencyResult:
        request_text = json.dumps(
            {
                "objective": context.goal,
                "image_description": context.image_description,
                "strategy": strategy.to_dict(),
                "test_case": test_case.to_dict(),
            },
            ensure_ascii=False,
        )
        content, _, _ = self.client.chat(
            [
                {
                    "role": "system",
                    "content": self.prompts.get("consistency_checker"),
                },
                {
                    "role": "user",
                    "content": multimodal_user_content(request_text, test_case.image_path),
                },
            ],
            json_mode=True,
            temperature=0.0,
        )
        data = extract_json(content)
        return ConsistencyResult(
            valid=bool(data.get("valid", False)),
            goal_preserved=bool(data.get("goal_preserved", False)),
            strategy_realized=bool(data.get("strategy_realized", False)),
            cross_modal_consistent=bool(data.get("cross_modal_consistent", False)),
            perceptible=bool(data.get("perceptible", False)),
            rationale=str(data.get("rationale", "")),
        )
