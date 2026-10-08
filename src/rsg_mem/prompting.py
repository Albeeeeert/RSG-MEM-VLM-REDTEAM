"""Versionable prompt loading with optional experiment-specific overrides."""

from __future__ import annotations

import hashlib
from importlib import resources
from pathlib import Path


PROMPT_NAMES = (
    "action_advisor",
    "strategy_designer",
    "testcase_realizer",
    "testcase_repair",
    "consistency_checker",
    "response_judge",
)


class PromptLibrary:
    """Loads built-in Markdown prompts or same-name files from an override directory."""

    def __init__(self, override_dir: str | Path | None = None) -> None:
        self.override_dir = Path(override_dir).expanduser().resolve() if override_dir else None
        if self.override_dir is not None and not self.override_dir.is_dir():
            raise FileNotFoundError(
                f"prompt override directory does not exist: {self.override_dir}"
            )

    def names(self) -> tuple[str, ...]:
        return PROMPT_NAMES

    def get(self, name: str) -> str:
        if name not in PROMPT_NAMES:
            raise KeyError(f"unknown prompt template: {name}")
        if self.override_dir is not None:
            override = self.override_dir / f"{name}.md"
            if override.is_file():
                return override.read_text(encoding="utf-8").strip()
        packaged = resources.files("rsg_mem.prompt_templates").joinpath(f"{name}.md")
        return packaged.read_text(encoding="utf-8").strip()

    def manifest(self) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for name in self.names():
            content = self.get(name)
            source = "built-in"
            if self.override_dir is not None and (self.override_dir / f"{name}.md").is_file():
                source = str(self.override_dir / f"{name}.md")
            result[name] = {
                "source": source,
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        return result
