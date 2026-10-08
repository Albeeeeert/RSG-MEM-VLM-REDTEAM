"""Constrained image operations and multimodal test-case materialization."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from .models import TestCase


class ImageOperationError(ValueError):
    """Raised when a requested image operation is outside the tool registry."""


class PillowToolRegistry:
    """Small, auditable registry for deterministic local image transformations."""

    allowed_operations = frozenset(
        {"rotate", "grayscale", "blur", "contrast", "resize", "overlay_text"}
    )

    @staticmethod
    def _number(operation: dict[str, Any], key: str, default: float) -> float:
        try:
            return float(operation.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ImageOperationError(f"{key} must be numeric") from exc

    def apply_one(self, image: Image.Image, operation: dict[str, Any]) -> Image.Image:
        name = str(operation.get("op", operation.get("name", ""))).strip().lower()
        if name not in self.allowed_operations:
            raise ImageOperationError(f"unsupported image operation: {name or '<empty>'}")

        if name == "rotate":
            degrees = max(-180.0, min(180.0, self._number(operation, "degrees", 0.0)))
            return image.rotate(degrees, expand=True)
        if name == "grayscale":
            return ImageOps.grayscale(image).convert("RGB")
        if name == "blur":
            radius = max(0.0, min(20.0, self._number(operation, "radius", 1.0)))
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        if name == "contrast":
            factor = max(0.1, min(4.0, self._number(operation, "factor", 1.0)))
            return ImageEnhance.Contrast(image).enhance(factor)
        if name == "resize":
            width = int(max(16, min(4096, self._number(operation, "width", image.width))))
            height = int(max(16, min(4096, self._number(operation, "height", image.height))))
            return image.resize((width, height), Image.Resampling.LANCZOS)

        text = str(operation.get("text", ""))[:500]
        if not text:
            return image
        result = image.convert("RGB")
        draw = ImageDraw.Draw(result)
        x = int(max(0, self._number(operation, "x", 12)))
        y = int(max(0, self._number(operation, "y", 12)))
        fill = operation.get("fill", "white")
        stroke_fill = operation.get("stroke_fill", "black")
        stroke_width = int(max(0, min(8, self._number(operation, "stroke_width", 2))))
        draw.text(
            (x, y),
            text,
            fill=fill,
            font=ImageFont.load_default(),
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        return result

    def apply(
        self,
        source: str | Path,
        destination: str | Path,
        operations: Iterable[dict[str, Any]],
    ) -> Path:
        source_path = Path(source)
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            for operation in operations:
                if not isinstance(operation, dict):
                    raise ImageOperationError("each image operation must be an object")
                image = self.apply_one(image, operation)
            image.save(destination_path, format="PNG", optimize=True)
        return destination_path


class MultimodalActuator:
    """Executes only registered image operations and records artifact provenance."""

    def __init__(self, artifacts_dir: str | Path, registry: PillowToolRegistry | None = None) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.registry = registry or PillowToolRegistry()

    def materialize(self, test_case: TestCase) -> TestCase:
        result = TestCase.from_dict(test_case.to_dict())
        operations = result.metadata.get("image_operations", [])
        if not isinstance(operations, list):
            raise ImageOperationError("image_operations must be a list")
        if not result.image_path or not operations:
            result.metadata["executed_image_operations"] = []
            return result

        source = Path(str(result.metadata.get("source_image_path", result.image_path)))
        if not source.is_file():
            raise FileNotFoundError(f"source image does not exist: {source}")
        operation_bytes = repr(operations).encode("utf-8")
        digest = hashlib.sha256(source.read_bytes() + operation_bytes).hexdigest()[:16]
        destination = self.artifacts_dir / f"{result.test_id}_{digest}.png"
        self.registry.apply(source, destination, operations)
        result.metadata["source_image_path"] = str(source)
        result.metadata["executed_image_operations"] = operations
        result.metadata["artifact_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        result.image_path = str(destination)
        return result
