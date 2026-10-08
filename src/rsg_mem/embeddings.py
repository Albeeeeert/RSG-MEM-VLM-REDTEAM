"""Small deterministic embeddings used by the offline prototype.

Production experiments can replace :class:`HashingEmbedder` with a CLIP/VLM
embedding adapter without changing graph or memory APIs.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Iterable, Protocol


class Embedder(Protocol):
    dimension: int

    def encode_text(self, text: str) -> list[float]: ...

    def encode_image(self, image_path: str | None, description: str = "") -> list[float]: ...

    def encode_cross_modal(self, text: str, image_description: str) -> list[float]: ...


def normalize(vector: Iterable[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


def cosine(left: Iterable[float], right: Iterable[float]) -> float:
    a = list(left)
    b = list(right)
    if not a or not b or len(a) != len(b):
        return 0.0
    denominator = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if denominator == 0.0:
        return 0.0
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / denominator))


def mean_vectors(vectors: Iterable[Iterable[float]]) -> list[float]:
    items = [list(vector) for vector in vectors if vector]
    if not items:
        return []
    dimension = len(items[0])
    if any(len(item) != dimension for item in items):
        raise ValueError("all vectors must have the same dimension")
    return normalize(sum(item[index] for item in items) / len(items) for index in range(dimension))


class HashingEmbedder:
    """Dependency-light feature hashing for deterministic tests and prototypes."""

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 16:
            raise ValueError("dimension must be at least 16")
        self.dimension = dimension

    def _tokens(self, text: str) -> list[str]:
        words = re.findall(r"[\w\-]+", text.lower(), flags=re.UNICODE)
        bigrams = [f"{left}::{right}" for left, right in zip(words, words[1:])]
        return words + bigrams

    def encode_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
        return normalize(vector)

    def encode_image(self, image_path: str | None, description: str = "") -> list[float]:
        descriptor = description
        if image_path:
            path = Path(image_path)
            descriptor = f"{descriptor}\nfilename:{path.name}"
            if path.exists() and path.is_file():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                descriptor += f"\nsha256:{digest}\nsize:{path.stat().st_size}"
        return self.encode_text(descriptor or "no-image")

    def encode_cross_modal(self, text: str, image_description: str) -> list[float]:
        return self.encode_text(f"text::{text}\nimage::{image_description}")
