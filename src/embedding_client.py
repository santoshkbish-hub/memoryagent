from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.config import Settings


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#.-]*")


class EmbeddingClient(Protocol):
    def embed_text(self, text: str) -> list[float]:
        ...


@dataclass
class HashEmbeddingClient:
    """Deterministic local embedding used for tests, demos, and offline runs.

    This is a bounded feature-hashing vectorizer. It is not as semantically rich
    as a hosted embedding model, but keeps retrieval behavior deterministic and
    fast at the assignment scale.
    """

    dimensions: int = 384

    def embed_text(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        tokens = [token.lower() for token in TOKEN_RE.findall(text)]
        if not tokens:
            return vector.tolist()

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big")
            index = raw % self.dimensions
            sign = 1.0 if (raw >> 8) & 1 else -1.0
            vector[index] += sign

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()


@dataclass
class OpenAICompatibleEmbeddingClient:
    api_key: str
    base_url: str | None
    model: str

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("EMBEDDING_API_KEY or OPENAI_API_KEY is required for hosted embeddings.")
        from openai import OpenAI

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    def embed_text(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)


@dataclass
class SentenceTransformerEmbeddingClient:
    model_name: str = "all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> list[float]:
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    denom = math.sqrt(left_norm) * math.sqrt(right_norm)
    return dot / denom if denom else 0.0


def get_embedding_client(settings: Settings) -> EmbeddingClient:
    if settings.embedding_provider == "openai_compatible":
        return OpenAICompatibleEmbeddingClient(
            api_key=settings.embedding_api_key or "",
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
        )
    if settings.embedding_provider == "local":
        return SentenceTransformerEmbeddingClient(model_name=settings.embedding_model)
    return HashEmbeddingClient(dimensions=settings.embedding_dimensions)
