from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KnowledgeDocument:
    """A source knowledge article or chunk ready for indexing."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedDocument:
    """A document returned by a retriever."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    distance: float | None = None
    embedding: list[float] | None = None


@dataclass
class RagResponse:
    answer: str
    sources: list[RetrievedDocument]
    mode: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
