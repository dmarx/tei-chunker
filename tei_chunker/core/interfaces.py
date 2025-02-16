# tei_chunker/core/interfaces.py
"""
Core interfaces for document processing and synthesis.
"""
from typing import Protocol, Dict, List, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum

class Strategy(Enum):
    """Available processing strategies."""
    TOP_DOWN_MAXIMAL = "top_down_maximal"
    BOTTOM_UP = "bottom_up"
    HYBRID = "hybrid"

@dataclass
class ProcessingContext:
    """Shared context for document processing."""
    max_tokens: int
    overlap_tokens: int = 100
    min_chunk_tokens: int = 500

@dataclass
class Span:
    """Represents a span of text in the document."""
    start: int
    end: int
    text: str
    metadata: Dict = field(default_factory=dict)

@dataclass
class Feature:
    """Represents a feature derived from document content."""
    name: str
    content: str
    span: Span
    metadata: Dict = field(default_factory=dict)

class ContentProcessor(Protocol):
    """Protocol for content processing functions."""
    def __call__(self, content: str) -> str: ...

class SynthesisStrategy(Protocol):
    """Protocol for synthesis strategies."""
    def synthesize(
        self,
        content: str,
        features: Dict[str, List[Feature]],
        processor: ContentProcessor,
        context: ProcessingContext
    ) -> str: ...
