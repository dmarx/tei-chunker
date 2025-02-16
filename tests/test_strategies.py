# tests/test_strategies.py
"""Tests for synthesis strategies."""
import pytest
from typing import Dict, List

from tei_chunker.core.strategies import Strategy, TopDownStrategy, BottomUpStrategy, HybridStrategy
from tei_chunker.core.interfaces import ProcessingContext, Feature, Span, ContentProcessor


@pytest.fixture
def context():
    """Basic processing context."""
    return ProcessingContext(
        max_tokens=100,
        overlap_tokens=20,
        min_chunk_tokens=50
    )

@pytest.fixture
def simple_processor():
    """Simple content processor for testing."""
    def process(content: str) -> str:
        return f"Processed: {content}"
    return process

@pytest.fixture
def sample_content():
    """Sample document content."""
    return """Section 1
    This is the first section of the document.
    It contains multiple paragraphs.

    Section 2
    This is the second section.
    It also has content.

    Section 3
    This is the final section.
    It completes the document."""

@pytest.fixture
def sample_features():
    """Sample features for testing."""
    return {
        "summary": [
            Feature(
                name="summary",
                content="Summary of section 1",
                span=Span(0, 100, "Section 1 content"),
                metadata={}
            ),
            Feature(
                name="summary",
                content="Summary of section 2",
                span=Span(101, 200, "Section 2 content"),
                metadata={}
            )
        ],
        "keywords": [
            Feature(
                name="keywords",
                content="keywords for all content",
                span=Span(0, 300, "Full content"),
                metadata={}
            )
        ]
    }

def test_top_down_strategy_fits_context(
    context: ProcessingContext,
    simple_processor: ContentProcessor,
    sample_content: str,
    sample_features: Dict[str, List[Feature]]
):
    """Test top-down strategy when content fits in context."""
    strategy = TopDownStrategy()
    # Use small content that will fit
    small_content = "Small test content"
    
    result = strategy.synthesize(
        small_content,
        sample_features,
        simple_processor,
        context
    )
    
    assert "Processed" in result
    assert "Small test content" in result
    assert "summary" in result.lower()
    assert "keywords" in result.lower()

def test_top_down_strategy_splits_content(
    context: ProcessingContext,
    simple_processor: ContentProcessor,
    sample_content: str,
    sample_features: Dict[str, List[Feature]]
):
    """Test top-down strategy splits content when needed."""
    strategy = TopDownStrategy()
    
    result = strategy.synthesize(
        sample_content,
        sample_features,
        simple_processor,
        context
    )
    
    # Should have processed multiple chunks
    assert result.count("Processed") > 1
    # Should contain content from different sections
    assert "Section 1" in result
    assert "Section 2" in result
    assert "Section 3" in result

def test_bottom_up_strategy(
    context: ProcessingContext,
    simple_processor: ContentProcessor,
    sample_content: str,
    sample_features: Dict[str, List[Feature]]
):
    """Test bottom-up strategy processing."""
    strategy = BottomUpStrategy()
    
    result = strategy.synthesize(
        sample_content,
        sample_features,
        simple_processor,
        context
    )
    
    # Should process sections separately and combine
    assert "Section 1" in result
    assert "Section 2" in result
    assert "Section 3" in result
    # Should include features
    assert "summary" in result.lower()
    assert "keywords" in result.lower()

def test_hybrid_strategy_small_content(
    context: ProcessingContext,
    simple_processor: ContentProcessor,
    sample_features: Dict[str, List[Feature]]
):
    """Test hybrid strategy with small content (should use top-down)."""
    strategy = HybridStrategy()
    small_content = "Small test content"
    
    result = strategy.synthesize(
        small_content,
        sample_features,
        simple_processor,
        context
    )
    
    # Should process everything at once
    assert result.count("Processed") == 1
    assert "Small test content" in result

def test_hybrid_strategy_large_content(
    context: ProcessingContext,
    simple_processor: ContentProcessor,
    sample_content: str,
    sample_features: Dict[str, List[Feature]]
):
    """Test hybrid strategy with large content (should fall back to bottom-up)."""
    strategy = HybridStrategy()
    
    result = strategy.synthesize(
        sample_content,
        sample_features,
        simple_processor,
        context
    )
    
    # Should have processed multiple chunks
    assert result.count("Processed") > 1
    # Should contain all sections
    assert "Section 1" in result
    assert "Section 2" in result
    assert "Section 3" in result

def test_strategy_with_overlapping_features(
    context: ProcessingContext,
    simple_processor: ContentProcessor,
    sample_content: str
):
    """Test handling of overlapping features."""
    features = {
        "analysis": [
            Feature(
                name="analysis",
                content="Analysis of sections 1-2",
                span=Span(0, 150, "Sections 1-2"),
                metadata={}
            ),
            Feature(
                name="analysis",
                content="Analysis of sections 2-3",
                span=Span(100, 300, "Sections 2-3"),
                metadata={}
            )
        ]
    }
    
    strategy = TopDownStrategy()
    result = strategy.synthesize(
        sample_content,
        features,
        simple_processor,
        context
    )
    
    # Should handle overlapping features appropriately
    assert "Analysis of sections 1-2" in result
    assert "Analysis of sections 2-3" in result

def test_strategy_respects_min_chunk_size(
    simple_processor: ContentProcessor,
    sample_content: str,
    sample_features: Dict[str, List[Feature]]
):
    """Test that strategies respect minimum chunk size."""
    # Set very small max tokens but large minimum chunk
    context = ProcessingContext(
        max_tokens=20,
        min_chunk_tokens=100  # Larger than max_tokens
    )
    
    strategy = TopDownStrategy()
    
    with pytest.raises(ValueError) as exc_info:
        strategy.synthesize(
            sample_content,
            sample_features,
            simple_processor,
            context
        )
    
    assert "cannot be subdivided further" in str(exc_info.value)

def test_strategy_handles_empty_features(
    context: ProcessingContext,
    simple_processor: ContentProcessor,
    sample_content: str
):
    """Test strategies work with no features."""
    strategy = TopDownStrategy()
    
    result = strategy.synthesize(
        sample_content,
        {},  # No features
        simple_processor,
        context
    )
    
    assert "Processed" in result
    assert "Section 1" in result
    assert "Section 2" in result

def test_strategy_error_handling(
    context: ProcessingContext,
    sample_content: str,
    sample_features: Dict[str, List[Feature]]
):
    """Test error handling in strategies."""
    def failing_processor(content: str) -> str:
        raise ValueError("Processing failed")
    
    strategy = TopDownStrategy()
    
    with pytest.raises(ValueError) as exc_info:
        strategy.synthesize(
            sample_content,
            sample_features,
            failing_processor,
            context
        )
    
    assert "Processing failed" in str(exc_info.value)
