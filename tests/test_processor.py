# tests/test_processor.py
"""Tests for main feature processor."""
import pytest
from pathlib import Path
from typing import List

from tei_chunker.core.interfaces import Strategy
from tei_chunker.features.processor import FeatureProcessor
from tei_chunker.features.manager import FeatureRequest

class MockLLMClient:
    """Mock LLM client for testing."""
    def complete(self, prompt: str) -> str:
        return f"LLM response for: {prompt}"

@pytest.fixture
def processor(tmp_path):
    """Create feature processor with test configuration."""
    return FeatureProcessor(
        data_dir=tmp_path,
        llm_client=MockLLMClient()
    )

@pytest.fixture
def sample_requests():
    """Create sample feature requests with dependencies."""
    return [
        FeatureRequest(
            name="summary",
            prompt_template="Summarize: {content}",
            strategy=Strategy.TOP_DOWN_MAXIMAL,
            required_features=[]
        ),
        FeatureRequest(
            name="analysis",
            prompt_template="Analyze with summary: {content}",
            strategy=Strategy.HYBRID,
            required_features=["summary"]
        ),
        FeatureRequest(
            name="final",
            prompt_template="Final analysis: {content}",
            strategy=Strategy.BOTTOM_UP,
            required_features=["summary", "analysis"]
        )
    ]

def test_process_document(processor, sample_requests):
    """Test processing multiple feature requests for a document."""
    content = "Test document content"
    
    feature_ids = processor.process_document(content, sample_requests)
    
    assert len(feature_ids) == 3
    assert "summary" in feature_ids
    assert "analysis" in feature_ids
    assert "final" in feature_ids
    
    # Check features were created in correct order
    features = processor.get_features("final")
    assert len(features) == 1
    assert all(dep in features[0]["metadata"]["required_features"] 
              for dep in ["summary", "analysis"])

def test_request_ordering(processor):
    """Test requests are processed in correct dependency order."""
    # Create requests in "wrong" order
    requests = [
        FeatureRequest(
            name="complex",
            prompt_template="Template",
            required_features=["basic", "intermediate"]
        ),
        FeatureRequest(
            name="intermediate",
            prompt_template="Template",
            required_features=["basic"]
        ),
        FeatureRequest(
            name="basic",
            prompt_template="Template",
            required_features=[]
        )
    ]
    
    feature_ids = processor.process_document("Test content", requests)
    
    # Should process in correct order despite input order
    assert feature_ids.index("basic") < feature_ids.index("intermediate")
    assert feature_ids.index("intermediate") < feature_ids.index("complex")

def test_circular_dependency_detection(processor):
    """Test detection of circular dependencies in requests."""
    requests = [
        FeatureRequest(
            name="feature_a",
            prompt_template="Template",
            required_features=["feature_b"]
        ),
        FeatureRequest(
            name="feature_b",
            prompt_template="Template",
            required_features=["feature_a"]
        )
    ]
    
    with pytest.raises(ValueError) as exc_info:
        processor.process_document("Test content", requests)
    
    assert "Circular dependency" in str(exc_info.value)

def test_error_handling(processor, sample_requests):
    """Test error handling during processing."""
    # Create processor with failing LLM client
    class FailingLLMClient:
        def complete(self, prompt: str) -> str:
            raise ValueError("LLM failed")
            
    failing_processor = FeatureProcessor(
        data_dir=processor.data_dir,
        llm_client=FailingLLMClient()
    )
    
    # Should continue processing despite errors
    feature_ids = failing_processor.process_document(
        "Test content",
        sample_requests
    )
    
    assert len(feature_ids) == 0  # No features created due to errors

def test_feature_retrieval(processor, sample_requests):
    """Test retrieving processed features."""
    content = "Test document content"
    processor.process_document(content, sample_requests)
    
    # Get features
    summaries = processor.get_features("summary")
    analyses = processor.get_features("analysis")
    
    assert len(summaries) == 1
    assert len(analyses) == 1
    assert "LLM response" in summaries[0]["content"]
    assert "metadata" in summaries[0]
    assert "span" in summaries[0]

def test_dependency_graph(processor, sample_requests):
    """Test getting feature dependency graph."""
    content = "Test document content"
    processor.process_document(content, sample_requests)
    
    graph = processor.get_feature_dependencies("final")
    
    assert "final" in graph
    assert "summary" in graph
    assert "analysis" in graph
    assert graph["final"]["summary"]  # final depends on summary
    assert graph["final"]["analysis"]  # final depends on analysis
    assert not graph["summary"]  # summary has no dependencies

def test_processing_with_existing_features(processor):
    """Test processing requests when some features already exist."""
    # Process basic feature first
    basic_request = FeatureRequest(
        name="basic",
        prompt_template="Basic: {content}",
        required_features=[]
    )
    processor.process_document("Test content", [basic_request])
    
    # Now process dependent feature
    dependent_request = FeatureRequest(
        name="dependent",
        prompt_template="Dependent: {content}",
        required_features=["basic"]
    )
    feature_ids = processor.process_document(
        "Test content",
        [dependent_request]
    )
    
    assert len(feature_ids) == 1
    assert "dependent" in feature_ids
    
    # Check dependency was properly handled
    features = processor.get_features("dependent")
    assert "basic" in features[0]["metadata"]["required_features"]
