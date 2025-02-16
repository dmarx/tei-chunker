# tests/test_feature_manager.py
"""Tests for feature management functionality."""
import pytest
from pathlib import Path
from typing import Any
from datetime import datetime

from tei_chunker.core.interfaces import Strategy, Feature, Span
from tei_chunker.features.manager import (
    FeatureManager,
    FeatureStore,
    FeatureRequest
)

class MockLLMClient:
    """Mock LLM client for testing."""
    def complete(self, prompt: str) -> str:
        return f"LLM response for: {prompt}"

@pytest.fixture
def tmp_feature_dir(tmp_path):
    """Create temporary feature directory."""
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    return feature_dir

@pytest.fixture
def feature_store(tmp_feature_dir):
    """Create feature store with some test data."""
    store = FeatureStore(tmp_feature_dir)
    
    # Add some test features
    summary_feature = Feature(
        name="summary",
        content="Test summary",
        span=Span(0, 100, "Test content"),
        metadata={"created_at": datetime.utcnow().isoformat()}
    )
    store.save_feature(summary_feature)
    
    return store

@pytest.fixture
def feature_manager(tmp_feature_dir):
    """Create feature manager with mock LLM client."""
    return FeatureManager(
        tmp_feature_dir,
        xml_processor=None
    )

@pytest.fixture
def sample_request():
    """Create sample feature request."""
    return FeatureRequest(
        name="analysis",
        prompt_template="Analyze: {content}",
        strategy=Strategy.TOP_DOWN_MAXIMAL,
        required_features=["summary"]
    )

def test_feature_store_save_load(tmp_feature_dir):
    """Test saving and loading features."""
    store = FeatureStore(tmp_feature_dir)
    
    feature = Feature(
        name="test",
        content="Test content",
        span=Span(0, 100, "Test content"),
        metadata={"test_meta": "value"}
    )
    
    # Save feature
    store.save_feature(feature)
    
    # Create new store instance and check feature loads
    new_store = FeatureStore(tmp_feature_dir)
    loaded_features = new_store.get_features("test")
    
    assert len(loaded_features) == 1
    assert loaded_features[0].content == "Test content"
    assert loaded_features[0].metadata["test_meta"] == "value"

def test_feature_store_get_by_span(feature_store):
    """Test getting features filtered by span."""
    # Add features with different spans
    feature1 = Feature(
        name="test",
        content="Content 1",
        span=Span(0, 50, "Content 1"),
        metadata={}
    )
    feature2 = Feature(
        name="test",
        content="Content 2",
        span=Span(40, 90, "Content 2"),
        metadata={}
    )
    
    feature_store.save_feature(feature1)
    feature_store.save_feature(feature2)
    
    # Get features overlapping span
    features = feature_store.get_features(
        "test",
        span=Span(30, 60, "test")
    )
    
    assert len(features) == 2  # Both features overlap

def test_feature_manager_process_request(
    feature_manager,
    sample_request
):
    """Test processing a feature request."""
    content = "Test document content"
    llm_client = MockLLMClient()
    
    feature = feature_manager.process_request(
        content,
        sample_request,
        llm_client
    )
    
    assert feature.name == "analysis"
    assert "LLM response" in feature.content
    assert feature.metadata["strategy"] == Strategy.TOP_DOWN_MAXIMAL.value
    assert "created_at" in feature.metadata

def test_feature_manager_validate_request(
    feature_manager,
    sample_request
):
    """Test feature request validation."""
    # Test missing required feature
    errors = feature_manager.validate_feature_request(sample_request)
    assert len(errors) == 1
    assert "summary" in errors[0]
    
    # Add required feature and test again
    summary_feature = Feature(
        name="summary",
        content="Summary content",
        span=Span(0, 100, "test"),
        metadata={}
    )
    feature_manager.store.save_feature(summary_feature)
    
    errors = feature_manager.validate_feature_request(sample_request)
    assert len(errors) == 0

def test_feature_manager_circular_dependencies(feature_manager):
    """Test detection of circular dependencies."""
    # Create features with circular dependencies
    feature_a = FeatureRequest(
        name="feature_a",
        prompt_template="Template",
        required_features=["feature_b"]
    )
    
    feature_b = FeatureRequest(
        name="feature_b",
        prompt_template="Template",
        required_features=["feature_a"]
    )
    
    # Add one feature first
    feature_manager.store.save_feature(Feature(
        name="feature_b",
        content="Content",
        span=Span(0, 100, "test"),
        metadata={"required_features": ["feature_a"]}
    ))
    
    # Validate should detect circular dependency
    errors = feature_manager.validate_feature_request(feature_a)
    assert len(errors) == 1
    assert "Circular dependency" in errors[0]

def test_feature_manager_get_feature_chain(
    feature_manager
):
    """Test getting feature dependency chain."""
    # Create features with dependencies
    feature_manager.store.save_feature(Feature(
        name="raw",
        content="Raw content",
        span=Span(0, 100, "test"),
        metadata={}
    ))
    
    feature_manager.store.save_feature(Feature(
        name="summary",
        content="Summary",
        span=Span(0, 100, "test"),
        metadata={"required_features": ["raw"]}
    ))
    
    feature_manager.store.save_feature(Feature(
        name="analysis",
        content="Analysis",
        span=Span(0, 100, "test"),
        metadata={"required_features": ["summary"]}
    ))
    
    # Get feature chain
    chain = feature_manager.get_feature_chain("analysis")
    
    assert len(chain) == 3
    assert "raw" in chain[0]
    assert "summary" in chain[1]
    assert "analysis" in chain[2]

def test_feature_manager_error_handling(
    feature_manager,
    sample_request
):
    """Test error handling in feature manager."""
    content = "Test content"
    
    class FailingLLMClient:
        def complete(self, prompt: str) -> str:
            raise ValueError("LLM processing failed")
            
    with pytest.raises(Exception) as exc_info:
        feature_manager.process_request(
            content,
            sample_request,
            FailingLLMClient()
        )
    
    assert "LLM processing failed" in str(exc_info.value)

def test_feature_persistence(feature_manager, sample_request):
    """Test feature persistence across manager instances."""
    content = "Test content"
    llm_client = MockLLMClient()
    
    # Process feature with first manager
    feature = feature_manager.process_request(
        content,
        sample_request,
        llm_client
    )
    
    # Create new manager instance
    new_manager = FeatureManager(
        feature_manager.store.storage_dir,
        xml_processor=None
    )
    
    # Check feature was loaded
    features = new_manager.store.get_features(feature.name)
    assert len(features) == 1
    assert features[0].content == feature.content
    
def test_feature_graph(feature_manager):
    """Test getting feature dependency graph."""
    # Add features with complex dependencies
    features = [
        Feature(
            name="base",
            content="Base content",
            span=Span(0, 100, "test"),
            metadata={}
        ),
        Feature(
            name="derived1",
            content="Derived 1",
            span=Span(0, 100, "test"),
            metadata={"required_features": ["base"]}
        ),
        Feature(
            name="derived2",
            content="Derived 2",
            span=Span(0, 100, "test"),
            metadata={"required_features": ["base"]}
        ),
        Feature(
            name="complex",
            content="Complex",
            span=Span(0, 100, "test"),
            metadata={"required_features": ["derived1", "derived2"]}
        )
    ]
    
    for feat in features:
        feature_manager.store.save_feature(feat)
        
    # Get dependency graph
    graph = feature_manager.get_feature_graph()
    
    assert len(graph) == 4
    assert "base" in graph
    assert not graph["base"]  # No dependencies
    assert "derived1" in graph
    assert "base" in graph["derived1"]
    assert "complex" in graph
    assert all(dep in graph["complex"] for dep in ["derived1", "derived2"])
