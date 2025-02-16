# tests/test_feature_manager.py
"""Tests for feature management functionality."""
import pytest
from pathlib import Path
from datetime import datetime

from tei_chunker.core.interfaces import Feature, Span, ProcessingContext
from tei_chunker.core.strategies import Strategy
from tei_chunker.features.manager import (
    FeatureManager,
    FeatureStore,
    FeatureRequest
)
from tei_chunker.synthesis.patterns import SynthesisMode

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
    
    # Add test features
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
def basic_request():
    """Create basic feature request."""
    return FeatureRequest(
        name="analysis",
        prompt_template="Analyze: {content}",
        strategy=Strategy.TOP_DOWN_MAXIMAL,
        required_features=["summary"]
    )

@pytest.fixture
def advanced_request():
    """Create request with synthesis configuration."""
    return FeatureRequest(
        name="synthesis",
        prompt_template="Synthesize: {content}",
        strategy=Strategy.TOP_DOWN_MAXIMAL,
        required_features=["summary", "analysis"],
        synthesis_config={
            "mode": SynthesisMode.HIERARCHICAL,
            "max_length": 500,
            "dependencies": [{
                "source_feature": "summary",
                "relationship": "informs"
            }]
        }
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
    
    store.save_feature(feature)
    
    # Load and verify
    new_store = FeatureStore(tmp_feature_dir)
    loaded = new_store.get_features("test")
    assert len(loaded) == 1
    assert loaded[0].content == feature.content
    assert loaded[0].metadata == feature.metadata

def test_feature_store_get_by_span(feature_store):
    """Test getting features filtered by span."""
    features = [
        Feature(
            name="test",
            content=f"Content {i}",
            span=Span(i*50, (i+1)*50, f"Content {i}"),
            metadata={}
        )
        for i in range(3)
    ]
    
    for f in features:
        feature_store.save_feature(f)
    
    # Get overlapping features
    results = feature_store.get_features(
        "test",
        span=Span(40, 110, "test")
    )
    
    assert len(results) == 2  # Should get features that overlap span

def test_feature_manager_process_request(
    feature_manager,
    basic_request
):
    """Test basic feature request processing."""
    content = "Test document content"
    llm_client = MockLLMClient()
    
    # Add required feature
    feature_manager.store.save_feature(Feature(
        name="summary",
        content="Summary content",
        span=Span(0, len(content), content),
        metadata={}
    ))
    
    # Process request
    feature = feature_manager.process_request(
        content,
        basic_request,
        llm_client
    )
    
    assert feature.name == "analysis"
    assert "LLM response" in feature.content
    assert feature.metadata["strategy"] == Strategy.TOP_DOWN_MAXIMAL.value

def test_feature_manager_advanced_request(
    feature_manager,
    advanced_request
):
    """Test processing request with synthesis configuration."""
    content = "Test document content"
    llm_client = MockLLMClient()
    
    # Add required features
    for name in ["summary", "analysis"]:
        feature_manager.store.save_feature(Feature(
            name=name,
            content=f"{name} content",
            span=Span(0, len(content), content),
            metadata={}
        ))
    
    feature = feature_manager.process_request(
        content,
        advanced_request,
        llm_client
    )
    
    assert feature.name == "synthesis"
    assert feature.metadata["synthesis_mode"] == SynthesisMode.HIERARCHICAL.value
    assert "dependencies" in feature.metadata

def test_feature_manager_validation(feature_manager):
    """Test feature request validation."""
    request = FeatureRequest(
        name="invalid",
        prompt_template="Template",
        required_features=["nonexistent"]
    )
    
    errors = feature_manager.validate_feature_request(request)
    assert len(errors) == 1
    assert "nonexistent" in errors[0]

def test_circular_dependency_detection(feature_manager):
    """Test detection of circular dependencies."""
    # Create circular dependency
    feature_a = FeatureRequest(
        name="feature_a",
        prompt_template="Template",
        required_features=["feature_b"]
    )
    
    feature_b = Feature(
        name="feature_b",
        content="Content",
        span=Span(0, 100, "test"),
        metadata={"required_features": ["feature_a"]}
    )
    
    feature_manager.store.save_feature(feature_b)
    
    errors = feature_manager.validate_feature_request(feature_a)
    assert len(errors) == 1
    assert "circular" in errors[0].lower()

def test_feature_chain_resolution(feature_manager):
    """Test resolving feature dependency chains."""
    # Create feature chain: raw -> processed -> analyzed
    features = [
        Feature(
            name="raw",
            content="Raw content",
            span=Span(0, 100, "test"),
            metadata={}
        ),
        Feature(
            name="processed",
            content="Processed content",
            span=Span(0, 100, "test"),
            metadata={"required_features": ["raw"]}
        ),
        Feature(
            name="analyzed",
            content="Analyzed content",
            span=Span(0, 100, "test"),
            metadata={"required_features": ["processed"]}
        )
    ]
    
    for f in features:
        feature_manager.store.save_feature(f)
        
    chain = feature_manager.get_feature_chain("analyzed")
    
    assert len(chain) == 3
    assert list(chain[0].keys())[0] == "raw"
    assert list(chain[1].keys())[0] == "processed"
    assert list(chain[2].keys())[0] == "analyzed"

def test_error_handling(feature_manager, basic_request):
    """Test error handling during feature processing."""
    class FailingLLMClient:
        def complete(self, prompt: str) -> str:
            raise ValueError("LLM processing failed")
    
    with pytest.raises(ValueError) as exc_info:
        feature_manager.process_request(
            "content",
            basic_request,
            FailingLLMClient()
        )
    
    assert "LLM processing failed" in str(exc_info.value)
