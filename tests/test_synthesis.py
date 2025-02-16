# tests/test_synthesis.py
"""Tests for document synthesis functionality."""
import pytest
from datetime import datetime

from tei_chunker.graph import DocumentGraph, Node
from tei_chunker.core.interfaces import Feature, Span
from tei_chunker.synthesis.base import Synthesizer, SynthesisNode
from tei_chunker.synthesis.patterns import (
    FeatureSynthesizer,
    SynthesisMode,
    FeatureDependency
)

@pytest.fixture
def sample_graph():
    """Create a sample document graph."""
    content = "This is a test document with multiple sections."
    graph = DocumentGraph(content)
    
    # Add root section
    root = graph.add_node(
        content="Root section",
        type="section",
        span=(0, len(content)),
        metadata={"level": 1}
    )
    
    # Add subsections
    section1 = graph.add_node(
        content="Section 1 content",
        type="section",
        span=(0, 20),
        parents=[root.id],
        metadata={"level": 2}
    )
    
    section2 = graph.add_node(
        content="Section 2 content",
        type="section",
        span=(21, len(content)),
        parents=[root.id],
        metadata={"level": 2}
    )
    
    # Add features
    graph.add_node(
        content="Summary of section 1",
        type="feature:summary",
        span=(0, 20),
        parents=[section1.id]
    )
    
    graph.add_node(
        content="Summary of section 2",
        type="feature:summary",
        span=(21, len(content)),
        parents=[section2.id]
    )
    
    return graph

def test_synthesis_tree_structure(sample_graph):
    """Test synthesis tree construction."""
    synthesizer = Synthesizer(sample_graph)
    root_node = sample_graph.get_nodes_by_type("section")[0]
    
    tree = synthesizer.get_synthesis_tree(
        root_node,
        feature_types=["summary"],
        max_depth=None
    )
    
    # Test tree structure
    assert tree.node_id == root_node.id
    assert len(tree.children) == 2
    assert tree.metadata['node_metadata']['level'] == 1
    assert all(child.metadata['node_metadata']['level'] == 2 
              for child in tree.children)

def test_feature_synthesis_modes(sample_graph):
    """Test different synthesis modes."""
    synthesizer = FeatureSynthesizer(sample_graph)
    root_node = sample_graph.get_nodes_by_type("section")[0]
    tree = synthesizer.get_synthesis_tree(root_node, ["summary"])
    
    # Test hierarchical synthesis
    synthesizer.synthesize_with_mode(
        tree,
        SynthesisMode.HIERARCHICAL,
        "hierarchical_summary",
        max_length=200
    )
    
    hier_features = sample_graph.get_feature_nodes("hierarchical_summary")
    assert len(hier_features) > 0
    
    # Test incremental synthesis
    synthesizer.synthesize_with_mode(
        tree,
        SynthesisMode.INCREMENTAL,
        "incremental_synthesis",
        feature_sequence=["summary"]
    )
    
    incr_features = sample_graph.get_feature_nodes("incremental_synthesis")
    assert len(incr_features) > 0

def test_feature_dependencies(sample_graph):
    """Test synthesis with feature dependencies."""
    synthesizer = FeatureSynthesizer(sample_graph)
    
    # Add dependent feature
    root = sample_graph.get_nodes_by_type("section")[0]
    sample_graph.add_node(
        content="Analysis based on summaries",
        type="feature:analysis",
        span=root.span,
        parents=[root.id],
        metadata={"depends_on": ["summary"]}
    )
    
    # Register dependency
    synthesizer.register_dependency(FeatureDependency(
        source_feature="summary",
        target_feature="analysis",
        relationship="informs"
    ))
    
    # Test dependency-aware synthesis
    tree = synthesizer.get_synthesis_tree(root, ["summary", "analysis"])
    synthesizer.synthesize_with_mode(
        tree,
        SynthesisMode.AGGREGATE,
        "aggregated_analysis",
        dependencies=[{
            "source_feature": "summary",
            "target_feature": "analysis",
            "relationship": "informs"
        }]
    )
    
    agg_features = sample_graph.get_feature_nodes("aggregated_analysis")
    assert len(agg_features) > 0

def test_overlapping_features(sample_graph):
    """Test handling of overlapping features."""
    synthesizer = FeatureSynthesizer(sample_graph)
    root = sample_graph.get_nodes_by_type("section")[0]
    
    # Add overlapping feature
    sample_graph.add_node(
        content="Overlapping analysis",
        type="feature:analysis",
        span=(15, 25),  # Overlaps sections
        parents=[root.id]
    )
    
    tree = synthesizer.get_synthesis_tree(root, ["analysis"])
    
    # Test cross-reference synthesis
    synthesizer.synthesize_with_mode(
        tree,
        SynthesisMode.CROSS_REFERENCE,
        "cross_referenced",
        feature_type="analysis"
    )
    
    features = sample_graph.get_feature_nodes("cross_referenced")
    assert len(features) > 0
    assert any("overlapping" in f.content.lower() for f in features)

def test_evidence_synthesis(sample_graph):
    """Test evidence-graded synthesis."""
    synthesizer = FeatureSynthesizer(sample_graph)
    root = sample_graph.get_nodes_by_type("section")[0]
    
    # Add evidence features
    sample_graph.add_node(
        content="Strong evidence for X",
        type="feature:evidence",
        span=root.span,
        parents=[root.id],
        metadata={"strength": "strong"}
    )
    
    tree = synthesizer.get_synthesis_tree(root, ["evidence"])
    synthesizer.evidence_graded_synthesis(
        tree,
        feature_types=["evidence"],
        confidence_threshold=0.8
    )
    
    features = sample_graph.get_feature_nodes("evidence_graded")
    assert len(features) > 0
    assert any("strong" in f.content.lower() for f in features)

def test_synthesis_persistence(tmp_path, sample_graph):
    """Test persistence of synthesized features."""
    synthesizer = FeatureSynthesizer(sample_graph)
    root = sample_graph.get_nodes_by_type("section")[0]
    
    # Create syntheses
    tree = synthesizer.get_synthesis_tree(root, ["summary"])
    synthesizer.synthesize_with_mode(
        tree,
        SynthesisMode.HIERARCHICAL,
        "hierarchical_summary"
    )
    
    # Save graph
    save_path = tmp_path / "test_graph.json"
    sample_graph.save(save_path)
    
    # Load graph
    loaded_graph = DocumentGraph.load(save_path)
    
    # Verify syntheses were preserved
    original_features = set(
        n.id for n in sample_graph.get_feature_nodes("hierarchical_summary")
    )
    loaded_features = set(
        n.id for n in loaded_graph.get_feature_nodes("hierarchical_summary")
    )
    assert original_features == loaded_features
    
    # Verify feature content and metadata preserved
    orig_feat = sample_graph.get_feature_nodes("hierarchical_summary")[0]
    loaded_feat = loaded_graph.get_feature_nodes("hierarchical_summary")[0]
    assert orig_feat.content == loaded_feat.content
    assert orig_feat.metadata == loaded_feat.metadata

def test_contextual_synthesis(sample_graph):
    """Test contextual synthesis considering document structure."""
    synthesizer = FeatureSynthesizer(sample_graph)
    root = sample_graph.get_nodes_by_type("section")[0]
    
    # Add some contextual features
    sample_graph.add_node(
        content="Document-level context",
        type="feature:context",
        span=root.span,
        parents=[root.id],
        metadata={"scope": "document"}
    )
    
    tree = synthesizer.get_synthesis_tree(root, ["summary", "context"])
    synthesizer.synthesize_with_mode(
        tree,
        SynthesisMode.CONTEXTUAL,
        "contextual_synthesis",
        dependencies=[{
            "source_feature": "context",
            "relationship": "provides_context"
        }]
    )
    
    features = sample_graph.get_feature_nodes("contextual_synthesis")
    assert len(features) > 0
    # Should incorporate both local and document-level content
    assert any(
        "context" in f.content.lower() and "summary" in f.content.lower() 
        for f in features
    )
