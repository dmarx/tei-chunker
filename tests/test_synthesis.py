# tests/test_synthesis.py
"""
Tests for document synthesis functionality.
File: tests/test_synthesis.py
"""
import pytest
from pathlib import Path
from datetime import datetime

from tei_chunker.graph import DocumentGraph, Node, Feature
from tei_chunker.synthesis.base import Synthesizer, SynthesisNode
from tei_chunker.synthesis.patterns import FeatureSynthesizer, SynthesisStrategy

@pytest.fixture
def sample_graph():
    """Create a sample document graph."""
    content = "This is a test document with multiple sections."
    graph = DocumentGraph(content)
    
    # Add root section
    root = graph.add_node(
        content="Root section",
        type="section",
        span=(0, len(content))
    )
    
    # Add subsections
    section1 = graph.add_node(
        content="Section 1",
        type="section",
        span=(0, 20),
        parents=[root.id]
    )
    
    section2 = graph.add_node(
        content="Section 2",
        type="section",
        span=(21, len(content)),
        parents=[root.id]
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

def test_synthesis_tree_creation(sample_graph):
    """Test creation of synthesis tree."""
    synthesizer = Synthesizer(sample_graph)
    root_node = sample_graph.get_nodes_by_type("section")[0]
    
    tree = synthesizer.get_synthesis_tree(
        root_node,
        feature_types=["summary"],
        max_depth=None
    )
    
    assert tree.node_id == root_node.id
    assert len(tree.children) == 2
    assert "summary" in tree.metadata["features"]

def test_hierarchical_summary(sample_graph):
    """Test hierarchical summary synthesis."""
    synthesizer = FeatureSynthesizer(sample_graph)
    root_node = sample_graph.get_nodes_by_type("section")[0]
    
    tree = synthesizer.get_synthesis_tree(
        root_node,
        feature_types=["summary"],
        max_depth=None
    )
    
    synthesizer.hierarchical_summary(tree, max_length=200)
    
    # Check that new feature was created
    summaries = sample_graph.get_feature_nodes("hierarchical_summary")
    assert len(summaries) > 0
    assert all(s.type == "feature:hierarchical_summary" for s in summaries)

def test_conflict_resolution(sample_graph):
    """Test conflict resolution between overlapping features."""
    synthesizer = FeatureSynthesizer(sample_graph)
    root_node = sample_graph.get_nodes_by_type("section")[0]
    
    # Add overlapping feature
    sample_graph.add_node(
        content="Overlapping summary",
        type="feature:summary",
        span=(15, 25),  # Overlaps sections 1 and 2
        parents=[root_node.id]
    )
    
    tree = synthesizer.get_synthesis_tree(
        root_node,
        feature_types=["summary"],
        max_depth=None
    )
    
    synthesizer.resolve_conflicts(tree, "summary")
    
    resolved = sample_graph.get_feature_nodes("resolved_summary")
    assert len(resolved) > 0
    assert any("overlapping" in n.metadata for n in resolved)

def test_evidence_graded_synthesis(sample_graph):
    """Test evidence-graded synthesis."""
    synthesizer = FeatureSynthesizer(sample_graph)
    root_node = sample_graph.get_nodes_by_type("section")[0]
    
    tree = synthesizer.get_synthesis_tree(
        root_node,
        feature_types=["summary"],
        max_depth=None
    )
    
    synthesizer.evidence_graded_synthesis(
        tree,
        feature_types=["summary"],
        confidence_threshold=0.8
    )
    
    graded = sample_graph.get_feature_nodes("evidence_graded")
    assert len(graded) > 0
    assert all("evidence" in n.content.lower() for n in graded)

def test_incremental_synthesis(sample_graph):
    """Test incremental feature synthesis."""
    # Add another feature type
    section = sample_graph.get_nodes_by_type("section")[0]
    sample_graph.add_node(
        content="Keywords for section",
        type="feature:keywords",
        span=section.span,
        parents=[section.id]
    )
    
    synthesizer = FeatureSynthesizer(sample_graph)
    tree = synthesizer.get_synthesis_tree(
        section,
        feature_types=["summary", "keywords"],
        max_depth=None
    )
    
    synthesizer.incremental_synthesis(
        tree,
        feature_sequence=["summary", "keywords"]
    )
    
    incremental = sample_graph.get_feature_nodes("incremental_2")
    assert len(incremental) > 0
    assert all(
        "summary" in n.content.lower() and "keywords" in n.content.lower()
        for n in incremental
    )

def test_graph_persistence(tmp_path, sample_graph):
    """Test saving and loading graph with syntheses."""
    save_path = tmp_path / "test_graph.json"
    
    # Create some syntheses
    synthesizer = FeatureSynthesizer(sample_graph)
    root_node = sample_graph.get_nodes_by_type("section")[0]
    tree = synthesizer.get_synthesis_tree(
        root_node,
        feature_types=["summary"],
        max_depth=None
    )
    synthesizer.hierarchical_summary(tree)
    
    # Save graph
    sample_graph.save(save_path)
    
    # Load graph
    loaded = DocumentGraph.load(save_path)
    
    # Check that syntheses were preserved
    original_features = set(n.id for n in sample_graph.get_feature_nodes("hierarchical_summary"))
    loaded_features = set(n.id for n in loaded.get_feature_nodes("hierarchical_summary"))
    assert original_features == loaded_features
