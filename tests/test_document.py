# tests/test_document.py
"""
Tests for paper document handling.
"""
import pytest
from pathlib import Path
from scripts.document import Paper

@pytest.fixture
def test_data_dir(tmp_path):
    """Create a test data directory with sample papers."""
    data_dir = tmp_path / "papers"
    data_dir.mkdir()
    
    # Create a paper with some features
    paper_dir = data_dir / "2101.00123"
    paper_dir.mkdir()
    
    features_dir = paper_dir / "features"
    features_dir.mkdir()
    
    # Add some features
    markdown_dir = features_dir / "markdown-grobid"
    markdown_dir.mkdir()
    (markdown_dir / "2101.00123.md").write_text("Test content")
    
    abstract_dir = features_dir / "abstract"
    abstract_dir.mkdir()
    (abstract_dir / "2101.00123.json").write_text("{}")
    
    return data_dir

def test_paper_initialization(test_data_dir):
    """Test basic paper object initialization."""
    paper = Paper("2101.00123", data_dir=test_data_dir)
    assert paper.arxiv_id == "2101.00123"
    assert paper.paper_dir == test_data_dir / "2101.00123"
    assert paper.features_dir == test_data_dir / "2101.00123" / "features"

def test_available_features(test_data_dir):
    """Test detection of available features."""
    paper = Paper("2101.00123", data_dir=test_data_dir)
    features = paper.available_features
    assert "markdown-grobid" in features
    assert "abstract" in features
    assert "nonexistent" not in features

def test_has_feature(test_data_dir):
    """Test feature availability checking."""
    paper = Paper("2101.00123", data_dir=test_data_dir)
    assert paper.has_feature("markdown-grobid")
    assert not paper.has_feature("nonexistent")

def test_feature_path(test_data_dir):
    """Test feature path resolution."""
    paper = Paper("2101.00123", data_dir=test_data_dir)
    
    # Test existing feature
    path = paper.feature_path("markdown-grobid")
    assert path is not None
    assert path.name == "2101.00123.md"
    assert path.is_file()
    
    # Test nonexistent feature
    path = paper.feature_path("nonexistent")
    assert path is None

def test_paper_iteration(test_data_dir):
    """Test iteration over papers in directory."""
    # Create another paper
    paper2_dir = test_data_dir / "2101.00456"
    paper2_dir.mkdir()
    (paper2_dir / "features").mkdir()
    
    papers = list(Paper.iter_papers(test_data_dir))
    assert len(papers) == 2
    paper_ids = {p.arxiv_id for p in papers}
    assert paper_ids == {"2101.00123", "2101.00456"}

def test_empty_directory(tmp_path):
    """Test handling of empty data directory."""
    papers = list(Paper.iter_papers(tmp_path))
    assert len(papers) == 0

def test_nonexistent_directory(tmp_path):
    """Test handling of nonexistent data directory."""
    nonexistent = tmp_path / "nonexistent"
    papers = list(Paper.iter_papers(nonexistent))
    assert len(papers) == 0
