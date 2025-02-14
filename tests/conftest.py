# tests/conftest.py
"""
Shared test fixtures.
"""
import pytest
from pathlib import Path


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a test data directory structure."""
    data_dir = tmp_path / "papers"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def test_paper(test_data_dir):
    """Create a test paper with basic structure."""
    paper_dir = test_data_dir / "2101.00123"
    paper_dir.mkdir()

    features_dir = paper_dir / "features"
    features_dir.mkdir()

    return Paper("2101.00123", data_dir=test_data_dir)


@pytest.fixture
def sample_xml_content():
    """Create sample XML content for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <teiHeader>
        <fileDesc>
            <titleStmt>
                <title>Test Paper</title>
            </titleStmt>
        </fileDesc>
    </teiHeader>
    <text>
        <body>
            <div>
                <head>Introduction</head>
                <p>Test introduction content.</p>
                <formula>E = mc^2</formula>
            </div>
            <div>
                <head>Methods</head>
                <p>Test methods content.</p>
            </div>
        </body>
    </text>
</TEI>"""
