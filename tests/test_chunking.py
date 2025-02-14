"""
Tests for hierarchical document chunking.
"""

import pytest
from tei_chunker.chunking import HierarchicalChunker, Section


@pytest.fixture
def sample_xml():
    """Create a sample XML document."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <teiHeader>
        <fileDesc>
            <titleStmt>
                <title level="a" type="main">Test Paper</title>
            </titleStmt>
        </fileDesc>
    </teiHeader>
    <text>
        <body>
            <div xmlns="http://www.tei-c.org/ns/1.0">
                <head>Introduction</head>
                <p>This is an introduction paragraph.</p>
                <p>This is another paragraph.</p>
                <div xmlns="http://www.tei-c.org/ns/1.0">
                    <head>Background</head>
                    <p>Some background information.</p>
                    <formula>E = mc^2</formula>
                </div>
            </div>
            <div xmlns="http://www.tei-c.org/ns/1.0">
                <head>Methods</head>
                <p>Our methodology is described here.</p>
                <div xmlns="http://www.tei-c.org/ns/1.0">
                    <head>Data Collection</head>
                    <p>We collected data as follows.</p>
                </div>
                <div xmlns="http://www.tei-c.org/ns/1.0">
                    <head>Analysis</head>
                    <p>Analysis was performed using...</p>
                </div>
            </div>
        </body>
    </text>
</TEI>"""


@pytest.fixture
def chunker():
    """Create a chunker instance."""
    return HierarchicalChunker(max_chunk_size=500, overlap_size=50)


def test_section_creation():
    """Test basic section object creation."""
    section = Section(title="Test", content="Content", level=1, subsections=[])
    assert section.title == "Test"
    assert section.content == "Content"
    assert section.level == 1
    assert len(section.subsections) == 0


def test_section_hierarchy():
    """Test section hierarchy handling."""
    subsection = Section(
        title="Subsection", content="Sub content", level=2, subsections=[]
    )
    section = Section(
        title="Main", content="Main content", level=1, subsections=[subsection]
    )
    subsection.parent = section

    assert section.subsections[0] == subsection
    assert subsection.parent == section
    assert "Main" in section.full_content
    assert "Sub content" in section.full_content


def test_parse_xml(chunker, sample_xml):
    """Test XML parsing into sections."""
    sections = chunker.parse_grobid_xml(sample_xml)

    # Check top-level sections
    assert len(sections) >= 2  # Introduction and Methods

    # Check Introduction section
    intro = next((s for s in sections if s.title == "Introduction"), None)
    assert intro is not None
    assert "introduction paragraph" in intro.content


def test_formula_handling(chunker, sample_xml):
    """Test handling of mathematical formulas."""
    sections = chunker.parse_grobid_xml(sample_xml)
    # Find the Background section which contains the formula
    intro = next((s for s in sections if s.title == "Introduction"), None)
    assert intro is not None
    assert len(intro.subsections) > 0
    background = intro.subsections[0]
    assert "E = mc^2" in background.content


def test_chunking_small_document(chunker):
    """Test chunking of a document smaller than chunk size."""
    sections = [
        Section(
            title="Small Section",
            content="This is a small section.",
            level=1,
            subsections=[],
        )
    ]
    chunks = chunker.chunk_document(sections)
    assert len(chunks) >= 1
    assert "Small Section" in chunks[0]


def test_chunking_large_section(chunker):
    """Test chunking of a section larger than chunk size."""
    chunker.max_chunk_size = 100  # Set a very small chunk size
    large_content = "word " * 200  # ~1000 characters
    sections = [
        Section(title="Large Section", content=large_content, level=1, subsections=[])
    ]
    chunks = chunker.chunk_document(sections)
    assert len(chunks) > 1
    assert any("Large Section" in chunk for chunk in chunks)


def test_chunking_with_subsections(chunker):
    """Test chunking with hierarchical sections."""
    sections = [
        Section(
            title="Main",
            content="Main content",
            level=1,
            subsections=[
                Section(title="Sub A", content="A content", level=2, subsections=[]),
                Section(title="Sub B", content="B content", level=2, subsections=[]),
            ],
        )
    ]
    chunks = chunker.chunk_document(sections)
    assert any("Main content" in chunk for chunk in chunks)
    assert any("Sub A" in chunk for chunk in chunks)
    assert any("Sub B" in chunk for chunk in chunks)


def test_invalid_xml(chunker):
    """Test handling of invalid XML."""
    invalid_xml = "<invalid>xml"
    sections = chunker.parse_grobid_xml(invalid_xml)
    assert len(sections) == 0


def test_empty_sections(chunker):
    """Test handling of empty sections."""
    empty_sections = []
    chunks = chunker.chunk_document(empty_sections)
    assert len(chunks) == 0
