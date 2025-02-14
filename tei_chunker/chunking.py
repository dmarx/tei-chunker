# tei_chunker/chunking.py
"""
Hierarchical document chunking based on XML structure.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict
import xml.etree.ElementTree as ET
from loguru import logger


# Define TEI namespace
NS = {"tei": "http://www.tei-c.org/ns/1.0"}


@dataclass
class Section:
    """
    Represents a document section with hierarchical structure.

    Args:
        title: Section title
        content: Direct content of this section (excluding subsections)
        level: Heading level (1 for main sections, 2+ for subsections)
        subsections: List of child sections
        parent: Parent section (None for top-level sections)
    """

    title: str
    content: str
    level: int
    subsections: List["Section"]
    parent: Optional["Section"] = None

    @property
    def full_content(self) -> str:
        """Get full content including all subsections."""
        result = [f"{'#' * self.level} {self.title}\n\n{self.content}"]
        for subsection in self.subsections:
            result.append(subsection.full_content)
        return "\n\n".join(result)

    @property
    def total_length(self) -> int:
        """Get total character length including all subsections."""
        return len(self.full_content)

    def __str__(self) -> str:
        return f"{self.title} ({self.total_length} chars, {len(self.subsections)} subsections)"


class HierarchicalChunker:
    """
    Chunks documents while respecting their hierarchical structure.
    Args:
        max_chunk_size: Maximum size in characters for each chunk
        overlap_size: Number of characters to overlap between chunks
        min_section_size: Minimum section size to keep intact
    """

    def __init__(
        self, max_chunk_size: int, overlap_size: int = 200, min_section_size: int = 1000
    ):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.min_section_size = min_section_size

    def parse_grobid_xml(self, xml_content: str) -> List[Section]:
        """
        Parse GROBID XML into hierarchical sections.

        Args:
            xml_content: Raw XML string from GROBID
        Returns:
            List of top-level sections with their subsections
        """
        try:
            root = ET.fromstring(xml_content)
            sections = []

            # Process abstract if present
            abstract = root.find(".//tei:abstract", NS)
            if abstract is not None:
                abstract_text = self._get_element_text(abstract)
                if abstract_text:
                    sections.append(
                        Section(
                            title="Abstract",
                            content=abstract_text,
                            level=1,
                            subsections=[],
                        )
                    )

            # Process main body
            body = root.find(".//tei:body", NS)
            if body is not None:
                sections.extend(self._process_divs(body))

            return sections

        except ET.ParseError as e:
            logger.error(f"Failed to parse XML: {e}")
            return []

    def _get_element_text(self, element: ET.Element) -> str:
        """Extract all text content from an element, preserving structure."""
        if element is None:
            return ""

        parts = []

        # Handle direct text
        if element.text and element.text.strip():
            parts.append(element.text.strip())

        # Process child elements
        for child in element:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            # Special handling for formulas
            if tag == 'formula':
                formula_text = child.text.strip() if child.text else ""
                parts.append(f"$${formula_text}$$")
            # Handle references
            elif tag == 'ref':
                ref_text = child.text.strip() if child.text else ""
                parts.append(f"[{ref_text}]")
            # Regular text content
            else:
                child_text = self._get_element_text(child)
                if child_text:
                    parts.append(child_text)

            # Handle tail text
            if child.tail and child.tail.strip():
                parts.append(child.tail.strip())

        return " ".join(parts)

    def _process_divs(self, element: ET.Element, level: int = 1) -> List[Section]:
        """
        Recursively process div elements into sections.

        Args:
            element: XML element to process
            level: Current heading level
        Returns:
            List of sections from this element
        """
        sections = []

        for div in element.findall(".//tei:div", NS):
            # Get section heading
            head = div.find("./tei:head", NS)
            title = head.text if head is not None and head.text else "Untitled Section"

            # Get immediate paragraph content
            paragraphs = []
            for p in div.findall("./tei:p", NS):
                text = self._get_element_text(p)
                if text:
                    paragraphs.append(text)

            # Create section
            section = Section(
                title=title,
                content="\n\n".join(paragraphs),
                level=level,
                subsections=[],
            )

            # Process subsections
            subsections = self._process_divs(div, level + 1)
            section.subsections = subsections
            for subsection in subsections:
                subsection.parent = section

            sections.append(section)

        return sections

    def chunk_document(self, sections: List[Section]) -> List[str]:
        """
        Create chunks while respecting section boundaries.

        Args:
            sections: List of document sections
        Returns:
            List of text chunks
        """
        if not sections:
            return []

        chunks = []
        current_chunk = []
        current_size = 0

        def process_section(section: Section):
            """Process a single section and its subsections."""
            nonlocal current_chunk, current_size

            section_content = section.full_content
            section_size = len(section_content)

            # Check if section should be split
            if current_size + section_size > self.max_chunk_size:
                # If section itself is too large
                if section_size > self.max_chunk_size:
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk = []
                        current_size = 0

                    # Add section content if substantial
                    if len(section.content) > self.min_section_size:
                        section_text = (
                            f"{'#' * section.level} {section.title}\n\n"
                            f"{section.content}"
                        )
                        current_chunk.append(section_text)
                        current_size = len(section_text)

                        if current_size >= self.max_chunk_size:
                            chunks.append("\n\n".join(current_chunk))
                            current_chunk = []
                            current_size = 0

                    # Process subsections
                    for subsection in section.subsections:
                        process_section(subsection)

                else:
                    # Create new chunk with overlap
                    if current_chunk:
                        # Find appropriate overlap point
                        chunk_text = "\n\n".join(current_chunk)
                        last_period = chunk_text.rfind(". ", -self.overlap_size)
                        if last_period > 0:
                            overlap_text = chunk_text[last_period + 2 :]
                        else:
                            overlap_text = chunk_text[-self.overlap_size :]

                        chunks.append(chunk_text)
                        current_chunk = [overlap_text]
                        current_size = len(overlap_text)

                    current_chunk.append(section_content)
                    current_size += section_size
            else:
                # Add to current chunk
                current_chunk.append(section_content)
                current_size += section_size

        # Process all top-level sections
        for section in sections:
            process_section(section)

        # Add final chunk if exists and not already added
        if current_chunk:
            final_chunk = "\n\n".join(current_chunk)
            if not chunks or chunks[-1] != final_chunk:
                chunks.append(final_chunk)

        return chunks

    def get_section_structure(self, sections: List[Section], indent: str = "") -> str:
        """
        Generate a readable outline of the document structure.

        Args:
            sections: List of sections to outline
            indent: Current indentation string
        Returns:
            Formatted string showing document structure
        """
        result = []
        for section in sections:
            result.append(f"{indent}{str(section)}")
            if section.subsections:
                result.append(
                    self.get_section_structure(section.subsections, indent + "  ")
                )
        return "\n".join(result)
