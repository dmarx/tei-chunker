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
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            # Special handling for formulas
            if tag == "formula":
                formula_text = child.text.strip() if child.text else ""
                parts.append(f"$${formula_text}$$")
            # Handle references
            elif tag == "ref":
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

        for div in element.findall("./tei:div", NS):
            # Get section heading
            head = div.find("./tei:head", NS)
            title = head.text if head is not None and head.text else "Untitled Section"

            # Gather content from child elements in order (p and formula)
            content_parts = []
            for child in div:
                tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if tag in ["p", "formula"]:
                    text = self._get_element_text(child)
                    if text:
                        content_parts.append(text)
                # Skip head and nested div (handled later)
                # You might also want to handle other tags here if needed.

            # Create section
            section = Section(
                title=title,
                content="\n\n".join(content_parts),
                level=level,
                subsections=[],
            )

            # Process subsections (nested divs)
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
            nonlocal current_chunk, current_size, chunks

            section_content = section.full_content
            section_size = len(section_content)

            # If current section is too large to fit in a single chunk
            if section_size > self.max_chunk_size:
                # First, add any existing content as a chunk
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Split section content into chunks
                words = section_content.split()
                current_words = []
                current_word_size = 0

                for word in words:
                    word_size = len(word) + 1  # +1 for space
                    if current_word_size + word_size > self.max_chunk_size:
                        if current_words:  # Create chunk from accumulated words
                            chunk_text = " ".join(current_words)
                            chunks.append(chunk_text)
                            # Keep some overlap
                            overlap_words = current_words[
                                -self.overlap_size // 10 :
                            ]  # Approximate words for overlap
                            current_words = overlap_words + [word]
                            current_word_size = sum(len(w) + 1 for w in current_words)
                    else:
                        current_words.append(word)
                        current_word_size += word_size

                # Add remaining words if any
                if current_words:
                    chunks.append(" ".join(current_words))

            # If section can fit in current chunk with room
            elif current_size + section_size <= self.max_chunk_size:
                current_chunk.append(section_content)
                current_size += section_size

            # If section needs a new chunk
            else:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [section_content]
                current_size = section_size

            # Process subsections
            for subsection in section.subsections:
                process_section(subsection)

        # Process all top-level sections
        for section in sections:
            process_section(section)

        # Add final chunk if it exists
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return [chunk for chunk in chunks if chunk.strip()]

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
