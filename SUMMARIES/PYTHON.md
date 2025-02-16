# Python Project Structure

## tei_chunker/chunking.py
```python
@dataclass
class Section
    """
    Represents a document section with hierarchical structure.
    Args:
        title: Section title
        content: Direct content of this section (excluding subsections)
        level: Heading level (1 for main sections, 2+ for subsections)
        subsections: List of child sections
        parent: Parent section (None for top-level sections)
    """

    @property
    def full_content(self) -> str
        """Get full content including all subsections."""

    @property
    def total_length(self) -> int
        """Get total character length including all subsections."""

    def __str__(self) -> str


class HierarchicalChunker
    """
    Chunks documents while respecting their hierarchical structure.
    Args:
        max_chunk_size: Maximum size in characters for each chunk
        overlap_size: Number of characters to overlap between chunks
        min_section_size: Minimum section size to keep intact
    """

    def __init__(self, max_chunk_size: int, overlap_size: int, min_section_size: int)

    def parse_grobid_xml(self, xml_content: str) -> List[Section]
        """
        Parse GROBID XML into hierarchical sections.
        Args:
            xml_content: Raw XML string from GROBID
        Returns:
            List of top-level sections with their subsections
        """

    def _get_element_text(self, element: Any) -> str
        """Extract all text content from an element, preserving structure."""

    def _process_divs(self, element: Any, level: int) -> List[Section]
        """
        Recursively process div elements into sections.
        Args:
            element: XML element to process
            level: Current heading level
        Returns:
            List of sections from this element
        """

    def chunk_document(self, sections: List[Section]) -> List[str]
        """
        Create chunks while respecting section boundaries.
        Args:
            sections: List of document sections
        Returns:
            List of text chunks
        """

    def get_section_structure(self, sections: List[Section], indent: str) -> str
        """
        Generate a readable outline of the document structure.
        Args:
            sections: List of sections to outline
            indent: Current indentation string
        Returns:
            Formatted string showing document structure
        """


def process_section(section: Section) -> None
    """Process a single section and its subsections."""

```

## tei_chunker/core/interfaces.py
```python
class Strategy(Enum)
    """Available processing strategies."""

@dataclass
class ProcessingContext
    """Shared context for document processing."""

@dataclass
class Span
    """Represents a span of text in the document."""

@dataclass
class Feature
    """Represents a feature derived from document content."""

class ContentProcessor(Protocol)
    """Protocol for content processing functions."""

    def __call__(self, content: str) -> str


class SynthesisStrategy(Protocol)
    """Protocol for synthesis strategies."""

    def synthesize(self, content: str, features: Dict[[str, List[Feature]]], processor: ContentProcessor, context: ProcessingContext) -> str


```

## tei_chunker/core/processor.py
```python
class FeatureAwareProcessor
    """Document processor with feature awareness but clean boundaries."""

    def __init__(self, strategy: Strategy, context: ProcessingContext)

    def process_with_features(self, content: str, available_features: Dict[[str, List[Feature]]], process_fn: ContentProcessor) -> str
        """
        Process content with awareness of available features.
        Args:
            content: Document content to process
            available_features: Map of feature_type -> features
            process_fn: Function to process content chunks
        Returns:
            Processed content
        """

    def _get_strategy_impl(self, strategy: Strategy) -> SynthesisStrategy
        """Get concrete strategy implementation."""

    def _estimate_tokens(self, text: str) -> int
        """Rough token count estimation."""

    def _can_fit_in_context(self, content: str, features: Dict[[str, List[Feature]]]) -> bool
        """Check if content and features fit in context."""


```

## tei_chunker/core/strategies.py
```python
class TopDownStrategy(SynthesisStrategy)
    """Try to process maximum content at once."""

    def synthesize(self, content: str, features: Dict[[str, List[Feature]]], processor: ContentProcessor, context: ProcessingContext, depth: int) -> str

    def _can_fit_in_context(self, content: str, features: Dict[[str, List[Feature]]], context: ProcessingContext) -> bool

    def _process_all_at_once(self, content: str, features: Dict[[str, List[Feature]]], processor: ContentProcessor) -> str

    def _split_into_sections(self, content: str) -> List[Span]
        """Split content into logical sections."""

    def _is_section_header(self, text: str) -> bool
        """Identify section headers."""

    def _split_section(self, section: Span, max_tokens: int, overlap_tokens: int) -> List[Span]
        """Split a section into overlapping chunks."""

    def _get_relevant_features(self, span: Span, features: Dict[[str, List[Feature]]]) -> Dict[[str, List[Feature]]]
        """Get features relevant to a span."""

    def _chunk_content(self, content_list: List[str], max_tokens: int, overlap_tokens: int) -> List[str]
        """Split content into overlapping chunks."""


class BottomUpStrategy(SynthesisStrategy)
    """Build synthesis incrementally from leaves up."""

    def synthesize(self, content: str, features: Dict[[str, List[Feature]]], processor: ContentProcessor, context: ProcessingContext, depth: int) -> str

    def _split_hierarchical(self, content: str, max_tokens: int) -> List[Span]
        """Split content preserving hierarchical structure."""

    def _process_spans_bottom_up(self, spans: List[Span], features: Dict[[str, List[Feature]]], processor: ContentProcessor, context: ProcessingContext, depth: int) -> str

    def _split_section(self, section: Span, max_tokens: int, overlap_tokens: int) -> List[Span]
        """Split a section into smaller spans."""

    def _format_for_processing(self, span: Span, features: Dict[[str, List[Feature]]]) -> str

    def _get_relevant_features(self, span: Span, features: Dict[[str, List[Feature]]]) -> Dict[[str, List[Feature]]]
        """Get features relevant to a span."""


class HybridStrategy(SynthesisStrategy)
    """Try top-down first, fall back to bottom-up when needed."""

    def __init__(self)

    def synthesize(self, content: str, features: Dict[[str, List[Feature]]], processor: ContentProcessor, context: ProcessingContext) -> str


```

## tei_chunker/features/manager.py
```python
@dataclass
class FeatureRequest
    """Request to generate a new feature."""

class FeatureStore
    """Handles feature persistence and retrieval."""

    def __init__(self, storage_dir: Path)

    def _load_features(self) -> None
        """Load existing features from storage."""

    def save_feature(self, feature: Feature) -> None
        """Save a feature to storage."""

    def get_features(self, feature_type: str, span: Optional[Span]) -> List[Feature]
        """Get features, optionally filtered by span."""


class FeatureManager
    """Manages feature creation and orchestration."""

    def __init__(self, storage_dir: Path, xml_processor: Optional[Any])

    def process_request(self, content: str, request: FeatureRequest, llm_client: Any) -> Feature
        """Process a feature request."""

    def get_feature_chain(self, feature_type: str, span: Optional[Span]) -> List[Dict[[str, List[Feature]]]]
        """
        Get a feature and all its dependencies.
        Returns list of feature maps in dependency order.
        """

    def get_feature_graph(self, span: Optional[Span]) -> Dict[[str, Dict[[str, List[Feature]]]]]
        """
        Get graph of all features and their relationships.
        Returns map of feature_type -> (dependency_type -> features).
        """

    def validate_feature_request(self, request: FeatureRequest) -> List[str]
        """
        Validate a feature request.
        Returns list of validation errors, empty if valid.
        """

    def _check_circular_deps(self, feature_type: str, dependencies: List[str], visited: set) -> None
        """Check for circular dependencies."""


def process_content(content: str) -> str

def add_dependencies(feat_type: str)

```

## tei_chunker/features/processor.py
```python
class FeatureProcessor
    """High-level interface for feature processing."""

    def __init__(self, data_dir: Path, llm_client: Any, xml_processor: Optional[Any])

    def process_document(self, content: str, requests: List[FeatureRequest]) -> List[str]
        """
        Process multiple feature requests for a document.
        Returns list of created feature IDs.
        """

    def get_features(self, feature_type: str, span: Optional[Span]) -> List[dict]
        """
        Get features with their metadata.
        Returns list of feature dictionaries.
        """

    def get_feature_dependencies(self, feature_type: str) -> dict
        """Get dependency graph for a feature type."""

    def _sort_requests(self, requests: List[FeatureRequest]) -> List[FeatureRequest]
        """Sort requests by dependencies."""


def visit(name: str)

```

## tei_chunker/graph.py
```python
@dataclass
class Node
    """Node in the document graph."""

    def to_dict(self) -> dict

    @classmethod
    def from_dict(cls, data: dict) -> 'Node'


@dataclass
class Feature
    """Metadata about a feature type."""

    def to_dict(self) -> dict

    @classmethod
    def from_dict(cls, data: dict) -> 'Feature'


class DocumentGraph
    """Graph structure for document content and features with persistence."""

    def __init__(self, content: str)

    def generate_id(self, content: str, type: str) -> str
        """Generate deterministic node ID."""

    def add_node(self, content: str, type: str, span: tuple[[int, int]], parents: Optional[List[str]], metadata: Optional[Dict[[str, Any]]]) -> Node
        """Add a new node to the graph."""

    def register_feature(self, feature: Feature) -> None
        """Register a new feature type."""

    def get_nodes_by_type(self, type: str) -> List[Node]
        """Get all nodes of a given type."""

    def get_feature_nodes(self, feature_name: str) -> List[Node]
        """Get all nodes for a specific feature."""

    def get_overlapping_nodes(self, span: tuple[[int, int]], exclude_ids: Optional[Set[str]]) -> List[Node]
        """Find nodes that overlap with the given span."""

    def get_node_ancestors(self, node_id: str) -> List[Node]
        """Get all ancestors of a node."""

    def get_node_descendants(self, node_id: str) -> List[Node]
        """Get all descendants of a node."""

    def save(self, path: Path) -> None
        """Save the graph to disk."""

    @classmethod
    def load(cls, path: Path) -> 'DocumentGraph'
        """Load a graph from disk."""


```

## tei_chunker/service.py
```python
class ChunkingHandler(BaseHTTPRequestHandler)
    """Handles HTTP requests for document chunking."""

    def do_POST(self) -> None
        """Handle POST requests with XML content."""

    def do_GET(self) -> None
        """Handle GET requests with simple health check."""


def run_server(host: str, port: int) -> None
    """Run the chunking service."""

```

## tei_chunker/synthesis/advanced.py
```python
class SynthesisMode(Enum)

@dataclass
class FeatureDependency
    """Defines relationships between features."""

class AdvancedSynthesizer(Synthesizer)
    """Advanced synthesis strategies for complex feature relationships."""

    def __init__(self, graph)

    def register_dependency(self, dependency: FeatureDependency) -> None
        """Register a feature dependency."""

    def synthesize_with_dependencies(self, tree: SynthesisNode, target_feature: str, mode: SynthesisMode) -> None
        """Synthesize features respecting dependencies."""

    def _process_dependency(self, tree: SynthesisNode, dependency: FeatureDependency, mode: SynthesisMode) -> None
        """Process a single dependency."""

    def _aggregate_synthesis(self, tree: SynthesisNode, dependency: FeatureDependency) -> None
        """Combine multiple features into a cohesive whole."""

    def _cross_reference_synthesis(self, tree: SynthesisNode, dependency: FeatureDependency) -> None
        """Cross-reference between related features."""

    def _comparative_synthesis(self, tree: SynthesisNode, dependency: FeatureDependency) -> None
        """Compare different feature perspectives."""

    def _temporal_synthesis(self, tree: SynthesisNode, dependency: FeatureDependency) -> None
        """Time-based synthesis of features."""

    def _contextual_synthesis(self, tree: SynthesisNode, dependency: FeatureDependency) -> None
        """Context-aware feature synthesis."""


def process_node(node: SynthesisNode) -> str

```

## tei_chunker/synthesis/base.py
```python
@dataclass
class SynthesisNode
    """Node in the synthesis tree."""

    def get_feature_content(self, feature_type: str) -> List[str]
        """Get all content of a specific feature type in this subtree."""

    def get_overlapping_content(self, feature_type: str) -> List[str]
        """Get feature content from overlapping nodes."""


class Synthesizer
    """Base class for document synthesis operations."""

    def __init__(self, graph: DocumentGraph)

    def get_synthesis_tree(self, root_node: Node, feature_types: List[str], max_depth: Optional[int], visited: Optional[Set[str]]) -> SynthesisNode
        """Build synthesis tree from document graph."""

    def synthesize(self, tree: SynthesisNode, process_fn: Callable[[Any, str]], feature_name: str, version: str, bottom_up: bool) -> None
        """
        Synthesize features across a subtree.
        Args:
            tree: Synthesis tree to process
            process_fn: Function to generate synthesized content
            feature_name: Name for the synthesized feature
            version: Version string for the feature
            bottom_up: If True, process children before parents
        """

    def format_for_llm(self, node: SynthesisNode, feature_types: List[str], max_depth: Optional[int], current_depth: int, include_overlapping: bool) -> str
        """Format a synthesis node's content for LLM input."""


def process_node(node: SynthesisNode) -> None

```

## tei_chunker/synthesis/patterns.py
```python
class SynthesisStrategy(Enum)
    """Available synthesis strategies."""

class FeatureSynthesizer(Synthesizer)
    """Implementation of common synthesis patterns."""

    def __init__(self, graph)

    def hierarchical_summary(self, tree: SynthesisNode, max_length: int) -> None
        """
        Create hierarchical summary synthesis.
        Args:
            tree: Root of synthesis tree
            max_length: Maximum length for each summary
        """

    def resolve_conflicts(self, tree: SynthesisNode, feature_type: str) -> None
        """
        Resolve conflicts between overlapping features.
        Args:
            tree: Root of synthesis tree
            feature_type: Type of feature to resolve
        """

    def evidence_graded_synthesis(self, tree: SynthesisNode, feature_types: List[str], confidence_threshold: float) -> None
        """
        Create synthesis with evidence grading.
        Args:
            tree: Root of synthesis tree
            feature_types: Types of features to synthesize
            confidence_threshold: Minimum confidence threshold
        """

    def incremental_synthesis(self, tree: SynthesisNode, feature_sequence: List[str]) -> None
        """
        Build up synthesis incrementally across features.
        Args:
            tree: Root of synthesis tree
            feature_sequence: Order of features to incorporate
        """

    def _format_structure(self, node: SynthesisNode, depth: int) -> str
        """Helper to format document structure."""


def process_node(node: SynthesisNode) -> str

def process_node(node: SynthesisNode) -> str

def process_node(node: SynthesisNode) -> str

def process_node(node: SynthesisNode) -> str

```

## tei_chunker/synthesis/prompts.py
```python
@dataclass
class SynthesisPrompt
    """Template for synthesis prompts."""

    def format(self) -> str
        """Format prompt with provided values."""


class PromptTemplates
    """Collection of common synthesis prompt templates."""

    @staticmethod
    def hierarchical_summary(max_length: int) -> SynthesisPrompt
        """Template for hierarchical summary synthesis."""

    @staticmethod
    def conflict_resolution() -> SynthesisPrompt
        """Template for resolving conflicts between features."""

    @staticmethod
    def evidence_graded(confidence_threshold: float) -> SynthesisPrompt
        """Template for evidence-graded synthesis."""

    @staticmethod
    def citation_preserving() -> SynthesisPrompt
        """Template for citation-preserving synthesis."""

    @staticmethod
    def incremental_feature(feature_type: str) -> SynthesisPrompt
        """Template for incremental feature synthesis."""


```

## tei_chunker/synthesis/strategies.py
```python
class TreeStrategy(Enum)
    """Available tree synthesis strategies."""

@dataclass
class SynthesisContext
    """Context for synthesis operations."""

class TreeSynthesizer
    """Implements different tree synthesis strategies."""

    def __init__(self, strategy: TreeStrategy, context: SynthesisContext, process_fn: Callable[[Any, str]])

    def synthesize_tree(self, tree: SynthesisNode, parent_result: Optional[str]) -> str
        """
        Synthesize a tree using the selected strategy.
        Args:
            tree: Root of the synthesis tree
            parent_result: Result from parent node (for hybrid strategy)
        Returns:
            Synthesized content
        """

    def _estimate_tokens(self, content: str) -> int
        """Rough estimate of token count."""

    def _can_fit_in_context(self, tree: SynthesisNode) -> bool
        """Check if tree content fits in context window."""

    def _synthesize_top_down(self, tree: SynthesisNode) -> str
        """Try to synthesize entire tree at once, subdividing only if necessary."""

    def _synthesize_bottom_up(self, tree: SynthesisNode) -> str
        """Build synthesis from leaves up to root."""

    def _chunk_content(self, content_list: List[str], max_tokens: int, overlap_tokens: int) -> List[str]
        """Split content into overlapping chunks."""


```

## tests/conftest.py
```python
def test_data_dir(tmp_path)
    """Create a test data directory structure."""

def sample_xml_content()
    """Create sample XML content for testing."""

```

## tests/test_chunking.py
```python
def sample_xml()
    """Create a sample XML document."""

def chunker()
    """Create a chunker instance."""

def test_section_creation()
    """Test basic section object creation."""

def test_section_hierarchy()
    """Test section hierarchy handling."""

def test_parse_xml(chunker, sample_xml)
    """Test XML parsing into sections."""

def test_formula_handling(chunker, sample_xml)
    """Test handling of mathematical formulas."""

def test_chunking_small_document(chunker)
    """Test chunking of a document smaller than chunk size."""

def test_chunking_large_section(chunker)
    """Test chunking of a section larger than chunk size."""

def test_chunking_with_subsections(chunker)
    """Test chunking with hierarchical sections."""

def test_invalid_xml(chunker)
    """Test handling of invalid XML."""

def test_empty_sections(chunker)
    """Test handling of empty sections."""

```

## tests/test_feature_manager.py
```python
class MockLLMClient
    """Mock LLM client for testing."""

    def complete(self, prompt: str) -> str


def tmp_feature_dir(tmp_path)
    """Create temporary feature directory."""

def feature_store(tmp_feature_dir)
    """Create feature store with some test data."""

def feature_manager(tmp_feature_dir)
    """Create feature manager with mock LLM client."""

def sample_request()
    """Create sample feature request."""

def test_feature_store_save_load(tmp_feature_dir)
    """Test saving and loading features."""

def test_feature_store_get_by_span(feature_store)
    """Test getting features filtered by span."""

def test_feature_manager_process_request(feature_manager, sample_request)
    """Test processing a feature request."""

def test_feature_manager_validate_request(feature_manager, sample_request)
    """Test feature request validation."""

def test_feature_manager_circular_dependencies(feature_manager)
    """Test detection of circular dependencies."""

def test_feature_manager_get_feature_chain(feature_manager)
    """Test getting feature dependency chain."""

def test_feature_manager_error_handling(feature_manager, sample_request)
    """Test error handling in feature manager."""

def test_feature_persistence(feature_manager, sample_request)
    """Test feature persistence across manager instances."""

def test_feature_graph(feature_manager)
    """Test getting feature dependency graph."""

class FailingLLMClient

    def complete(self, prompt: str) -> str


```

## tests/test_processor.py
```python
class MockLLMClient
    """Mock LLM client for testing."""

    def complete(self, prompt: str) -> str


def processor(tmp_path)
    """Create feature processor with test configuration."""

def sample_requests()
    """Create sample feature requests with dependencies."""

def test_process_document(processor, sample_requests)
    """Test processing multiple feature requests for a document."""

def test_request_ordering(processor)
    """Test requests are processed in correct dependency order."""

def test_circular_dependency_detection(processor)
    """Test detection of circular dependencies in requests."""

def test_error_handling(processor, sample_requests)
    """Test error handling during processing."""

def test_feature_retrieval(processor, sample_requests)
    """Test retrieving processed features."""

def test_dependency_graph(processor, sample_requests)
    """Test getting feature dependency graph."""

def test_processing_with_existing_features(processor)
    """Test processing requests when some features already exist."""

class FailingLLMClient

    def complete(self, prompt: str) -> str


```

## tests/test_strategies.py
```python
def context()
    """Basic processing context."""

def simple_processor()
    """Simple content processor for testing."""

def sample_content()
    """Sample document content."""

def sample_features()
    """Sample features for testing."""

def test_top_down_strategy_fits_context(context: ProcessingContext, simple_processor: ContentProcessor, sample_content: str, sample_features: Dict[[str, List[Feature]]])
    """Test top-down strategy when content fits in context."""

def test_top_down_strategy_splits_content(context: ProcessingContext, simple_processor: ContentProcessor, sample_content: str, sample_features: Dict[[str, List[Feature]]])
    """Test top-down strategy splits content when needed."""

def test_bottom_up_strategy(context: ProcessingContext, simple_processor: ContentProcessor, sample_content: str, sample_features: Dict[[str, List[Feature]]])
    """Test bottom-up strategy processing."""

def test_hybrid_strategy_small_content(context: ProcessingContext, simple_processor: ContentProcessor, sample_features: Dict[[str, List[Feature]]])
    """Test hybrid strategy with small content (should use top-down)."""

def test_hybrid_strategy_large_content(context: ProcessingContext, simple_processor: ContentProcessor, sample_content: str, sample_features: Dict[[str, List[Feature]]])
    """Test hybrid strategy with large content (should fall back to bottom-up)."""

def test_strategy_with_overlapping_features(context: ProcessingContext, simple_processor: ContentProcessor, sample_content: str)
    """Test handling of overlapping features."""

def test_strategy_respects_min_chunk_size(simple_processor: ContentProcessor, sample_content: str, sample_features: Dict[[str, List[Feature]]])
    """Test that strategies respect minimum chunk size."""

def test_strategy_handles_empty_features(context: ProcessingContext, simple_processor: ContentProcessor, sample_content: str)
    """Test strategies work with no features."""

def test_strategy_error_handling(context: ProcessingContext, sample_content: str, sample_features: Dict[[str, List[Feature]]])
    """Test error handling in strategies."""

def process(content: str) -> str

def failing_processor(content: str) -> str

```

## tests/test_synthesis.py
```python
def sample_graph()
    """Create a sample document graph."""

def test_synthesis_tree_creation(sample_graph)
    """Test creation of synthesis tree."""

def test_hierarchical_summary(sample_graph)
    """Test hierarchical summary synthesis."""

def test_conflict_resolution(sample_graph)
    """Test conflict resolution between overlapping features."""

def test_evidence_graded_synthesis(sample_graph)
    """Test evidence-graded synthesis."""

def test_incremental_synthesis(sample_graph)
    """Test incremental feature synthesis."""

def test_graph_persistence(tmp_path, sample_graph)
    """Test saving and loading graph with syntheses."""

```
