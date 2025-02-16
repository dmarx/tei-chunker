# tei_chunker/features/manager.py
"""
Feature management and processing.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from loguru import logger

from ..core.interfaces import (
    Strategy,
    ProcessingContext,
    Feature,
    Span,
    ContentProcessor
)
from ..core.processor import FeatureAwareProcessor

@dataclass
class FeatureRequest:
    """Request to generate a new feature."""
    name: str
    prompt_template: str
    strategy: Strategy = Strategy.TOP_DOWN_MAXIMAL
    context_size: int = 8000
    overlap_size: int = 200
    min_chunk_size: int = 500
    required_features: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class FeatureStore:
    """Handles feature persistence and retrieval."""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.features: Dict[str, List[Feature]] = {}
        self._load_features()
        
    def _load_features(self) -> None:
        """Load existing features from storage."""
        if not self.storage_dir.exists():
            return
            
        for feature_dir in self.storage_dir.iterdir():
            if not feature_dir.is_dir():
                continue
                
            feature_type = feature_dir.name
            self.features[feature_type] = []
            
            for content_file in feature_dir.glob("*.md"):
                meta_file = content_file.with_suffix(".json")
                if not meta_file.exists():
                    continue
                    
                try:
                    content = content_file.read_text()
                    metadata = json.loads(meta_file.read_text())
                    
                    feature = Feature(
                        name=feature_type,
                        content=content,
                        span=Span(
                            start=metadata['span']['start'],
                            end=metadata['span']['end'],
                            text=content
                        ),
                        metadata=metadata['metadata']
                    )
                    self.features[feature_type].append(feature)
                except Exception as e:
                    logger.error(f"Error loading feature {content_file}: {e}")
                    
    def save_feature(self, feature: Feature) -> None:
        """Save a feature to storage."""
        feature_dir = self.storage_dir / feature.name
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save content
        content_path = feature_dir / f"{timestamp}.md"
        content_path.write_text(feature.content)
        
        # Save metadata
        meta_path = feature_dir / f"{timestamp}.json"
        meta_path.write_text(json.dumps({
            'span': {
                'start': feature.span.start,
                'end': feature.span.end
            },
            'metadata': feature.metadata
        }, indent=2))
        
    def get_features(
        self,
        feature_type: str,
        span: Optional[Span] = None
    ) -> List[Feature]:
        """Get features, optionally filtered by span."""
        features = self.features.get(feature_type, [])
        if span is None:
            return features
            
        return [
            f for f in features
            if f.span.start < span.end and f.span.end > span.start
        ]

class FeatureManager:
    """
    Manages feature creation and orchestration.
    """
    def __init__(
        self,
        storage_dir: Path,
        xml_processor: Optional[Any] = None  # Your XML processor type
    ):
        self.store = FeatureStore(storage_dir)
        self.xml_processor = xml_processor
        
    def process_request(
        self,
        content: str,
        request: FeatureRequest,
        llm_client: Any  # Your LLM client type
    ) -> Feature:
        """Process a feature request."""
        # Setup context
        context = ProcessingContext(
            max_tokens=request.context_size,
            overlap_tokens=request.overlap_size,
            min_chunk_tokens=request.min_chunk_size
        )
        
        # Create processor function
        def process_content(content: str) -> str:
            return llm_client.complete(
                request.prompt_template.format(content=content)
            )
        
        # Get required features
        available_features = {
            name: self.store.get_features(name)
            for name in request.required_features
        }
        
        # Create processor
        processor = FeatureAwareProcessor(
            strategy=request.strategy,
            context=context
        )
        
        try:
            # Process content with features
            result = processor.process_with_features(
                content,
                available_features,
                process_content
            )
            
            # Create feature
            feature = Feature(
                name=request.name,
                content=result,
                span=Span(0, len(content), content),
                metadata={
                    'created_at': datetime.utcnow().isoformat(),
                    'strategy': request.strategy.value,
                    'required_features': request.required_features,
                    'user_metadata': request.metadata,
                    'context': {
                        'max_tokens': context.max_tokens,
                        'overlap_tokens': context.overlap_tokens,
                        'min_chunk_tokens': context.min_chunk_tokens
                    }
                }
            )
            
            # Store feature
            self.store.save_feature(feature)
            
            return feature
            
        except Exception as e:
            logger.error(f"Error processing feature request: {e}")
            raise
            
    def get_feature_chain(
        self,
        feature_type: str,
        span: Optional[Span] = None
    ) -> List[Dict[str, List[Feature]]]:
        """
        Get a feature and all its dependencies.
        Returns list of feature maps in dependency order.
        """
        chain = []
        visited = set()
        
        def add_dependencies(feat_type: str):
            if feat_type in visited:
                return
                
            # Get features
            features = self.store.get_features(feat_type, span)
            if not features:
                return
                
            # Get required features
            required = set()
            for feat in features:
                required.update(
                    feat.metadata.get('required_features', [])
                )
                
            # Add dependencies first
            for dep in required:
                add_dependencies(dep)
                
            # Add this feature type
            chain.append({feat_type: features})
            visited.add(feat_type)
            
        add_dependencies(feature_type)
        return chain
        
    def get_feature_graph(
        self,
        span: Optional[Span] = None
    ) -> Dict[str, Dict[str, List[Feature]]]:
        """
        Get graph of all features and their relationships.
        Returns map of feature_type -> (dependency_type -> features).
        """
        graph = {}
        
        # Get all feature types
        feature_types = set(self.store.features.keys())
        
        for feat_type in feature_types:
            features = self.store.get_features(feat_type, span)
            if not features:
                continue
                
            # Get required features
            required = set()
            for feat in features:
                required.update(
                    feat.metadata.get('required_features', [])
                )
                
            # Build dependency map
            deps = {}
            for dep in required:
                dep_features = self.store.get_features(dep, span)
                if dep_features:
                    deps[dep] = dep_features
                    
            graph[feat_type] = deps
            
        return graph
        
    def validate_feature_request(
        self,
        request: FeatureRequest
    ) -> List[str]:
        """
        Validate a feature request.
        Returns list of validation errors, empty if valid.
        """
        errors = []
        
        # Check required features exist
        for feat_type in request.required_features:
            if not self.store.features.get(feat_type):
                errors.append(
                    f"Required feature type '{feat_type}' does not exist"
                )
                
        # Check for circular dependencies
        try:
            self._check_circular_deps(
                request.name,
                request.required_features,
                set()
            )
        except ValueError as e:
            errors.append(str(e))
            
        return errors
        
    def _check_circular_deps(
        self,
        feature_type: str,
        dependencies: List[str],
        visited: set
    ) -> None:
        """Check for circular dependencies."""
        if feature_type in visited:
            path = " -> ".join(visited | {feature_type})
            raise ValueError(
                f"Circular dependency detected: {path}"
            )
            
        visited.add(feature_type)
        
        for dep in dependencies:
            # Get features of this type
            features = self.store.get_features(dep)
            if not features:
                continue
                
            # Get their dependencies
            for feat in features:
                required = feat.metadata.get('required_features', [])
                self._check_circular_deps(dep, required, visited.copy())
