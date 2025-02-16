# tei_chunker/features/processor.py
"""
Main entry point for feature processing.
"""
from pathlib import Path
from typing import Optional, List, Any
from loguru import logger

from .manager import FeatureManager, FeatureRequest
from ..core.interfaces import Strategy, Span

class FeatureProcessor:
    """
    High-level interface for feature processing.
    """
    def __init__(
        self,
        data_dir: Path,
        llm_client: Any,  # Your LLM client type
        xml_processor: Optional[Any] = None  # Your XML processor type
    ):
        self.data_dir = Path(data_dir)
        self.llm_client = llm_client
        self.feature_manager = FeatureManager(
            self.data_dir / "features",
            xml_processor
        )
        
    def process_document(
        self,
        content: str,
        requests: List[FeatureRequest]
    ) -> List[str]:
        """
        Process multiple feature requests for a document.
        Returns list of created feature IDs.
        """
        feature_ids = []
        
        # Sort requests by dependencies
        sorted_requests = self._sort_requests(requests)
        
        # Process each request
        for request in sorted_requests:
            # Validate request
            errors = self.feature_manager.validate_feature_request(request)
            if errors:
                logger.error(
                    f"Invalid feature request '{request.name}': {errors}"
                )
                continue
                
            try:
                # Process request
                feature = self.feature_manager.process_request(
                    content,
                    request,
                    self.llm_client
                )
                feature_ids.append(feature.name)
                
            except Exception as e:
                logger.error(
                    f"Error processing feature '{request.name}': {e}"
                )
                
        return feature_ids
        
    def get_features(
        self,
        feature_type: str,
        span: Optional[Span] = None
    ) -> List[dict]:
        """
        Get features with their metadata.
        Returns list of feature dictionaries.
        """
        features = self.feature_manager.store.get_features(
            feature_type,
            span
        )
        
        return [
            {
                'id': feat.name,
                'content': feat.content,
                'span': {
                    'start': feat.span.start,
                    'end': feat.span.end
                },
                'metadata': feat.metadata
            }
            for feat in features
        ]
        
    def get_feature_dependencies(
        self,
        feature_type: str
    ) -> dict:
        """Get dependency graph for a feature type."""
        return self.feature_manager.get_feature_graph()
        
    def _sort_requests(
        self,
        requests: List[FeatureRequest]
    ) -> List[FeatureRequest]:
        """Sort requests by dependencies."""
        graph = {}
        for req in requests:
            graph[req.name] = set(req.required_features)
            
        # Find order using topological sort
        visited = set()
        temp = set()
        order = []
        
        def visit(name: str):
            if name in temp:
                raise ValueError(f"Circular dependency involving {name}")
            if name in visited:
                return
                
            temp.add(name)
            
            # Visit dependencies
            for dep in graph.get(name, set()):
                visit(dep)
                
            temp.remove(name)
            visited.add(name)
            order.append(name)
            
        for req in requests:
            if req.name not in visited:
                visit(req.name)
                
        # Map back to requests
        name_to_request = {r.name: r for r in requests}
        return [name_to_request[name] for name in order]
