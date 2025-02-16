# tei_chunker/core/processor.py
"""
Core document processing functionality.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger

from .interfaces import (
    Strategy,
    ProcessingContext,
    Feature,
    Span,
    ContentProcessor,
    SynthesisStrategy
)

.strategies import TopDownStrategy, BottomUpStrategy, HybridStrategy

class FeatureAwareProcessor:
    """
    Document processor with feature awareness but clean boundaries.
    """
    def __init__(
        self,
        strategy: Strategy,
        context: ProcessingContext
    ):
        self.strategy = strategy
        self.context = context
        
    def process_with_features(
        self,
        content: str,
        available_features: Dict[str, List[Feature]],
        process_fn: ContentProcessor
    ) -> str:
        """
        Process content with awareness of available features.
        
        Args:
            content: Document content to process
            available_features: Map of feature_type -> features
            process_fn: Function to process content chunks
        Returns:
            Processed content
        """
        strategy_impl = self._get_strategy_impl(self.strategy)
        return strategy_impl.synthesize(
            content,
            available_features,
            process_fn,
            self.context
        )
        
    def _get_strategy_impl(self, strategy: Strategy) -> SynthesisStrategy:
        """Get concrete strategy implementation."""
        if strategy == Strategy.TOP_DOWN_MAXIMAL:
            return TopDownStrategy()
        elif strategy == Strategy.BOTTOM_UP:
            return BottomUpStrategy()
        else:
            return HybridStrategy()
            
    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimation."""
        return len(text.split())
        
    def _can_fit_in_context(
        self,
        content: str,
        features: Dict[str, List[Feature]]
    ) -> bool:
        """Check if content and features fit in context."""
        total_tokens = self._estimate_tokens(content)
        
        # Add tokens from features
        for feature_list in features.values():
            for feature in feature_list:
                total_tokens += self._estimate_tokens(feature.content)
                
        return total_tokens <= self.context.max_tokens
