# tei_chunker/synthesis/strategies.py
"""
Implementation of tree synthesis strategies.
File: tei_chunker/synthesis/strategies.py
"""
from enum import Enum
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from loguru import logger

from .base import SynthesisNode

class TreeStrategy(Enum):
    """Available tree synthesis strategies."""
    TOP_DOWN_MAXIMAL = "top_down_maximal"  # Try root first, subdivide if needed
    BOTTOM_UP = "bottom_up"                # Build up from leaves
    HYBRID = "hybrid"                      # Try top-down, fall back to bottom-up if needed

@dataclass
class SynthesisContext:
    """Context for synthesis operations."""
    max_tokens: int
    feature_types: List[str]
    overlap_tokens: int = 100
    min_chunk_tokens: int = 500  # Don't subdivide below this size

class TreeSynthesizer:
    """
    Implements different tree synthesis strategies.
    """
    def __init__(
        self,
        strategy: TreeStrategy,
        context: SynthesisContext,
        process_fn: Callable[[str], str]
    ):
        self.strategy = strategy
        self.context = context
        self.process_fn = process_fn
        
    def synthesize_tree(
        self,
        tree: SynthesisNode,
        parent_result: Optional[str] = None
    ) -> str:
        """
        Synthesize a tree using the selected strategy.
        
        Args:
            tree: Root of the synthesis tree
            parent_result: Result from parent node (for hybrid strategy)
        Returns:
            Synthesized content
        """
        if self.strategy == TreeStrategy.TOP_DOWN_MAXIMAL:
            return self._synthesize_top_down(tree)
        elif self.strategy == TreeStrategy.BOTTOM_UP:
            return self._synthesize_bottom_up(tree)
        else:  # HYBRID
            try:
                return self._synthesize_top_down(tree)
            except ValueError as e:
                logger.info(f"Falling back to bottom-up strategy: {e}")
                return self._synthesize_bottom_up(tree)
                
    def _estimate_tokens(self, content: str) -> int:
        """Rough estimate of token count."""
        return len(content.split())
        
    def _can_fit_in_context(self, tree: SynthesisNode) -> bool:
        """Check if tree content fits in context window."""
        total_tokens = 0
        
        # Get all feature content
        for feat_type in self.context.feature_types:
            content = tree.get_feature_content(feat_type)
            total_tokens += sum(self._estimate_tokens(c) for c in content)
            
        # Check overlapping content
        for overlap in tree.overlapping:
            for feat_type in self.context.feature_types:
                content = overlap.get_feature_content(feat_type)
                total_tokens += sum(self._estimate_tokens(c) for c in content)
                
        return total_tokens <= self.context.max_tokens
        
    def _synthesize_top_down(self, tree: SynthesisNode) -> str:
        """
        Try to synthesize entire tree at once, subdividing only if necessary.
        """
        # Check if we can process the entire tree
        if self._can_fit_in_context(tree):
            # Collect all content
            all_content = []
            for feat_type in self.context.feature_types:
                content = tree.get_feature_content(feat_type)
                if content:
                    all_content.extend(content)
                    
            # Include relevant overlapping content
            for overlap in tree.overlapping:
                for feat_type in self.context.feature_types:
                    content = overlap.get_feature_content(feat_type)
                    if content:
                        all_content.extend(content)
                        
            # Process everything at once
            return self.process_fn("\n\n".join(all_content))
            
        # If tree is too large, check if it's subdividable
        if not tree.children or self._estimate_tokens(tree.content) <= self.context.min_chunk_tokens:
            raise ValueError(
                f"Content size ({self._estimate_tokens(tree.content)} tokens) "
                f"exceeds context window ({self.context.max_tokens} tokens) "
                "and cannot be subdivided further"
            )
            
        # Process children separately
        child_results = []
        for child in tree.children:
            result = self._synthesize_top_down(child)
            child_results.append(result)
            
        # Combine child results
        combined = "\n\n".join(child_results)
        if self._estimate_tokens(combined) <= self.context.max_tokens:
            return self.process_fn(combined)
        else:
            # Need to synthesize child results in chunks
            chunks = self._chunk_content(
                child_results,
                self.context.max_tokens,
                self.context.overlap_tokens
            )
            chunk_results = [self.process_fn(chunk) for chunk in chunks]
            return self.process_fn("\n\n".join(chunk_results))
            
    def _synthesize_bottom_up(self, tree: SynthesisNode) -> str:
        """
        Build synthesis from leaves up to root.
        """
        # Process leaves first
        if tree.children:
            child_results = []
            for child in tree.children:
                result = self._synthesize_bottom_up(child)
                child_results.append(result)
                
            # Combine child results with current node
            all_content = [tree.content] + child_results
            
            # Check if we need to chunk
            if sum(self._estimate_tokens(c) for c in all_content) <= self.context.max_tokens:
                return self.process_fn("\n\n".join(all_content))
            else:
                # Process in chunks
                chunks = self._chunk_content(
                    all_content,
                    self.context.max_tokens,
                    self.context.overlap_tokens
                )
                chunk_results = [self.process_fn(chunk) for chunk in chunks]
                return self.process_fn("\n\n".join(chunk_results))
        else:
            # Leaf node - process directly
            return self.process_fn(tree.content)
            
    def _chunk_content(
        self,
        content_list: List[str],
        max_tokens: int,
        overlap_tokens: int
    ) -> List[str]:
        """Split content into overlapping chunks."""
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for content in content_list:
            content_tokens = self._estimate_tokens(content)
            
            if current_tokens + content_tokens <= max_tokens:
                current_chunk.append(content)
                current_tokens += content_tokens
            else:
                # Add current chunk if it exists
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    
                # Start new chunk with overlap
                if overlap_tokens > 0:
                    # Find overlap content from previous chunk
                    overlap_content = []
                    overlap_size = 0
                    for c in reversed(current_chunk):
                        size = self._estimate_tokens(c)
                        if overlap_size + size <= overlap_tokens:
                            overlap_content.insert(0, c)
                            overlap_size += size
                        else:
                            break
                    current_chunk = overlap_content
                else:
                    current_chunk = []
                    
                current_chunk.append(content)
                current_tokens = sum(self._estimate_tokens(c) for c in current_chunk)
                
        # Add final chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        return chunks
