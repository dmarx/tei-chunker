# tei_chunker/synthesis/base.py
"""
Base classes for document synthesis.
File: tei_chunker/synthesis/base.py
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from pathlib import Path
from loguru import logger

from ..graph import DocumentGraph, Node, Feature

@dataclass
class SynthesisNode:
    """Node in the synthesis tree."""
    node_id: str
    feature_type: str
    content: str
    children: List['SynthesisNode']
    overlapping: List['SynthesisNode']
    metadata: Dict[str, Any]
    
    def get_feature_content(self, feature_type: str) -> List[str]:
        """Get all content of a specific feature type in this subtree."""
        content = []
        
        # Get features from this node
        if features := self.metadata.get('features', {}).get(feature_type, []):
            content.extend(f['content'] for f in features)
            
        # Get features from children
        for child in self.children:
            content.extend(child.get_feature_content(feature_type))
            
        return content
        
    def get_overlapping_content(self, feature_type: str) -> List[str]:
        """Get feature content from overlapping nodes."""
        content = []
        
        for overlap in self.overlapping:
            if features := overlap.metadata.get('features', {}).get(feature_type, []):
                content.extend(f['content'] for f in features)
                
        return content

class Synthesizer:
    """
    Base class for document synthesis operations.
    """
    def __init__(self, graph: DocumentGraph):
        self.graph = graph
        self.synthesis_cache: Dict[str, SynthesisNode] = {}
        
    def get_synthesis_tree(
        self,
        root_node: Node,
        feature_types: List[str],
        max_depth: Optional[int] = None
    ) -> SynthesisNode:
        """Build synthesis tree from document graph."""
        # Check cache
        cache_key = f"{root_node.id}:{':'.join(feature_types)}:{max_depth}"
        if cache_key in self.synthesis_cache:
            return self.synthesis_cache[cache_key]
            
        # Get features for this node
        features = {}
        for feat_type in feature_types:
            features[feat_type] = [
                n for n in self.graph.get_feature_nodes(feat_type)
                if root_node.span[0] <= n.span[0] and root_node.span[1] >= n.span[1]
            ]
            
        # Process children if within depth limit
        children = []
        if max_depth != 0:
            next_depth = max_depth - 1 if max_depth else None
            for child_id in root_node.children:
                if child := self.graph.nodes.get(child_id):
                    child_tree = self.get_synthesis_tree(
                        child,
                        feature_types,
                        next_depth
                    )
                    children.append(child_tree)
                    
        # Get overlapping nodes
        overlapping = []
        overlap_nodes = self.graph.get_overlapping_nodes(
            root_node.span,
            exclude_ids={root_node.id}
        )
        for node in overlap_nodes:
            overlap_tree = self.get_synthesis_tree(
                node,
                feature_types,
                max_depth=1  # Limit overlap depth
            )
            overlapping.append(overlap_tree)
            
        # Create synthesis node
        syn_node = SynthesisNode(
            node_id=root_node.id,
            feature_type=root_node.type,
            content=root_node.content,
            children=children,
            overlapping=overlapping,
            metadata={
                'features': features,
                'span': root_node.span,
                'node_metadata': root_node.metadata
            }
        )
        
        self.synthesis_cache[cache_key] = syn_node
        return syn_node
        
    def synthesize(
        self,
        tree: SynthesisNode,
        process_fn: Callable[[SynthesisNode], str],
        feature_name: str,
        version: str = "1.0",
        bottom_up: bool = True
    ) -> None:
        """
        Synthesize features across a subtree.
        
        Args:
            tree: Synthesis tree to process
            process_fn: Function to generate synthesized content
            feature_name: Name for the synthesized feature
            version: Version string for the feature
            bottom_up: If True, process children before parents
        """
        def process_node(node: SynthesisNode) -> None:
            # Process children first if bottom-up
            if bottom_up:
                for child in node.children:
                    process_node(child)
                    
            # Generate synthesis
            synthesized = process_fn(node)
            
            # Add to graph
            self.graph.add_node(
          # tei_chunker/synthesis/base.py (continued)
                content=synthesized,
                type=f"feature:{feature_name}",
                span=node.metadata['span'],
                parents=[node.node_id],
                metadata={
                    'synthesized_from': [
                        n.node_id for n in node.children + node.overlapping
                    ],
                    'feature_version': version,
                    'synthesized_at': datetime.utcnow().isoformat()
                }
            )
            
            # Process children last if top-down
            if not bottom_up:
                for child in node.children:
                    process_node(child)
        
        # Process entire tree
        process_node(tree)
        
    def format_for_llm(
        self,
        node: SynthesisNode,
        feature_types: List[str],
        max_depth: Optional[int] = None,
        current_depth: int = 0,
        include_overlapping: bool = True
    ) -> str:
        """Format a synthesis node's content for LLM input."""
        if max_depth is not None and current_depth > max_depth:
            return ""
            
        parts = []
        indent = "  " * current_depth
        
        # Add node type and content
        parts.append(f"{indent}[{node.feature_type}]")
        parts.append(f"{indent}{node.content}\n")
        
        # Add features
        for feat_type in feature_types:
            if features := node.metadata['features'].get(feat_type, []):
                parts.append(f"{indent}[{feat_type}]")
                for feat in features:
                    parts.append(f"{indent}{feat.content}\n")
                    
        # Add overlapping content if requested
        if include_overlapping and node.overlapping:
            parts.append(f"{indent}[overlapping content]")
            for overlap in node.overlapping:
                overlap_text = self.format_for_llm(
                    overlap,
                    feature_types,
                    max_depth=1,
                    current_depth=current_depth + 1
                )
                if overlap_text:
                    parts.append(overlap_text)
                    
        # Add children
        for child in node.children:
            child_text = self.format_for_llm(
                child,
                feature_types,
                max_depth,
                current_depth + 1,
                include_overlapping
            )
            if child_text:
                parts.append(child_text)
                
        return "\n".join(parts)
