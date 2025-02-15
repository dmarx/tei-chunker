# tei_chunker/synthesis/patterns.py
"""
Implementation of common synthesis patterns.
File: tei_chunker/synthesis/patterns.py
"""
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from .base import Synthesizer, SynthesisNode
from .prompts import PromptTemplates

class SynthesisStrategy(Enum):
    """Available synthesis strategies."""
    HIERARCHICAL = "hierarchical"  # Maintain document hierarchy
    FLAT = "flat"                 # Flatten and synthesize all at once
    INCREMENTAL = "incremental"   # Build up synthesis gradually

class FeatureSynthesizer(Synthesizer):
    """
    Implementation of common synthesis patterns.
    """
    def __init__(self, graph):
        super().__init__(graph)
        self.prompts = PromptTemplates()
        
    def hierarchical_summary(
        self,
        tree: SynthesisNode,
        max_length: int = 500
    ) -> None:
        """
        Create hierarchical summary synthesis.
        
        Args:
            tree: Root of synthesis tree
            max_length: Maximum length for each summary
        """
        prompt = self.prompts.hierarchical_summary(max_length)
        
        def process_node(node: SynthesisNode) -> str:
            # Format input for LLM
            context = self.format_for_llm(
                node,
                feature_types=["summary", "key_findings"]
            )
            
            return prompt.format(
                structure=self._format_structure(node),
                features=context
            )
            
        self.synthesize(
            tree,
            process_node,
            feature_name="hierarchical_summary",
            version="1.0",
            bottom_up=True
        )
        
    def resolve_conflicts(
        self,
        tree: SynthesisNode,
        feature_type: str
    ) -> None:
        """
        Resolve conflicts between overlapping features.
        
        Args:
            tree: Root of synthesis tree
            feature_type: Type of feature to resolve
        """
        prompt = self.prompts.conflict_resolution()
        
        def process_node(node: SynthesisNode) -> str:
            main_content = node.get_feature_content(feature_type)
            overlapping = node.get_overlapping_content(feature_type)
            
            if not overlapping:
                return "\n".join(main_content)
                
            return prompt.format(
                main_content="\n".join(main_content),
                overlapping_content="\n".join(overlapping)
            )
            
        self.synthesize(
            tree,
            process_node,
            feature_name=f"resolved_{feature_type}",
            version="1.0",
            bottom_up=True
        )
        
    def evidence_graded_synthesis(
        self,
        tree: SynthesisNode,
        feature_types: List[str],
        confidence_threshold: float = 0.8
    ) -> None:
        """
        Create synthesis with evidence grading.
        
        Args:
            tree: Root of synthesis tree
            feature_types: Types of features to synthesize
            confidence_threshold: Minimum confidence threshold
        """
        prompt = self.prompts.evidence_graded(confidence_threshold)
        
        def process_node(node: SynthesisNode) -> str:
            findings = []
            for feat_type in feature_types:
                findings.extend(node.get_feature_content(feat_type))
                findings.extend(node.get_overlapping_content(feat_type))
                
            return prompt.format(
                findings="\n\n".join(findings),
                confidence_threshold=confidence_threshold
            )
            
        self.synthesize(
            tree,
            process_node,
            feature_name="evidence_graded",
            version="1.0",
            bottom_up=True
        )
        
    def incremental_synthesis(
        self,
        tree: SynthesisNode,
        feature_sequence: List[str]
    ) -> None:
        """
        Build up synthesis incrementally across features.
        
        Args:
            tree: Root of synthesis tree
            feature_sequence: Order of features to incorporate
        """
        current_synthesis = ""
        
        for feature_type in feature_sequence:
            prompt = self.prompts.incremental_feature(feature_type)
            
            def process_node(node: SynthesisNode) -> str:
                new_content = node.get_feature_content(feature_type)
                return prompt.format(
                    current_synthesis=current_synthesis,
                    feature_type=feature_type,
                    new_feature="\n".join(new_content)
                )
                
            self.synthesize(
                tree,
                process_node,
                feature_name=f"incremental_{len(feature_sequence)}",
                version="1.0",
                bottom_up=True
            )
            
            # Update current synthesis with new feature
            current_synthesis = self.graph.get_feature_nodes(
                f"incremental_{len(feature_sequence)}"
            )[0].content
            
    def _format_structure(self, node: SynthesisNode, depth: int = 0) -> str:
        """Helper to format document structure."""
        parts = [f"{'  ' * depth}{node.feature_type}: {node.content[:100]}..."]
        for child in node.children:
            parts.append(self._format_structure(child, depth + 1))
        return "\n".join(parts)
