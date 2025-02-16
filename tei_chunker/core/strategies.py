# tei_chunker/core/strategies.py
"""
Implementation of synthesis strategies.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger

from .interfaces import (
    SynthesisStrategy,
    Feature,
    Span,
    ProcessingContext,
    ContentProcessor
)

class BottomUpStrategy(SynthesisStrategy):
    """Build synthesis incrementally from leaves up."""
    
    def synthesize(
        self,
        content: str,
        features: Dict[str, List[Feature]],
        processor: ContentProcessor,
        context: ProcessingContext
    ) -> str:
        spans = self._split_hierarchical(content, context.max_tokens)
        return self._process_spans_bottom_up(
            spans,
            features,
            processor,
            context
        )
        
    def _split_hierarchical(
        self,
        content: str,
        max_tokens: int
    ) -> List[Span]:
        """Split content preserving hierarchical structure."""
        # In practice, this would use the XML structure
        # For now, using simple paragraph-based splitting
        paragraphs = content.split("\n\n")
        spans = []
        current_pos = 0
        
        for para in paragraphs:
            span = Span(
                start=current_pos,
                end=current_pos + len(para),
                text=para,
                metadata={'type': 'paragraph'}
            )
            spans.append(span)
            current_pos = span.end + 2  # Account for "\n\n"
            
        return spans
        
    def _process_spans_bottom_up(
        self,
        spans: List[Span],
        features: Dict[str, List[Feature]],
        processor: ContentProcessor,
        context: ProcessingContext
    ) -> str:
        processed_spans = []
        
        for span in spans:
            # Get relevant features for this span
            span_features = self._get_relevant_features(span, features)
            
            # Process span with its features
            result = processor(
                self._format_for_processing(span, span_features)
            )
            
            processed_spans.append(result)
            
        # Combine processed spans
        if len(processed_spans) == 1:
            return processed_spans[0]
            
        # Recursively combine if needed
        return self.synthesize(
            "\n\n".join(processed_spans),
            features,
            processor,
            context
        )
        
    def _format_for_processing(
        self,
        span: Span,
        features: Dict[str, List[Feature]]
    ) -> str:
        parts = [span.text]
        
        for feat_type, feats in features.items():
            for feat in feats:
                parts.append(f"\n\n{feat_type}: {feat.content}")
                
        return "\n\n".join(parts)

class HybridStrategy(SynthesisStrategy):
    """Try top-down first, fall back to bottom-up when needed."""
    
    def __init__(self):
        self.top_down = TopDownStrategy()
        self.bottom_up = BottomUpStrategy()
        
    def synthesize(
        self,
        content: str,
        features: Dict[str, List[Feature]],
        processor: ContentProcessor,
        context: ProcessingContext
    ) -> str:
        try:
            return self.top_down.synthesize(
                content,
                features,
                processor,
                context
            )
        except ValueError as e:
            logger.info(f"Falling back to bottom-up strategy: {e}")
            return self.bottom_up.synthesize(
                content,
                features,
                processor,
                context
            )
            
class TopDownStrategy(SynthesisStrategy):
    """Try to process maximum content at once."""
    
    def synthesize(
        self,
        content: str,
        features: Dict[str, List[Feature]],
        processor: ContentProcessor,
        context: ProcessingContext
    ) -> str:
        if self._can_fit_in_context(content, features, context):
            return self._process_all_at_once(
                content,
                features,
                processor
            )
            
        sections = self._split_into_sections(content)
        results = []
        
        for section in sections:
            section_features = self._get_relevant_features(section, features)
            try:
                result = self.synthesize(
                    section.text,
                    section_features,
                    processor,
                    context
                )
                results.append(result)
            except ValueError as e:
                if len(section.text.split()) <= context.min_chunk_tokens:
                    raise ValueError(
                        f"Section of size {len(section.text.split())} tokens "
                        f"exceeds context window ({context.max_tokens} tokens) "
                        "and cannot be subdivided further"
                    ) from e
                    
                # Split section and process parts
                subsections = self._split_section(
                    section,
                    context.max_tokens,
                    context.overlap_tokens
                )
                for sub in subsections:
                    sub_features = self._get_relevant_features(sub, features)
                    sub_result = self.synthesize(
                        sub.text,
                        sub_features,
                        processor,
                        context
                    )
                    results.append(sub_result)
                    
        # Final combination of results
        if len(results) == 1:
            return results[0]
            
        combined = "\n\n".join(results)
        if self._can_fit_in_context(combined, {}, context):
            return processor(combined)
            
        # Need to process combination in chunks
        return self.synthesize(
            combined,
            {},  # No features for final combination
            processor,
            context
        )
        
    def _can_fit_in_context(
        self,
        content: str,
        features: Dict[str, List[Feature]],
        context: ProcessingContext
    ) -> bool:
        total_tokens = len(content.split())
        
        for feats in features.values():
            for feat in feats:
                total_tokens += len(feat.content.split())
                
        return total_tokens <= context.max_tokens
        
    def _process_all_at_once(
        self,
        content: str,
        features: Dict[str, List[Feature]],
        processor: ContentProcessor
    ) -> str:
        parts = [content]
        
        for feat_type, feats in features.items():
            for feat in feats:
                parts.append(f"{feat_type}:\n{feat.content}")
                
        return processor("\n\n".join(parts))
        
    def _split_into_sections(self, content: str) -> List[Span]:
        """Split content into logical sections."""
        # In practice, this would use XML structure
        # For now, using simple heuristics
        sections = []
        current_pos = 0
        
        paragraphs = content.split("\n\n")
        current_section = []
        
        for para in paragraphs:
            if self._is_section_header(para):
                if current_section:
                    text = "\n\n".join(current_section)
                    sections.append(Span(
                        start=current_pos,
                        end=current_pos + len(text),
                        text=text,
                        metadata={'type': 'section'}
                    ))
                    current_pos = current_pos + len(text) + 2
                    current_section = []
            
            current_section.append(para)
            
        # Add final section
        if current_section:
            text = "\n\n".join(current_section)
            sections.append(Span(
                start=current_pos,
                end=current_pos + len(text),
                text=text,
                metadata={'type': 'section'}
            ))
            
        return sections
        
    def _is_section_header(self, text: str) -> bool:
        """Identify section headers."""
        # Simple heuristic - could be more sophisticated
        lines = text.strip().split("\n")
        if len(lines) == 1 and len(lines[0].split()) <= 5:
            return True
        return False
        
    def _split_section(
        self,
        section: Span,
        max_tokens: int,
        overlap_tokens: int
    ) -> List[Span]:
        """Split a section into overlapping chunks."""
        words = section.text.split()
        chunks = []
        start_idx = 0
        
        while start_idx < len(words):
            # Calculate end index for this chunk
            end_idx = start_idx + max_tokens
            if end_idx > len(words):
                end_idx = len(words)
                
            # Create chunk
            chunk_text = " ".join(words[start_idx:end_idx])
            chunk_start = section.start + len(" ".join(words[:start_idx]))
            if start_idx > 0:
                chunk_start += 1  # Account for space
                
            chunks.append(Span(
                start=chunk_start,
                end=chunk_start + len(chunk_text),
                text=chunk_text,
                metadata={'type': 'chunk'}
            ))
            
            # Move to next chunk with overlap
            start_idx = end_idx - overlap_tokens
            if start_idx < 0:
                start_idx = 0
                
        return chunks
        
    def _get_relevant_features(
        self,
        span: Span,
        features: Dict[str, List[Feature]]
    ) -> Dict[str, List[Feature]]:
        """Get features relevant to a span."""
        relevant = {}
        for name, feature_list in features.items():
            relevant_features = [
                f for f in feature_list
                if (f.span.start < span.end and f.span.end > span.start)
            ]
            if relevant_features:
                relevant[name] = relevant_features
        return relevant
