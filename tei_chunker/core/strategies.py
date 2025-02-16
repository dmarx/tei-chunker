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

class TopDownStrategy(SynthesisStrategy):
    """Try to process maximum content at once."""
    
    def synthesize(
        self,
        content: str,
        features: Dict[str, List[Feature]],
        processor: ContentProcessor,
        context: ProcessingContext,
        depth: int = 0  # Add depth parameter
    ) -> str:
        # Add recursion limit
        if depth > 10:  # reasonable limit
            raise ValueError("Maximum recursion depth exceeded")
            
        if self._can_fit_in_context(content, features, context):
            return self._process_all_at_once(content, features, processor)
            
        # Check minimum chunk size
        if len(content.split()) <= context.min_chunk_tokens:
            raise ValueError(
                f"Content size ({len(content.split())} tokens) "
                f"exceeds context window ({context.max_tokens} tokens) "
                "and cannot be subdivided further"
            )
            
        sections = self._split_into_sections(content)
        results = []
        
        for section in sections:
            section_features = self._get_relevant_features(section, features)
            try:
                # Pass incremented depth
                result = self.synthesize(
                    section.text,
                    section_features,
                    processor,
                    context,
                    depth + 1
                )
                results.append(result)
            except ValueError as e:
                # If we hit minimum chunk size, propagate error
                if "cannot be subdivided further" in str(e):
                    raise
                # Otherwise try to split section
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
                        context,
                        depth + 1
                    )
                    results.append(sub_result)
                    
        # Final combination
        combined = "\n\n".join(results)
        if self._can_fit_in_context(combined, {}, context):
            return processor(combined)
            
        # Split final combination if needed
        chunks = self._chunk_content(
            [combined],
            context.max_tokens,
            context.overlap_tokens
        )
        chunk_results = [processor(chunk) for chunk in chunks]
        return "\n\n".join(chunk_results)
        
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
        lines = content.split("\n")
        sections: List[Span] = []
        current_section: List[str] = []
        start_pos = 0
        
        for line in lines:
            if self._is_section_header(line) and current_section:
                # Complete current section
                section_text = "\n".join(current_section)
                sections.append(Span(
                    start=start_pos,
                    end=start_pos + len(section_text),
                    text=section_text
                ))
                start_pos += len(section_text) + 1  # +1 for newline
                current_section = []
                
            current_section.append(line)
            
        # Add final section
        if current_section:
            section_text = "\n".join(current_section)
            sections.append(Span(
                start=start_pos,
                end=start_pos + len(section_text),
                text=section_text
            ))
            
        return sections if sections else [Span(0, len(content), content)]
        
    def _is_section_header(self, text: str) -> bool:
        """Identify section headers."""
        text = text.strip()
        if not text:
            return False
            
        # Simple heuristics for headers
        starts_with_number = any(text.startswith(str(i)) for i in range(10))
        is_short = len(text.split()) <= 5
        is_capitalized = text.istitle() or text.isupper()
        
        return (starts_with_number or is_capitalized) and is_short
        
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
                if f.span.start < span.end and f.span.end > span.start
            ]
            if relevant_features:
                relevant[name] = relevant_features
        return relevant
        
    def _chunk_content(
        self,
        content_list: List[str],
        max_tokens: int,
        overlap_tokens: int
    ) -> List[str]:
        """Split content into overlapping chunks."""
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_tokens = 0
        
        for content in content_list:
            content_tokens = len(content.split())
            
            if current_tokens + content_tokens <= max_tokens:
                current_chunk.append(content)
                current_tokens += content_tokens
            else:
                # Add current chunk if it exists
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    
                # Start new chunk with overlap
                if overlap_tokens > 0 and current_chunk:
                    # Keep some overlap from previous chunk
                    words = " ".join(current_chunk).split()
                    overlap_words = words[-overlap_tokens:]
                    current_chunk = [" ".join(overlap_words)]
                    current_tokens = len(overlap_words)
                else:
                    current_chunk = []
                    current_tokens = 0
                    
                current_chunk.append(content)
                current_tokens += content_tokens
                
        # Add final chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        return chunks

class BottomUpStrategy(SynthesisStrategy):
    """Build synthesis incrementally from leaves up."""
    def synthesize(
        self,
        content: str,
        features: Dict[str, List[Feature]],
        processor: ContentProcessor,
        context: ProcessingContext,
        depth: int = 0  # Add depth parameter
    ) -> str:
        # Prevent infinite recursion
        if depth > 10:  # reasonable limit
            raise ValueError("Maximum recursion depth exceeded")
            
        # Split into sections
        spans = self._split_hierarchical(content, context.max_tokens)
        return self._process_spans_bottom_up(
            spans, 
            features, 
            processor, 
            context,
            depth + 1
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
        context: ProcessingContext,
        depth: int
    ) -> str:
        processed_spans = []
        
        for span in spans:
            # Get relevant features for this span
            span_features = self._get_relevant_features(span, features)
            
            # Process span with its features
            content = span.text
            if len(content.split()) <= context.max_tokens:
                result = processor(self._format_for_processing(span, span_features))
            else:
                # Split if too large
                sub_spans = self._split_section(span, context.max_tokens, context.overlap_tokens)
                sub_results = []
                for sub in sub_spans:
                    sub_features = self._get_relevant_features(sub, features)
                    sub_result = processor(self._format_for_processing(sub, sub_features))
                    sub_results.append(sub_result)
                result = "\n\n".join(sub_results)
                
            processed_spans.append(result)
            
        # Combine processed spans
        return "\n\n".join(processed_spans)
    
    def _split_section(
        self,
        section: Span,
        max_tokens: int,
        overlap_tokens: int
    ) -> List[Span]:
        """Split a section into smaller spans."""
        words = section.text.split()
        spans: List[Span] = []
        
        start_idx = 0
        while start_idx < len(words):
            end_idx = min(start_idx + max_tokens, len(words))
            chunk_text = " ".join(words[start_idx:end_idx])
            
            spans.append(Span(
                start=section.start + start_idx,
                end=section.start + end_idx,
                text=chunk_text
            ))
            
            start_idx = end_idx - overlap_tokens
            if start_idx < 0:
                start_idx = 0
                
        return spans
        
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
                if f.span.start < span.end and f.span.end > span.start
            ]
            if relevant_features:
                relevant[name] = relevant_features
        return relevant

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
