# tei_chunker/core/strategies.py
"""
Implementation of synthesis strategies.
"""
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from loguru import logger

from .interfaces import (
    SynthesisStrategy,
    Feature,
    Span,
    ProcessingContext,
    ContentProcessor
)

class Strategy(SynthesisStrategy):
    """Base class for synthesis strategies."""
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

class TopDownStrategy(Strategy):
    """Try to process maximum content at once."""
    
    def synthesize(
        self,
        content: str,
        features: Dict[str, List[Feature]],
        processor: ContentProcessor,
        context: ProcessingContext,
        depth: int = 0
    ) -> str:
        if depth > 10:  # Reasonable recursion limit
            raise ValueError("Maximum recursion depth exceeded")
            
        if self._can_fit_in_context(content, features, context):
            return self._process_all_at_once(content, features, processor)
            
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
                    
        combined = "\n\n".join(results)
        if self._can_fit_in_context(combined, {}, context):
            return processor(combined)
            
        chunks = self._chunk_content(
            [combined],
            context.max_tokens,
            context.overlap_tokens
        )
        return "\n\n".join(processor(chunk) for chunk in chunks)
        
    def _can_fit_in_context(
        self,
        content: str,
        features: Dict[str, List[Feature]],
        context: ProcessingContext
    ) -> bool:
        total_tokens = len(content.split())
        for feats in features.values():
            total_tokens += sum(len(f.content.split()) for f in feats)
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
                section_text = "\n".join(current_section)
                sections.append(Span(
                    start=start_pos,
                    end=start_pos + len(section_text),
                    text=section_text
                ))
                start_pos += len(section_text) + 1
                current_section = []
            current_section.append(line)
            
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
        return (
            (any(text.startswith(str(i)) for i in range(10)) or text.istitle() or text.isupper())
            and len(text.split()) <= 5
        )
        
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
            end_idx = min(start_idx + max_tokens, len(words))
            chunk_text = " ".join(words[start_idx:end_idx])
            chunk_start = section.start + len(" ".join(words[:start_idx]))
            if start_idx > 0:
                chunk_start += 1
                
            chunks.append(Span(
                start=chunk_start,
                end=chunk_start + len(chunk_text),
                text=chunk_text,
                metadata={'type': 'chunk'}
            ))
            
            start_idx = end_idx - overlap_tokens
            if start_idx < 0:
                start_idx = 0
                
        return chunks

class BottomUpStrategy(Strategy):
    """Build synthesis incrementally from leaves up."""
    
    def synthesize(
        self,
        content: str,
        features: Dict[str, List[Feature]],
        processor: ContentProcessor,
        context: ProcessingContext,
        depth: int = 0
    ) -> str:
        if depth > 10:
            raise ValueError("Maximum recursion depth exceeded")
            
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
        paragraphs = content.split("\n\n")
        spans = []
        current_pos = 0
        
        for para in paragraphs:
            spans.append(Span(
                start=current_pos,
                end=current_pos + len(para),
                text=para,
                metadata={'type': 'paragraph'}
            ))
            current_pos += len(para) + 2  # Account for "\n\n"
            
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
            span_features = self._get_relevant_features(span, features)
            content = span.text
            
            if len(content.split()) <= context.max_tokens:
                result = self._format_for_processing(span, span_features, processor)
            else:
                sub_spans = self._split_section(span, context.max_tokens, context.overlap_tokens)
                sub_results = []
                for sub in sub_spans:
                    sub_features = self._get_relevant_features(sub, features)
                    sub_result = self._format_for_processing(sub, sub_features, processor)
                    sub_results.append(sub_result)
                result = "\n\n".join(sub_results)
                
            processed_spans.append(result)
            
        return "\n\n".join(processed_spans)
    
    def _format_for_processing(
        self,
        span: Span,
        features: Dict[str, List[Feature]],
        processor: ContentProcessor
    ) -> str:
        parts = [span.text]
        for feat_type, feats in features.items():
            for feat in feats:
                parts.append(f"\n\n{feat_type}: {feat.content}")
        return processor("\n\n".join(parts))

class HybridStrategy(Strategy):
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
            return self.top_down.synthesize(content, features, processor, context)
        except ValueError as e:
            logger.info(f"Falling back to bottom-up strategy: {e}")
            return self.bottom_up.synthesize(content, features, processor, context)
