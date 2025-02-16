# tei_chunker/synthesis/prompts.py
"""
LLM prompting templates for document synthesis.
File: tei_chunker/synthesis/prompts.py
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class SynthesisPrompt:
    """Template for synthesis prompts."""
    template: str
    examples: Optional[List[Dict[str, str]]] = None
    constraints: Optional[List[str]] = None
    
    def format(self, **kwargs) -> str:
        """Format prompt with provided values."""
        prompt_parts = [self.template.format(**kwargs)]
        
        if self.examples:
            prompt_parts.append("\nExamples:")
            for example in self.examples:
                for key, value in example.items():
                    prompt_parts.append(f"{key}:\n{value}")
                prompt_parts.append("")
                
        if self.constraints:
            prompt_parts.append("\nConstraints:")
            for constraint in self.constraints:
                prompt_parts.append(f"- {constraint}")
                
        return "\n".join(prompt_parts)

class PromptTemplates:
    """Collection of common synthesis prompt templates."""
    
    @staticmethod
    def hierarchical_summary(max_length: int = 500) -> SynthesisPrompt:
        """Template for hierarchical summary synthesis."""
        return SynthesisPrompt(
            template="""
            Synthesize a coherent summary from these section summaries and their relationships.
            Preserve key insights while resolving any conflicts.
            
            Section Structure:
            {structure}
            
            Features to Synthesize:
            {features}
            
            Synthesized Summary:
            """,
            constraints=[
                f"Maximum length: {max_length} characters",
                "Maintain narrative flow between sections",
                "Resolve any contradictions between sections",
                "Preserve specific numbers and key findings"
            ]
        )
    
    @staticmethod
    def conflict_resolution() -> SynthesisPrompt:
        """Template for resolving conflicts between features."""
        return SynthesisPrompt(
            template="""
            Review these potentially conflicting analyses and synthesize a coherent view.
            Explicitly address any contradictions or inconsistencies.
            
            Main Analysis:
            {main_content}
            
            Overlapping Analyses:
            {overlapping_content}
            
            Please:
            1. Identify any conflicts between these analyses
            2. Evaluate the evidence for conflicting claims
            3. Provide a synthesized analysis that:
               - Resolves conflicts with clear reasoning
               - Preserves well-supported findings
               - Acknowledges uncertainty where appropriate
            
            Synthesized Analysis:
            """,
            constraints=[
                "Must explicitly address each conflict",
                "Must preserve source evidence",
                "Must indicate confidence levels"
            ]
        )
    
    @staticmethod
    def evidence_graded(confidence_threshold: float = 0.8) -> SynthesisPrompt:
        """Template for evidence-graded synthesis."""
        return SynthesisPrompt(
            template="""
            Synthesize these findings while evaluating evidence strength.
            
            Findings:
            {findings}
            
            For each finding, provide:
            1. Synthesized statement
            2. Evidence strength (Strong|Moderate|Weak)
            3. Supporting/Conflicting evidence
            
            Confidence Threshold: {confidence_threshold}
            
            Evidence-Graded Synthesis:
            """,
            constraints=[
                f"Must meet {confidence_threshold} confidence threshold",
                "Must grade all evidence",
                "Must explain confidence ratings"
            ]
        )
    
    @staticmethod
    def citation_preserving() -> SynthesisPrompt:
        """Template for citation-preserving synthesis."""
        return SynthesisPrompt(
            template="""
            Synthesize these findings while preserving citation links.
            
            Source Material:
            {source_material}
            
            Required Citation Types:
            {citation_types}
            
            Guidelines:
            1. Maintain all relevant citations
            2. Group related findings
            3. Indicate strength of evidence
            
            Synthesized Result (with citations):
            """,
            constraints=[
                "Must preserve all citation links",
                "Must indicate evidence strength",
                "Must group related findings"
            ]
        )
    
    @staticmethod
    def incremental_feature(feature_type: str) -> SynthesisPrompt:
        """Template for incremental feature synthesis."""
        return SynthesisPrompt(
            template="""
            Incorporate this new feature into the existing synthesis.
            
            Current Synthesis:
            {current_synthesis}
            
            New Feature to Incorporate ({feature_type}):
            {new_feature}
            
            Updated Synthesis:
            """,
            constraints=[
                "Must preserve key information from current synthesis",
                "Must integrate new feature naturally",
                "Must maintain overall coherence"
            ]
        )
