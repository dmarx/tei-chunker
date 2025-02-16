# tei_chunker/__init__.py
"""
TEI document chunking and synthesis library.
"""
from .__about__ import __version__

from .graph import DocumentGraph, Node, Feature
from .synthesis.base import Synthesizer, SynthesisNode
from .synthesis.patterns import FeatureSynthesizer, SynthesisStrategy
from .synthesis.prompts import SynthesisPrompt, PromptTemplates

__all__ = [
    "__version__",
    "DocumentGraph",
    "Node",
    "Feature",
    "Synthesizer",
    "SynthesisNode",
    "FeatureSynthesizer",
    "SynthesisStrategy",
    "SynthesisPrompt",
    "PromptTemplates"
]
