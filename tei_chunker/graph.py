# tei_chunker/graph.py
"""
Core document graph representation with persistence.
File: tei_chunker/graph.py
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import json
from datetime import datetime
import hashlib
from loguru import logger

@dataclass
class Node:
    """Node in the document graph."""
    id: str
    content: str
    type: str
    span: tuple[int, int]
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'content': self.content,
            'type': self.type,
            'span': self.span,
            'parents': self.parents,
            'children': self.children,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Node':
        return cls(**data)

@dataclass
class Feature:
    """Metadata about a feature type."""
    name: str
    version: str
    source_types: List[str]
    parameters: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Feature':
        return cls(**data)

class DocumentGraph:
    """
    Graph structure for document content and features with persistence.
    """
    def __init__(self, content: str):
        self.content = content
        self.nodes: Dict[str, Node] = {}
        self.features: Dict[str, Feature] = {}
        
    def generate_id(self, content: str, type: str) -> str:
        """Generate deterministic node ID."""
        hash_input = f"{content}:{type}".encode('utf-8')
        return hashlib.sha256(hash_input).hexdigest()[:12]
        
    def add_node(
        self,
        content: str,
        type: str,
        span: tuple[int, int],
        parents: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Node:
        """Add a new node to the graph."""
        node_id = self.generate_id(content, type)
        
        # Update existing node if it exists
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.content = content
            node.metadata.update(metadata or {})
            if parents:
                for parent_id in parents:
                    if parent_id not in node.parents:
                        node.parents.append(parent_id)
                        self.nodes[parent_id].children.append(node_id)
            return node
            
        # Create new node
        node = Node(
            id=node_id,
            content=content,
            type=type,
            span=span,
            parents=parents or [],
            children=[],
            metadata=metadata or {}
        )
        
        # Update parent-child relationships
        if parents:
            for parent_id in parents:
                if parent_id in self.nodes:
                    self.nodes[parent_id].children.append(node_id)
                    
        self.nodes[node_id] = node
        return node
        
    def register_feature(self, feature: Feature) -> None:
        """Register a new feature type."""
        self.features[feature.name] = feature
        
    def get_nodes_by_type(self, type: str) -> List[Node]:
        """Get all nodes of a given type."""
        return [n for n in self.nodes.values() if n.type == type]
        
    def get_feature_nodes(self, feature_name: str) -> List[Node]:
        """Get all nodes for a specific feature."""
        return self.get_nodes_by_type(f"feature:{feature_name}")
        
    def get_overlapping_nodes(
        self,
        span: tuple[int, int],
        exclude_ids: Optional[Set[str]] = None
    ) -> List[Node]:
        """Find nodes that overlap with the given span."""
        start, end = span
        exclude_ids = exclude_ids or set()
        
        return [
            node for node in self.nodes.values()
            if node.id not in exclude_ids
            and not (end <= node.span[0] or start >= node.span[1])
        ]
        
    def get_node_ancestors(self, node_id: str) -> List[Node]:
        """Get all ancestors of a node."""
        ancestors = []
        queue = [node_id]
        seen = set()
        
        while queue:
            current_id = queue.pop(0)
            if current_id in seen:
                continue
                
            seen.add(current_id)
            if current_id in self.nodes:
                node = self.nodes[current_id]
                ancestors.append(node)
                queue.extend(node.parents)
                
        return ancestors[1:]  # Exclude the starting node
        
    def get_node_descendants(self, node_id: str) -> List[Node]:
        """Get all descendants of a node."""
        descendants = []
        queue = [node_id]
        seen = set()
        
        while queue:
            current_id = queue.pop(0)
            if current_id in seen:
                continue
                
            seen.add(current_id)
            if current_id in self.nodes:
                node = self.nodes[current_id]
                descendants.append(node)
                queue.extend(node.children)
                
        return descendants[1:]  # Exclude the starting node
        
    def save(self, path: Path) -> None:
        """Save the graph to disk."""
        data = {
            'content': self.content,
            'nodes': {id: node.to_dict() for id, node in self.nodes.items()},
            'features': {name: feat.to_dict() for name, feat in self.features.items()}
        }
        
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved document graph to {path}")
        
    @classmethod
    def load(cls, path: Path) -> 'DocumentGraph':
        """Load a graph from disk."""
        data = json.loads(path.read_text())
        
        graph = cls(content=data['content'])
        graph.nodes = {
            id: Node.from_dict(node_data)
            for id, node_data in data['nodes'].items()
        }
        graph.features = {
            name: Feature.from_dict(feat_data)
            for name, feat_data in data['features'].items()
        }
        
        logger.info(f"Loaded document graph from {path}")
        return graph
