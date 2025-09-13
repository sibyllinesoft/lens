#!/usr/bin/env python3
"""
Symbol Graph Boosting System for v2.2.0 Algorithm Sprint
Implements LSIF/SCIP integration with call graph expansion and symbol relationship mapping
"""

import json
import sqlite3
import networkx as nx
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from pathlib import Path
from enum import Enum
import hashlib
import ast
import re
from collections import defaultdict, deque


class SymbolType(Enum):
    """Types of code symbols"""
    FUNCTION = "function"
    CLASS = "class" 
    VARIABLE = "variable"
    MODULE = "module"
    METHOD = "method"
    PROPERTY = "property"
    INTERFACE = "interface"
    ENUM = "enum"
    CONSTANT = "constant"


class RelationType(Enum):
    """Types of symbol relationships"""
    DEFINES = "defines"          # A defines B
    REFERENCES = "references"    # A references B  
    CALLS = "calls"             # A calls B
    INHERITS = "inherits"       # A inherits from B
    IMPLEMENTS = "implements"   # A implements B
    CONTAINS = "contains"       # A contains B
    IMPORTS = "imports"         # A imports B
    TYPE_OF = "type_of"         # A is type of B


@dataclass
class SymbolNode:
    """Represents a symbol in the graph"""
    id: str
    name: str
    symbol_type: SymbolType
    file_path: str
    start_line: int
    end_line: int
    start_char: int
    end_char: int
    signature: Optional[str] = None
    docstring: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SymbolEdge:
    """Represents a relationship between symbols"""
    from_symbol: str
    to_symbol: str
    relation_type: RelationType
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GraphExpansionResult:
    """Result of graph expansion for a query"""
    original_results: List[Dict]
    expanded_symbols: List[str]
    expansion_paths: Dict[str, List[str]]
    boost_scores: Dict[str, float]
    total_boost_factor: float


class BaseSymbolGraphBuilder(ABC):
    """Base class for symbol graph builders"""
    
    def __init__(self, corpus_path: str, cache_path: Optional[str] = None):
        self.corpus_path = Path(corpus_path)
        self.cache_path = Path(cache_path) if cache_path else None
        self.graph = nx.MultiDiGraph()  # Allow multiple edge types between nodes
        self.symbol_index: Dict[str, SymbolNode] = {}
        
    @abstractmethod
    def build_graph(self, files: List[str] = None) -> nx.MultiDiGraph:
        """Build the symbol graph from source code"""
        pass
    
    @abstractmethod
    def extract_symbols(self, file_path: str, content: str) -> Tuple[List[SymbolNode], List[SymbolEdge]]:
        """Extract symbols and relationships from a file"""
        pass
    
    def get_symbol_neighbors(self, symbol_id: str, hops: int = 1, 
                           relation_types: List[RelationType] = None) -> Dict[str, float]:
        """Get neighboring symbols with decay weights"""
        if symbol_id not in self.graph:
            return {}
        
        neighbors = {}
        visited = {symbol_id}
        queue = deque([(symbol_id, 1.0, 0)])  # (node, weight, hop_count)
        
        while queue:
            current_id, current_weight, hop_count = queue.popleft()
            
            if hop_count >= hops:
                continue
            
            # Get outgoing edges (what this symbol references/calls)
            for neighbor_id in self.graph.successors(current_id):
                if neighbor_id in visited:
                    continue
                
                # Filter by relation types if specified
                edge_data = self.graph.get_edge_data(current_id, neighbor_id)
                if relation_types:
                    valid_edge = any(
                        edge.get('relation_type') in relation_types 
                        for edge in edge_data.values()
                    )
                    if not valid_edge:
                        continue
                
                # Calculate weight decay
                decay_factor = 0.7 ** (hop_count + 1)  # Exponential decay
                neighbor_weight = current_weight * decay_factor
                
                # Boost weight based on relationship strength
                edge_boost = max(edge.get('confidence', 1.0) for edge in edge_data.values())
                neighbor_weight *= edge_boost
                
                neighbors[neighbor_id] = max(neighbors.get(neighbor_id, 0), neighbor_weight)
                
                visited.add(neighbor_id)
                queue.append((neighbor_id, neighbor_weight, hop_count + 1))
            
            # Also check incoming edges (what references/calls this symbol)
            for predecessor_id in self.graph.predecessors(current_id):
                if predecessor_id in visited:
                    continue
                
                edge_data = self.graph.get_edge_data(predecessor_id, current_id)
                if relation_types:
                    valid_edge = any(
                        edge.get('relation_type') in relation_types 
                        for edge in edge_data.values()
                    )
                    if not valid_edge:
                        continue
                
                decay_factor = 0.8 ** (hop_count + 1)  # Slightly less decay for incoming
                predecessor_weight = current_weight * decay_factor
                edge_boost = max(edge.get('confidence', 1.0) for edge in edge_data.values())
                predecessor_weight *= edge_boost
                
                neighbors[predecessor_id] = max(neighbors.get(predecessor_id, 0), predecessor_weight)
                
                visited.add(predecessor_id)
                queue.append((predecessor_id, predecessor_weight, hop_count + 1))
        
        return neighbors
    
    def save_graph(self, output_path: str):
        """Save graph to disk for caching"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as GraphML for networkx compatibility
        nx.write_graphml(self.graph, str(output_path) + ".graphml")
        
        # Save symbol index as JSON
        symbol_data = {
            symbol_id: asdict(symbol) for symbol_id, symbol in self.symbol_index.items()
        }
        with open(str(output_path) + "_symbols.json", 'w') as f:
            json.dump(symbol_data, f, indent=2, default=str)
    
    def load_graph(self, input_path: str) -> bool:
        """Load graph from disk"""
        try:
            input_path = Path(input_path)
            
            # Load NetworkX graph
            self.graph = nx.read_graphml(str(input_path) + ".graphml")
            
            # Load symbol index
            with open(str(input_path) + "_symbols.json", 'r') as f:
                symbol_data = json.load(f)
                self.symbol_index = {
                    symbol_id: SymbolNode(**data) 
                    for symbol_id, data in symbol_data.items()
                }
            
            return True
        except Exception as e:
            print(f"Failed to load graph: {e}")
            return False


class LSIFGraphBuilder(BaseSymbolGraphBuilder):
    """
    LSIF (Language Server Index Format) graph builder
    Processes LSIF dumps for precise symbol relationships
    """
    
    def __init__(self, corpus_path: str, lsif_dump_path: str, cache_path: Optional[str] = None):
        super().__init__(corpus_path, cache_path)
        self.lsif_dump_path = Path(lsif_dump_path)
        
    def build_graph(self, files: List[str] = None) -> nx.MultiDiGraph:
        """Build graph from LSIF dump"""
        
        if self.cache_path and self.cache_path.exists():
            if self.load_graph(str(self.cache_path)):
                print(f"Loaded cached graph with {len(self.graph.nodes)} nodes")
                return self.graph
        
        print("Building LSIF symbol graph...")
        
        # Parse LSIF dump (JSONL format)
        lsif_data = self._parse_lsif_dump()
        
        # Build document mapping
        documents = {}
        for entry in lsif_data:
            if entry.get('label') == 'document':
                documents[entry['id']] = entry['uri']
        
        # Build symbol nodes
        for entry in lsif_data:
            if entry.get('label') == 'range':
                symbol_node = self._create_symbol_from_range(entry, documents, lsif_data)
                if symbol_node:
                    self.symbol_index[symbol_node.id] = symbol_node
                    self.graph.add_node(symbol_node.id, **asdict(symbol_node))
        
        # Build relationships
        for entry in lsif_data:
            if entry.get('label') in ['next', 'textDocument/definition', 'textDocument/references']:
                edges = self._create_edges_from_lsif(entry, lsif_data)
                for edge in edges:
                    self.graph.add_edge(
                        edge.from_symbol, edge.to_symbol,
                        relation_type=edge.relation_type,
                        confidence=edge.confidence,
                        **edge.metadata
                    )
        
        print(f"Built LSIF graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        
        # Cache if path provided
        if self.cache_path:
            self.save_graph(str(self.cache_path))
        
        return self.graph
    
    def _parse_lsif_dump(self) -> List[Dict]:
        """Parse LSIF JSONL dump file"""
        lsif_entries = []
        
        with open(self.lsif_dump_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    lsif_entries.append(entry)
                except:
                    continue
        
        return lsif_entries
    
    def _create_symbol_from_range(self, range_entry: Dict, documents: Dict, 
                                 all_entries: List[Dict]) -> Optional[SymbolNode]:
        """Create symbol node from LSIF range entry"""
        
        document_id = range_entry.get('document')
        if not document_id or document_id not in documents:
            return None
        
        file_path = documents[document_id]
        start = range_entry.get('start', {})
        end = range_entry.get('end', {})
        
        # Try to determine symbol type and name from LSIF data
        symbol_name = f"range_{range_entry['id']}"  # Default fallback
        symbol_type = SymbolType.VARIABLE  # Default
        
        # Look for associated definition or hover information
        for entry in all_entries:
            if (entry.get('outV') == range_entry['id'] and 
                entry.get('label') == 'textDocument/hover'):
                hover_info = entry.get('inV', {})
                if isinstance(hover_info, dict) and 'contents' in hover_info:
                    # Parse hover contents to determine symbol type
                    contents = hover_info['contents']
                    if 'function' in contents.lower():
                        symbol_type = SymbolType.FUNCTION
                    elif 'class' in contents.lower():
                        symbol_type = SymbolType.CLASS
        
        symbol_id = f"{file_path}:{start.get('line', 0)}:{start.get('character', 0)}"
        
        return SymbolNode(
            id=symbol_id,
            name=symbol_name,
            symbol_type=symbol_type,
            file_path=file_path,
            start_line=start.get('line', 0),
            end_line=end.get('line', 0),
            start_char=start.get('character', 0),
            end_char=end.get('character', 0),
            metadata={'lsif_range_id': range_entry['id']}
        )
    
    def _create_edges_from_lsif(self, entry: Dict, all_entries: List[Dict]) -> List[SymbolEdge]:
        """Create symbol edges from LSIF relationship entries"""
        edges = []
        
        label = entry.get('label')
        out_v = entry.get('outV')
        in_vs = entry.get('inVs', [])
        
        if not out_v or not in_vs:
            return edges
        
        # Map LSIF labels to our relation types
        relation_mapping = {
            'textDocument/definition': RelationType.DEFINES,
            'textDocument/references': RelationType.REFERENCES,
            'next': RelationType.CALLS  # Simplified
        }
        
        relation_type = relation_mapping.get(label, RelationType.REFERENCES)
        
        for in_v in in_vs:
            from_symbol = f"lsif_{out_v}"
            to_symbol = f"lsif_{in_v}"
            
            edge = SymbolEdge(
                from_symbol=from_symbol,
                to_symbol=to_symbol,
                relation_type=relation_type,
                confidence=0.9,  # LSIF is high confidence
                metadata={'lsif_label': label}
            )
            edges.append(edge)
        
        return edges
    
    def extract_symbols(self, file_path: str, content: str) -> Tuple[List[SymbolNode], List[SymbolEdge]]:
        """Extract symbols from file content (fallback when no LSIF)"""
        # This is a fallback - LSIF builder primarily uses dump data
        return [], []


class SCIPGraphBuilder(BaseSymbolGraphBuilder):
    """
    SCIP (SCIP Code Intelligence Protocol) graph builder
    Processes SCIP index files for symbol relationships
    """
    
    def __init__(self, corpus_path: str, scip_index_path: str, cache_path: Optional[str] = None):
        super().__init__(corpus_path, cache_path)
        self.scip_index_path = Path(scip_index_path)
        
    def build_graph(self, files: List[str] = None) -> nx.MultiDiGraph:
        """Build graph from SCIP index"""
        
        if self.cache_path and self.cache_path.exists():
            if self.load_graph(str(self.cache_path)):
                print(f"Loaded cached SCIP graph with {len(self.graph.nodes)} nodes")
                return self.graph
        
        print("Building SCIP symbol graph...")
        
        # Parse SCIP index (protobuf format, simplified JSON representation)
        scip_data = self._parse_scip_index()
        
        # Process documents and symbols
        for document in scip_data.get('documents', []):
            file_path = document.get('relative_path', '')
            symbols, edges = self._process_scip_document(document, file_path)
            
            # Add symbols to graph
            for symbol in symbols:
                self.symbol_index[symbol.id] = symbol
                self.graph.add_node(symbol.id, **asdict(symbol))
            
            # Add edges to graph
            for edge in edges:
                self.graph.add_edge(
                    edge.from_symbol, edge.to_symbol,
                    relation_type=edge.relation_type,
                    confidence=edge.confidence,
                    **edge.metadata
                )
        
        print(f"Built SCIP graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        
        if self.cache_path:
            self.save_graph(str(self.cache_path))
        
        return self.graph
    
    def _parse_scip_index(self) -> Dict:
        """Parse SCIP index file"""
        # Simplified - in reality would use protobuf parsing
        try:
            with open(self.scip_index_path, 'r') as f:
                return json.load(f)
        except:
            # Fallback to empty structure
            return {'documents': []}
    
    def _process_scip_document(self, document: Dict, file_path: str) -> Tuple[List[SymbolNode], List[SymbolEdge]]:
        """Process SCIP document to extract symbols and relationships"""
        symbols = []
        edges = []
        
        # Process symbol definitions
        for symbol_def in document.get('symbols', []):
            symbol_node = self._create_symbol_from_scip(symbol_def, file_path)
            if symbol_node:
                symbols.append(symbol_node)
        
        # Process occurrences (references)
        for occurrence in document.get('occurrences', []):
            occurrence_edges = self._create_edges_from_scip_occurrence(occurrence, file_path)
            edges.extend(occurrence_edges)
        
        return symbols, edges
    
    def _create_symbol_from_scip(self, symbol_def: Dict, file_path: str) -> Optional[SymbolNode]:
        """Create symbol node from SCIP symbol definition"""
        
        symbol_info = symbol_def.get('symbol', '')
        display_name = symbol_def.get('display_name', symbol_info.split('.')[-1])
        
        # Parse SCIP symbol to determine type
        symbol_type = SymbolType.VARIABLE
        if 'function' in symbol_info.lower() or '()' in symbol_info:
            symbol_type = SymbolType.FUNCTION
        elif 'class' in symbol_info.lower() or symbol_info.isupper():
            symbol_type = SymbolType.CLASS
        
        # Get location info
        location = symbol_def.get('location', [])
        start_line = location[0] if len(location) > 0 else 0
        start_char = location[1] if len(location) > 1 else 0
        end_line = location[2] if len(location) > 2 else start_line
        end_char = location[3] if len(location) > 3 else start_char
        
        symbol_id = f"{file_path}:{symbol_info}"
        
        return SymbolNode(
            id=symbol_id,
            name=display_name,
            symbol_type=symbol_type,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            start_char=start_char,
            end_char=end_char,
            signature=symbol_def.get('signature'),
            docstring=symbol_def.get('documentation'),
            metadata={'scip_symbol': symbol_info}
        )
    
    def _create_edges_from_scip_occurrence(self, occurrence: Dict, file_path: str) -> List[SymbolEdge]:
        """Create edges from SCIP occurrence data"""
        edges = []
        
        symbol_roles = occurrence.get('symbol_roles', [])
        symbol_ref = occurrence.get('symbol', '')
        
        # Map SCIP roles to relation types
        for role in symbol_roles:
            relation_type = RelationType.REFERENCES
            if role == 'Definition':
                relation_type = RelationType.DEFINES
            elif role == 'Reference':
                relation_type = RelationType.REFERENCES
            
            # Create edge (simplified - would need more context for from_symbol)
            from_symbol_id = f"{file_path}:context"  # Placeholder
            to_symbol_id = f"{file_path}:{symbol_ref}"
            
            edge = SymbolEdge(
                from_symbol=from_symbol_id,
                to_symbol=to_symbol_id,
                relation_type=relation_type,
                confidence=0.85,
                metadata={'scip_role': role}
            )
            edges.append(edge)
        
        return edges
    
    def extract_symbols(self, file_path: str, content: str) -> Tuple[List[SymbolNode], List[SymbolEdge]]:
        """Extract symbols from file content (fallback)"""
        return [], []


class TreeSitterGraphBuilder(BaseSymbolGraphBuilder):
    """
    Tree-sitter based graph builder using AST parsing
    Fallback when LSIF/SCIP are not available
    """
    
    def __init__(self, corpus_path: str, cache_path: Optional[str] = None):
        super().__init__(corpus_path, cache_path)
        
    def build_graph(self, files: List[str] = None) -> nx.MultiDiGraph:
        """Build graph using AST analysis"""
        
        if self.cache_path and self.cache_path.exists():
            if self.load_graph(str(self.cache_path)):
                print(f"Loaded cached tree-sitter graph with {len(self.graph.nodes)} nodes")
                return self.graph
        
        print("Building tree-sitter symbol graph...")
        
        target_files = files or []
        if not target_files:
            # Discover files in corpus
            for ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c']:
                target_files.extend(self.corpus_path.rglob(f'*{ext}'))
        
        for file_path in target_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                symbols, edges = self.extract_symbols(str(file_path), content)
                
                # Add symbols to graph
                for symbol in symbols:
                    self.symbol_index[symbol.id] = symbol
                    self.graph.add_node(symbol.id, **asdict(symbol))
                
                # Add edges to graph
                for edge in edges:
                    self.graph.add_edge(
                        edge.from_symbol, edge.to_symbol,
                        relation_type=edge.relation_type,
                        confidence=edge.confidence,
                        **edge.metadata
                    )
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        print(f"Built tree-sitter graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        
        if self.cache_path:
            self.save_graph(str(self.cache_path))
        
        return self.graph
    
    def extract_symbols(self, file_path: str, content: str) -> Tuple[List[SymbolNode], List[SymbolEdge]]:
        """Extract symbols using AST analysis"""
        symbols = []
        edges = []
        
        if file_path.endswith('.py'):
            symbols, edges = self._extract_python_symbols(file_path, content)
        elif file_path.endswith(('.js', '.ts')):
            symbols, edges = self._extract_javascript_symbols(file_path, content)
        
        return symbols, edges
    
    def _extract_python_symbols(self, file_path: str, content: str) -> Tuple[List[SymbolNode], List[SymbolEdge]]:
        """Extract Python symbols using AST"""
        symbols = []
        edges = []
        
        try:
            tree = ast.parse(content)
            
            # Track current scope for context
            scope_stack = [file_path]
            
            for node in ast.walk(tree):
                symbol_node = None
                
                if isinstance(node, ast.FunctionDef):
                    symbol_node = SymbolNode(
                        id=f"{file_path}:func:{node.name}:{node.lineno}",
                        name=node.name,
                        symbol_type=SymbolType.FUNCTION,
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=getattr(node, 'end_lineno', node.lineno),
                        start_char=getattr(node, 'col_offset', 0),
                        end_char=getattr(node, 'end_col_offset', 0),
                        signature=self._build_function_signature(node),
                        docstring=ast.get_docstring(node)
                    )
                
                elif isinstance(node, ast.ClassDef):
                    symbol_node = SymbolNode(
                        id=f"{file_path}:class:{node.name}:{node.lineno}",
                        name=node.name,
                        symbol_type=SymbolType.CLASS,
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=getattr(node, 'end_lineno', node.lineno),
                        start_char=getattr(node, 'col_offset', 0),
                        end_char=getattr(node, 'end_col_offset', 0),
                        docstring=ast.get_docstring(node)
                    )
                
                if symbol_node:
                    symbols.append(symbol_node)
            
            # Extract relationships (calls, references)
            edges.extend(self._extract_python_relationships(tree, file_path, symbols))
            
        except Exception as e:
            print(f"Error parsing Python file {file_path}: {e}")
        
        return symbols, edges
    
    def _extract_javascript_symbols(self, file_path: str, content: str) -> Tuple[List[SymbolNode], List[SymbolEdge]]:
        """Extract JavaScript/TypeScript symbols (simplified regex-based)"""
        symbols = []
        edges = []
        
        # Function definitions
        func_pattern = r'(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))'
        for match in re.finditer(func_pattern, content):
            name = match.group(1) or match.group(2)
            line_no = content[:match.start()].count('\n') + 1
            
            symbol = SymbolNode(
                id=f"{file_path}:func:{name}:{line_no}",
                name=name,
                symbol_type=SymbolType.FUNCTION,
                file_path=file_path,
                start_line=line_no,
                end_line=line_no,
                start_char=match.start(),
                end_char=match.end()
            )
            symbols.append(symbol)
        
        # Class definitions
        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            name = match.group(1)
            line_no = content[:match.start()].count('\n') + 1
            
            symbol = SymbolNode(
                id=f"{file_path}:class:{name}:{line_no}",
                name=name,
                symbol_type=SymbolType.CLASS,
                file_path=file_path,
                start_line=line_no,
                end_line=line_no,
                start_char=match.start(),
                end_char=match.end()
            )
            symbols.append(symbol)
        
        return symbols, edges
    
    def _build_function_signature(self, node: ast.FunctionDef) -> str:
        """Build function signature from AST node"""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        return f"{node.name}({', '.join(args)})"
    
    def _extract_python_relationships(self, tree: ast.AST, file_path: str, 
                                    symbols: List[SymbolNode]) -> List[SymbolEdge]:
        """Extract relationships from Python AST"""
        edges = []
        symbol_names = {s.name: s.id for s in symbols}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Function call relationship
                if isinstance(node.func, ast.Name):
                    caller_context = self._find_containing_symbol(node, symbols)
                    callee_name = node.func.id
                    
                    if caller_context and callee_name in symbol_names:
                        edge = SymbolEdge(
                            from_symbol=caller_context,
                            to_symbol=symbol_names[callee_name],
                            relation_type=RelationType.CALLS,
                            confidence=0.8
                        )
                        edges.append(edge)
        
        return edges
    
    def _find_containing_symbol(self, node: ast.AST, symbols: List[SymbolNode]) -> Optional[str]:
        """Find which symbol contains a given AST node"""
        node_line = getattr(node, 'lineno', 0)
        
        for symbol in symbols:
            if symbol.start_line <= node_line <= symbol.end_line:
                return symbol.id
        
        return None


class CallGraphExpander:
    """
    Expands search results using call graph relationships
    Key innovation: Multi-hop expansion with confidence decay
    """
    
    def __init__(self, symbol_graph: nx.MultiDiGraph, symbol_index: Dict[str, SymbolNode]):
        self.graph = symbol_graph
        self.symbol_index = symbol_index
        
    def expand_query_results(self, query: str, initial_results: List[Dict], 
                           max_hops: int = 2, min_boost: float = 0.1,
                           boost_factor: float = 1.5) -> GraphExpansionResult:
        """
        Expand query results using symbol graph relationships
        
        Args:
            query: Original search query
            initial_results: Initial retrieval results 
            max_hops: Maximum graph expansion hops
            min_boost: Minimum boost threshold
            boost_factor: Base boost multiplier
            
        Returns:
            GraphExpansionResult with expanded symbols and boost scores
        """
        
        expanded_symbols = set()
        expansion_paths = {}
        boost_scores = {}
        
        # Extract symbols from initial results
        result_symbols = self._extract_symbols_from_results(initial_results)
        
        # Expand each symbol
        for symbol_id in result_symbols:
            if symbol_id not in self.graph:
                continue
                
            # Get neighbors with confidence weights
            neighbors = self._get_weighted_neighbors(symbol_id, max_hops)
            
            for neighbor_id, weight in neighbors.items():
                if weight >= min_boost:
                    expanded_symbols.add(neighbor_id)
                    boost_scores[neighbor_id] = weight * boost_factor
                    
                    # Track expansion path
                    if neighbor_id not in expansion_paths:
                        expansion_paths[neighbor_id] = []
                    expansion_paths[neighbor_id].append(symbol_id)
        
        # Calculate total boost factor
        total_boost = sum(boost_scores.values()) / len(boost_scores) if boost_scores else 0
        
        return GraphExpansionResult(
            original_results=initial_results,
            expanded_symbols=list(expanded_symbols),
            expansion_paths=expansion_paths,
            boost_scores=boost_scores,
            total_boost_factor=total_boost
        )
    
    def _extract_symbols_from_results(self, results: List[Dict]) -> List[str]:
        """Extract symbol IDs from search results"""
        symbols = []
        
        for result in results:
            file_path = result.get('file_path', '')
            content = result.get('content', '')
            
            # Try to match symbols in this file
            for symbol_id, symbol in self.symbol_index.items():
                if (symbol.file_path == file_path and 
                    symbol.name.lower() in content.lower()):
                    symbols.append(symbol_id)
        
        return symbols
    
    def _get_weighted_neighbors(self, symbol_id: str, max_hops: int) -> Dict[str, float]:
        """Get neighbors with decay weights"""
        neighbors = {}
        visited = {symbol_id}
        queue = deque([(symbol_id, 1.0, 0)])
        
        while queue:
            current_id, current_weight, hop_count = queue.popleft()
            
            if hop_count >= max_hops:
                continue
            
            # Explore outgoing edges (calls, references)
            for neighbor_id in self.graph.successors(current_id):
                if neighbor_id in visited:
                    continue
                
                # Get edge data for confidence calculation
                edge_data = self.graph.get_edge_data(current_id, neighbor_id)
                max_confidence = max(
                    edge.get('confidence', 0.5) for edge in edge_data.values()
                )
                
                # Calculate decayed weight
                decay_factor = 0.7 ** (hop_count + 1)
                neighbor_weight = current_weight * decay_factor * max_confidence
                
                neighbors[neighbor_id] = max(neighbors.get(neighbor_id, 0), neighbor_weight)
                
                visited.add(neighbor_id)
                queue.append((neighbor_id, neighbor_weight, hop_count + 1))
        
        return neighbors


# Factory function for builder selection
def create_symbol_graph_builder(builder_type: str, corpus_path: str, **kwargs) -> BaseSymbolGraphBuilder:
    """Factory function to create symbol graph builders"""
    
    builders = {
        'lsif': LSIFGraphBuilder,
        'scip': SCIPGraphBuilder, 
        'treesitter': TreeSitterGraphBuilder
    }
    
    if builder_type not in builders:
        raise ValueError(f"Unknown builder type: {builder_type}. Available: {list(builders.keys())}")
    
    return builders[builder_type](corpus_path=corpus_path, **kwargs)


# Example usage
if __name__ == "__main__":
    # Test symbol graph building
    corpus_path = "/path/to/code/corpus"
    
    # Test tree-sitter builder (works without external deps)
    builder = create_symbol_graph_builder('treesitter', corpus_path, cache_path="symbol_graph_cache")
    
    # Build graph
    graph = builder.build_graph()
    
    # Test expansion
    expander = CallGraphExpander(graph, builder.symbol_index)
    
    # Mock query results for testing
    mock_results = [
        {
            'file_path': 'test.py',
            'content': 'def process_data(data): return data.transform()',
            'score': 0.9
        }
    ]
    
    expansion_result = expander.expand_query_results(
        query="data processing", 
        initial_results=mock_results,
        max_hops=2
    )
    
    print(f"Expanded symbols: {len(expansion_result.expanded_symbols)}")
    print(f"Average boost factor: {expansion_result.total_boost_factor:.3f}")
    print(f"Boost scores: {list(expansion_result.boost_scores.items())[:3]}")