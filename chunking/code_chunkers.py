#!/usr/bin/env python3
"""
Advanced Code Chunking Strategies for v2.2.0 Algorithm Sprint
Implements AST-aligned boundaries, function scope chunking, and hybrid approaches
"""

import ast
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import tree_sitter
from tree_sitter import Language, Parser


class ChunkBoundary(Enum):
    """Types of chunk boundaries"""
    STATEMENT = "statement"
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    SEMANTIC = "semantic"


@dataclass
class CodeChunk:
    """Represents a code chunk with metadata"""
    content: str
    start_line: int
    end_line: int
    start_char: int
    end_char: int
    boundary_type: ChunkBoundary
    scope_info: Optional[Dict[str, Any]] = None
    overlap_with_prev: float = 0.0
    overlap_with_next: float = 0.0
    token_count: int = 0
    symbols: List[str] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = []


class BaseCodeChunker(ABC):
    """Base class for all code chunkers"""
    
    def __init__(self, max_tokens: int = 512, overlap_pct: float = 0.2):
        self.max_tokens = max_tokens
        self.overlap_pct = overlap_pct
    
    @abstractmethod
    def chunk(self, code: str, file_path: str = None) -> List[CodeChunk]:
        """Chunk code into semantically meaningful units"""
        pass
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token average)"""
        return len(text) // 4
    
    def _extract_symbols(self, code: str) -> List[str]:
        """Extract symbols (functions, classes, variables) from code"""
        symbols = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    symbols.append(f"func:{node.name}")
                elif isinstance(node, ast.ClassDef):
                    symbols.append(f"class:{node.name}")
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    symbols.append(f"var:{node.id}")
        except:
            # Fallback to regex for non-Python code
            func_pattern = r'\b(?:def|function|func)\s+(\w+)'
            class_pattern = r'\b(?:class|struct|interface)\s+(\w+)'
            var_pattern = r'\b(?:let|var|const|auto)\s+(\w+)'
            
            symbols.extend([f"func:{m.group(1)}" for m in re.finditer(func_pattern, code)])
            symbols.extend([f"class:{m.group(1)}" for m in re.finditer(class_pattern, code)])
            symbols.extend([f"var:{m.group(1)}" for m in re.finditer(var_pattern, code)])
        
        return symbols


class CodeUnitsV2BoundariesChunker(BaseCodeChunker):
    """
    Advanced chunker that respects AST boundaries with dynamic overlap
    Key innovation: AST-aligned boundaries with semantic overlap detection
    """
    
    def __init__(self, max_tokens: int = 512, overlap_strategy: str = "dynamic_ast"):
        super().__init__(max_tokens)
        self.overlap_strategy = overlap_strategy
        
    def chunk(self, code: str, file_path: str = None) -> List[CodeChunk]:
        """Chunk code using AST boundaries with intelligent overlap"""
        chunks = []
        
        try:
            # Parse AST for Python (extend for other languages)
            tree = ast.parse(code)
            statements = [node for node in tree.body]
            
            current_chunk = ""
            current_start_line = 1
            current_symbols = []
            
            for i, stmt in enumerate(statements):
                stmt_text = ast.get_source_segment(code, stmt) or self._fallback_extract(code, stmt)
                stmt_tokens = self._estimate_tokens(stmt_text)
                stmt_symbols = self._extract_symbols(stmt_text)
                
                # Check if adding this statement exceeds token limit
                if (self._estimate_tokens(current_chunk + stmt_text) > self.max_tokens 
                    and current_chunk.strip()):
                    
                    # Create chunk with current content
                    chunk = self._create_chunk(
                        current_chunk, current_start_line, stmt.lineno - 1,
                        current_symbols, ChunkBoundary.STATEMENT
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    overlap_content, overlap_symbols = self._compute_overlap(
                        statements, i, code
                    )
                    current_chunk = overlap_content + stmt_text
                    current_start_line = stmt.lineno
                    current_symbols = overlap_symbols + stmt_symbols
                else:
                    current_chunk += stmt_text + "\n"
                    current_symbols.extend(stmt_symbols)
            
            # Add final chunk
            if current_chunk.strip():
                final_chunk = self._create_chunk(
                    current_chunk, current_start_line, len(code.splitlines()),
                    current_symbols, ChunkBoundary.STATEMENT
                )
                chunks.append(final_chunk)
                
        except SyntaxError:
            # Fallback to line-based chunking for non-Python or invalid syntax
            chunks = self._fallback_line_chunking(code)
        
        # Post-process to add overlap information
        self._compute_overlap_metrics(chunks)
        return chunks
    
    def _compute_overlap(self, statements: List[ast.stmt], current_idx: int, code: str) -> Tuple[str, List[str]]:
        """Compute semantic overlap for chunk boundaries"""
        overlap_content = ""
        overlap_symbols = []
        
        if self.overlap_strategy == "dynamic_ast":
            # Look back for related statements (imports, class definitions, etc.)
            lookback_limit = max(1, int(len(statements) * 0.1))  # 10% lookback
            start_idx = max(0, current_idx - lookback_limit)
            
            for stmt in statements[start_idx:current_idx]:
                if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef)):
                    stmt_text = ast.get_source_segment(code, stmt) or ""
                    overlap_content += stmt_text + "\n"
                    overlap_symbols.extend(self._extract_symbols(stmt_text))
        
        elif self.overlap_strategy == "static_20pct":
            # Fixed 20% overlap with previous statements
            overlap_count = max(1, int(len(statements) * 0.2))
            start_idx = max(0, current_idx - overlap_count)
            
            for stmt in statements[start_idx:current_idx]:
                stmt_text = ast.get_source_segment(code, stmt) or ""
                overlap_content += stmt_text + "\n"
                overlap_symbols.extend(self._extract_symbols(stmt_text))
        
        return overlap_content, overlap_symbols
    
    def _create_chunk(self, content: str, start_line: int, end_line: int, 
                     symbols: List[str], boundary_type: ChunkBoundary) -> CodeChunk:
        """Create a CodeChunk with metadata"""
        return CodeChunk(
            content=content.strip(),
            start_line=start_line,
            end_line=end_line,
            start_char=0,  # Would need more sophisticated calculation
            end_char=len(content),
            boundary_type=boundary_type,
            token_count=self._estimate_tokens(content),
            symbols=symbols
        )
    
    def _fallback_extract(self, code: str, stmt: ast.stmt) -> str:
        """Fallback method to extract statement text"""
        lines = code.splitlines()
        if hasattr(stmt, 'lineno') and hasattr(stmt, 'end_lineno'):
            start = stmt.lineno - 1
            end = stmt.end_lineno if stmt.end_lineno else start + 1
            return "\n".join(lines[start:end])
        return ""
    
    def _fallback_line_chunking(self, code: str) -> List[CodeChunk]:
        """Fallback to simple line-based chunking"""
        lines = code.splitlines()
        chunks = []
        current_chunk = ""
        start_line = 1
        
        for i, line in enumerate(lines, 1):
            if self._estimate_tokens(current_chunk + line) > self.max_tokens and current_chunk.strip():
                chunk = CodeChunk(
                    content=current_chunk.strip(),
                    start_line=start_line,
                    end_line=i-1,
                    start_char=0,
                    end_char=len(current_chunk),
                    boundary_type=ChunkBoundary.STATEMENT,
                    token_count=self._estimate_tokens(current_chunk)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_lines = max(1, int(self.overlap_pct * 20))  # ~20% line overlap
                overlap_start = max(0, i - overlap_lines - 1)
                current_chunk = "\n".join(lines[overlap_start:i-1]) + "\n" + line
                start_line = overlap_start + 1
            else:
                current_chunk += line + "\n"
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(CodeChunk(
                content=current_chunk.strip(),
                start_line=start_line,
                end_line=len(lines),
                start_char=0,
                end_char=len(current_chunk),
                boundary_type=ChunkBoundary.STATEMENT,
                token_count=self._estimate_tokens(current_chunk)
            ))
        
        return chunks
    
    def _compute_overlap_metrics(self, chunks: List[CodeChunk]):
        """Compute overlap percentages between adjacent chunks"""
        for i in range(len(chunks)):
            if i > 0:
                prev_chunk = chunks[i-1]
                current_chunk = chunks[i]
                
                # Simple overlap detection based on common lines
                prev_lines = set(prev_chunk.content.splitlines())
                current_lines = set(current_chunk.content.splitlines())
                common_lines = prev_lines.intersection(current_lines)
                
                if current_lines:
                    current_chunk.overlap_with_prev = len(common_lines) / len(current_lines)
                if prev_lines:
                    prev_chunk.overlap_with_next = len(common_lines) / len(prev_lines)


class FnScopeChunker(BaseCodeChunker):
    """
    Function-scope based chunker that treats each function as a semantic unit
    Key innovation: Preserves function boundaries while managing size constraints
    """
    
    def __init__(self, max_tokens: int = 768, include_context: bool = True):
        super().__init__(max_tokens)
        self.include_context = include_context
    
    def chunk(self, code: str, file_path: str = None) -> List[CodeChunk]:
        """Chunk code by function scope"""
        chunks = []
        
        try:
            tree = ast.parse(code)
            
            # Extract top-level imports and globals as context
            context_nodes = []
            function_nodes = []
            class_nodes = []
            
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    context_nodes.append(node)
                elif isinstance(node, ast.FunctionDef):
                    function_nodes.append(node)
                elif isinstance(node, ast.ClassDef):
                    class_nodes.append(node)
                else:
                    context_nodes.append(node)
            
            # Build context string if needed
            context_content = ""
            if self.include_context and context_nodes:
                for node in context_nodes:
                    context_content += (ast.get_source_segment(code, node) or "") + "\n"
            
            # Process functions
            for func_node in function_nodes:
                func_content = ast.get_source_segment(code, func_node) or ""
                full_content = context_content + func_content if self.include_context else func_content
                
                if self._estimate_tokens(full_content) <= self.max_tokens:
                    # Function fits in one chunk
                    chunk = CodeChunk(
                        content=full_content.strip(),
                        start_line=func_node.lineno,
                        end_line=func_node.end_lineno or func_node.lineno,
                        start_char=0,
                        end_char=len(full_content),
                        boundary_type=ChunkBoundary.FUNCTION,
                        scope_info={'function_name': func_node.name, 'type': 'function'},
                        token_count=self._estimate_tokens(full_content),
                        symbols=self._extract_symbols(func_content)
                    )
                    chunks.append(chunk)
                else:
                    # Function too large, split into sub-chunks
                    sub_chunks = self._split_large_function(func_node, code, context_content)
                    chunks.extend(sub_chunks)
            
            # Process classes
            for class_node in class_nodes:
                class_chunks = self._process_class(class_node, code, context_content)
                chunks.extend(class_chunks)
            
        except SyntaxError:
            # Fallback to pattern-based function detection
            chunks = self._fallback_function_chunking(code)
        
        return chunks
    
    def _split_large_function(self, func_node: ast.FunctionDef, code: str, context: str) -> List[CodeChunk]:
        """Split large functions into smaller semantic chunks"""
        chunks = []
        func_content = ast.get_source_segment(code, func_node) or ""
        func_lines = func_content.splitlines()
        
        # Try to split by logical blocks (if/for/while/try blocks)
        current_chunk = context
        current_start = func_node.lineno
        indent_stack = []
        
        for i, line in enumerate(func_lines):
            stripped = line.strip()
            if stripped:
                indent = len(line) - len(line.lstrip())
                
                # Detect block boundaries
                if any(stripped.startswith(keyword) for keyword in 
                       ['if ', 'for ', 'while ', 'try:', 'except:', 'finally:', 'with ', 'def ']):
                    
                    # Check if adding this block would exceed token limit
                    if (self._estimate_tokens(current_chunk + line) > self.max_tokens 
                        and current_chunk.strip() != context.strip()):
                        
                        # Create chunk for previous content
                        chunk = CodeChunk(
                            content=current_chunk.strip(),
                            start_line=current_start,
                            end_line=func_node.lineno + i - 1,
                            start_char=0,
                            end_char=len(current_chunk),
                            boundary_type=ChunkBoundary.FUNCTION,
                            scope_info={
                                'function_name': func_node.name, 
                                'type': 'function_fragment'
                            },
                            token_count=self._estimate_tokens(current_chunk)
                        )
                        chunks.append(chunk)
                        
                        # Start new chunk
                        current_chunk = context + line + "\n"
                        current_start = func_node.lineno + i
                    else:
                        current_chunk += line + "\n"
                else:
                    current_chunk += line + "\n"
        
        # Add final chunk
        if current_chunk.strip() != context.strip():
            final_chunk = CodeChunk(
                content=current_chunk.strip(),
                start_line=current_start,
                end_line=func_node.end_lineno or func_node.lineno,
                start_char=0,
                end_char=len(current_chunk),
                boundary_type=ChunkBoundary.FUNCTION,
                scope_info={'function_name': func_node.name, 'type': 'function_fragment'},
                token_count=self._estimate_tokens(current_chunk)
            )
            chunks.append(final_chunk)
        
        return chunks
    
    def _process_class(self, class_node: ast.ClassDef, code: str, context: str) -> List[CodeChunk]:
        """Process class definitions"""
        chunks = []
        class_content = ast.get_source_segment(code, class_node) or ""
        
        # Include class header and methods separately
        class_header = f"class {class_node.name}"
        if class_node.bases:
            class_header += f"({', '.join(ast.get_source_segment(code, base) or '' for base in class_node.bases)})"
        class_header += ":"
        
        # Process class methods
        for method in class_node.body:
            if isinstance(method, ast.FunctionDef):
                method_content = ast.get_source_segment(code, method) or ""
                full_content = context + class_header + "\n" + method_content
                
                chunk = CodeChunk(
                    content=full_content.strip(),
                    start_line=method.lineno,
                    end_line=method.end_lineno or method.lineno,
                    start_char=0,
                    end_char=len(full_content),
                    boundary_type=ChunkBoundary.FUNCTION,
                    scope_info={
                        'class_name': class_node.name,
                        'method_name': method.name,
                        'type': 'method'
                    },
                    token_count=self._estimate_tokens(full_content),
                    symbols=[f"class:{class_node.name}", f"method:{method.name}"]
                )
                chunks.append(chunk)
        
        return chunks
    
    def _fallback_function_chunking(self, code: str) -> List[CodeChunk]:
        """Fallback function chunking using regex patterns"""
        chunks = []
        
        # Pattern for function definitions (multi-language)
        func_patterns = [
            r'^\s*def\s+(\w+)\s*\([^)]*\)\s*:',  # Python
            r'^\s*function\s+(\w+)\s*\([^)]*\)\s*{',  # JavaScript
            r'^\s*(?:public|private|protected)?\s*\w+\s+(\w+)\s*\([^)]*\)\s*{',  # Java/C#
        ]
        
        lines = code.splitlines()
        current_func = ""
        current_start = 1
        in_function = False
        brace_count = 0
        func_name = ""
        
        for i, line in enumerate(lines, 1):
            # Check for function start
            for pattern in func_patterns:
                match = re.search(pattern, line)
                if match:
                    # Save previous function if exists
                    if in_function and current_func.strip():
                        chunk = CodeChunk(
                            content=current_func.strip(),
                            start_line=current_start,
                            end_line=i-1,
                            start_char=0,
                            end_char=len(current_func),
                            boundary_type=ChunkBoundary.FUNCTION,
                            scope_info={'function_name': func_name, 'type': 'function'},
                            token_count=self._estimate_tokens(current_func)
                        )
                        chunks.append(chunk)
                    
                    # Start new function
                    current_func = line + "\n"
                    current_start = i
                    in_function = True
                    func_name = match.group(1)
                    brace_count = line.count('{') - line.count('}')
                    break
            else:
                if in_function:
                    current_func += line + "\n"
                    brace_count += line.count('{') - line.count('}')
                    
                    # Check for function end (simplified)
                    if (brace_count <= 0 and '{' in line) or (line.strip() == '' and brace_count == 0):
                        in_function = False
        
        # Add final function
        if in_function and current_func.strip():
            chunk = CodeChunk(
                content=current_func.strip(),
                start_line=current_start,
                end_line=len(lines),
                start_char=0,
                end_char=len(current_func),
                boundary_type=ChunkBoundary.FUNCTION,
                scope_info={'function_name': func_name, 'type': 'function'},
                token_count=self._estimate_tokens(current_func)
            )
            chunks.append(chunk)
        
        return chunks


class HybridASTChunker(BaseCodeChunker):
    """
    Hybrid chunker that combines AST analysis with semantic boundary detection
    Key innovation: Adapts chunking strategy based on code structure and complexity
    """
    
    def __init__(self, max_tokens: int = 512, adaptive_threshold: float = 0.7):
        super().__init__(max_tokens)
        self.adaptive_threshold = adaptive_threshold
        self.ast_chunker = CodeUnitsV2BoundariesChunker(max_tokens)
        self.fn_chunker = FnScopeChunker(max_tokens)
    
    def chunk(self, code: str, file_path: str = None) -> List[CodeChunk]:
        """Adaptively choose chunking strategy based on code characteristics"""
        
        # Analyze code structure
        analysis = self._analyze_code_structure(code)
        
        # Choose strategy based on analysis
        if analysis['function_density'] > self.adaptive_threshold:
            # High function density -> use function-scope chunking
            chunks = self.fn_chunker.chunk(code, file_path)
            for chunk in chunks:
                chunk.scope_info = chunk.scope_info or {}
                chunk.scope_info['strategy'] = 'function_focused'
        elif analysis['complexity_score'] > self.adaptive_threshold:
            # High complexity -> use AST-based chunking  
            chunks = self.ast_chunker.chunk(code, file_path)
            for chunk in chunks:
                chunk.scope_info = {'strategy': 'ast_boundaries'}
        else:
            # Mixed approach -> combine both
            chunks = self._hybrid_approach(code, file_path, analysis)
        
        # Post-process to add hybrid metadata
        for chunk in chunks:
            chunk.scope_info = chunk.scope_info or {}
            chunk.scope_info['code_analysis'] = analysis
        
        return chunks
    
    def _analyze_code_structure(self, code: str) -> Dict[str, float]:
        """Analyze code structure to determine optimal chunking strategy"""
        analysis = {
            'function_density': 0.0,
            'class_density': 0.0, 
            'complexity_score': 0.0,
            'avg_function_size': 0.0,
            'import_ratio': 0.0
        }
        
        lines = code.splitlines()
        total_lines = len(lines)
        
        if total_lines == 0:
            return analysis
        
        try:
            tree = ast.parse(code)
            
            # Count different elements
            functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
            
            # Calculate densities
            analysis['function_density'] = len(functions) / max(1, total_lines / 10)  # functions per 10 lines
            analysis['class_density'] = len(classes) / max(1, total_lines / 20)      # classes per 20 lines
            analysis['import_ratio'] = len(imports) / max(1, len(tree.body))         # import ratio
            
            # Calculate average function size
            if functions:
                func_sizes = []
                for func in functions:
                    if hasattr(func, 'end_lineno') and func.end_lineno:
                        size = func.end_lineno - func.lineno + 1
                        func_sizes.append(size)
                
                if func_sizes:
                    analysis['avg_function_size'] = sum(func_sizes) / len(func_sizes)
            
            # Calculate complexity score (simplified)
            complexity_nodes = [n for n in ast.walk(tree) if isinstance(n, (
                ast.If, ast.While, ast.For, ast.Try, ast.With, ast.Lambda, ast.ListComp, ast.DictComp
            ))]
            analysis['complexity_score'] = len(complexity_nodes) / max(1, total_lines / 5)
            
        except:
            # Fallback to regex-based analysis
            func_count = len(re.findall(r'^\s*def\s+\w+', code, re.MULTILINE))
            class_count = len(re.findall(r'^\s*class\s+\w+', code, re.MULTILINE))
            import_count = len(re.findall(r'^\s*(?:import|from)\s+', code, re.MULTILINE))
            
            analysis['function_density'] = func_count / max(1, total_lines / 10)
            analysis['class_density'] = class_count / max(1, total_lines / 20)
            analysis['import_ratio'] = import_count / max(1, total_lines / 10)
        
        return analysis
    
    def _hybrid_approach(self, code: str, file_path: str, analysis: Dict[str, float]) -> List[CodeChunk]:
        """Use a hybrid of both approaches based on code characteristics"""
        
        # Get chunks from both approaches
        ast_chunks = self.ast_chunker.chunk(code, file_path)
        fn_chunks = self.fn_chunker.chunk(code, file_path)
        
        # Merge strategies: use function chunks for clear functions, AST for other parts
        hybrid_chunks = []
        covered_lines = set()
        
        # First pass: add clear function chunks
        for chunk in fn_chunks:
            if (chunk.boundary_type == ChunkBoundary.FUNCTION and 
                chunk.scope_info and 
                chunk.scope_info.get('type') == 'function'):
                
                hybrid_chunks.append(chunk)
                for line_num in range(chunk.start_line, chunk.end_line + 1):
                    covered_lines.add(line_num)
        
        # Second pass: fill gaps with AST chunks
        for chunk in ast_chunks:
            chunk_lines = set(range(chunk.start_line, chunk.end_line + 1))
            
            # If this chunk covers mostly uncovered lines, include it
            uncovered_lines = chunk_lines - covered_lines
            if len(uncovered_lines) / len(chunk_lines) > 0.5:  # >50% uncovered
                chunk.scope_info = {'strategy': 'ast_gap_fill'}
                hybrid_chunks.append(chunk)
                covered_lines.update(chunk_lines)
        
        # Sort by start line
        hybrid_chunks.sort(key=lambda c: c.start_line)
        
        return hybrid_chunks


# Factory function for easy chunker selection
def create_chunker(chunker_type: str, max_tokens: int = 512, **kwargs) -> BaseCodeChunker:
    """Factory function to create chunkers"""
    
    chunkers = {
        'code_units_v2_boundaries': CodeUnitsV2BoundariesChunker,
        'fn_scope': FnScopeChunker,
        'hybrid_ast': HybridASTChunker
    }
    
    if chunker_type not in chunkers:
        raise ValueError(f"Unknown chunker type: {chunker_type}. Available: {list(chunkers.keys())}")
    
    return chunkers[chunker_type](max_tokens=max_tokens, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test with sample Python code
    sample_code = '''
import os
import sys
from typing import List, Dict

class CodeProcessor:
    def __init__(self, config: Dict):
        self.config = config
    
    def process_file(self, file_path: str) -> List[str]:
        """Process a single file and return results"""
        with open(file_path, 'r') as f:
            content = f.read()
        
        results = []
        for line in content.splitlines():
            if line.strip():
                processed = self.process_line(line)
                results.append(processed)
        
        return results
    
    def process_line(self, line: str) -> str:
        """Process a single line"""
        if line.startswith('#'):
            return f"Comment: {line[1:].strip()}"
        else:
            return f"Code: {line.strip()}"

def main():
    processor = CodeProcessor({'debug': True})
    results = processor.process_file('sample.py')
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
'''
    
    # Test all chunkers
    chunkers = [
        ('AST Boundaries', create_chunker('code_units_v2_boundaries', max_tokens=256)),
        ('Function Scope', create_chunker('fn_scope', max_tokens=256)),
        ('Hybrid', create_chunker('hybrid_ast', max_tokens=256))
    ]
    
    for name, chunker in chunkers:
        print(f"\n=== {name} Chunker ===")
        chunks = chunker.chunk(sample_code)
        
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1}:")
            print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
            print(f"  Tokens: {chunk.token_count}")
            print(f"  Boundary: {chunk.boundary_type.value}")
            print(f"  Symbols: {chunk.symbols[:3]}...")  # First 3 symbols
            if chunk.scope_info:
                print(f"  Scope: {chunk.scope_info}")
            print(f"  Preview: {chunk.content[:100]}...")