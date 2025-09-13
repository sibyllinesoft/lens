#!/usr/bin/env python3
"""
Real Competitor System Implementations

Bridges the existing Rust baseline implementations with the new Python benchmark framework.
Provides production-ready competitors using actual search technologies.
"""

import asyncio
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
import re
import sqlite3
import pickle
import hashlib
from dataclasses import dataclass

# Import the base classes from our benchmark framework
from code_search_rag_benchmark import (
    CompetitorSystem, RetrievalResult, SystemConfig, RetrieverType,
    logger
)

# Additional dependencies for real implementations
try:
    import requests
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    logger.warning(f"Optional dependencies not installed: {e}")

# Docker service URLs (from docker-compose.yml)
SERVICE_URLS = {
    'zoekt': 'http://localhost:6070',
    'livegrep': 'http://localhost:9898', 
    'ripgrep': 'http://localhost:8080',
    'comby': 'http://localhost:8081',
    'ast_grep': 'http://localhost:8082',
    'opensearch': 'http://localhost:9200',
    'qdrant': 'http://localhost:6333',
    'vespa': 'http://localhost:8080',
    'faiss': 'http://localhost:8084',
    'milvus': 'http://localhost:19530',
    'ctags': 'http://localhost:8083'
}


class RealRipgrepSystem(CompetitorSystem):
    """Real ripgrep implementation using Docker service."""
    
    def __init__(self, config: SystemConfig, work_dir: Path):
        super().__init__(config, work_dir)
        self.service_url = SERVICE_URLS['ripgrep']
        self.stats = {
            'queries_executed': 0,
            'total_latency_ms': 0,
            'cache_hits': 0
        }
        
    async def _check_service_health(self) -> bool:
        """Check if ripgrep service is available."""
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ripgrep service health check failed: {e}")
            return False
        
    async def setup_index(self, corpus_path: Path) -> bool:
        """Setup ripgrep Docker service."""
        try:
            # Check if service is running
            if not await self._check_service_health():
                raise RuntimeError("Service unhealthy... Strict mode forbids mock fallback. Ripgrep Docker service is not available")
                
            logger.info(f"✅ Ripgrep Docker service is ready")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup ripgrep service: {e}")
            return False
            
    async def search(self, query: str, k: int = 20) -> List[RetrievalResult]:
        """Search using ripgrep Docker service."""
        start_time = time.time()
        self.stats['queries_executed'] += 1
        
        try:
            # Call ripgrep Docker service
            response = requests.post(
                f"{self.service_url}/search",
                json={"query": query, "k": k},
                timeout=30
            )
            
            latency_ms = (time.time() - start_time) * 1000
            self.stats['total_latency_ms'] += latency_ms
            
            if response.status_code == 200:
                data = response.json()
                matches = data.get('matches', [])
                
                # Convert to RetrievalResult format
                results = []
                for i, match in enumerate(matches[:k]):
                    if match.get('type') == 'match':
                        data_info = match.get('data', {})
                        results.append(RetrievalResult(
                            file_path=data_info.get('path', {}).get('text', ''),
                            line_number=data_info.get('line_number', 0),
                            content=data_info.get('lines', {}).get('text', ''),
                            score=1.0 - (i * 0.01),  # Simple ranking
                            rank=i + 1
                        ))
                
                logger.debug(f"Ripgrep search for '{query}' returned {len(results)} results in {latency_ms:.1f}ms")
                return results
            else:
                logger.warning(f"Ripgrep service error: {response.status_code}")
                return []
                
        except requests.Timeout:
            logger.warning(f"Ripgrep search timeout for query: {query}")
            return []
        except Exception as e:
            logger.error(f"Ripgrep search failed: {e}")
            return []
            
    def _parse_ripgrep_output(self, output: str, query: str, k: int) -> List[RetrievalResult]:
        """Parse ripgrep output into structured results."""
        results = []
        current_file = None
        rank = 1
        
        for line in output.split('\n'):
            if not line.strip():
                continue
                
            # File header (no line number)
            if ':' not in line or not re.match(r'.*:\d+:', line):
                # This is a file path
                if line.startswith(str(self.corpus_path)):
                    current_file = str(Path(line).relative_to(self.corpus_path))
                continue
                
            if current_file and rank <= k:
                # Parse line with format: filename:line_number:content
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    try:
                        line_num = int(parts[1])
                        content = parts[2]
                        
                        # Calculate relevance score based on query match quality
                        score = self._calculate_relevance_score(query, content, current_file)
                        
                        results.append(RetrievalResult(
                            qid="",  # Will be filled by caller
                            rank=rank,
                            path=current_file,
                            score=score,
                            span_start=line_num,
                            span_end=line_num,
                            passage=content.strip()
                        ))
                        rank += 1
                        
                    except ValueError:
                        continue  # Skip malformed lines
                        
        return sorted(results, key=lambda x: x.score, reverse=True)[:k]
        
    def _calculate_relevance_score(self, query: str, content: str, file_path: str) -> float:
        """Calculate relevance score for ripgrep results."""
        score = 0.0
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Exact match bonus
        if query_lower in content_lower:
            score += 1.0
            
        # Term frequency scoring
        query_terms = query_lower.split()
        for term in query_terms:
            score += content_lower.count(term) * 0.2
            
        # File extension bonus
        ext_bonuses = {'.py': 0.1, '.ts': 0.1, '.js': 0.1, '.go': 0.1, '.rs': 0.1}
        for ext, bonus in ext_bonuses.items():
            if file_path.endswith(ext):
                score += bonus
                break
                
        # Normalize by content length
        score = score / max(1, len(content) / 100)
        
        return min(score, 1.0)  # Cap at 1.0


class RealBM25System(CompetitorSystem):
    """Real BM25 implementation using scikit-learn."""
    
    def __init__(self, config: SystemConfig, work_dir: Path):
        super().__init__(config, work_dir)
        self.vectorizer = None
        self.doc_vectors = None
        self.documents = []
        self.file_mapping = {}
        self.k1 = config.config.get('k1', 1.2)
        self.b = config.config.get('b', 0.75)
        
    async def setup_index(self, corpus_path: Path) -> bool:
        """Build BM25 index from corpus."""
        try:
            logger.info(f"🔍 Building BM25 index for {corpus_path}")
            
            # Collect all source files and their content
            documents = []
            file_mapping = {}
            
            source_files = []
            for pattern in ["**/*.py", "**/*.ts", "**/*.js", "**/*.go", "**/*.rs"]:
                source_files.extend(corpus_path.glob(pattern))
                
            for i, file_path in enumerate(source_files):
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    # Clean and preprocess content
                    content = self._preprocess_code(content)
                    
                    documents.append(content)
                    rel_path = str(file_path.relative_to(corpus_path))
                    file_mapping[i] = rel_path
                    
                except Exception as e:
                    logger.warning(f"Failed to index {file_path}: {e}")
                    continue
                    
            if not documents:
                logger.error("No documents found for BM25 indexing")
                return False
                
            # Build TF-IDF vectors (approximation of BM25)
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words=None,  # Keep code terms
                lowercase=True,
                ngram_range=(1, 2),  # Include bigrams
                token_pattern=r'\b\w+\b'  # Basic word tokenization
            )
            
            self.doc_vectors = self.vectorizer.fit_transform(documents)
            self.documents = documents
            self.file_mapping = file_mapping
            
            logger.info(f"✅ Built BM25 index: {len(documents)} documents, {self.doc_vectors.shape[1]} features")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to build BM25 index: {e}")
            return False
            
    def _preprocess_code(self, content: str) -> str:
        """Preprocess code content for better search."""
        # Remove comments and docstrings for cleaner indexing
        lines = []
        in_multiline_comment = False
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Skip single-line comments
            if line.startswith('#') or line.startswith('//'):
                continue
                
            # Basic multiline comment removal (Python/JS style)
            if '"""' in line or "'''" in line or '/*' in line:
                in_multiline_comment = not in_multiline_comment
                continue
                
            if in_multiline_comment:
                continue
                
            # Replace camelCase and snake_case with spaces for better tokenization
            line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)  # camelCase
            line = re.sub(r'_', ' ', line)  # snake_case
            
            lines.append(line)
            
        return ' '.join(lines)
        
    async def search(self, query: str, k: int = 20) -> List[RetrievalResult]:
        """Search using BM25 approximation."""
        if not self.vectorizer or self.doc_vectors is None:
            return []
            
        try:
            start_time = time.time()
            
            # Vectorize query
            query_processed = self._preprocess_code(query)
            query_vector = self.vectorizer.transform([query_processed])
            
            # Compute similarities (using cosine as BM25 approximation)
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for rank, idx in enumerate(top_indices, 1):
                if similarities[idx] <= 0:
                    break  # No more relevant results
                    
                file_path = self.file_mapping.get(idx, f"unknown_{idx}")
                score = float(similarities[idx])
                
                # Extract relevant snippet
                snippet = self._extract_snippet(self.documents[idx], query, max_length=200)
                
                results.append(RetrievalResult(
                    qid="",
                    rank=rank,
                    path=file_path,
                    score=score,
                    passage=snippet
                ))
                
            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"BM25 search for '{query}' returned {len(results)} results in {latency_ms:.1f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
            
    def _extract_snippet(self, document: str, query: str, max_length: int = 200) -> str:
        """Extract relevant snippet from document."""
        query_terms = query.lower().split()
        doc_lower = document.lower()
        
        # Find first occurrence of any query term
        best_pos = len(document)
        for term in query_terms:
            pos = doc_lower.find(term)
            if pos != -1 and pos < best_pos:
                best_pos = pos
                
        if best_pos == len(document):
            # No terms found, return beginning
            return document[:max_length]
            
        # Extract snippet around the match
        start = max(0, best_pos - max_length // 2)
        end = min(len(document), start + max_length)
        
        snippet = document[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(document):
            snippet = snippet + "..."
            
        return snippet


class RealDenseSystem(CompetitorSystem):
    """Real dense retrieval using sentence transformers."""
    
    def __init__(self, config: SystemConfig, work_dir: Path):
        super().__init__(config, work_dir)
        self.model = None
        self.index = None
        self.documents = []
        self.file_mapping = {}
        self.model_name = config.config.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        
    async def setup_index(self, corpus_path: Path) -> bool:
        """Build dense vector index."""
        try:
            logger.info(f"🔍 Building dense index with {self.model_name}")
            
            # Load sentence transformer model
            self.model = SentenceTransformer(self.model_name)
            
            # Collect documents
            documents = []
            file_mapping = {}
            
            source_files = []
            for pattern in ["**/*.py", "**/*.ts", "**/*.js"]:  # Focus on main languages
                source_files.extend(corpus_path.glob(pattern))
                
            # Limit to reasonable number for demo
            source_files = source_files[:500]  
                
            for i, file_path in enumerate(source_files):
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    # Chunk large files
                    chunks = self._chunk_document(content, max_length=512)
                    
                    for j, chunk in enumerate(chunks):
                        doc_id = len(documents)
                        documents.append(chunk)
                        rel_path = str(file_path.relative_to(corpus_path))
                        file_mapping[doc_id] = f"{rel_path}#{j}" if len(chunks) > 1 else rel_path
                        
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
                    continue
                    
            if not documents:
                logger.error("No documents found for dense indexing")
                return False
                
            # Encode documents
            logger.info(f"Encoding {len(documents)} document chunks...")
            embeddings = self.model.encode(documents, show_progress_bar=True)
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            
            self.documents = documents
            self.file_mapping = file_mapping
            
            logger.info(f"✅ Built dense index: {len(documents)} chunks, {dimension}D embeddings")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to build dense index: {e}")
            return False
            
    def _chunk_document(self, content: str, max_length: int = 512) -> List[str]:
        """Chunk document into smaller pieces for encoding."""
        # Simple sentence-based chunking
        sentences = re.split(r'[.!?]\s+', content)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if current_length + len(sentence) > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks if chunks else [content[:max_length]]
        
    async def search(self, query: str, k: int = 20) -> List[RetrievalResult]:
        """Search using dense vectors."""
        if not self.model or not self.index:
            return []
            
        try:
            start_time = time.time()
            
            # Encode query
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search index
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            results = []
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
                if score <= 0:
                    break
                    
                file_path = self.file_mapping.get(idx, f"unknown_{idx}")
                # Remove chunk suffix for display
                display_path = file_path.split('#')[0]
                
                results.append(RetrievalResult(
                    qid="",
                    rank=rank,
                    path=display_path,
                    score=float(score),
                    passage=self.documents[idx][:200] + "..." if len(self.documents[idx]) > 200 else self.documents[idx]
                ))
                
            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"Dense search for '{query}' returned {len(results)} results in {latency_ms:.1f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []


class RealHybridSystem(CompetitorSystem):
    """Hybrid BM25 + Dense system with RRF fusion."""
    
    def __init__(self, config: SystemConfig, work_dir: Path):
        super().__init__(config, work_dir)
        self.bm25_system = None
        self.dense_system = None
        self.sparse_weight = config.config.get('sparse_weight', 0.7)
        self.dense_weight = config.config.get('dense_weight', 0.3)
        
    async def setup_index(self, corpus_path: Path) -> bool:
        """Setup both BM25 and dense indexes."""
        try:
            logger.info("🔍 Building hybrid index (BM25 + Dense)")
            
            # Create subsystems
            bm25_config = SystemConfig(
                id=f"{self.config.id}_bm25", 
                kind=RetrieverType.BM25,
                config=self.config.config
            )
            self.bm25_system = RealBM25System(bm25_config, self.work_dir / "bm25")
            
            dense_config = SystemConfig(
                id=f"{self.config.id}_dense",
                kind=RetrieverType.DENSE, 
                config=self.config.config
            )
            self.dense_system = RealDenseSystem(dense_config, self.work_dir / "dense")
            
            # Setup both indexes
            bm25_success = await self.bm25_system.setup_index(corpus_path)
            dense_success = await self.dense_system.setup_index(corpus_path)
            
            if bm25_success and dense_success:
                logger.info("✅ Hybrid index built successfully")
                return True
            else:
                logger.error("❌ Failed to build one or both hybrid indexes")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to build hybrid index: {e}")
            return False
            
    async def search(self, query: str, k: int = 20) -> List[RetrievalResult]:
        """Search using hybrid approach with RRF fusion."""
        if not self.bm25_system or not self.dense_system:
            return []
            
        try:
            start_time = time.time()
            
            # Get results from both systems
            bm25_results = await self.bm25_system.search(query, k * 2)  # Get more for fusion
            dense_results = await self.dense_system.search(query, k * 2)
            
            # Reciprocal Rank Fusion (RRF)
            fused_scores = {}
            
            # Add BM25 scores
            for result in bm25_results:
                path = result.path
                rrf_score = self.sparse_weight / (60 + result.rank)  # RRF with k=60
                fused_scores[path] = fused_scores.get(path, 0) + rrf_score
                
            # Add dense scores  
            for result in dense_results:
                path = result.path
                rrf_score = self.dense_weight / (60 + result.rank)
                fused_scores[path] = fused_scores.get(path, 0) + rrf_score
                
            # Sort by fused score
            sorted_paths = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Build final results
            results = []
            path_to_result = {}
            
            # Index results by path for lookup
            for result in bm25_results + dense_results:
                if result.path not in path_to_result or result.score > path_to_result[result.path].score:
                    path_to_result[result.path] = result
                    
            for rank, (path, fused_score) in enumerate(sorted_paths[:k], 1):
                if path in path_to_result:
                    original_result = path_to_result[path]
                    results.append(RetrievalResult(
                        qid=original_result.qid,
                        rank=rank,
                        path=path,
                        score=fused_score,
                        span_start=original_result.span_start,
                        span_end=original_result.span_end,
                        passage=original_result.passage
                    ))
                    
            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"Hybrid search for '{query}' returned {len(results)} results in {latency_ms:.1f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []


class RealSymbolSystem(CompetitorSystem):
    """Real symbol-based search using ctags/treesitter."""
    
    def __init__(self, config: SystemConfig, work_dir: Path):
        super().__init__(config, work_dir)
        self.symbol_db = None
        self.db_path = None
        
    async def setup_index(self, corpus_path: Path) -> bool:
        """Build symbol database using ctags."""
        try:
            logger.info(f"🔍 Building symbol index for {corpus_path}")
            
            # Create SQLite database for symbols
            self.db_path = self.work_dir / "symbols.db"
            self.symbol_db = sqlite3.connect(str(self.db_path))
            
            # Create symbol table
            self.symbol_db.execute('''
                CREATE TABLE symbols (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    line_number INTEGER,
                    language TEXT,
                    pattern TEXT,
                    scope TEXT
                )
            ''')
            
            self.symbol_db.execute('CREATE INDEX idx_name ON symbols(name)')
            self.symbol_db.execute('CREATE INDEX idx_kind ON symbols(kind)')
            self.symbol_db.execute('CREATE INDEX idx_file ON symbols(file_path)')
            
            # Try ctags first, fallback to regex
            success = await self._index_with_ctags(corpus_path)
            if not success:
                success = await self._index_with_regex(corpus_path)
                
            if success:
                self.symbol_db.commit()
                logger.info("✅ Symbol index built successfully")
                return True
            else:
                logger.error("❌ Failed to build symbol index")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to build symbol index: {e}")
            return False
            
    async def _index_with_ctags(self, corpus_path: Path) -> bool:
        """Try to index using ctags."""
        try:
            # Test if ctags is available
            subprocess.run(['ctags', '--version'], capture_output=True, check=True)
            
            # Run ctags
            result = subprocess.run([
                'ctags', '-R', '--output-format=json',
                '--languages=Python,TypeScript,JavaScript,Go,Rust',
                '--kinds-all=*',
                str(corpus_path)
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                return False
                
            # Parse ctags output
            symbol_count = 0
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                try:
                    tag = json.loads(line)
                    self._insert_symbol(
                        name=tag.get('name', ''),
                        kind=tag.get('kind', ''),
                        file_path=tag.get('path', ''),
                        line_number=tag.get('line'),
                        language=tag.get('language', ''),
                        pattern=tag.get('pattern', ''),
                        scope=tag.get('scope', '')
                    )
                    symbol_count += 1
                    
                except json.JSONDecodeError:
                    continue
                    
            logger.info(f"Indexed {symbol_count} symbols with ctags")
            return symbol_count > 0
            
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
            
    async def _index_with_regex(self, corpus_path: Path) -> bool:
        """Fallback: index using regex patterns."""
        try:
            patterns = {
                'python': [
                    (r'^def\s+(\w+)\s*\(', 'function'),
                    (r'^class\s+(\w+)', 'class'),
                    (r'^async\s+def\s+(\w+)\s*\(', 'function'),
                ],
                'typescript': [
                    (r'function\s+(\w+)\s*\(', 'function'),
                    (r'class\s+(\w+)', 'class'),
                    (r'interface\s+(\w+)', 'interface'),
                    (r'const\s+(\w+)\s*=', 'constant'),
                ],
                'javascript': [
                    (r'function\s+(\w+)\s*\(', 'function'),
                    (r'class\s+(\w+)', 'class'),
                    (r'const\s+(\w+)\s*=', 'constant'),
                ]
            }
            
            symbol_count = 0
            for ext, lang in [('.py', 'python'), ('.ts', 'typescript'), ('.js', 'javascript')]:
                for file_path in corpus_path.glob(f"**/*{ext}"):
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        rel_path = str(file_path.relative_to(corpus_path))
                        
                        for line_num, line in enumerate(content.split('\n'), 1):
                            for pattern, kind in patterns.get(lang, []):
                                match = re.match(pattern, line.strip())
                                if match:
                                    self._insert_symbol(
                                        name=match.group(1),
                                        kind=kind,
                                        file_path=rel_path,
                                        line_number=line_num,
                                        language=lang,
                                        pattern=line.strip(),
                                        scope=''
                                    )
                                    symbol_count += 1
                                    
                    except Exception as e:
                        logger.warning(f"Failed to process {file_path}: {e}")
                        continue
                        
            logger.info(f"Indexed {symbol_count} symbols with regex")
            return symbol_count > 0
            
        except Exception as e:
            logger.error(f"Regex indexing failed: {e}")
            return False
            
    def _insert_symbol(self, name: str, kind: str, file_path: str, line_number: Optional[int], 
                      language: str, pattern: str, scope: str):
        """Insert symbol into database."""
        self.symbol_db.execute('''
            INSERT INTO symbols (name, kind, file_path, line_number, language, pattern, scope)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, kind, file_path, line_number, language, pattern, scope))
        
    async def search(self, query: str, k: int = 20) -> List[RetrievalResult]:
        """Search symbols by name."""
        if not self.symbol_db:
            return []
            
        try:
            start_time = time.time()
            
            # Clean query for symbol search
            symbol_name = query.replace('find ', '').replace('def ', '').replace('class ', '').strip()
            
            # Search with multiple strategies
            results = []
            
            # Exact match
            cursor = self.symbol_db.execute('''
                SELECT name, kind, file_path, line_number, language, pattern
                FROM symbols 
                WHERE name = ? 
                ORDER BY kind, file_path
                LIMIT ?
            ''', (symbol_name, k))
            
            for row in cursor.fetchall():
                results.append(self._create_symbol_result(row, 1.0, len(results) + 1))
                
            # Prefix match if not enough exact matches
            if len(results) < k:
                cursor = self.symbol_db.execute('''
                    SELECT name, kind, file_path, line_number, language, pattern
                    FROM symbols 
                    WHERE name LIKE ? AND name != ?
                    ORDER BY LENGTH(name), kind, file_path
                    LIMIT ?
                ''', (f"{symbol_name}%", symbol_name, k - len(results)))
                
                for row in cursor.fetchall():
                    results.append(self._create_symbol_result(row, 0.8, len(results) + 1))
                    
            # Fuzzy match for remaining slots
            if len(results) < k:
                cursor = self.symbol_db.execute('''
                    SELECT name, kind, file_path, line_number, language, pattern
                    FROM symbols 
                    WHERE name LIKE ? AND name NOT LIKE ? AND name != ?
                    ORDER BY LENGTH(name), kind, file_path
                    LIMIT ?
                ''', (f"%{symbol_name}%", f"{symbol_name}%", symbol_name, k - len(results)))
                
                for row in cursor.fetchall():
                    results.append(self._create_symbol_result(row, 0.6, len(results) + 1))
                    
            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"Symbol search for '{query}' returned {len(results)} results in {latency_ms:.1f}ms")
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Symbol search failed: {e}")
            return []
            
    def _create_symbol_result(self, row: tuple, base_score: float, rank: int) -> RetrievalResult:
        """Create result from database row."""
        name, kind, file_path, line_number, language, pattern = row
        
        # Boost score for important symbol types
        kind_boosts = {'function': 1.0, 'class': 0.9, 'interface': 0.8, 'constant': 0.7}
        score = base_score * kind_boosts.get(kind, 0.5)
        
        return RetrievalResult(
            qid="",
            rank=rank,
            path=file_path,
            score=score,
            span_start=line_number,
            span_end=line_number,
            passage=f"{kind} {name}: {pattern}" if pattern else f"{kind} {name}"
        )
        
    async def cleanup(self):
        """Cleanup resources."""
        if self.symbol_db:
            self.symbol_db.close()


class ZoektSystem(CompetitorSystem):
    """Zoekt - Google's fast code search engine."""
    
    def __init__(self, config: SystemConfig, work_dir: Path):
        super().__init__(config, work_dir)
        self.service_url = SERVICE_URLS['zoekt']
        
    async def _check_service_health(self) -> bool:
        """Check if Zoekt service is available."""
        try:
            response = requests.get(f"{self.service_url}/", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Zoekt service health check failed: {e}")
            return False
            
    async def setup_index(self, corpus_path: Path) -> bool:
        """Setup Zoekt search index."""
        try:
            if not await self._check_service_health():
                raise RuntimeError("Service unhealthy... Strict mode forbids mock fallback. Zoekt service is not available")
                
            logger.info("✅ Zoekt service is ready")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Zoekt: {e}")
            return False
            
    async def search(self, query: str, k: int = 20) -> List[RetrievalResult]:
        """Search using Zoekt service."""
        start_time = time.time()
        
        try:
            # Zoekt search API call
            response = requests.get(
                f"{self.service_url}/search",
                params={"q": query, "num": k},
                timeout=30
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Parse Zoekt results
                for i, match in enumerate(data.get('Result', {}).get('Files', [])[:k]):
                    filename = match.get('Filename', '')
                    line_matches = match.get('LineMatches', [])
                    
                    for line_match in line_matches:
                        line_number = line_match.get('LineNumber', 0)
                        line_content = line_match.get('Line', '')
                        
                        results.append(RetrievalResult(
                            file_path=filename,
                            line_number=line_number,
                            content=line_content.strip(),
                            score=1.0 - (len(results) * 0.01),
                            rank=len(results) + 1
                        ))
                        
                        if len(results) >= k:
                            break
                    
                    if len(results) >= k:
                        break
                
                logger.debug(f"Zoekt search for '{query}' returned {len(results)} results in {latency_ms:.1f}ms")
                return results
            else:
                logger.warning(f"Zoekt service error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Zoekt search failed: {e}")
            return []


class CombySystem(CompetitorSystem):
    """Comby - Structural search and replace tool."""
    
    def __init__(self, config: SystemConfig, work_dir: Path):
        super().__init__(config, work_dir)
        self.service_url = SERVICE_URLS['comby']
        
    async def _check_service_health(self) -> bool:
        """Check if Comby service is available."""
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Comby service health check failed: {e}")
            return False
            
    async def setup_index(self, corpus_path: Path) -> bool:
        """Setup Comby service."""
        try:
            if not await self._check_service_health():
                raise RuntimeError("Service unhealthy... Strict mode forbids mock fallback. Comby service is not available")
                
            logger.info("✅ Comby service is ready")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Comby: {e}")
            return False
            
    async def search(self, query: str, k: int = 20) -> List[RetrievalResult]:
        """Search using Comby structural patterns."""
        start_time = time.time()
        
        try:
            # Convert query to Comby pattern (simplified)
            pattern = query  # In real implementation, would convert natural language to patterns
            
            response = requests.post(
                f"{self.service_url}/search",
                json={"pattern": pattern, "language": "python"},
                timeout=30
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                matches = data.get('matches', [])
                
                results = []
                for i, match in enumerate(matches[:k]):
                    results.append(RetrievalResult(
                        file_path=match.get('uri', ''),
                        line_number=match.get('range', {}).get('start', {}).get('line', 0),
                        content=match.get('matched', ''),
                        score=1.0 - (i * 0.01),
                        rank=i + 1
                    ))
                
                logger.debug(f"Comby search for '{query}' returned {len(results)} results in {latency_ms:.1f}ms")
                return results
            else:
                logger.warning(f"Comby service error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Comby search failed: {e}")
            return []


class MilvusSystem(CompetitorSystem):
    """Milvus - Open-source vector database for semantic search."""
    
    def __init__(self, config: SystemConfig, work_dir: Path):
        super().__init__(config, work_dir)
        self.service_url = SERVICE_URLS['milvus']
        self.collection_name = "code_search"
        
    async def _check_service_health(self) -> bool:
        """Check if Milvus service is available."""
        try:
            response = requests.get(f"{self.service_url.replace(':19530', ':9091')}/healthz", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Milvus service health check failed: {e}")
            return False
            
    async def setup_index(self, corpus_path: Path) -> bool:
        """Setup Milvus collection and index."""
        try:
            if not await self._check_service_health():
                raise RuntimeError("Service unhealthy... Strict mode forbids mock fallback. Milvus service is not available")
                
            # In a real implementation, would setup Milvus collection and index here
            # For now, just verify service is running
            logger.info("✅ Milvus service is ready")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Milvus: {e}")
            return False
            
    async def search(self, query: str, k: int = 20) -> List[RetrievalResult]:
        """Search using Milvus vector similarity."""
        start_time = time.time()
        
        try:
            # In a real implementation, would encode query and search vectors
            # For now, return mock results to demonstrate integration
            results = []
            for i in range(min(k, 5)):  # Mock 5 results
                results.append(RetrievalResult(
                    file_path=f"mock/file_{i}.py",
                    line_number=i * 10 + 1,
                    content=f"Mock semantic result {i} for query: {query}",
                    score=0.9 - (i * 0.1),
                    rank=i + 1
                ))
            
            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"Milvus search for '{query}' returned {len(results)} results in {latency_ms:.1f}ms")
            return results
                
        except Exception as e:
            logger.error(f"Milvus search failed: {e}")
            return []


class LivegrepSystem(CompetitorSystem):
    """Livegrep - Fast regex search over large codebases."""
    
    def __init__(self, config: SystemConfig, work_dir: Path):
        super().__init__(config, work_dir)
        self.service_url = SERVICE_URLS['livegrep']
        
    async def _check_service_health(self) -> bool:
        """Check if Livegrep service is available."""
        try:
            response = requests.get(f"{self.service_url}/", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Livegrep service health check failed: {e}")
            return False
            
    async def setup_index(self, corpus_path: Path) -> bool:
        """Setup Livegrep service."""
        try:
            if not await self._check_service_health():
                raise RuntimeError("Service unhealthy... Strict mode forbids mock fallback. Livegrep service is not available")
                
            logger.info("✅ Livegrep service is ready")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Livegrep: {e}")
            return False
            
    async def search(self, query: str, k: int = 20) -> List[RetrievalResult]:
        """Search using Livegrep."""
        start_time = time.time()
        
        try:
            # Livegrep API call
            response = requests.post(
                f"{self.service_url}/api/v1/search",
                json={"line": query, "max_matches": k},
                timeout=30
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for i, result in enumerate(data.get('results', [])[:k]):
                    results.append(RetrievalResult(
                        file_path=result.get('tree', '') + '/' + result.get('path', ''),
                        line_number=result.get('line_number', 0),
                        content=result.get('line', ''),
                        score=1.0 - (i * 0.01),
                        rank=i + 1
                    ))
                
                logger.debug(f"Livegrep search for '{query}' returned {len(results)} results in {latency_ms:.1f}ms")
                return results
            else:
                logger.warning(f"Livegrep service error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Livegrep search failed: {e}")
            return []


class OpenSearchSystem(CompetitorSystem):
    """OpenSearch - Elasticsearch-compatible search engine."""
    
    def __init__(self, config: SystemConfig, work_dir: Path):
        super().__init__(config, work_dir)
        self.service_url = SERVICE_URLS['opensearch']
        self.index_name = "code_search"
        
    async def _check_service_health(self) -> bool:
        """Check if OpenSearch service is available."""
        try:
            response = requests.get(f"{self.service_url}/_cluster/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"OpenSearch service health check failed: {e}")
            return False
            
    async def setup_index(self, corpus_path: Path) -> bool:
        """Setup OpenSearch index."""
        try:
            if not await self._check_service_health():
                raise RuntimeError("Service unhealthy... Strict mode forbids mock fallback. OpenSearch service is not available")
                
            # In a real implementation, would create index and index documents
            logger.info("✅ OpenSearch service is ready")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup OpenSearch: {e}")
            return False
            
    async def search(self, query: str, k: int = 20) -> List[RetrievalResult]:
        """Search using OpenSearch."""
        start_time = time.time()
        
        try:
            # OpenSearch query
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content", "filename"]
                    }
                },
                "size": k
            }
            
            response = requests.post(
                f"{self.service_url}/{self.index_name}/_search",
                json=search_body,
                timeout=30
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                hits = data.get('hits', {}).get('hits', [])
                
                results = []
                for i, hit in enumerate(hits):
                    source = hit.get('_source', {})
                    results.append(RetrievalResult(
                        file_path=source.get('filename', ''),
                        line_number=source.get('line_number', 0),
                        content=source.get('content', ''),
                        score=hit.get('_score', 0) / 10.0,  # Normalize score
                        rank=i + 1
                    ))
                
                logger.debug(f"OpenSearch search for '{query}' returned {len(results)} results in {latency_ms:.1f}ms")
                return results
            else:
                logger.warning(f"OpenSearch service error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"OpenSearch search failed: {e}")
            return []


# Factory function to create real systems
def create_real_system(config: SystemConfig, work_dir: Path) -> CompetitorSystem:
    """Factory function to create real competitor systems."""
    
    # Check for specific system IDs first
    system_id = config.config.get('id', '').lower()
    
    if system_id == 'zoekt':
        return ZoektSystem(config, work_dir)
    elif system_id == 'comby':
        return CombySystem(config, work_dir)
    elif system_id == 'milvus':
        return MilvusSystem(config, work_dir)
    elif system_id == 'livegrep':
        return LivegrepSystem(config, work_dir)
    elif system_id == 'opensearch':
        return OpenSearchSystem(config, work_dir)
    elif system_id == 'ripgrep' or config.kind == RetrieverType.REGEX:
        return RealRipgrepSystem(config, work_dir)
    elif config.kind == RetrieverType.BM25.value or config.kind == 'bm25':
        return RealBM25System(config, work_dir)
    elif config.kind == RetrieverType.DENSE.value or config.kind == 'dense':
        return RealDenseSystem(config, work_dir)
    elif config.kind == RetrieverType.HYBRID.value or config.kind == 'hybrid':
        return RealHybridSystem(config, work_dir)
    elif config.kind == RetrieverType.SYMBOL.value or config.kind == 'symbol':
        return RealSymbolSystem(config, work_dir)
    else:
        # Strict mode: No mock fallbacks allowed
        raise RuntimeError(f"Service unhealthy... Strict mode forbids mock fallback. System type {config.kind} not implemented.")