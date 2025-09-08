#!/usr/bin/env python3
"""
FAISS Server - Pure ANN library wrapper for authentic benchmarking
Implements IVF, HNSW, and PQ indices for scientific comparison
"""

import os
import sys
import time
import json
import logging
import numpy as np
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, List, Tuple, Any, Optional
import uuid
from sentence_transformers import SentenceTransformer
import psutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FAISSServer:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Configuration from environment
        self.port = int(os.environ.get('FAISS_SERVER_PORT', 8084))
        self.index_type = os.environ.get('INDEX_TYPE', 'IVF_HNSW')
        self.vector_dim = int(os.environ.get('VECTOR_DIM', 512))
        self.indices_dir = Path(os.environ.get('INDICES_DIR', '/indices'))
        self.datasets_dir = Path(os.environ.get('DATASETS_DIR', '/datasets'))
        
        # Ensure directories exist
        self.indices_dir.mkdir(exist_ok=True)
        self.datasets_dir.mkdir(exist_ok=True)
        
        # FAISS indices storage
        self.indices: Dict[str, faiss.Index] = {}
        self.index_metadata: Dict[str, Dict] = {}
        
        # Embedding model for code embeddings
        self.embedding_model = None
        self.load_embedding_model()
        
        # Setup routes
        self.setup_routes()
        
        logger.info(f"FAISS Server initialized - Index Type: {self.index_type}, Vector Dim: {self.vector_dim}")
    
    def load_embedding_model(self):
        """Load pre-trained model for code embeddings"""
        try:
            # Use CodeBERT for code embeddings (authentic model used in research)
            model_name = "microsoft/codebert-base"
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    def create_faiss_index(self, index_type: str, dimension: int, num_vectors: int = 100000) -> faiss.Index:
        """Create FAISS index based on type specification"""
        
        if index_type == "FLAT":
            # Brute force exact search
            index = faiss.IndexFlatL2(dimension)
            
        elif index_type == "IVF":
            # Inverted file index  
            nlist = min(4096, num_vectors // 39)  # Typical ratio
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
        elif index_type == "HNSW":
            # Hierarchical Navigable Small World
            M = 16  # Number of bi-directional links for each node
            index = faiss.IndexHNSWFlat(dimension, M)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 50
            
        elif index_type == "IVF_HNSW":
            # Combined IVF + HNSW (best of both worlds)
            nlist = min(1024, num_vectors // 100)
            quantizer = faiss.IndexHNSWFlat(dimension, 32)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
        elif index_type == "PQ":
            # Product Quantization
            m = 8  # Number of subquantizers
            bits = 8  # Bits per subquantizer
            index = faiss.IndexPQ(dimension, m, bits)
            
        elif index_type == "IVF_PQ":
            # IVF with Product Quantization
            nlist = min(4096, num_vectors // 39)
            m = 8
            bits = 8
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, bits)
            
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        logger.info(f"Created FAISS index: {index_type} (dim={dimension})")
        return index
    
    def embed_code_snippets(self, code_snippets: List[str]) -> np.ndarray:
        """Generate embeddings for code snippets using pre-trained model"""
        if self.embedding_model is None:
            # Fallback to random embeddings for testing
            logger.warning("No embedding model available, using random vectors")
            return np.random.random((len(code_snippets), self.vector_dim)).astype('float32')
        
        try:
            embeddings = self.embedding_model.encode(code_snippets, convert_to_numpy=True)
            return embeddings.astype('float32')
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return np.random.random((len(code_snippets), self.vector_dim)).astype('float32')
    
    def build_index_from_dataset(self, corpus_name: str, max_samples: int = 100000) -> Dict[str, Any]:
        """Build FAISS index from dataset corpus"""
        start_time = time.time()
        
        # Load code snippets from corpus
        code_snippets = self.load_corpus_data(corpus_name, max_samples)
        if not code_snippets:
            return {"status": "failed", "error": "No corpus data found"}
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(code_snippets)} code snippets...")
        embeddings = self.embed_code_snippets(code_snippets)
        
        # Create and train index
        index = self.create_faiss_index(self.index_type, embeddings.shape[1], len(code_snippets))
        
        # Train index if needed
        if hasattr(index, 'train') and not index.is_trained:
            logger.info("Training index...")
            index.train(embeddings)
        
        # Add vectors to index
        logger.info("Adding vectors to index...")
        index.add(embeddings)
        
        # Store index and metadata
        self.indices[corpus_name] = index
        self.index_metadata[corpus_name] = {
            "index_type": self.index_type,
            "dimension": embeddings.shape[1],
            "num_vectors": embeddings.shape[0],
            "created_at": time.time(),
            "corpus_name": corpus_name,
            "build_time_seconds": time.time() - start_time
        }
        
        # Save index to disk
        index_path = self.indices_dir / f"{corpus_name}_{self.index_type}.index"
        faiss.write_index(index, str(index_path))
        
        # Save metadata
        metadata_path = self.indices_dir / f"{corpus_name}_{self.index_type}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.index_metadata[corpus_name], f, indent=2)
        
        build_time = time.time() - start_time
        logger.info(f"Index built successfully in {build_time:.2f}s: {len(code_snippets)} vectors")
        
        return {
            "status": "completed",
            "corpus_name": corpus_name,
            "index_type": self.index_type,
            "num_vectors": len(code_snippets),
            "dimension": embeddings.shape[1],
            "build_time_seconds": build_time,
            "index_path": str(index_path)
        }
    
    def load_corpus_data(self, corpus_name: str, max_samples: int) -> List[str]:
        """Load code snippets from corpus data"""
        code_snippets = []
        
        # Look for corpus files
        corpus_files = []
        for ext in ['.json', '.jsonl', '.txt']:
            corpus_files.extend(self.datasets_dir.glob(f"*{corpus_name}*{ext}"))
            corpus_files.extend(self.datasets_dir.glob(f"{corpus_name}/**/*{ext}"))
        
        if not corpus_files:
            logger.warning(f"No corpus files found for {corpus_name}")
            return []
        
        logger.info(f"Loading corpus data from {len(corpus_files)} files")
        
        for corpus_file in corpus_files:
            try:
                if corpus_file.suffix == '.jsonl':
                    with open(corpus_file, 'r') as f:
                        for line in f:
                            if len(code_snippets) >= max_samples:
                                break
                            try:
                                data = json.loads(line)
                                # Extract code content (adapt to different corpus formats)
                                code = self.extract_code_from_item(data)
                                if code:
                                    code_snippets.append(code)
                            except json.JSONDecodeError:
                                continue
                                
                elif corpus_file.suffix == '.json':
                    with open(corpus_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if len(code_snippets) >= max_samples:
                                    break
                                code = self.extract_code_from_item(item)
                                if code:
                                    code_snippets.append(code)
                
                elif corpus_file.suffix == '.txt':
                    with open(corpus_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines[:max_samples]:
                            line = line.strip()
                            if line:
                                code_snippets.append(line)
                        
            except Exception as e:
                logger.error(f"Error loading {corpus_file}: {e}")
                continue
                
            if len(code_snippets) >= max_samples:
                break
        
        logger.info(f"Loaded {len(code_snippets)} code snippets from corpus")
        return code_snippets
    
    def extract_code_from_item(self, item: Dict) -> Optional[str]:
        """Extract code content from different corpus formats"""
        # Try different field names used in various corpora
        for field in ['code', 'function', 'body', 'content', 'text', 'docstring', 'snippet']:
            if field in item and item[field]:
                return str(item[field]).strip()
        
        # If it's a simple string, return it
        if isinstance(item, str):
            return item.strip()
            
        return None
    
    def search_index(self, query: str, corpus_name: str, k: int = 10) -> Dict[str, Any]:
        """Search FAISS index with query"""
        query_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Check if index exists
        if corpus_name not in self.indices:
            return {
                "query_id": query_id,
                "status": "failed",
                "error": f"Index for corpus '{corpus_name}' not found"
            }
        
        index = self.indices[corpus_name]
        
        # Generate query embedding
        query_embedding = self.embed_code_snippets([query])
        if query_embedding.shape[0] == 0:
            return {
                "query_id": query_id,
                "status": "failed", 
                "error": "Failed to generate query embedding"
            }
        
        # Perform search
        try:
            distances, indices = index.search(query_embedding, k)
            
            # Convert results
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= 0:  # Valid result
                    results.append({
                        "rank": i + 1,
                        "index": int(idx),
                        "distance": float(dist),
                        "similarity": 1.0 / (1.0 + dist)  # Convert distance to similarity
                    })
            
            latency_ms = (time.time() - start_time) * 1000
            sla_violated = latency_ms > 150.0
            
            response = {
                "query_id": query_id,
                "system": "faiss",
                "version": faiss.__version__,
                "index_type": self.index_type,
                "corpus": corpus_name,
                "query": query,
                "k": k,
                "latency_ms": latency_ms,
                "sla_violated": sla_violated,
                "total_hits": len(results),
                "results": results
            }
            
            logger.info(f"Search completed: corpus={corpus_name}, k={k}, hits={len(results)}, latency={latency_ms:.2f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "query_id": query_id,
                "status": "failed",
                "error": str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health information"""
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            "status": "healthy",
            "system": "faiss", 
            "version": faiss.__version__,
            "index_type": self.index_type,
            "vector_dimension": self.vector_dim,
            "available_indices": list(self.indices.keys()),
            "system_resources": {
                "memory_usage_percent": memory_info.percent,
                "cpu_usage_percent": cpu_percent,
                "available_memory_gb": memory_info.available / (1024**3)
            },
            "embedding_model": "microsoft/codebert-base" if self.embedding_model else None
        }
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify(self.get_system_status())
        
        @self.app.route('/build-index', methods=['POST'])
        def build_index():
            data = request.get_json()
            corpus_name = data.get('corpus_name', 'default')
            max_samples = data.get('max_samples', 100000)
            
            try:
                result = self.build_index_from_dataset(corpus_name, max_samples)
                return jsonify(result)
            except Exception as e:
                logger.error(f"Failed to build index: {e}")
                return jsonify({
                    "status": "failed",
                    "error": str(e)
                }), 500
        
        @self.app.route('/search', methods=['POST'])
        def search():
            data = request.get_json()
            query = data.get('query', '')
            corpus_name = data.get('corpus', 'default')
            k = data.get('k', 10)
            
            if not query:
                return jsonify({
                    "status": "failed",
                    "error": "Query is required"
                }), 400
            
            result = self.search_index(query, corpus_name, k)
            return jsonify(result)
        
        @self.app.route('/indices', methods=['GET'])
        def list_indices():
            return jsonify({
                "available_indices": {
                    name: {
                        "metadata": self.index_metadata.get(name, {}),
                        "num_vectors": self.indices[name].ntotal if name in self.indices else 0
                    }
                    for name in self.indices.keys()
                }
            })
    
    def run(self):
        """Start the FAISS server"""
        logger.info(f"Starting FAISS server on port {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=False)

def main():
    """Main entry point"""
    server = FAISSServer()
    server.run()

if __name__ == "__main__":
    main()