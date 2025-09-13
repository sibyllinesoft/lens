#!/usr/bin/env python3
"""
Cross-encoder Reranking and Graph Expansion for v2.2.0 Algorithm Sprint
Implements neural reranking with graph-based expansion for RAG scenarios
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from pathlib import Path
import json
import pickle
import networkx as nx
from collections import defaultdict, deque
import logging
from sklearn.metrics import ndcg_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RankingExample:
    """Training example for ranking models"""
    query: str
    doc_id: str
    content: str
    relevance_score: float
    features: Dict[str, float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RerankedResult:
    """Result after cross-encoder reranking"""
    doc_id: str
    original_score: float
    original_rank: int
    reranked_score: float
    reranked_rank: int
    content: str
    confidence: float
    reranking_features: Dict[str, float] = None
    
    def __post_init__(self):
        if self.reranking_features is None:
            self.reranking_features = {}


@dataclass
class GraphExpandedResult:
    """Result after graph-based expansion"""
    doc_id: str
    original_score: float
    expansion_boost: float
    final_score: float
    expansion_paths: List[str]
    hop_count: int
    expansion_confidence: float
    related_symbols: List[str] = None
    
    def __post_init__(self):
        if self.related_symbols is None:
            self.related_symbols = []


class CrossEncoderReranker(nn.Module):
    """
    Neural cross-encoder for query-document reranking
    Key innovation: BERT-based relevance scoring with code-specific features
    """
    
    def __init__(self, model_name: str = "microsoft/codebert-base", hidden_dim: int = 768):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Add special tokens for code
        special_tokens = ['<CODE>', '</CODE>', '<FUNC>', '</FUNC>', '<CLASS>', '</CLASS>']
        self.tokenizer.add_tokens(special_tokens)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        
        # Ranking head
        self.ranking_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Feature fusion layer (for combining neural + handcrafted features)
        self.feature_fusion = nn.Linear(hidden_dim + 10, hidden_dim)  # +10 for handcrafted features
        
    def forward(self, query_doc_pairs: List[str], handcrafted_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for ranking"""
        
        # Tokenize query-document pairs
        inputs = self.tokenizer(
            query_doc_pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Get contextual embeddings
        outputs = self.encoder(**inputs)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        
        # Fuse with handcrafted features if provided
        if handcrafted_features is not None:
            fused_features = torch.cat([pooled_output, handcrafted_features], dim=1)
            pooled_output = self.feature_fusion(fused_features)
        
        # Generate relevance scores
        scores = self.ranking_head(pooled_output)
        return scores.squeeze(-1)
    
    def encode_query_doc_pair(self, query: str, document: str) -> str:
        """Encode query-document pair with special tokens"""
        # Add code structure markers
        if any(keyword in document.lower() for keyword in ['def ', 'class ', 'function ', 'import ']):
            document = f"<CODE>{document}</CODE>"
        
        return f"Query: {query} [SEP] Document: {document}"
    
    def extract_handcrafted_features(self, query: str, document: str) -> List[float]:
        """Extract handcrafted features for query-document pairs"""
        
        features = []
        
        # Query-document overlap features
        query_tokens = set(query.lower().split())
        doc_tokens = set(document.lower().split())
        
        overlap_count = len(query_tokens.intersection(doc_tokens))
        overlap_ratio = overlap_count / max(len(query_tokens), 1)
        jaccard_similarity = overlap_count / len(query_tokens.union(doc_tokens)) if query_tokens.union(doc_tokens) else 0
        
        features.extend([overlap_count, overlap_ratio, jaccard_similarity])
        
        # Document length features
        doc_length_chars = len(document)
        doc_length_words = len(document.split())
        avg_word_length = doc_length_chars / max(doc_length_words, 1)
        
        features.extend([doc_length_chars / 1000, doc_length_words / 100, avg_word_length / 10])  # Normalized
        
        # Code-specific features
        code_indicators = [
            'def ', 'class ', 'import ', 'from ', 'return ', '()', '{', '}', ';'
        ]
        code_score = sum(1 for indicator in code_indicators if indicator in document.lower()) / len(code_indicators)
        
        # Query type features (simplified)
        is_question = any(word in query.lower() for word in ['what', 'how', 'why', 'where', 'when'])
        has_quotes = '"' in query or "'" in query
        
        features.extend([code_score, float(is_question), float(has_quotes)])
        
        # Pad to fixed size (10 features)
        while len(features) < 10:
            features.append(0.0)
        
        return features[:10]


class OfflineLTRTrainer:
    """
    Offline Learning-to-Rank trainer for cross-encoder reranking
    Key innovation: Listwise training with NDCG optimization
    """
    
    def __init__(self, model: CrossEncoderReranker, learning_rate: float = 2e-5, device: str = "cuda"):
        self.model = model
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        # Loss function - using ListNet for listwise learning
        self.criterion = nn.BCELoss()
        
    def train(self, training_data: List[List[RankingExample]], num_epochs: int = 10, 
              batch_size: int = 8) -> Dict[str, List[float]]:
        """
        Train the cross-encoder model
        
        Args:
            training_data: List of query groups, each containing RankingExamples
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history with loss and metrics
        """
        
        logger.info(f"Training cross-encoder on {len(training_data)} query groups")
        
        history = {'loss': [], 'ndcg': []}
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_ndcg = 0.0
            num_batches = 0
            
            self.model.train()
            
            # Process each query group
            for query_group in training_data:
                if len(query_group) < 2:
                    continue  # Skip groups with too few examples
                
                # Prepare batch data
                query_doc_pairs = []
                handcrafted_features = []
                relevance_scores = []
                
                query = query_group[0].query
                
                for example in query_group:
                    pair_text = self.model.encode_query_doc_pair(example.query, example.content)
                    query_doc_pairs.append(pair_text)
                    
                    hc_features = self.model.extract_handcrafted_features(example.query, example.content)
                    handcrafted_features.append(hc_features)
                    
                    relevance_scores.append(example.relevance_score)
                
                # Convert to tensors
                hc_tensor = torch.tensor(handcrafted_features, dtype=torch.float32).to(self.device)
                relevance_tensor = torch.tensor(relevance_scores, dtype=torch.float32).to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                predicted_scores = self.model(query_doc_pairs, hc_tensor)
                
                # Compute listwise loss (simplified)
                loss = self.criterion(predicted_scores, relevance_tensor)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Compute NDCG for this query group
                if len(relevance_scores) > 1:
                    ndcg = ndcg_score([relevance_scores], [predicted_scores.detach().cpu().numpy()])
                    epoch_ndcg += ndcg
                
                num_batches += 1
            
            # Update scheduler
            self.scheduler.step()
            
            # Record metrics
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_ndcg = epoch_ndcg / max(num_batches, 1)
            
            history['loss'].append(avg_loss)
            history['ndcg'].append(avg_ndcg)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, NDCG: {avg_ndcg:.4f}")
        
        return history
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 100) -> List[RerankedResult]:
        """
        Rerank candidate documents using trained model
        
        Args:
            query: Search query
            candidates: List of candidate documents with 'doc_id', 'content', 'score', 'rank'
            top_k: Number of top results to return
            
        Returns:
            List of reranked results
        """
        
        self.model.eval()
        
        if not candidates:
            return []
        
        # Prepare data
        query_doc_pairs = []
        handcrafted_features = []
        
        for candidate in candidates:
            pair_text = self.model.encode_query_doc_pair(query, candidate['content'])
            query_doc_pairs.append(pair_text)
            
            hc_features = self.model.extract_handcrafted_features(query, candidate['content'])
            handcrafted_features.append(hc_features)
        
        # Batch processing
        batch_size = 16
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(query_doc_pairs), batch_size):
                batch_pairs = query_doc_pairs[i:i+batch_size]
                batch_features = handcrafted_features[i:i+batch_size]
                
                hc_tensor = torch.tensor(batch_features, dtype=torch.float32).to(self.device)
                scores = self.model(batch_pairs, hc_tensor)
                
                all_scores.extend(scores.cpu().numpy())
        
        # Create reranked results
        reranked_results = []
        
        for i, (candidate, reranked_score) in enumerate(zip(candidates, all_scores)):
            result = RerankedResult(
                doc_id=candidate['doc_id'],
                original_score=candidate['score'],
                original_rank=candidate['rank'],
                reranked_score=float(reranked_score),
                reranked_rank=0,  # Will be set after sorting
                content=candidate['content'],
                confidence=float(reranked_score),
                reranking_features=dict(zip(
                    ['neural_score', 'original_score', 'score_delta'],
                    [float(reranked_score), candidate['score'], float(reranked_score) - candidate['score']]
                ))
            )
            reranked_results.append(result)
        
        # Sort by reranked scores and update ranks
        reranked_results.sort(key=lambda x: x.reranked_score, reverse=True)
        
        for i, result in enumerate(reranked_results[:top_k], 1):
            result.reranked_rank = i
        
        return reranked_results[:top_k]
    
    def save_model(self, path: str) -> None:
        """Save trained model"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_name': self.model.model_name
        }
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")


class GraphExpandReranker:
    """
    Graph-based expansion for reranking using symbol relationships
    Key innovation: Multi-hop graph expansion with relevance propagation
    """
    
    def __init__(self, symbol_graph: nx.MultiDiGraph, expansion_weight: float = 0.3, max_hops: int = 2):
        self.symbol_graph = symbol_graph
        self.expansion_weight = expansion_weight
        self.max_hops = max_hops
        self.decay_factor = 0.7
        
    def expand_and_rerank(self, query: str, candidates: List[Dict], 
                         expansion_mode: str = "both") -> List[GraphExpandedResult]:
        """
        Expand candidates using graph relationships and rerank
        
        Args:
            query: Search query
            candidates: List of candidate documents
            expansion_mode: "file_neighbors", "symbol_neighbors", "both"
            
        Returns:
            List of graph-expanded and reranked results
        """
        
        logger.info(f"Graph expansion with mode: {expansion_mode}")
        
        expanded_results = []
        
        for candidate in candidates:
            doc_id = candidate['doc_id']
            content = candidate['content']
            original_score = candidate['score']
            
            # Find relevant symbols in the document
            doc_symbols = self._extract_document_symbols(content, candidate.get('file_path', ''))
            
            # Expand using graph relationships
            expansion_boost = 0.0
            expansion_paths = []
            related_symbols = []
            
            if doc_symbols:
                if expansion_mode in ["symbol_neighbors", "both"]:
                    symbol_boost, symbol_paths, symbol_related = self._expand_via_symbols(
                        query, doc_symbols, original_score
                    )
                    expansion_boost += symbol_boost
                    expansion_paths.extend(symbol_paths)
                    related_symbols.extend(symbol_related)
                
                if expansion_mode in ["file_neighbors", "both"]:
                    file_boost, file_paths, file_related = self._expand_via_file_neighbors(
                        query, candidate.get('file_path', ''), original_score
                    )
                    expansion_boost += file_boost
                    expansion_paths.extend(file_paths)
                    related_symbols.extend(file_related)
            
            # Calculate final score
            final_score = original_score + (expansion_boost * self.expansion_weight)
            
            # Calculate confidence based on expansion evidence
            expansion_confidence = min(1.0, expansion_boost / original_score) if original_score > 0 else 0.0
            
            result = GraphExpandedResult(
                doc_id=doc_id,
                original_score=original_score,
                expansion_boost=expansion_boost,
                final_score=final_score,
                expansion_paths=expansion_paths,
                hop_count=min(len(expansion_paths), self.max_hops),
                expansion_confidence=expansion_confidence,
                related_symbols=list(set(related_symbols))  # Remove duplicates
            )
            
            expanded_results.append(result)
        
        # Sort by final scores
        expanded_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return expanded_results
    
    def _extract_document_symbols(self, content: str, file_path: str = "") -> List[str]:
        """Extract symbols that might be in the graph from document content"""
        
        symbols = []
        
        # Look for symbols in the graph that match content
        for symbol_id in self.symbol_graph.nodes():
            symbol_data = self.symbol_graph.nodes[symbol_id]
            symbol_name = symbol_data.get('name', symbol_id.split(':')[-1])
            
            # Check if symbol name appears in content (case-insensitive)
            if symbol_name.lower() in content.lower():
                symbols.append(symbol_id)
            
            # Check file path match if available
            if file_path and symbol_data.get('file_path', '') == file_path:
                symbols.append(symbol_id)
        
        return symbols
    
    def _expand_via_symbols(self, query: str, doc_symbols: List[str], 
                           original_score: float) -> Tuple[float, List[str], List[str]]:
        """Expand via symbol graph relationships"""
        
        total_boost = 0.0
        expansion_paths = []
        related_symbols = []
        
        query_words = set(query.lower().split())
        
        for symbol_id in doc_symbols:
            if symbol_id not in self.symbol_graph:
                continue
            
            # Perform multi-hop expansion
            visited = {symbol_id}
            queue = deque([(symbol_id, 1.0, 0, [symbol_id])])  # (id, weight, hops, path)
            
            while queue:
                current_id, current_weight, hop_count, path = queue.popleft()
                
                if hop_count >= self.max_hops:
                    continue
                
                # Explore neighbors
                for neighbor_id in self.symbol_graph.neighbors(current_id):
                    if neighbor_id in visited:
                        continue
                    
                    # Calculate relevance boost
                    neighbor_data = self.symbol_graph.nodes[neighbor_id]
                    neighbor_name = neighbor_data.get('name', neighbor_id.split(':')[-1])
                    
                    # Check relevance to query
                    name_words = set(neighbor_name.lower().split('_'))
                    word_overlap = len(query_words.intersection(name_words))
                    
                    if word_overlap > 0:
                        # Calculate weighted boost
                        hop_decay = self.decay_factor ** (hop_count + 1)
                        relevance_boost = (word_overlap / max(len(query_words), 1)) * hop_decay
                        boost_amount = relevance_boost * original_score * 0.5  # Scale factor
                        
                        total_boost += boost_amount
                        expansion_paths.append(" -> ".join(path + [neighbor_id]))
                        related_symbols.append(neighbor_id)
                        
                        # Continue expansion from this neighbor
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, current_weight * hop_decay, hop_count + 1, path + [neighbor_id]))
        
        return total_boost, expansion_paths, related_symbols
    
    def _expand_via_file_neighbors(self, query: str, file_path: str, 
                                  original_score: float) -> Tuple[float, List[str], List[str]]:
        """Expand via file-based relationships"""
        
        if not file_path:
            return 0.0, [], []
        
        total_boost = 0.0
        expansion_paths = []
        related_symbols = []
        
        query_words = set(query.lower().split())
        
        # Find symbols in the same file
        file_symbols = [
            symbol_id for symbol_id in self.symbol_graph.nodes()
            if self.symbol_graph.nodes[symbol_id].get('file_path', '') == file_path
        ]
        
        # Find related files through symbol relationships
        related_files = set()
        for symbol_id in file_symbols:
            for neighbor_id in self.symbol_graph.neighbors(symbol_id):
                neighbor_file = self.symbol_graph.nodes[neighbor_id].get('file_path', '')
                if neighbor_file and neighbor_file != file_path:
                    related_files.add(neighbor_file)
        
        # Calculate boost from related files
        for related_file in related_files:
            # Simple heuristic: boost based on file name similarity to query
            file_name = Path(related_file).stem.lower()
            file_words = set(file_name.replace('_', ' ').split())
            word_overlap = len(query_words.intersection(file_words))
            
            if word_overlap > 0:
                boost_amount = (word_overlap / max(len(query_words), 1)) * original_score * 0.2
                total_boost += boost_amount
                expansion_paths.append(f"file:{file_path} -> file:{related_file}")
                
                # Add symbols from related file
                related_file_symbols = [
                    symbol_id for symbol_id in self.symbol_graph.nodes()
                    if self.symbol_graph.nodes[symbol_id].get('file_path', '') == related_file
                ]
                related_symbols.extend(related_file_symbols[:5])  # Limit to avoid explosion
        
        return total_boost, expansion_paths, related_symbols


# Integrated RAG system combining both approaches
class IntegratedRAGReranker:
    """
    Integrated system combining cross-encoder reranking with graph expansion
    Key innovation: Sequential application with confidence-weighted combination
    """
    
    def __init__(self, cross_encoder_trainer: OfflineLTRTrainer, 
                 graph_expander: GraphExpandReranker, 
                 combination_strategy: str = "weighted"):
        
        self.cross_encoder = cross_encoder_trainer
        self.graph_expander = graph_expander
        self.combination_strategy = combination_strategy
        
    def rerank_with_expansion(self, query: str, candidates: List[Dict], 
                             cross_encoder_weight: float = 0.7,
                             graph_weight: float = 0.3,
                             top_k: int = 50) -> List[Dict]:
        """
        Integrated reranking with both neural and graph methods
        
        Args:
            query: Search query
            candidates: Candidate documents
            cross_encoder_weight: Weight for cross-encoder scores
            graph_weight: Weight for graph expansion scores
            top_k: Number of results to return
            
        Returns:
            Combined and reranked results
        """
        
        logger.info("Starting integrated RAG reranking")
        
        # Step 1: Cross-encoder reranking
        logger.info("Applying cross-encoder reranking...")
        cross_encoder_results = self.cross_encoder.rerank(query, candidates, top_k=len(candidates))
        
        # Convert to format for graph expansion
        ce_candidates = []
        for result in cross_encoder_results:
            ce_candidates.append({
                'doc_id': result.doc_id,
                'content': result.content,
                'score': result.reranked_score,
                'rank': result.reranked_rank,
                'original_score': result.original_score,
                'file_path': next((c['file_path'] for c in candidates if c['doc_id'] == result.doc_id), '')
            })
        
        # Step 2: Graph expansion
        logger.info("Applying graph expansion...")
        graph_results = self.graph_expander.expand_and_rerank(query, ce_candidates[:top_k*2])
        
        # Step 3: Combine scores
        final_results = []
        
        for i, graph_result in enumerate(graph_results):
            # Find corresponding cross-encoder result
            ce_result = next((r for r in cross_encoder_results if r.doc_id == graph_result.doc_id), None)
            
            if ce_result is None:
                continue
            
            # Combine scores based on strategy
            if self.combination_strategy == "weighted":
                final_score = (cross_encoder_weight * ce_result.reranked_score + 
                              graph_weight * graph_result.final_score)
            elif self.combination_strategy == "multiplicative":
                final_score = ce_result.reranked_score * (1.0 + graph_result.expansion_boost)
            else:  # additive
                final_score = ce_result.reranked_score + graph_result.expansion_boost
            
            # Create combined result
            combined_result = {
                'doc_id': graph_result.doc_id,
                'final_score': final_score,
                'cross_encoder_score': ce_result.reranked_score,
                'graph_expansion_score': graph_result.final_score,
                'expansion_boost': graph_result.expansion_boost,
                'content': ce_result.content,
                'original_rank': ce_result.original_rank,
                'cross_encoder_rank': ce_result.reranked_rank,
                'expansion_confidence': graph_result.expansion_confidence,
                'related_symbols': graph_result.related_symbols,
                'expansion_paths': graph_result.expansion_paths,
                'reranking_metadata': {
                    'cross_encoder_confidence': ce_result.confidence,
                    'graph_expansion_boost': graph_result.expansion_boost,
                    'combination_strategy': self.combination_strategy,
                    'weights': {
                        'cross_encoder': cross_encoder_weight,
                        'graph': graph_weight
                    }
                }
            }
            
            final_results.append(combined_result)
        
        # Final sort by combined scores
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Update final ranks
        for i, result in enumerate(final_results[:top_k], 1):
            result['final_rank'] = i
        
        logger.info(f"Integrated reranking complete. Returned {min(len(final_results), top_k)} results")
        
        return final_results[:top_k]


# Example usage and testing
if __name__ == "__main__":
    # Mock training data for cross-encoder
    mock_training_data = [
        [
            RankingExample("find data processing function", "doc1", 
                          "def process_data(data): return data.transform()", 1.0),
            RankingExample("find data processing function", "doc2", 
                          "class DataProcessor: def init(self): pass", 0.6),
            RankingExample("find data processing function", "doc3", 
                          "import pandas as pd", 0.2)
        ],
        [
            RankingExample("authentication logic", "doc4", 
                          "def authenticate_user(username, password): return verify(username, password)", 1.0),
            RankingExample("authentication logic", "doc5", 
                          "def login_handler(request): return authenticate(request)", 0.8),
            RankingExample("authentication logic", "doc6", 
                          "def logout(): session.clear()", 0.3)
        ]
    ]
    
    # Test cross-encoder training
    print("=== Cross-encoder Training ===")
    model = CrossEncoderReranker()
    trainer = OfflineLTRTrainer(model, device="cpu")  # Use CPU for testing
    
    history = trainer.train(mock_training_data, num_epochs=2, batch_size=2)
    print(f"Training history: {history}")
    
    # Test reranking
    print("\n=== Cross-encoder Reranking ===")
    test_candidates = [
        {'doc_id': 'doc1', 'content': 'def process_data(data): return data.transform()', 'score': 0.7, 'rank': 1},
        {'doc_id': 'doc2', 'content': 'class DataProcessor: def init(self): pass', 'score': 0.6, 'rank': 2},
        {'doc_id': 'doc3', 'content': 'import pandas as pd', 'score': 0.5, 'rank': 3}
    ]
    
    reranked = trainer.rerank("find data processing function", test_candidates)
    for result in reranked:
        print(f"Rank {result.reranked_rank}: {result.doc_id} (score: {result.reranked_score:.3f}, "
              f"original: {result.original_score:.3f})")
    
    # Test graph expansion (with mock graph)
    print("\n=== Graph Expansion ===")
    mock_graph = nx.MultiDiGraph()
    mock_graph.add_node("doc1:func:process_data", name="process_data", file_path="data_utils.py")
    mock_graph.add_node("doc2:class:DataProcessor", name="DataProcessor", file_path="processors.py")
    mock_graph.add_node("related_func", name="transform_data", file_path="data_utils.py")
    mock_graph.add_edge("doc1:func:process_data", "related_func", relation_type="calls")
    
    graph_expander = GraphExpandReranker(mock_graph)
    
    expanded = graph_expander.expand_and_rerank(
        "data processing function", 
        [{'doc_id': 'doc1', 'content': 'def process_data(data): return data.transform()', 
          'score': 0.7, 'file_path': 'data_utils.py'}]
    )
    
    for result in expanded:
        print(f"Doc {result.doc_id}: final_score={result.final_score:.3f}, "
              f"boost={result.expansion_boost:.3f}, paths={result.expansion_paths}")
    
    print("\nAlgorithmic implementations complete!")