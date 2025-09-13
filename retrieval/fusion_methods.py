#!/usr/bin/env python3
"""
Advanced Fusion Methods for v2.2.0 Algorithm Sprint
Implements Weighted-RRF, QSF (Query-Score Fusion), and learned fusion approaches
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum
from scipy import stats
from collections import defaultdict
import json
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class FusionStrategy(Enum):
    """Types of fusion strategies"""
    WEIGHTED_RRF = "weighted_rrf"
    QSF = "qsf"  # Query-Score Fusion
    LEARNED_FUSION = "learned_fusion"
    RANK_SVM = "rank_svm"
    NEURAL_FUSION = "neural_fusion"


@dataclass
class SearchResult:
    """Represents a search result from a single retrieval system"""
    doc_id: str
    score: float
    rank: int
    content: str
    metadata: Dict[str, Any] = None
    source_system: str = "unknown"
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FusedResult:
    """Represents a fused result from multiple systems"""
    doc_id: str
    final_score: float
    final_rank: int
    content: str
    fusion_metadata: Dict[str, Any] = None
    source_scores: Dict[str, float] = None
    source_ranks: Dict[str, int] = None
    
    def __post_init__(self):
        if self.fusion_metadata is None:
            self.fusion_metadata = {}
        if self.source_scores is None:
            self.source_scores = {}
        if self.source_ranks is None:
            self.source_ranks = {}


@dataclass
class FusionConfig:
    """Configuration for fusion methods"""
    strategy: FusionStrategy
    weights: List[float] = None
    k0: int = 30  # RRF parameter
    alpha: float = 0.5  # QSF mixing parameter
    normalization: str = "z_score"  # "z_score", "min_max", "none"
    min_overlap: int = 1  # Minimum systems that must return a doc
    max_results: int = 100
    learning_rate: float = 0.01
    regularization: float = 0.001
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = [1.0, 1.0, 1.0]  # Default equal weights


class BaseFusionMethod(ABC):
    """Base class for all fusion methods"""
    
    def __init__(self, config: FusionConfig):
        self.config = config
        self.system_names = []
        self.is_trained = False
        
    @abstractmethod
    def fuse(self, results_by_system: Dict[str, List[SearchResult]]) -> List[FusedResult]:
        """Fuse results from multiple retrieval systems"""
        pass
    
    def _normalize_scores(self, scores: np.ndarray, method: str = "z_score") -> np.ndarray:
        """Normalize scores using specified method"""
        if method == "z_score":
            return stats.zscore(scores) if len(scores) > 1 else scores
        elif method == "min_max":
            if len(scores) <= 1:
                return scores
            min_score, max_score = np.min(scores), np.max(scores)
            if max_score > min_score:
                return (scores - min_score) / (max_score - min_score)
            return scores
        elif method == "none":
            return scores
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def _rank_to_score(self, rank: int, total_results: int) -> float:
        """Convert rank to normalized score"""
        return (total_results - rank + 1) / total_results
    
    def _compute_rrf_score(self, rank: int, k0: int = 30) -> float:
        """Compute Reciprocal Rank Fusion score"""
        return 1.0 / (k0 + rank)


class WeightedRRFFusion(BaseFusionMethod):
    """
    Weighted Reciprocal Rank Fusion with z-score normalization
    Key innovation: System-specific weights with score normalization
    """
    
    def __init__(self, config: FusionConfig):
        super().__init__(config)
        
    def fuse(self, results_by_system: Dict[str, List[SearchResult]]) -> List[FusedResult]:
        """Fuse results using weighted RRF"""
        
        # Initialize system tracking
        if not self.system_names:
            self.system_names = list(results_by_system.keys())
        
        # Ensure we have weights for all systems
        if len(self.config.weights) != len(self.system_names):
            self.config.weights = [1.0] * len(self.system_names)
        
        # Collect all unique documents
        all_docs = set()
        for results in results_by_system.values():
            all_docs.update(result.doc_id for result in results)
        
        # Compute fusion scores
        doc_scores = {}
        doc_metadata = {}
        
        for doc_id in all_docs:
            system_scores = []
            system_ranks = {}
            system_raw_scores = {}
            contributing_systems = 0
            
            # Collect scores from each system
            for i, (system_name, results) in enumerate(results_by_system.items()):
                doc_result = next((r for r in results if r.doc_id == doc_id), None)
                
                if doc_result:
                    # Compute RRF score
                    rrf_score = self._compute_rrf_score(doc_result.rank, self.config.k0)
                    
                    # Apply system weight
                    weighted_score = rrf_score * self.config.weights[i]
                    system_scores.append(weighted_score)
                    
                    system_ranks[system_name] = doc_result.rank
                    system_raw_scores[system_name] = doc_result.score
                    contributing_systems += 1
                else:
                    # Document not found in this system
                    system_scores.append(0.0)
            
            # Skip if not enough systems returned this document
            if contributing_systems < self.config.min_overlap:
                continue
            
            # Normalize and combine scores
            if self.config.normalization != "none":
                system_scores = self._normalize_scores(np.array(system_scores), self.config.normalization)
            
            final_score = np.sum(system_scores)
            
            doc_scores[doc_id] = final_score
            doc_metadata[doc_id] = {
                'source_ranks': system_ranks,
                'source_scores': system_raw_scores,
                'contributing_systems': contributing_systems,
                'fusion_method': 'weighted_rrf',
                'system_weights': dict(zip(self.system_names, self.config.weights))
            }
        
        # Sort by final scores and create FusedResult objects
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for final_rank, (doc_id, final_score) in enumerate(sorted_docs[:self.config.max_results], 1):
            # Get document content (from first system that has it)
            content = ""
            for results in results_by_system.values():
                doc_result = next((r for r in results if r.doc_id == doc_id), None)
                if doc_result:
                    content = doc_result.content
                    break
            
            fused_result = FusedResult(
                doc_id=doc_id,
                final_score=final_score,
                final_rank=final_rank,
                content=content,
                fusion_metadata=doc_metadata[doc_id],
                source_scores=doc_metadata[doc_id]['source_scores'],
                source_ranks=doc_metadata[doc_id]['source_ranks']
            )
            fused_results.append(fused_result)
        
        return fused_results


class QSFFusion(BaseFusionMethod):
    """
    Query-Score Fusion (QSF) that mixes rank-based and score-based signals
    Key innovation: Adaptive mixing based on score distribution variance
    """
    
    def __init__(self, config: FusionConfig):
        super().__init__(config)
        self.score_variance_threshold = 0.1  # Threshold for score vs rank preference
        
    def fuse(self, results_by_system: Dict[str, List[SearchResult]]) -> List[FusedResult]:
        """Fuse results using Query-Score Fusion"""
        
        if not self.system_names:
            self.system_names = list(results_by_system.keys())
        
        # Ensure we have weights for all systems
        if len(self.config.weights) != len(self.system_names):
            self.config.weights = [1.0] * len(self.system_names)
        
        all_docs = set()
        for results in results_by_system.values():
            all_docs.update(result.doc_id for result in results)
        
        doc_scores = {}
        doc_metadata = {}
        
        # Analyze score distributions per system to adapt alpha
        system_score_variances = {}
        for system_name, results in results_by_system.items():
            if results:
                scores = [r.score for r in results]
                system_score_variances[system_name] = np.var(scores) if len(scores) > 1 else 0
        
        for doc_id in all_docs:
            rank_component = 0.0
            score_component = 0.0
            contributing_systems = 0
            system_ranks = {}
            system_raw_scores = {}
            
            total_weight = 0.0
            
            for i, (system_name, results) in enumerate(results_by_system.items()):
                doc_result = next((r for r in results if r.doc_id == doc_id), None)
                
                if doc_result:
                    system_weight = self.config.weights[i]
                    
                    # Compute rank-based component (RRF)
                    rank_score = self._compute_rrf_score(doc_result.rank, self.config.k0)
                    
                    # Compute score-based component (normalized)
                    all_scores = [r.score for r in results]
                    norm_scores = self._normalize_scores(np.array(all_scores), self.config.normalization)
                    score_idx = next(i for i, r in enumerate(results) if r.doc_id == doc_id)
                    norm_score = norm_scores[score_idx] if len(norm_scores) > score_idx else doc_result.score
                    
                    # Adaptive alpha based on score variance
                    score_variance = system_score_variances.get(system_name, 0)
                    adaptive_alpha = self.config.alpha
                    if score_variance < self.score_variance_threshold:
                        # Low variance -> prefer rank-based
                        adaptive_alpha = max(0.2, self.config.alpha - 0.2)
                    else:
                        # High variance -> prefer score-based  
                        adaptive_alpha = min(0.8, self.config.alpha + 0.2)
                    
                    # QSF combination
                    qsf_score = adaptive_alpha * norm_score + (1 - adaptive_alpha) * rank_score
                    
                    rank_component += rank_score * system_weight
                    score_component += norm_score * system_weight
                    total_weight += system_weight
                    
                    system_ranks[system_name] = doc_result.rank
                    system_raw_scores[system_name] = doc_result.score
                    contributing_systems += 1
            
            if contributing_systems < self.config.min_overlap:
                continue
            
            # Normalize by total weight
            if total_weight > 0:
                final_score = (rank_component + score_component) / total_weight
            else:
                final_score = 0.0
            
            doc_scores[doc_id] = final_score
            doc_metadata[doc_id] = {
                'source_ranks': system_ranks,
                'source_scores': system_raw_scores,
                'contributing_systems': contributing_systems,
                'fusion_method': 'qsf',
                'rank_component': rank_component / total_weight if total_weight > 0 else 0,
                'score_component': score_component / total_weight if total_weight > 0 else 0,
                'adaptive_alpha': adaptive_alpha if 'adaptive_alpha' in locals() else self.config.alpha
            }
        
        # Sort and create results
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for final_rank, (doc_id, final_score) in enumerate(sorted_docs[:self.config.max_results], 1):
            content = ""
            for results in results_by_system.values():
                doc_result = next((r for r in results if r.doc_id == doc_id), None)
                if doc_result:
                    content = doc_result.content
                    break
            
            fused_result = FusedResult(
                doc_id=doc_id,
                final_score=final_score,
                final_rank=final_rank,
                content=content,
                fusion_metadata=doc_metadata[doc_id],
                source_scores=doc_metadata[doc_id]['source_scores'],
                source_ranks=doc_metadata[doc_id]['source_ranks']
            )
            fused_results.append(fused_result)
        
        return fused_results


class LearnedFusion(BaseFusionMethod):
    """
    Machine-learned fusion using features from multiple retrieval systems
    Key innovation: Feature extraction from retrieval systems + ML optimization
    """
    
    def __init__(self, config: FusionConfig):
        super().__init__(config)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.training_data = []
        
    def train(self, training_examples: List[Dict]) -> None:
        """
        Train the fusion model on labeled examples
        
        Args:
            training_examples: List of dicts with 'query', 'results_by_system', 'relevance_labels'
        """
        print("Training learned fusion model...")
        
        features = []
        labels = []
        
        for example in training_examples:
            query = example['query']
            results_by_system = example['results_by_system']
            relevance_labels = example['relevance_labels']  # Dict[doc_id, relevance_score]
            
            # Extract features for each document
            doc_features = self._extract_features(query, results_by_system)
            
            for doc_id, feature_vector in doc_features.items():
                if doc_id in relevance_labels:
                    features.append(feature_vector)
                    labels.append(relevance_labels[doc_id])
        
        if not features:
            raise ValueError("No training features extracted")
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model (using Random Forest for robustness)
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        
        self.is_trained = True
        print(f"Trained fusion model on {len(features)} examples")
        print(f"Feature importance: {dict(zip(self.feature_names, self.model.feature_importances_))}")
    
    def _extract_features(self, query: str, results_by_system: Dict[str, List[SearchResult]]) -> Dict[str, List[float]]:
        """Extract features for each document from multiple systems"""
        
        # Initialize feature names on first call
        if not self.feature_names:
            system_names = list(results_by_system.keys())
            self.feature_names = []
            
            # Per-system features
            for system in system_names:
                self.feature_names.extend([
                    f"{system}_score",
                    f"{system}_rank_norm",
                    f"{system}_rrf_score",
                    f"{system}_present"
                ])
            
            # Cross-system features
            self.feature_names.extend([
                "system_agreement",      # How many systems returned this doc
                "score_variance",        # Variance across system scores
                "rank_variance",         # Variance across system ranks
                "max_score",            # Highest score across systems
                "min_rank",             # Best rank across systems
                "query_doc_overlap",    # Query-document term overlap (simplified)
                "doc_length",           # Document length in characters
            ])
        
        # Collect all documents
        all_docs = set()
        for results in results_by_system.values():
            all_docs.update(r.doc_id for r in results)
        
        doc_features = {}
        
        for doc_id in all_docs:
            feature_vector = []
            
            # Per-system features
            system_scores = []
            system_ranks = []
            
            for system_name, results in results_by_system.items():
                doc_result = next((r for r in results if r.doc_id == doc_id), None)
                
                if doc_result:
                    # System-specific features
                    feature_vector.extend([
                        doc_result.score,
                        self._rank_to_score(doc_result.rank, len(results)),
                        self._compute_rrf_score(doc_result.rank, self.config.k0),
                        1.0  # Present in this system
                    ])
                    
                    system_scores.append(doc_result.score)
                    system_ranks.append(doc_result.rank)
                else:
                    # Document not in this system
                    feature_vector.extend([0.0, 0.0, 0.0, 0.0])
            
            # Cross-system features
            system_agreement = len(system_scores)
            score_variance = np.var(system_scores) if len(system_scores) > 1 else 0
            rank_variance = np.var(system_ranks) if len(system_ranks) > 1 else 0
            max_score = max(system_scores) if system_scores else 0
            min_rank = min(system_ranks) if system_ranks else float('inf')
            
            # Get document content for text features
            content = ""
            for results in results_by_system.values():
                doc_result = next((r for r in results if r.doc_id == doc_id), None)
                if doc_result:
                    content = doc_result.content
                    break
            
            # Simple query-document overlap
            query_words = set(query.lower().split())
            doc_words = set(content.lower().split())
            overlap = len(query_words.intersection(doc_words)) / max(len(query_words), 1)
            
            feature_vector.extend([
                system_agreement,
                score_variance,
                rank_variance,
                max_score,
                min_rank if min_rank != float('inf') else 0,
                overlap,
                len(content)
            ])
            
            doc_features[doc_id] = feature_vector
        
        return doc_features
    
    def fuse(self, results_by_system: Dict[str, List[SearchResult]], query: str = "") -> List[FusedResult]:
        """Fuse results using learned model"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before fusion")
        
        if not self.system_names:
            self.system_names = list(results_by_system.keys())
        
        # Extract features for all documents
        doc_features = self._extract_features(query, results_by_system)
        
        # Predict scores using trained model
        doc_scores = {}
        doc_metadata = {}
        
        for doc_id, feature_vector in doc_features.items():
            # Scale features
            X_scaled = self.scaler.transform([feature_vector])
            
            # Predict relevance score
            predicted_score = self.model.predict(X_scaled)[0]
            
            doc_scores[doc_id] = predicted_score
            
            # Collect metadata
            system_ranks = {}
            system_raw_scores = {}
            for system_name, results in results_by_system.items():
                doc_result = next((r for r in results if r.doc_id == doc_id), None)
                if doc_result:
                    system_ranks[system_name] = doc_result.rank
                    system_raw_scores[system_name] = doc_result.score
            
            doc_metadata[doc_id] = {
                'source_ranks': system_ranks,
                'source_scores': system_raw_scores,
                'fusion_method': 'learned_fusion',
                'predicted_relevance': predicted_score,
                'feature_vector': feature_vector
            }
        
        # Sort and create results
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for final_rank, (doc_id, final_score) in enumerate(sorted_docs[:self.config.max_results], 1):
            content = ""
            for results in results_by_system.values():
                doc_result = next((r for r in results if r.doc_id == doc_id), None)
                if doc_result:
                    content = doc_result.content
                    break
            
            fused_result = FusedResult(
                doc_id=doc_id,
                final_score=final_score,
                final_rank=final_rank,
                content=content,
                fusion_metadata=doc_metadata[doc_id],
                source_scores=doc_metadata[doc_id]['source_scores'],
                source_ranks=doc_metadata[doc_id]['source_ranks']
            )
            fused_results.append(fused_result)
        
        return fused_results
    
    def save_model(self, path: str) -> None:
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config,
            'system_names': self.system_names
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path: str) -> None:
        """Load trained model from disk"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.system_names = model_data['system_names']
        self.is_trained = True


# Factory function for fusion method selection
def create_fusion_method(strategy: FusionStrategy, config: FusionConfig = None) -> BaseFusionMethod:
    """Factory function to create fusion methods"""
    
    if config is None:
        config = FusionConfig(strategy=strategy)
    
    fusion_methods = {
        FusionStrategy.WEIGHTED_RRF: WeightedRRFFusion,
        FusionStrategy.QSF: QSFFusion,
        FusionStrategy.LEARNED_FUSION: LearnedFusion
    }
    
    if strategy not in fusion_methods:
        raise ValueError(f"Unknown fusion strategy: {strategy}. Available: {list(fusion_methods.keys())}")
    
    return fusion_methods[strategy](config)


# Evaluation utilities
class FusionEvaluator:
    """Evaluates fusion method performance"""
    
    @staticmethod
    def compute_metrics(fused_results: List[FusedResult], ground_truth: Dict[str, float]) -> Dict[str, float]:
        """Compute evaluation metrics for fused results"""
        
        # Extract doc IDs and scores
        doc_ids = [r.doc_id for r in fused_results]
        scores = [r.final_score for r in fused_results]
        
        # Compute metrics where ground truth is available
        relevant_docs = [doc_id for doc_id in doc_ids if doc_id in ground_truth]
        
        if not relevant_docs:
            return {'ndcg_10': 0.0, 'map': 0.0, 'precision_10': 0.0}
        
        # NDCG@10
        ndcg_10 = FusionEvaluator._compute_ndcg(doc_ids[:10], ground_truth)
        
        # MAP (Mean Average Precision)
        map_score = FusionEvaluator._compute_map(doc_ids, ground_truth)
        
        # Precision@10
        relevant_in_top10 = len([d for d in doc_ids[:10] if ground_truth.get(d, 0) > 0])
        precision_10 = relevant_in_top10 / min(10, len(doc_ids))
        
        return {
            'ndcg_10': ndcg_10,
            'map': map_score,
            'precision_10': precision_10,
            'total_results': len(fused_results),
            'relevant_retrieved': len(relevant_docs)
        }
    
    @staticmethod
    def _compute_ndcg(doc_ids: List[str], ground_truth: Dict[str, float], k: int = 10) -> float:
        """Compute NDCG@k"""
        if not doc_ids:
            return 0.0
        
        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(doc_ids[:k]):
            relevance = ground_truth.get(doc_id, 0)
            dcg += (2**relevance - 1) / np.log2(i + 2)
        
        # IDCG
        ideal_relevances = sorted(ground_truth.values(), reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevances):
            idcg += (2**relevance - 1) / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def _compute_map(doc_ids: List[str], ground_truth: Dict[str, float]) -> float:
        """Compute Mean Average Precision"""
        relevant_retrieved = 0
        avg_precision = 0.0
        total_relevant = sum(1 for r in ground_truth.values() if r > 0)
        
        if total_relevant == 0:
            return 0.0
        
        for i, doc_id in enumerate(doc_ids):
            if ground_truth.get(doc_id, 0) > 0:
                relevant_retrieved += 1
                precision_at_i = relevant_retrieved / (i + 1)
                avg_precision += precision_at_i
        
        return avg_precision / total_relevant


# Example usage and testing
if __name__ == "__main__":
    # Create mock results from different systems
    sparse_results = [
        SearchResult("doc1", 0.9, 1, "def process_data(data): return data.transform()", source_system="sparse"),
        SearchResult("doc2", 0.8, 2, "class DataProcessor: def __init__(self):", source_system="sparse"),
        SearchResult("doc3", 0.7, 3, "import pandas as pd", source_system="sparse")
    ]
    
    dense_results = [
        SearchResult("doc2", 0.95, 1, "class DataProcessor: def __init__(self):", source_system="dense"),
        SearchResult("doc1", 0.85, 2, "def process_data(data): return data.transform()", source_system="dense"),
        SearchResult("doc4", 0.75, 3, "data.groupby('category').sum()", source_system="dense")
    ]
    
    symbol_results = [
        SearchResult("doc1", 0.88, 1, "def process_data(data): return data.transform()", source_system="symbol"),
        SearchResult("doc4", 0.82, 2, "data.groupby('category').sum()", source_system="symbol"),
        SearchResult("doc5", 0.76, 3, "from sklearn.preprocessing import StandardScaler", source_system="symbol")
    ]
    
    results_by_system = {
        "sparse": sparse_results,
        "dense": dense_results,
        "symbol": symbol_results
    }
    
    # Test Weighted RRF
    print("=== Weighted RRF Fusion ===")
    rrf_config = FusionConfig(
        strategy=FusionStrategy.WEIGHTED_RRF,
        weights=[1.2, 1.0, 0.8],  # Boost sparse, balance dense, lower symbol
        k0=30,
        normalization="z_score"
    )
    rrf_fusion = create_fusion_method(FusionStrategy.WEIGHTED_RRF, rrf_config)
    rrf_results = rrf_fusion.fuse(results_by_system)
    
    for i, result in enumerate(rrf_results[:5], 1):
        print(f"{i}. {result.doc_id} (score: {result.final_score:.3f})")
        print(f"   Sources: {result.source_ranks}")
    
    # Test QSF
    print("\n=== QSF Fusion ===")
    qsf_config = FusionConfig(
        strategy=FusionStrategy.QSF,
        alpha=0.6,  # Favor scores over ranks
        normalization="z_score"
    )
    qsf_fusion = create_fusion_method(FusionStrategy.QSF, qsf_config)
    qsf_results = qsf_fusion.fuse(results_by_system)
    
    for i, result in enumerate(qsf_results[:5], 1):
        print(f"{i}. {result.doc_id} (score: {result.final_score:.3f})")
        print(f"   Metadata: {result.fusion_metadata.get('adaptive_alpha', 'N/A')}")
    
    # Test evaluation
    print("\n=== Evaluation ===")
    ground_truth = {
        "doc1": 1.0,  # Highly relevant
        "doc2": 0.8,  # Relevant
        "doc3": 0.2,  # Slightly relevant
        "doc4": 0.6,  # Moderately relevant
        "doc5": 0.1   # Barely relevant
    }
    
    rrf_metrics = FusionEvaluator.compute_metrics(rrf_results, ground_truth)
    qsf_metrics = FusionEvaluator.compute_metrics(qsf_results, ground_truth)
    
    print(f"RRF Metrics: {rrf_metrics}")
    print(f"QSF Metrics: {qsf_metrics}")