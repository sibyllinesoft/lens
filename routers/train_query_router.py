#!/usr/bin/env python3
"""
Query Routing System for v2.2.0 Algorithm Sprint
Routes queries to specialized retrieval pipelines based on query characteristics
"""

import re
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
from collections import defaultdict, Counter
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


class QueryType(Enum):
    """Types of queries that can be routed"""
    LEXICAL = "lexical"           # Exact string matching queries
    SEMANTIC = "semantic"         # Meaning-based queries
    SYMBOL = "symbol"            # Symbol/identifier queries
    HYBRID = "hybrid"            # Mixed queries requiring multiple approaches
    STRUCTURAL = "structural"    # Code structure queries (class X extends Y)
    FUNCTIONAL = "functional"    # Behavior-based queries (what does this do)


class RetrievalPipeline(Enum):
    """Available retrieval pipelines"""
    LEXICAL_EXACT = "lexical_exact"        # BM25/TF-IDF exact matching
    SEMANTIC_DENSE = "semantic_dense"      # Dense vector similarity
    SYMBOL_GRAPH = "symbol_graph"          # Symbol graph traversal
    HYBRID_FUSION = "hybrid_fusion"        # Multi-system fusion
    STRUCTURAL_AST = "structural_ast"      # AST-based structural search
    CODE_UNDERSTANDING = "code_understanding"  # LLM-powered understanding


@dataclass
class QueryFeatures:
    """Extracted features from a query"""
    query_text: str
    
    # Lexical features
    exact_matches: int = 0           # Number of exact string patterns
    quoted_terms: int = 0            # Number of quoted exact terms
    
    # Symbol features
    identifiers: List[str] = None    # Programming identifiers found
    operators: List[str] = None      # Programming operators
    
    # Semantic features
    question_words: List[str] = None # What, how, why, where, etc.
    intent_keywords: List[str] = None # find, search, show, etc.
    
    # Structural features
    structural_patterns: List[str] = None # class X extends, function Y calls
    
    # Statistical features
    query_length: int = 0            # Character length
    word_count: int = 0              # Word count
    avg_word_length: float = 0.0     # Average word length
    code_like_ratio: float = 0.0     # Ratio of code-like terms
    
    def __post_init__(self):
        if self.identifiers is None:
            self.identifiers = []
        if self.operators is None:
            self.operators = []
        if self.question_words is None:
            self.question_words = []
        if self.intent_keywords is None:
            self.intent_keywords = []
        if self.structural_patterns is None:
            self.structural_patterns = []


@dataclass
class RoutingDecision:
    """Routing decision with confidence and metadata"""
    query: str
    primary_pipeline: RetrievalPipeline
    secondary_pipelines: List[RetrievalPipeline]
    confidence: float
    query_type: QueryType
    features: QueryFeatures
    routing_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.routing_metadata is None:
            self.routing_metadata = {}


@dataclass
class RoutingConfig:
    """Configuration for query routing"""
    confidence_threshold: float = 0.7
    max_secondary_pipelines: int = 2
    fallback_strategy: str = "uniform_blend"  # uniform_blend, weighted_cascade, ensemble
    feature_weights: Dict[str, float] = None
    pipeline_weights: Dict[RetrievalPipeline, float] = None
    
    def __post_init__(self):
        if self.feature_weights is None:
            self.feature_weights = {
                'lexical': 1.0,
                'semantic': 1.0,
                'symbol': 1.2,      # Boost symbol detection
                'structural': 1.1   # Boost structural patterns
            }
        if self.pipeline_weights is None:
            self.pipeline_weights = {
                RetrievalPipeline.LEXICAL_EXACT: 1.0,
                RetrievalPipeline.SEMANTIC_DENSE: 1.0,
                RetrievalPipeline.SYMBOL_GRAPH: 1.2,
                RetrievalPipeline.HYBRID_FUSION: 0.9,
                RetrievalPipeline.STRUCTURAL_AST: 1.1,
                RetrievalPipeline.CODE_UNDERSTANDING: 0.8
            }


class BaseQueryRouter(ABC):
    """Base class for query routers"""
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        self.is_trained = False
        
    @abstractmethod
    def route_query(self, query: str) -> RoutingDecision:
        """Route a query to appropriate retrieval pipeline(s)"""
        pass
    
    def extract_features(self, query: str) -> QueryFeatures:
        """Extract features from a query"""
        
        # Basic statistics
        query_length = len(query)
        words = query.split()
        word_count = len(words)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # Lexical features
        exact_matches = len(re.findall(r'"[^"]*"', query))  # Quoted strings
        quoted_terms = exact_matches
        
        # Symbol features (programming identifiers)
        identifier_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        identifiers = re.findall(identifier_pattern, query)
        
        # Programming operators
        operator_patterns = [
            r'==|!=|<=|>=|<|>',  # Comparison
            r'\+\+|--|->|\.',     # Increment, decrement, arrow, dot
            r'&&|\|\||!',         # Logical
            r'[+\-*/=]'          # Arithmetic and assignment
        ]
        operators = []
        for pattern in operator_patterns:
            operators.extend(re.findall(pattern, query))
        
        # Semantic features
        question_words = [w.lower() for w in words if w.lower() in 
                         {'what', 'how', 'why', 'where', 'when', 'which', 'who'}]
        
        intent_keywords = [w.lower() for w in words if w.lower() in
                          {'find', 'search', 'show', 'get', 'list', 'retrieve', 'fetch', 'locate'}]
        
        # Structural patterns
        structural_patterns = []
        
        # Class inheritance patterns
        if re.search(r'\bclass\s+\w+\s+extends\s+\w+', query, re.IGNORECASE):
            structural_patterns.append('class_extends')
        if re.search(r'\bclass\s+\w+\s+implements\s+\w+', query, re.IGNORECASE):
            structural_patterns.append('class_implements')
        
        # Function patterns
        if re.search(r'\bfunction\s+\w+\s*\(|\bdef\s+\w+\s*\(', query, re.IGNORECASE):
            structural_patterns.append('function_definition')
        if re.search(r'\w+\s*\([^)]*\)', query):
            structural_patterns.append('function_call')
        
        # Import patterns
        if re.search(r'\b(?:import|from)\s+\w+', query, re.IGNORECASE):
            structural_patterns.append('import_statement')
        
        # Code-like ratio (heuristic)
        code_indicators = len(identifiers) + len(operators) + len(structural_patterns)
        code_like_ratio = code_indicators / max(word_count, 1)
        
        return QueryFeatures(
            query_text=query,
            exact_matches=exact_matches,
            quoted_terms=quoted_terms,
            identifiers=identifiers,
            operators=operators,
            question_words=question_words,
            intent_keywords=intent_keywords,
            structural_patterns=structural_patterns,
            query_length=query_length,
            word_count=word_count,
            avg_word_length=avg_word_length,
            code_like_ratio=code_like_ratio
        )


class RuleBasedRouter(BaseQueryRouter):
    """
    Rule-based query router using heuristics and patterns
    Key innovation: Hand-crafted rules with confidence scoring
    """
    
    def __init__(self, config: RoutingConfig):
        super().__init__(config)
        self.is_trained = True  # Rule-based doesn't need training
        
        # Define routing rules with priorities
        self.routing_rules = [
            # Symbol-specific rules (highest priority)
            {
                'name': 'exact_identifier',
                'condition': lambda f: len(f.identifiers) > 0 and f.exact_matches > 0,
                'pipeline': RetrievalPipeline.SYMBOL_GRAPH,
                'confidence': 0.9,
                'query_type': QueryType.SYMBOL
            },
            {
                'name': 'high_code_ratio',
                'condition': lambda f: f.code_like_ratio > 0.5,
                'pipeline': RetrievalPipeline.SYMBOL_GRAPH,
                'confidence': 0.8,
                'query_type': QueryType.SYMBOL
            },
            
            # Structural rules
            {
                'name': 'structural_patterns',
                'condition': lambda f: len(f.structural_patterns) > 0,
                'pipeline': RetrievalPipeline.STRUCTURAL_AST,
                'confidence': 0.85,
                'query_type': QueryType.STRUCTURAL
            },
            
            # Semantic rules
            {
                'name': 'question_based',
                'condition': lambda f: len(f.question_words) > 0,
                'pipeline': RetrievalPipeline.CODE_UNDERSTANDING,
                'confidence': 0.75,
                'query_type': QueryType.SEMANTIC
            },
            {
                'name': 'intent_based',
                'condition': lambda f: len(f.intent_keywords) > 0 and f.code_like_ratio < 0.3,
                'pipeline': RetrievalPipeline.SEMANTIC_DENSE,
                'confidence': 0.7,
                'query_type': QueryType.SEMANTIC
            },
            
            # Lexical rules
            {
                'name': 'exact_quoted',
                'condition': lambda f: f.quoted_terms > 0,
                'pipeline': RetrievalPipeline.LEXICAL_EXACT,
                'confidence': 0.8,
                'query_type': QueryType.LEXICAL
            },
            {
                'name': 'short_exact',
                'condition': lambda f: f.word_count <= 3 and f.code_like_ratio < 0.2,
                'pipeline': RetrievalPipeline.LEXICAL_EXACT,
                'confidence': 0.6,
                'query_type': QueryType.LEXICAL
            },
            
            # Hybrid/fallback rules
            {
                'name': 'mixed_complexity',
                'condition': lambda f: (f.code_like_ratio > 0.2 and 
                                      (len(f.question_words) > 0 or len(f.intent_keywords) > 0)),
                'pipeline': RetrievalPipeline.HYBRID_FUSION,
                'confidence': 0.6,
                'query_type': QueryType.HYBRID
            }
        ]
        
    def route_query(self, query: str) -> RoutingDecision:
        """Route query using rule-based logic"""
        
        features = self.extract_features(query)
        
        # Apply rules in order of priority
        matched_rules = []
        for rule in self.routing_rules:
            if rule['condition'](features):
                matched_rules.append(rule)
        
        if not matched_rules:
            # Fallback to hybrid approach with low confidence
            return RoutingDecision(
                query=query,
                primary_pipeline=RetrievalPipeline.HYBRID_FUSION,
                secondary_pipelines=[RetrievalPipeline.SEMANTIC_DENSE],
                confidence=0.3,
                query_type=QueryType.HYBRID,
                features=features,
                routing_metadata={'fallback': True, 'reason': 'no_rules_matched'}
            )
        
        # Select primary rule (highest confidence)
        primary_rule = max(matched_rules, key=lambda r: r['confidence'])
        
        # Select secondary pipelines from other matched rules
        secondary_rules = [r for r in matched_rules if r != primary_rule]
        secondary_pipelines = [r['pipeline'] for r in secondary_rules[:self.config.max_secondary_pipelines]]
        
        # Apply feature weights to confidence
        confidence = primary_rule['confidence']
        if primary_rule['query_type'] == QueryType.SYMBOL:
            confidence *= self.config.feature_weights.get('symbol', 1.0)
        elif primary_rule['query_type'] == QueryType.STRUCTURAL:
            confidence *= self.config.feature_weights.get('structural', 1.0)
        elif primary_rule['query_type'] == QueryType.SEMANTIC:
            confidence *= self.config.feature_weights.get('semantic', 1.0)
        elif primary_rule['query_type'] == QueryType.LEXICAL:
            confidence *= self.config.feature_weights.get('lexical', 1.0)
        
        confidence = min(confidence, 1.0)  # Cap at 1.0
        
        return RoutingDecision(
            query=query,
            primary_pipeline=primary_rule['pipeline'],
            secondary_pipelines=secondary_pipelines,
            confidence=confidence,
            query_type=primary_rule['query_type'],
            features=features,
            routing_metadata={
                'matched_rules': [r['name'] for r in matched_rules],
                'primary_rule': primary_rule['name']
            }
        )


class LearnedRouter(BaseQueryRouter):
    """
    Machine learned query router using supervised learning
    Key innovation: Feature engineering + ML for routing decisions
    """
    
    def __init__(self, config: RoutingConfig):
        super().__init__(config)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.pipeline_encoder = {}
        self.feature_names = []
        
    def train(self, training_data: List[Dict]) -> None:
        """
        Train the routing model on labeled examples
        
        Args:
            training_data: List of dicts with 'query', 'optimal_pipeline', 'performance_score'
        """
        print("Training learned router...")
        
        queries = []
        labels = []
        
        for example in training_data:
            query = example['query']
            optimal_pipeline = example['optimal_pipeline']
            
            queries.append(query)
            labels.append(optimal_pipeline.value if isinstance(optimal_pipeline, RetrievalPipeline) else optimal_pipeline)
        
        # Create label encoding
        unique_pipelines = list(set(labels))
        self.pipeline_encoder = {pipeline: i for i, pipeline in enumerate(unique_pipelines)}
        self.inverse_pipeline_encoder = {i: pipeline for pipeline, i in self.pipeline_encoder.items()}
        
        # Extract features
        feature_matrix = []
        for query in queries:
            features = self._extract_ml_features(query)
            feature_matrix.append(features)
        
        X = np.array(feature_matrix)
        y = np.array([self.pipeline_encoder[label] for label in labels])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        self.classifier.fit(X_scaled, y)
        
        # Evaluate on training data
        y_pred = self.classifier.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        
        self.is_trained = True
        print(f"Trained router with accuracy: {accuracy:.3f}")
        print(f"Feature importance: {dict(zip(self.feature_names, self.classifier.feature_importances_))}")
    
    def _extract_ml_features(self, query: str) -> List[float]:
        """Extract ML-ready features from query"""
        
        # Extract structured features
        features = self.extract_features(query)
        
        # Convert to numerical feature vector
        feature_vector = []
        
        # Initialize feature names on first call
        if not self.feature_names:
            self.feature_names = [
                'query_length', 'word_count', 'avg_word_length',
                'exact_matches', 'quoted_terms', 'code_like_ratio',
                'identifier_count', 'operator_count', 'structural_pattern_count',
                'question_word_count', 'intent_keyword_count',
                'has_function_pattern', 'has_class_pattern', 'has_import_pattern'
            ]
        
        # Basic features
        feature_vector.extend([
            features.query_length,
            features.word_count,
            features.avg_word_length,
            features.exact_matches,
            features.quoted_terms,
            features.code_like_ratio,
            len(features.identifiers),
            len(features.operators),
            len(features.structural_patterns),
            len(features.question_words),
            len(features.intent_keywords)
        ])
        
        # Binary pattern features
        feature_vector.extend([
            1.0 if any('function' in p for p in features.structural_patterns) else 0.0,
            1.0 if any('class' in p for p in features.structural_patterns) else 0.0,
            1.0 if any('import' in p for p in features.structural_patterns) else 0.0
        ])
        
        return feature_vector
    
    def route_query(self, query: str) -> RoutingDecision:
        """Route query using learned model"""
        
        if not self.is_trained:
            raise ValueError("Router must be trained before use")
        
        features = self.extract_features(query)
        ml_features = self._extract_ml_features(query)
        
        # Scale features
        X_scaled = self.scaler.transform([ml_features])
        
        # Predict pipeline and confidence
        predicted_class = self.classifier.predict(X_scaled)[0]
        class_probabilities = self.classifier.predict_proba(X_scaled)[0]
        
        primary_pipeline_str = self.inverse_pipeline_encoder[predicted_class]
        primary_pipeline = RetrievalPipeline(primary_pipeline_str)
        confidence = float(class_probabilities[predicted_class])
        
        # Select secondary pipelines from other high-probability options
        secondary_pipelines = []
        sorted_indices = np.argsort(class_probabilities)[::-1]  # Sort descending
        
        for idx in sorted_indices[1:self.config.max_secondary_pipelines + 1]:
            if class_probabilities[idx] > 0.2:  # Minimum threshold for secondary
                secondary_pipeline_str = self.inverse_pipeline_encoder[idx]
                secondary_pipelines.append(RetrievalPipeline(secondary_pipeline_str))
        
        # Infer query type from primary pipeline
        pipeline_to_type = {
            RetrievalPipeline.LEXICAL_EXACT: QueryType.LEXICAL,
            RetrievalPipeline.SEMANTIC_DENSE: QueryType.SEMANTIC,
            RetrievalPipeline.SYMBOL_GRAPH: QueryType.SYMBOL,
            RetrievalPipeline.STRUCTURAL_AST: QueryType.STRUCTURAL,
            RetrievalPipeline.HYBRID_FUSION: QueryType.HYBRID,
            RetrievalPipeline.CODE_UNDERSTANDING: QueryType.FUNCTIONAL
        }
        
        query_type = pipeline_to_type.get(primary_pipeline, QueryType.HYBRID)
        
        return RoutingDecision(
            query=query,
            primary_pipeline=primary_pipeline,
            secondary_pipelines=secondary_pipelines,
            confidence=confidence,
            query_type=query_type,
            features=features,
            routing_metadata={
                'all_probabilities': dict(zip(self.inverse_pipeline_encoder.values(), class_probabilities)),
                'ml_features': ml_features
            }
        )
    
    def save_model(self, path: str) -> None:
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'vectorizer': self.vectorizer,
            'pipeline_encoder': self.pipeline_encoder,
            'inverse_pipeline_encoder': self.inverse_pipeline_encoder,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path: str) -> None:
        """Load trained model from disk"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.vectorizer = model_data['vectorizer']
        self.pipeline_encoder = model_data['pipeline_encoder']
        self.inverse_pipeline_encoder = model_data['inverse_pipeline_encoder']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.is_trained = True


class HybridRouter(BaseQueryRouter):
    """
    Hybrid router that combines rule-based and learned approaches
    Key innovation: Ensemble routing with confidence-based weighting
    """
    
    def __init__(self, config: RoutingConfig):
        super().__init__(config)
        self.rule_router = RuleBasedRouter(config)
        self.learned_router = LearnedRouter(config)
        self.ensemble_weight = 0.6  # Weight for learned router (0.4 for rules)
        
    def train(self, training_data: List[Dict]) -> None:
        """Train the learned component"""
        self.learned_router.train(training_data)
        self.is_trained = True
    
    def route_query(self, query: str) -> RoutingDecision:
        """Route using ensemble of rule-based and learned approaches"""
        
        # Get decisions from both routers
        rule_decision = self.rule_router.route_query(query)
        
        if self.learned_router.is_trained:
            learned_decision = self.learned_router.route_query(query)
        else:
            # Fallback to rule-based only
            return rule_decision
        
        # Ensemble the decisions
        rule_weight = 1.0 - self.ensemble_weight
        learned_weight = self.ensemble_weight
        
        # Weight confidences
        rule_weighted_conf = rule_decision.confidence * rule_weight
        learned_weighted_conf = learned_decision.confidence * learned_weight
        
        # Select primary decision based on weighted confidence
        if learned_weighted_conf > rule_weighted_conf:
            primary_decision = learned_decision
            secondary_decision = rule_decision
            final_confidence = learned_weighted_conf
            method = "learned_primary"
        else:
            primary_decision = rule_decision
            secondary_decision = learned_decision
            final_confidence = rule_weighted_conf
            method = "rule_primary"
        
        # Combine secondary pipelines (remove duplicates)
        combined_secondary = list(set(
            primary_decision.secondary_pipelines + 
            secondary_decision.secondary_pipelines +
            ([secondary_decision.primary_pipeline] if secondary_decision.primary_pipeline != primary_decision.primary_pipeline else [])
        ))[:self.config.max_secondary_pipelines]
        
        # Create hybrid metadata
        hybrid_metadata = {
            'ensemble_method': method,
            'rule_decision': asdict(rule_decision),
            'learned_decision': asdict(learned_decision) if self.learned_router.is_trained else None,
            'rule_confidence': rule_decision.confidence,
            'learned_confidence': learned_decision.confidence if self.learned_router.is_trained else 0,
            'weights': {'rule': rule_weight, 'learned': learned_weight}
        }
        
        return RoutingDecision(
            query=query,
            primary_pipeline=primary_decision.primary_pipeline,
            secondary_pipelines=combined_secondary,
            confidence=final_confidence,
            query_type=primary_decision.query_type,
            features=primary_decision.features,
            routing_metadata=hybrid_metadata
        )
    
    def save_model(self, path: str) -> None:
        """Save hybrid model components"""
        if self.learned_router.is_trained:
            learned_path = str(Path(path).with_suffix('.learned.pkl'))
            self.learned_router.save_model(learned_path)
        
        hybrid_data = {
            'ensemble_weight': self.ensemble_weight,
            'config': self.config,
            'is_trained': self.is_trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(hybrid_data, f)
    
    def load_model(self, path: str) -> None:
        """Load hybrid model components"""
        with open(path, 'rb') as f:
            hybrid_data = pickle.load(f)
        
        self.ensemble_weight = hybrid_data['ensemble_weight']
        self.config = hybrid_data['config']
        self.is_trained = hybrid_data['is_trained']
        
        # Try to load learned component
        learned_path = str(Path(path).with_suffix('.learned.pkl'))
        if Path(learned_path).exists():
            self.learned_router.load_model(learned_path)


# Factory function for router creation
def create_query_router(router_type: str, config: RoutingConfig = None) -> BaseQueryRouter:
    """Factory function to create query routers"""
    
    if config is None:
        config = RoutingConfig()
    
    routers = {
        'rule_based': RuleBasedRouter,
        'learned': LearnedRouter,
        'hybrid': HybridRouter
    }
    
    if router_type not in routers:
        raise ValueError(f"Unknown router type: {router_type}. Available: {list(routers.keys())}")
    
    return routers[router_type](config)


# Evaluation utilities
class RouterEvaluator:
    """Evaluates router performance"""
    
    @staticmethod
    def evaluate_routing_accuracy(router: BaseQueryRouter, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate routing accuracy against ground truth"""
        
        correct_primary = 0
        correct_in_any = 0
        total = 0
        confidence_scores = []
        
        for example in test_data:
            query = example['query']
            ground_truth = example['optimal_pipeline']
            
            if isinstance(ground_truth, str):
                ground_truth = RetrievalPipeline(ground_truth)
            
            decision = router.route_query(query)
            
            # Check primary pipeline accuracy
            if decision.primary_pipeline == ground_truth:
                correct_primary += 1
                correct_in_any += 1
            # Check if ground truth is in any selected pipeline
            elif ground_truth in decision.secondary_pipelines:
                correct_in_any += 1
            
            confidence_scores.append(decision.confidence)
            total += 1
        
        return {
            'primary_accuracy': correct_primary / total if total > 0 else 0,
            'any_pipeline_accuracy': correct_in_any / total if total > 0 else 0,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'total_queries': total
        }


# Example usage and testing
if __name__ == "__main__":
    # Test queries with different characteristics
    test_queries = [
        "find function processData",                    # Symbol query
        "class DataProcessor extends BaseProcessor",    # Structural query  
        "What does this function do?",                  # Semantic question
        '"def calculate_mean"',                        # Exact lexical query
        "how to implement error handling",             # Semantic intent
        "import pandas as pd",                         # Structural import
        "search for authentication logic",            # Hybrid query
        "user.getName().toLowerCase()",                # Complex symbol chain
    ]
    
    # Test rule-based router
    print("=== Rule-Based Router ===")
    config = RoutingConfig(confidence_threshold=0.6)
    rule_router = create_query_router('rule_based', config)
    
    for query in test_queries:
        decision = rule_router.route_query(query)
        print(f"\nQuery: '{query}'")
        print(f"Primary: {decision.primary_pipeline.value} (confidence: {decision.confidence:.2f})")
        print(f"Type: {decision.query_type.value}")
        print(f"Secondary: {[p.value for p in decision.secondary_pipelines]}")
        print(f"Features: identifiers={len(decision.features.identifiers)}, "
              f"code_ratio={decision.features.code_like_ratio:.2f}")
    
    # Generate mock training data for learned router
    print("\n=== Learned Router Training ===")
    mock_training_data = []
    ground_truth_mapping = {
        "find function processData": RetrievalPipeline.SYMBOL_GRAPH,
        "class DataProcessor extends BaseProcessor": RetrievalPipeline.STRUCTURAL_AST,
        "What does this function do?": RetrievalPipeline.CODE_UNDERSTANDING,
        '"def calculate_mean"': RetrievalPipeline.LEXICAL_EXACT,
        "how to implement error handling": RetrievalPipeline.SEMANTIC_DENSE,
        "import pandas as pd": RetrievalPipeline.STRUCTURAL_AST,
        "search for authentication logic": RetrievalPipeline.HYBRID_FUSION,
        "user.getName().toLowerCase()": RetrievalPipeline.SYMBOL_GRAPH,
    }
    
    for query, optimal_pipeline in ground_truth_mapping.items():
        mock_training_data.append({
            'query': query,
            'optimal_pipeline': optimal_pipeline,
            'performance_score': 0.9  # Mock performance
        })
    
    # Train learned router
    learned_router = create_query_router('learned', config)
    learned_router.train(mock_training_data)
    
    # Test learned router
    print("\n=== Learned Router Results ===")
    for query in test_queries[:4]:  # Test subset
        decision = learned_router.route_query(query)
        print(f"\nQuery: '{query}'")
        print(f"Primary: {decision.primary_pipeline.value} (confidence: {decision.confidence:.2f})")
        print(f"All probabilities: {decision.routing_metadata.get('all_probabilities', {})}")
    
    # Test hybrid router
    print("\n=== Hybrid Router Results ===")
    hybrid_router = create_query_router('hybrid', config)
    hybrid_router.train(mock_training_data)
    
    for query in test_queries[:3]:  # Test subset
        decision = hybrid_router.route_query(query)
        print(f"\nQuery: '{query}'")
        print(f"Primary: {decision.primary_pipeline.value} (confidence: {decision.confidence:.2f})")
        print(f"Method: {decision.routing_metadata.get('ensemble_method')}")
    
    # Evaluate router accuracy
    print("\n=== Router Evaluation ===")
    test_data = [{'query': q, 'optimal_pipeline': p} for q, p in ground_truth_mapping.items()]
    
    rule_metrics = RouterEvaluator.evaluate_routing_accuracy(rule_router, test_data)
    learned_metrics = RouterEvaluator.evaluate_routing_accuracy(learned_router, test_data)
    hybrid_metrics = RouterEvaluator.evaluate_routing_accuracy(hybrid_router, test_data)
    
    print(f"Rule-based: {rule_metrics}")
    print(f"Learned: {learned_metrics}")
    print(f"Hybrid: {hybrid_metrics}")