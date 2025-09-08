#!/usr/bin/env python3
"""
Benchmark Corpus Router

This script integrates domain-specific corpus routing into the lens search system.
It routes queries to the appropriate corpus based on the benchmark type.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkCorpusRouter:
    """Routes benchmark queries to the appropriate domain-specific corpus"""
    
    def __init__(self):
        self.corpus_mapping = {
            'swe-bench': {
                'directory': 'benchmark-corpus/swe-bench',
                'query_patterns': [
                    r'.*__(astropy|django|sympy|matplotlib|sklearn|sphinx).*',  # SWE-bench repo patterns
                    r'swe_bench_.*',  # Explicit SWE-bench prefix
                    r'.*\w+__\w+.*',  # Double underscore pattern typical in SWE-bench
                ],
                'description': 'Real GitHub repository code for software engineering issues'
            },
            'codesearchnet': {
                'directory': 'benchmark-corpus/codesearchnet', 
                'query_patterns': [
                    r'codesearchnet_.*',  # Explicit CodeSearchNet prefix
                    r'csn_.*',  # CodeSearchNet abbreviation
                ],
                'description': 'General purpose code search examples'
            },
            'coir': {
                'directory': 'benchmark-corpus/coir',
                'query_patterns': [
                    r'coir_.*',  # Explicit CoIR prefix
                ],
                'description': 'Technical documentation for information retrieval'
            },
            'cosqa': {
                'directory': 'benchmark-corpus/cosqa',
                'query_patterns': [
                    r'cosqa_.*',  # Explicit CoSQA prefix
                ],
                'description': 'Code question-answering pairs'
            }
        }
        
        self.default_corpus = 'codesearchnet'  # Fallback for unknown queries
    
    def identify_benchmark_type(self, query_id: str) -> str:
        """Identify which benchmark corpus to use for a given query ID"""
        
        query_lower = query_id.lower()
        
        # Check explicit patterns for each benchmark type
        for benchmark_type, config in self.corpus_mapping.items():
            for pattern in config['query_patterns']:
                if re.match(pattern, query_lower):
                    logger.info(f"Query '{query_id}' matched {benchmark_type} corpus (pattern: {pattern})")
                    return benchmark_type
        
        # Fallback logic based on query content
        if '__' in query_id and any(repo in query_lower for repo in ['astropy', 'django', 'sympy', 'matplotlib']):
            logger.info(f"Query '{query_id}' identified as SWE-bench based on repository name")
            return 'swe-bench'
        
        # Default fallback
        logger.info(f"Query '{query_id}' using default corpus: {self.default_corpus}")
        return self.default_corpus
    
    def get_corpus_path(self, benchmark_type: str) -> Path:
        """Get the filesystem path for a benchmark corpus"""
        if benchmark_type not in self.corpus_mapping:
            benchmark_type = self.default_corpus
        
        path = Path(self.corpus_mapping[benchmark_type]['directory'])
        return path
    
    def get_corpus_info(self, benchmark_type: str) -> Dict[str, Any]:
        """Get detailed information about a corpus"""
        if benchmark_type not in self.corpus_mapping:
            benchmark_type = self.default_corpus
        
        config = self.corpus_mapping[benchmark_type].copy()
        corpus_path = self.get_corpus_path(benchmark_type)
        
        # Add runtime statistics
        if corpus_path.exists():
            files = list(corpus_path.rglob('*'))
            source_files = [f for f in files if f.is_file() and f.suffix in ['.py', '.js', '.md', '.java', '.ts']]
            
            total_size = sum(f.stat().st_size for f in source_files)
            
            config.update({
                'path': str(corpus_path),
                'exists': True,
                'file_count': len(source_files),
                'total_size_mb': total_size / (1024 * 1024),
                'avg_file_size_kb': (total_size / len(source_files) / 1024) if source_files else 0,
            })
        else:
            config.update({
                'path': str(corpus_path),
                'exists': False,
                'file_count': 0,
                'total_size_mb': 0,
                'avg_file_size_kb': 0,
            })
        
        return config
    
    def route_query(self, query_id: str, query_text: str = "") -> Dict[str, Any]:
        """Route a query to the appropriate corpus and return routing information"""
        
        benchmark_type = self.identify_benchmark_type(query_id)
        corpus_info = self.get_corpus_info(benchmark_type)
        
        return {
            'query_id': query_id,
            'query_text': query_text,
            'benchmark_type': benchmark_type,
            'corpus_path': corpus_info['path'],
            'corpus_exists': corpus_info['exists'],
            'corpus_stats': {
                'file_count': corpus_info['file_count'],
                'size_mb': corpus_info['total_size_mb'],
                'avg_file_size_kb': corpus_info['avg_file_size_kb'],
            },
            'corpus_description': corpus_info['description']
        }
    
    def generate_routing_config(self) -> Dict[str, Any]:
        """Generate a configuration file for the Rust search engine"""
        
        config = {
            'corpus_routing': {
                'default_corpus': self.default_corpus,
                'benchmark_corpuses': {}
            }
        }
        
        for benchmark_type, corpus_config in self.corpus_mapping.items():
            info = self.get_corpus_info(benchmark_type)
            
            config['corpus_routing']['benchmark_corpuses'][benchmark_type] = {
                'directory': corpus_config['directory'],
                'patterns': corpus_config['query_patterns'],
                'description': corpus_config['description'],
                'exists': info['exists'],
                'file_count': info['file_count']
            }
        
        return config
    
    def test_routing(self, test_queries: List[str]) -> None:
        """Test the routing logic with sample queries"""
        
        print("\n" + "="*60)
        print("🧭 BENCHMARK CORPUS ROUTING TEST")
        print("="*60)
        
        for query_id in test_queries:
            route_info = self.route_query(query_id)
            
            print(f"\n📋 Query: {query_id}")
            print(f"   Benchmark: {route_info['benchmark_type']}")
            print(f"   Corpus: {route_info['corpus_path']}")
            print(f"   Files: {route_info['corpus_stats']['file_count']:,}")
            print(f"   Size: {route_info['corpus_stats']['size_mb']:.1f} MB")
            print(f"   Status: {'✅ Available' if route_info['corpus_exists'] else '❌ Missing'}")

def main():
    router = BenchmarkCorpusRouter()
    
    # Test with realistic query IDs
    test_queries = [
        "astropy__astropy-12907",  # SWE-bench query
        "django__django-13401",    # SWE-bench query  
        "codesearchnet_0001",      # CodeSearchNet query
        "coir_documentation_001",  # CoIR query
        "cosqa_python_sorting",    # CoSQA query
        "unknown_query_type",      # Should use default
    ]
    
    # Test routing logic
    router.test_routing(test_queries)
    
    # Generate configuration file for Rust integration
    config = router.generate_routing_config()
    
    config_file = "benchmark_corpus_routing.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n📄 Routing configuration written to: {config_file}")
    
    # Show overall corpus statistics
    print("\n" + "="*60)
    print("📊 CORPUS SUMMARY")
    print("="*60)
    
    total_files = 0
    total_size = 0
    
    for benchmark_type in router.corpus_mapping.keys():
        info = router.get_corpus_info(benchmark_type)
        total_files += info['file_count']
        total_size += info['total_size_mb']
        
        status = "✅" if info['exists'] else "❌"
        print(f"{status} {benchmark_type.upper():<15} {info['file_count']:>6,} files  {info['total_size_mb']:>8.1f} MB")
    
    print("-" * 60)
    print(f"   TOTAL{'':<15} {total_files:>6,} files  {total_size:>8.1f} MB")
    
    print(f"\n🎯 Result: Domain-specific corpus routing configured!")
    print(f"   This should dramatically improve semantic search relevance")
    print(f"   by matching queries to appropriate content domains.")

if __name__ == '__main__':
    main()