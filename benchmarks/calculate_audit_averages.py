#!/usr/bin/env python3
"""
Calculate average nDCG@10 performance from audit-proof results
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import statistics

def load_provenance_data():
    """Load the provenance data from audit results"""
    provenance_file = Path("demo_audit_results/provenance.jsonl")
    
    if not provenance_file.exists():
        raise FileNotFoundError(f"Provenance file not found: {provenance_file}")
    
    records = []
    with open(provenance_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    
    return records

def calculate_system_averages():
    """Calculate average performance per system across all datasets"""
    records = load_provenance_data()
    
    # Group by system
    system_results = defaultdict(list)
    system_metadata = {}
    
    for record in records:
        system = record['system']
        
        # Store metadata
        if system not in system_metadata:
            system_metadata[system] = {
                'provenance': record['provenance'],
                'status': record['status'],
                'impl': 'api' if record['provenance'] == 'api' else 'local'
            }
        
        # Only include available systems with valid metrics
        if record['status'] == 'AVAILABLE' and record['ndcg_10'] is not None:
            system_results[system].append({
                'ndcg_10': record['ndcg_10'],
                'recall_50': record['recall_50'],
                'p95_latency': record['p95_latency'],
                'dataset': record['dataset']
            })
    
    # Calculate averages
    averages = {}
    for system, results in system_results.items():
        if results:  # Only if we have valid results
            avg_ndcg = statistics.mean([r['ndcg_10'] for r in results])
            avg_recall = statistics.mean([r['recall_50'] for r in results])
            avg_latency = statistics.mean([r['p95_latency'] for r in results])
            
            averages[system] = {
                'ndcg_10_mean': avg_ndcg,
                'recall_50_mean': avg_recall,
                'p95_latency_mean': avg_latency,
                'dataset_count': len(results),
                'datasets': [r['dataset'] for r in results],
                **system_metadata[system]
            }
    
    # Include quarantined systems
    for system, metadata in system_metadata.items():
        if system not in averages and metadata['status'] != 'AVAILABLE':
            averages[system] = {
                'ndcg_10_mean': None,
                'recall_50_mean': None, 
                'p95_latency_mean': None,
                'dataset_count': 0,
                'datasets': [],
                **metadata
            }
    
    return averages

def print_results():
    """Print formatted results for marketing reports"""
    averages = calculate_system_averages()
    
    print("=== AUDIT-PROOF BENCHMARK RESULTS ===\n")
    
    # Sort by nDCG@10 performance (available systems first)
    available_systems = [(k, v) for k, v in averages.items() if v['status'] == 'AVAILABLE']
    unavailable_systems = [(k, v) for k, v in averages.items() if v['status'] != 'AVAILABLE']
    
    available_systems.sort(key=lambda x: x[1]['ndcg_10_mean'], reverse=True)
    
    print("AVAILABLE SYSTEMS (Ranked by nDCG@10):")
    print("=" * 50)
    
    for i, (system, data) in enumerate(available_systems):
        rank = i + 1
        print(f"{rank}. {system}")
        print(f"   nDCG@10: {data['ndcg_10_mean']:.3f}")
        print(f"   Recall@50: {data['recall_50_mean']:.3f}") 
        print(f"   P95 Latency: {data['p95_latency_mean']:.1f}ms")
        print(f"   Provenance: {data['provenance']}")
        print(f"   Datasets: {data['dataset_count']} ({', '.join(data['datasets'])})")
        print()
    
    print("QUARANTINED SYSTEMS:")
    print("=" * 20)
    
    for system, data in unavailable_systems:
        print(f"⚠️  {system}")
        print(f"   Status: {data['status']}")
        print(f"   Provenance: {data['provenance']}")
        print(f"   Metrics: None (unavailable)")
        print()
    
    # Calculate performance gaps for marketing
    if len(available_systems) >= 2:
        leader = available_systems[0]
        others = available_systems[1:]
        
        print("COMPETITIVE ADVANTAGES:")
        print("=" * 25)
        
        leader_score = leader[1]['ndcg_10_mean']
        print(f"Market Leader: {leader[0]} ({leader_score:.3f} nDCG@10)")
        
        for system, data in others:
            competitor_score = data['ndcg_10_mean']
            improvement = ((leader_score - competitor_score) / competitor_score) * 100
            print(f"  vs {system}: +{improvement:.1f}% improvement")
        
        print()
    
    return averages

if __name__ == "__main__":
    results = print_results()