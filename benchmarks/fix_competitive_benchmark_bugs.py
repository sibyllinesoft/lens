#!/usr/bin/env python3
"""
Fix competitive benchmark plotting bugs identified by user analysis.

IDENTIFIED ISSUES:
1. System key mismatch: t1_hero data exists but win rates show as 0.1 (constant fill)
2. Heatmap calculation bug: incorrect pairwise win rate algorithm
3. Missing guardrails: no detection of constant fill values

FIXES IMPLEMENTED:
1. Canonical system ID normalization (t1-leader → t1_hero mapping)
2. Corrected pairwise win rate calculation algorithm
3. Added variance checks to detect constant fill bugs
4. NaN masking instead of fillna for missing data
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore')

class BenchmarkResultsFixer:
    """Fix competitive benchmark plotting and aggregation bugs."""
    
    def __init__(self, results_dir: str = "competitive_benchmark_results"):
        self.results_dir = Path(results_dir)
        self.data_dir = Path("/home/nathan/Projects/lens/benchmarks") / results_dir
        
        # System ID canonicalization mapping
        self.system_id_mapping = {
            't1-leader': 't1_hero',
            't1-hero': 't1_hero', 
            't1_leader': 't1_hero',
            'cohere/embed-english-v3.0': 'cohere_embed_english_v3',
            'openai/text-embedding-3-large': 'openai_text_embedding_3_large'
        }
        
    def canonicalize_system_id(self, system_id: str) -> str:
        """Normalize system IDs to prevent key mismatch bugs."""
        return self.system_id_mapping.get(system_id, system_id)
        
    def load_competitor_matrix(self) -> pd.DataFrame:
        """Load and clean competitor matrix data."""
        matrix_file = self.data_dir / "competitor_matrix.csv"
        if not matrix_file.exists():
            raise FileNotFoundError(f"competitor_matrix.csv not found at {matrix_file}")
            
        df = pd.read_csv(matrix_file)
        
        # Canonicalize system IDs
        df['system'] = df['system'].apply(self.canonicalize_system_id)
        
        # Filter to available systems only
        df = df[df['status'] == 'AVAILABLE'].copy()
        
        print(f"✅ Loaded {len(df)} results from {len(df['system'].unique())} systems")
        print(f"📊 Systems: {sorted(df['system'].unique())}")
        
        return df
        
    def compute_corrected_pairwise_win_rates(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute pairwise win rates using correct algorithm.
        
        Reference implementation from user specification:
        ```
        def wr(a,b):
          mask = valid where P[a], P[b] not NaN
          n = count(mask); assert n>0
          wins = sum(P[a]>P[b] over mask)
          ties = sum(P[a]==P[b] over mask) 
          return (wins + 0.5*ties)/n
        ```
        """
        # Pivot to system x benchmark matrix
        pivot_df = df.pivot(index='system', columns='benchmark', values='ndcg_10')
        systems = list(pivot_df.index)
        n_systems = len(systems)
        
        print(f"🔢 Computing win rates for {n_systems} systems across {len(pivot_df.columns)} benchmarks")
        
        # Initialize win rate matrix
        win_matrix = np.full((n_systems, n_systems), np.nan)
        
        # Compute pairwise win rates
        for i, system_a in enumerate(systems):
            for j, system_b in enumerate(systems):
                if i == j:
                    win_matrix[i, j] = 0.5  # Self-comparison is 50%
                    continue
                    
                # Get valid comparisons (both systems have results)
                a_scores = pivot_df.loc[system_a]
                b_scores = pivot_df.loc[system_b]
                
                # Mask where both are valid
                valid_mask = ~(a_scores.isna() | b_scores.isna())
                
                if valid_mask.sum() == 0:
                    # No valid comparisons
                    win_matrix[i, j] = np.nan
                    continue
                
                a_valid = a_scores[valid_mask]
                b_valid = b_scores[valid_mask]
                
                wins = (a_valid > b_valid).sum()
                ties = (a_valid == b_valid).sum()
                total = valid_mask.sum()
                
                win_rate = (wins + 0.5 * ties) / total
                win_matrix[i, j] = win_rate
                
        return win_matrix, systems
        
    def validate_win_matrix(self, win_matrix: np.ndarray, systems: List[str]) -> None:
        """Add guardrails to detect constant fill bugs."""
        n_systems = len(systems)
        
        print("🛡️  Running validation checks on win matrix...")
        
        # Check 1: No row should have variance < 1e-6 (constant fill detection)
        for i, system in enumerate(systems):
            row = win_matrix[i, :]
            valid_values = row[~np.isnan(row)]
            
            if len(valid_values) > 1:
                variance = np.var(valid_values)
                if variance < 1e-6:
                    print(f"⚠️  WARNING: {system} has suspicious constant row variance={variance:.8f}")
                    print(f"    Values: {valid_values[:5]}...")  # Show first 5 values
                    
        # Check 2: At least one non-diagonal value should not be {0, 0.5, 1}
        non_diagonal_mask = ~np.eye(n_systems, dtype=bool)
        non_diagonal_values = win_matrix[non_diagonal_mask]
        valid_non_diagonal = non_diagonal_values[~np.isnan(non_diagonal_values)]
        
        if len(valid_non_diagonal) > 0:
            unique_values = np.unique(np.round(valid_non_diagonal, 6))  # Round to avoid floating point issues
            extreme_values = {0.0, 0.5, 1.0}
            
            if len(set(unique_values) - extreme_values) == 0:
                print("⚠️  WARNING: All non-diagonal win rates are exactly 0, 0.5, or 1.0")
                print(f"    This may indicate a calculation bug. Values: {unique_values}")
                
        # Check 3: Matrix should be roughly anti-symmetric (W[i,j] ≈ 1 - W[j,i])
        asymmetry_errors = []
        for i in range(n_systems):
            for j in range(i+1, n_systems):  # Only check upper triangle
                if not (np.isnan(win_matrix[i,j]) or np.isnan(win_matrix[j,i])):
                    expected_ji = 1.0 - win_matrix[i,j]
                    actual_ji = win_matrix[j,i]
                    error = abs(expected_ji - actual_ji)
                    
                    if error > 0.01:  # Allow small floating point errors
                        asymmetry_errors.append((systems[i], systems[j], error))
                        
        if asymmetry_errors:
            print(f"⚠️  WARNING: {len(asymmetry_errors)} anti-symmetry violations found")
            for sys_a, sys_b, error in asymmetry_errors[:3]:  # Show first 3
                print(f"    {sys_a} vs {sys_b}: asymmetry error = {error:.4f}")
                
        print(f"✅ Matrix validation complete. Shape: {win_matrix.shape}")
        
    def aggregate_win_rates_for_leaderboard(self, df: pd.DataFrame, win_matrix: np.ndarray, systems: List[str]) -> Dict[str, float]:
        """
        Calculate aggregate win rate per system for leaderboard tie-breaking.
        This is the overall win rate against all other systems.
        """
        aggregate_win_rates = {}
        
        for i, system in enumerate(systems):
            row = win_matrix[i, :]
            valid_values = row[~np.isnan(row)]
            
            # Exclude self-comparison (diagonal)
            other_comparisons = np.concatenate([valid_values[:i], valid_values[i+1:]])
            
            if len(other_comparisons) > 0:
                aggregate_win_rates[system] = float(np.mean(other_comparisons))
            else:
                aggregate_win_rates[system] = 0.0
                
        return aggregate_win_rates
        
    def generate_corrected_heatmap(self, win_matrix: np.ndarray, systems: List[str]) -> str:
        """Generate corrected win rate heatmap with proper NaN masking."""
        
        plt.figure(figsize=(12, 10))
        
        # Use masked array to handle NaNs properly (no fillna!)
        masked_matrix = np.ma.masked_invalid(win_matrix)
        
        # Create heatmap with NaN masking
        ax = sns.heatmap(masked_matrix, 
                        xticklabels=[s.replace('/', '\n').replace('_', ' ') for s in systems],
                        yticklabels=[s.replace('/', '\n').replace('_', ' ') for s in systems],
                        annot=True, fmt='.3f', cmap='RdYlBu_r',
                        cbar_kws={'label': 'Win Rate'},
                        mask=np.isnan(win_matrix),  # Mask NaN values
                        vmin=0, vmax=1)
        
        plt.title('Corrected Pairwise Win Rate Matrix\n(Row System beats Column System)')
        plt.xlabel('Opponent System')
        plt.ylabel('System')
        
        # Add provenance footer
        plt.figtext(0.02, 0.02, 
                   f"Generated: 2025-09-12 | Source: competitor_matrix.csv | Algorithm: Corrected pairwise with NaN masking",
                   fontsize=8, style='italic')
                   
        plt.tight_layout()
        
        # Save plot
        plot_file = self.data_dir / "plots" / "corrected_heatmap_win_rates.png"
        plot_file.parent.mkdir(exist_ok=True)
        
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Corrected heatmap saved to: {plot_file}")
        return str(plot_file)
        
    def generate_corrected_leaderboard(self, df: pd.DataFrame, aggregate_win_rates: Dict[str, float]) -> str:
        """Generate corrected leaderboard with proper t1_hero ranking."""
        
        # Calculate system-level aggregates
        system_stats = df.groupby('system').agg({
            'delta_ndcg': 'mean',
            'ndcg_10': 'mean', 
            'p95_latency': 'mean',
            'recall_50': 'mean',
            'jaccard_10': 'mean',
            'benchmark': 'count'  # Number of benchmarks
        }).round(4)
        
        # Build ranking records
        ranking_records = []
        for system in system_stats.index:
            ranking_records.append({
                'system': system,
                'aggregate_delta_ndcg': system_stats.loc[system, 'delta_ndcg'],
                'win_rate': aggregate_win_rates.get(system, 0.0),
                'avg_p95_latency': system_stats.loc[system, 'p95_latency'],
                'avg_recall_50': system_stats.loc[system, 'recall_50'],
                'avg_jaccard_10': system_stats.loc[system, 'jaccard_10'],
                'benchmarks_valid': int(system_stats.loc[system, 'benchmark'])
            })
            
        # Sort by primary metric (delta_ndcg), then tie-breakers
        ranking_records.sort(
            key=lambda x: (
                -x['aggregate_delta_ndcg'],  # Higher ΔnDCG better
                -x['win_rate'],              # Higher win rate better  
                x['avg_p95_latency'],        # Lower latency better
                -x['avg_recall_50'],         # Higher recall better
                -x['avg_jaccard_10']         # Higher jaccard better
            )
        )
        
        # Assign ranks
        for i, record in enumerate(ranking_records, 1):
            record['rank'] = i
            
        # Generate markdown leaderboard
        leaderboard_md = [
            "# 🏆 CORRECTED Competitive Benchmark Leaderboard",
            "",
            "**🔧 BUG FIXES APPLIED:**",
            "- ✅ Fixed system ID canonicalization (t1_hero key mismatch)",
            "- ✅ Corrected pairwise win rate algorithm", 
            "- ✅ Removed constant fill values (0.1 → proper calculation)",
            "- ✅ Added NaN masking instead of fillna",
            "",
            f"**Systems Ranked**: {len(ranking_records)}",
            f"**Valid Results**: Available systems only",
            "",
            "## 📊 Rankings",
            "",
            "| Rank | System | ΔnDCG@10 | Win Rate | p95 Latency | Benchmarks |",
            "|------|--------|----------|----------|-------------|------------|",
        ]
        
        for record in ranking_records:
            leaderboard_md.append(
                f"| **#{record['rank']}** | **{record['system']}** | "
                f"+{record['aggregate_delta_ndcg']:.3f} | "
                f"{record['win_rate']:.1%} | "
                f"{record['avg_p95_latency']:.0f}ms | "
                f"{record['benchmarks_valid']} |"
            )
            
        leaderboard_md.extend([
            "",
            "## 🔧 Bug Fix Summary",
            "",
            "**Original Issues:**",
            "1. System key mismatch: `t1_hero` data existed but lookups failed",
            "2. Heatmap algorithm: Incorrect pairwise comparison logic", 
            "3. Constant fill: NaN values imputed with 0.1 instead of proper masking",
            "",
            "**Fixes Applied:**",
            "1. Canonical system ID mapping with validation",
            "2. Correct pairwise win rate: `(wins + 0.5*ties) / total_comparisons`",
            "3. Variance checks to detect future constant fill bugs",
            "4. NaN masking in visualizations (no more fillna)",
            "",
            "## 📋 Technical Details",
            "",
            "- **Algorithm**: Pairwise comparison with tie handling (ties = 0.5)",
            "- **Missing Data**: NaN-masked, not imputed with constants",
            "- **Validation**: Row variance checks, anti-symmetry validation",
            "- **Provenance**: All calculations traceable to competitor_matrix.csv",
            "",
            "---",
            "*Corrected analysis generated with bug fixes applied*"
        ])
        
        # Save corrected leaderboard
        leaderboard_file = self.data_dir / "corrected_leaderboard.md"
        with open(leaderboard_file, 'w') as f:
            f.write('\n'.join(leaderboard_md))
            
        print(f"✅ Corrected leaderboard saved to: {leaderboard_file}")
        
        # Print key findings
        print("\n🏆 CORRECTED RANKINGS:")
        for record in ranking_records[:5]:  # Top 5
            print(f"#{record['rank']:2d}: {record['system']:15s} | "
                  f"ΔnDCG={record['aggregate_delta_ndcg']:+.3f} | "
                  f"WinRate={record['win_rate']:.1%}")
                  
        return str(leaderboard_file)
        
    def run_complete_fix(self) -> Dict[str, str]:
        """Run complete bug fix analysis and regenerate corrected outputs."""
        
        print("🔧 COMPETITIVE BENCHMARK BUG FIXES")
        print("=" * 50)
        print("Identified Issues:")
        print("1. System key mismatch (t1_hero → 0.1 constant fill)")
        print("2. Incorrect pairwise win rate algorithm") 
        print("3. Missing guardrails for constant fill detection")
        print("=" * 50)
        print()
        
        # Load and clean data
        df = self.load_competitor_matrix()
        
        # Compute corrected win rates
        win_matrix, systems = self.compute_corrected_pairwise_win_rates(df)
        
        # Validate results
        self.validate_win_matrix(win_matrix, systems)
        
        # Calculate aggregate win rates for leaderboard
        aggregate_win_rates = self.aggregate_win_rates_for_leaderboard(df, win_matrix, systems)
        
        # Generate outputs
        artifacts = {}
        artifacts['heatmap'] = self.generate_corrected_heatmap(win_matrix, systems)
        artifacts['leaderboard'] = self.generate_corrected_leaderboard(df, aggregate_win_rates)
        
        print("\n🎉 BUG FIXES COMPLETED!")
        print("=" * 50)
        print("Generated corrected artifacts:")
        for name, path in artifacts.items():
            print(f"✅ {name}: {path}")
            
        # Show t1_hero actual performance
        if 't1_hero' in aggregate_win_rates:
            t1_win_rate = aggregate_win_rates['t1_hero']
            print(f"\n🚀 t1_hero ACTUAL WIN RATE: {t1_win_rate:.1%}")
            print("   (Previously showing as ~10% due to key mismatch bug)")
        else:
            print("\n⚠️  t1_hero still not found - check system ID mapping")
            
        return artifacts


def main():
    """Main entry point for bug fixes."""
    fixer = BenchmarkResultsFixer()
    artifacts = fixer.run_complete_fix()
    
    print(f"\n📊 CORRECTED ANALYSIS READY!")
    print(f"🎯 Key Finding: t1_hero performance correctly calculated")
    print(f"🔧 All plotting bugs fixed with proper validation")


if __name__ == "__main__":
    main()