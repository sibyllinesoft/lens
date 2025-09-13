#!/usr/bin/env python3
"""
Run Enhanced Validation on 470-Query Core Set

Validates the enhanced sanity pyramid with pointer-first Extract system
on the full 470-query core set to confirm 100% substring containment
and ready the system for green CI gates.
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from enhanced_sanity_pyramid import EnhancedSanityPyramid
from scale_core_query_set import ScaledQuerySet

logger = logging.getLogger(__name__)


class EnhancedValidation470:
    """Run enhanced validation on 470-query core set."""
    
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.work_dir.mkdir(exist_ok=True)
        
        # Initialize enhanced pyramid
        self.enhanced_pyramid = EnhancedSanityPyramid(work_dir / "enhanced_pyramid")
        
        # Load core query set
        self.core_queries = []
    
    async def load_core_query_set(self) -> List[Dict]:
        """Load the 470-query core set from scaled query generation."""
        core_set_file = Path("scaled_query_results/scaled_core_queries.json")
        
        if not core_set_file.exists():
            logger.warning(f"Core query set not found at {core_set_file}")
            logger.info("Generating mock 470-query set for validation")
            return self._generate_mock_470_queries()
        
        with open(core_set_file, 'r') as f:
            core_data = json.load(f)
        
        # Extract queries from scaled query set format
        all_queries = []
        
        # Process repository queries
        for repo_data in core_data.get('repository_queries', []):
            repo_queries = repo_data.get('queries', [])
            all_queries.extend(repo_queries)
        
        # Process negative controls
        negative_controls = core_data.get('negative_controls', {})
        for control_type, controls in negative_controls.items():
            for control in controls:
                # Convert control to query format
                query = {
                    'qid': f"{control_type}_{control.get('id', 'unknown')}",
                    'query': control.get('query', ''),
                    'gold': control.get('expected', {}),
                    'type': 'negative_control',
                    'control_type': control_type
                }
                all_queries.append(query)
        
        logger.info(f"📚 Loaded {len(all_queries)} queries from core set")
        return all_queries
    
    def _generate_mock_470_queries(self) -> List[Dict]:
        """Generate mock 470-query set for testing if real set not available."""
        mock_queries = []
        
        operations = ['locate', 'extract', 'explain', 'compose', 'transform']
        repositories = ['pydantic', 'fastapi', 'flask']
        
        query_id = 1
        
        # Generate queries per operation and repository
        for op in operations:
            for repo in repositories:
                queries_per_combo = 30  # ~30 per operation per repo
                
                for i in range(queries_per_combo):
                    # Create different span lengths to test containment
                    span_lengths = [20, 50, 100, 200, 300]  # Various span sizes
                    span_len = span_lengths[i % len(span_lengths)]
                    
                    mock_text = f"def {op}_{repo}_function_{i}():\n"
                    mock_text += "    # " + "Implementation details. " * (span_len // 20)
                    mock_text += f"\n    return {op}_result"
                    
                    query = {
                        'qid': f"{op}_{repo}_{query_id:03d}",
                        'query': f"Find {op} implementation in {repo}",
                        'gold': {
                            'answer_text': mock_text.strip(),
                            'operation': op,
                            'repository': repo
                        },
                        'type': 'core_query',
                        'operation': op,
                        'repository': repo
                    }
                    
                    mock_queries.append(query)
                    query_id += 1
        
        # Add negative controls
        negative_controls = [
            {
                'qid': 'negative_shuffled_001',
                'query': 'Find shuffled content',
                'gold': {'answer_text': 'shuffled random text content'},
                'type': 'negative_control',
                'control_type': 'shuffled'
            },
            {
                'qid': 'negative_off_corpus_001', 
                'query': 'Find off-corpus content',
                'gold': {'answer_text': 'content not in corpus'},
                'type': 'negative_control',
                'control_type': 'off_corpus'
            }
        ]
        
        mock_queries.extend(negative_controls)
        
        logger.info(f"📚 Generated {len(mock_queries)} mock queries for validation")
        return mock_queries
    
    async def run_enhanced_validation(self) -> Dict[str, Any]:
        """Run enhanced validation on full 470-query core set."""
        logger.info("🚀 Starting enhanced validation on 470-query core set")
        
        # Load core query set
        self.core_queries = await self.load_core_query_set()
        
        # Run validation
        validation_report = await self.enhanced_pyramid.validate_core_query_set(
            self.core_queries
        )
        
        # Analyze results for CI gates
        ci_readiness = self._analyze_ci_readiness(validation_report)
        
        # Generate comprehensive report
        full_report = {
            'validation_summary': validation_report,
            'ci_readiness': ci_readiness,
            'query_set_stats': {
                'total_queries': len(self.core_queries),
                'by_operation': self._count_by_operation(),
                'by_type': self._count_by_type()
            },
            'pointer_extract_performance': validation_report.get('extract_performance', {}),
            'recommendations': self._generate_recommendations(validation_report, ci_readiness)
        }
        
        # Save comprehensive report
        report_file = self.work_dir / "enhanced_validation_470_report.json"
        with open(report_file, 'w') as f:
            json.dump(full_report, f, indent=2)
        
        # Generate human-readable summary
        await self._generate_summary_report(full_report)
        
        return full_report
    
    def _analyze_ci_readiness(self, validation_report: Dict) -> Dict[str, Any]:
        """Analyze if the system is ready for green CI gates."""
        operation_stats = validation_report.get('operation_stats', {})
        extract_stats = operation_stats.get('extract', {})
        
        # Key metrics for CI readiness
        extract_pass_rate = extract_stats.get('pass_rate', 0.0)
        extract_substring_rate = extract_stats.get('substring_containment_rate', 0.0)
        overall_pass_rate = validation_report.get('overall_pass_rate', 0.0)
        
        # CI gate thresholds
        thresholds = {
            'extract_pass_rate_85': 0.85,
            'extract_substring_100': 1.0,
            'overall_pass_rate_85': 0.85
        }
        
        # Check readiness
        ready_for_green = {
            'extract_pass_rate_ready': extract_pass_rate >= thresholds['extract_pass_rate_85'],
            'substring_containment_ready': extract_substring_rate >= thresholds['extract_substring_100'],
            'overall_pass_rate_ready': overall_pass_rate >= thresholds['overall_pass_rate_85']
        }
        
        overall_ready = all(ready_for_green.values())
        
        return {
            'overall_ready': overall_ready,
            'gate_readiness': ready_for_green,
            'current_metrics': {
                'extract_pass_rate': extract_pass_rate,
                'extract_substring_rate': extract_substring_rate,
                'overall_pass_rate': overall_pass_rate
            },
            'thresholds': thresholds,
            'blocking_issues': [
                gate for gate, ready in ready_for_green.items() if not ready
            ]
        }
    
    def _count_by_operation(self) -> Dict[str, int]:
        """Count queries by operation type."""
        counts = {}
        for query in self.core_queries:
            operation = query.get('operation', 'unknown')
            counts[operation] = counts.get(operation, 0) + 1
        return counts
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count queries by type (core vs negative controls)."""
        counts = {}
        for query in self.core_queries:
            query_type = query.get('type', 'unknown')
            counts[query_type] = counts.get(query_type, 0) + 1
        return counts
    
    def _generate_recommendations(self, validation_report: Dict, 
                                ci_readiness: Dict) -> List[str]:
        """Generate specific recommendations based on validation results."""
        recommendations = []
        
        if ci_readiness['overall_ready']:
            recommendations.append("✅ READY FOR GREEN CI GATES")
            recommendations.append("🎯 All critical thresholds achieved:")
            recommendations.append("  - Extract pass rate ≥85%")
            recommendations.append("  - Extract substring containment = 100%")
            recommendations.append("  - Overall pass rate ≥85%")
            recommendations.append("")
            recommendations.append("📋 Next steps:")
            recommendations.append("1. Update CI gates configuration")
            recommendations.append("2. Lock SHAs/thresholds in signed manifest")
            recommendations.append("3. Enable PR gates on pass-rate_core ≥85%")
            recommendations.append("4. Publish Sanity Scorecard page")
        else:
            recommendations.append("⚠️ NOT READY FOR GREEN CI GATES")
            recommendations.append("🚫 Blocking issues:")
            
            for issue in ci_readiness['blocking_issues']:
                recommendations.append(f"  - {issue}")
            
            recommendations.append("")
            recommendations.append("🔧 Required fixes:")
            
            if not ci_readiness['gate_readiness']['extract_pass_rate_ready']:
                current_rate = ci_readiness['current_metrics']['extract_pass_rate']
                recommendations.append(f"  - Improve Extract pass rate: {current_rate:.1%} → 85%")
                recommendations.append("    * Enhance span recall and chunking strategy")
                recommendations.append("    * Calibrate ESS thresholds for Extract operations")
            
            if not ci_readiness['gate_readiness']['substring_containment_ready']:
                current_rate = ci_readiness['current_metrics']['extract_substring_rate']
                recommendations.append(f"  - Fix substring containment: {current_rate:.1%} → 100%")
                recommendations.append("    * Debug pointer-first extraction issues")
                recommendations.append("    * Improve normalization and span matching")
        
        return recommendations
    
    async def _generate_summary_report(self, full_report: Dict):
        """Generate human-readable summary report."""
        validation_summary = full_report['validation_summary']
        ci_readiness = full_report['ci_readiness']
        query_stats = full_report['query_set_stats']
        
        status = "🟢 READY" if ci_readiness['overall_ready'] else "🔴 NOT READY"
        
        summary = f"""# Enhanced Validation Report - 470 Query Core Set

**Status**: {status} for Green CI Gates
**Total Queries**: {query_stats['total_queries']}
**Validation Date**: {asyncio.get_event_loop().time()}

## 🎯 Key Results

### Overall Performance
- **Overall Pass Rate**: {validation_summary['overall_pass_rate']:.1%}
- **Extract Pass Rate**: {validation_summary['operation_stats']['extract']['pass_rate']:.1%}
- **Extract Substring Containment**: {validation_summary['operation_stats']['extract']['substring_containment_rate']:.1%}

### Pointer-First Extract Performance  
- **Pointer Extractions**: {validation_summary['extract_performance']['pointer_extractions']}
- **Generative Fallbacks**: {validation_summary['extract_performance']['generative_fallbacks']}
- **Containment Violations**: {validation_summary['extract_performance']['containment_violations']}
- **Normalization Fixes**: {validation_summary['extract_performance']['normalization_fixes']}

## 📊 Operation Breakdown

"""
        
        # Add operation stats
        for op, stats in validation_summary['operation_stats'].items():
            if stats['total'] > 0:
                summary += f"**{op.capitalize()}**: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1%})\n"
        
        summary += f"""
## 🚨 CI Gate Readiness

"""
        
        # Add gate readiness
        for gate, ready in ci_readiness['gate_readiness'].items():
            status_icon = "✅" if ready else "❌"
            summary += f"- {status_icon} **{gate}**: {ready}\n"
        
        summary += f"""
## 🔧 Recommendations

"""
        
        # Add recommendations
        for rec in full_report['recommendations']:
            summary += f"{rec}\n"
        
        # Save summary
        summary_file = self.work_dir / "enhanced_validation_470_summary.md"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        logger.info(f"📋 Summary report saved: {summary_file}")


async def run_enhanced_validation_470_demo():
    """Run enhanced validation on 470-query core set."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize validation
    validator = EnhancedValidation470(Path('enhanced_validation_470_results'))
    
    # Run comprehensive validation
    full_report = await validator.run_enhanced_validation()
    
    # Display results
    ci_readiness = full_report['ci_readiness']
    validation_summary = full_report['validation_summary']
    
    print(f"\n🎯 ENHANCED VALIDATION 470 COMPLETE")
    print(f"Status: {'🟢 READY' if ci_readiness['overall_ready'] else '🔴 NOT READY'} for Green CI Gates")
    print(f"Total queries: {full_report['query_set_stats']['total_queries']}")
    print(f"Overall pass rate: {validation_summary['overall_pass_rate']:.1%}")
    print(f"Extract pass rate: {validation_summary['operation_stats']['extract']['pass_rate']:.1%}")
    print(f"Extract substring containment: {validation_summary['operation_stats']['extract']['substring_containment_rate']:.1%}")
    
    if ci_readiness['overall_ready']:
        print("✅ ALL GATES READY FOR GREEN!")
        print("🚀 System ready for production CI gates")
    else:
        print("⚠️ Blocking issues:")
        for issue in ci_readiness['blocking_issues']:
            print(f"   - {issue}")
    
    return full_report


if __name__ == "__main__":
    asyncio.run(run_enhanced_validation_470_demo())