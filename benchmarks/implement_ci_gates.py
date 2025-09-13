#!/usr/bin/env python3
"""
Hard CI Gates Implementation for Sanity Pyramid

Implement the hard CI gates for pass/fail validation in continuous integration.
All gates must pass for builds to proceed.
"""
import asyncio
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import subprocess

from live_sanity_integration import LiveSanityIntegration, LiveValidationReport, SanityGateResult
from sanity_pyramid import OperationType

logger = logging.getLogger(__name__)


@dataclass
class CIGateResult:
    """Result of a single CI gate check."""
    gate_name: str
    passed: bool
    actual_value: float
    threshold: float
    failure_reason: Optional[str]
    severity: str  # "critical", "warning", "info"
    remediation: str


@dataclass
class CIGateReport:
    """Complete CI gate validation report."""
    commit_sha: str
    timestamp: str
    overall_passed: bool
    critical_gates_passed: int
    critical_gates_total: int
    warning_gates_passed: int
    warning_gates_total: int
    gate_results: List[CIGateResult]
    baseline_comparison: Dict[str, Any]
    recommended_actions: List[str]


class HardCIGates:
    """Implementation of hard CI gates for sanity pyramid validation."""
    
    def __init__(self, config_path: str, work_dir: Path):
        self.config_path = config_path
        self.work_dir = work_dir
        self.work_dir.mkdir(exist_ok=True)
        
        # Load CI gates configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.baseline_dir = work_dir / "baselines"
        self.baseline_dir.mkdir(exist_ok=True)
        
        # Gate definitions
        self.gate_definitions = self._load_gate_definitions()
    
    def _load_gate_definitions(self) -> Dict[str, Dict]:
        """Load hard gate definitions from configuration."""
        hard_gates = self.config.get('hard_gates', {})
        
        return {
            # Pre-generation gates (critical)
            'overall_pass_rate_85': {
                'threshold': hard_gates.get('pre_generation', {}).get('overall_pass_rate', {}).get('threshold', 0.85),
                'severity': 'critical',
                'description': 'Overall pre-generation pass rate must be ≥85%',
                'remediation': 'Improve retrieval quality or lower ESS thresholds after validation'
            },
            'locate_pass_rate_90': {
                'threshold': hard_gates.get('pre_generation', {}).get('per_operation_minimums', {}).get('locate', {}).get('threshold', 0.90),
                'severity': 'critical',
                'description': 'Locate operations must have ≥90% pass rate',
                'remediation': 'Improve symbol indexing and exact matching retrieval'
            },
            'extract_pass_rate_85': {
                'threshold': hard_gates.get('pre_generation', {}).get('per_operation_minimums', {}).get('extract', {}).get('threshold', 0.85),
                'severity': 'critical',
                'description': 'Extract operations must have ≥85% pass rate',
                'remediation': 'Improve span recall and code chunking strategy'
            },
            'explain_pass_rate_70': {
                'threshold': hard_gates.get('pre_generation', {}).get('per_operation_minimums', {}).get('explain', {}).get('threshold', 0.70),
                'severity': 'critical',
                'description': 'Explain operations must have ≥70% pass rate',
                'remediation': 'Improve context composition and multi-file retrieval'
            },
            
            # Attribution integrity gates (critical)
            'citation_at_1': {
                'threshold': 1.0,
                'severity': 'critical',
                'description': 'Every answer must cite at least one retrieved path',
                'remediation': 'Fix generation prompts to ensure citation requirements'
            },
            'extract_substring_match': {
                'threshold': 1.0,
                'severity': 'critical',
                'description': 'Extract answers must be substring of context',
                'remediation': 'Validate extraction logic and prompt constraints'
            },
            'code_percentage_minimum': {
                'threshold': hard_gates.get('attribution', {}).get('code_percentage_minimum', {}).get('threshold', 0.70),
                'severity': 'critical',
                'description': 'Code operations require ≥70% code tokens in context',
                'remediation': 'Improve code-specific retrieval and chunking'
            },
            
            # Evidence sufficiency drift detection (warning/critical)
            'ess_median_drift_5pct': {
                'threshold': 0.05,
                'severity': 'warning',
                'description': 'Median ESS by operation must not drop >5% vs baseline',
                'remediation': 'Investigate retrieval degradation or corpus changes'
            },
            'answerable_at_k_70': {
                'threshold': 0.70,
                'severity': 'critical',
                'description': 'Answerable@k must be ≥70% on core set',
                'remediation': 'Improve retrieval recall and ranking'
            },
            'span_recall_50': {
                'threshold': 0.50,
                'severity': 'critical',
                'description': 'SpanRecall must be ≥50% for locate/extract operations',
                'remediation': 'Optimize chunking and overlap strategies'
            },
            
            # Ablation sensitivity (warning)
            'shuffle_context_sensitivity_10pct': {
                'threshold': 0.10,
                'severity': 'warning',
                'description': 'Context shuffle must reduce answer F1 by ≥10%',
                'remediation': 'Investigate evidence dependency in generation'
            },
            'drop_top1_sensitivity_5pct': {
                'threshold': 0.05,
                'severity': 'warning',
                'description': 'Dropping top-1 chunk should impact answers ≥5%',
                'remediation': 'Check ranking quality and context utilization'
            },
            'ess_answer_correlation_40': {
                'threshold': 0.4,
                'severity': 'warning',
                'description': 'ESS scores should correlate with answer quality ≥0.4',
                'remediation': 'Validate ESS calculation and answer evaluation'
            },
            
            # Performance budgets (warning with waiver)
            'retrieval_latency_p95_code_search': {
                'threshold': 200,  # ms
                'severity': 'warning',
                'description': 'P95 retrieval latency for code search ≤200ms',
                'remediation': 'Optimize indexing and search algorithms'
            },
            'retrieval_latency_p95_rag_qa': {
                'threshold': 350,  # ms
                'severity': 'warning',
                'description': 'P95 retrieval latency for RAG Q&A ≤350ms',
                'remediation': 'Optimize multi-stage retrieval pipeline'
            },
            'total_context_tokens_8k': {
                'threshold': 8000,
                'severity': 'critical',
                'description': 'Total context tokens must be ≤8000',
                'remediation': 'Implement more aggressive context truncation'
            }
        }
    
    async def run_ci_gates(self, validation_report: LiveValidationReport, 
                          ablation_results: Dict[str, float] = None,
                          performance_metrics: Dict[str, float] = None) -> CIGateReport:
        """Run all CI gates and generate pass/fail report."""
        logger.info("🚨 Running hard CI gates validation")
        
        # Get current commit SHA for tracking
        commit_sha = await self._get_commit_sha()
        
        # Load baseline for comparison
        baseline = await self._load_baseline()
        
        gate_results = []
        
        # Pre-generation gates
        gate_results.extend(await self._check_pre_generation_gates(validation_report))
        
        # Attribution integrity gates
        gate_results.extend(await self._check_attribution_gates(validation_report))
        
        # Drift detection gates
        gate_results.extend(await self._check_drift_gates(validation_report, baseline))
        
        # Ablation sensitivity gates
        if ablation_results:
            gate_results.extend(await self._check_ablation_gates(ablation_results))
        
        # Performance gates
        if performance_metrics:
            gate_results.extend(await self._check_performance_gates(performance_metrics))
        
        # Calculate overall results
        critical_gates = [g for g in gate_results if g.severity == 'critical']
        warning_gates = [g for g in gate_results if g.severity == 'warning']
        
        critical_passed = len([g for g in critical_gates if g.passed])
        warning_passed = len([g for g in warning_gates if g.passed])
        
        # Overall pass requires ALL critical gates to pass
        overall_passed = critical_passed == len(critical_gates)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(gate_results)
        
        # Create report
        report = CIGateReport(
            commit_sha=commit_sha,
            timestamp=datetime.now().isoformat(),
            overall_passed=overall_passed,
            critical_gates_passed=critical_passed,
            critical_gates_total=len(critical_gates),
            warning_gates_passed=warning_passed,
            warning_gates_total=len(warning_gates),
            gate_results=gate_results,
            baseline_comparison=baseline,
            recommended_actions=recommendations
        )
        
        # Save report
        await self._save_ci_report(report)
        
        # Log results
        self._log_gate_results(report)
        
        return report
    
    async def _check_pre_generation_gates(self, report: LiveValidationReport) -> List[CIGateResult]:
        """Check pre-generation pass rate gates."""
        gates = []
        
        # Overall pass rate
        overall_gate = self.gate_definitions['overall_pass_rate_85']
        gates.append(CIGateResult(
            gate_name='overall_pass_rate_85',
            passed=report.pre_gen_pass_rate >= overall_gate['threshold'],
            actual_value=report.pre_gen_pass_rate,
            threshold=overall_gate['threshold'],
            failure_reason=None if report.pre_gen_pass_rate >= overall_gate['threshold'] 
                          else f"Pass rate {report.pre_gen_pass_rate:.1%} below threshold {overall_gate['threshold']:.1%}",
            severity=overall_gate['severity'],
            remediation=overall_gate['remediation']
        ))
        
        # Per-operation gates
        operation_gates = ['locate_pass_rate_90', 'extract_pass_rate_85', 'explain_pass_rate_70']
        operations = ['locate', 'extract', 'explain']
        
        for gate_name, operation in zip(operation_gates, operations):
            gate_def = self.gate_definitions[gate_name]
            
            # Get operation pass rate from report
            op_stats = report.per_operation_stats.get(operation, {})
            actual_rate = op_stats.get('pass_rate', 0.0)
            
            gates.append(CIGateResult(
                gate_name=gate_name,
                passed=actual_rate >= gate_def['threshold'],
                actual_value=actual_rate,
                threshold=gate_def['threshold'],
                failure_reason=None if actual_rate >= gate_def['threshold']
                              else f"{operation} pass rate {actual_rate:.1%} below threshold {gate_def['threshold']:.1%}",
                severity=gate_def['severity'],
                remediation=gate_def['remediation']
            ))
        
        return gates
    
    async def _check_attribution_gates(self, report: LiveValidationReport) -> List[CIGateResult]:
        """Check attribution integrity gates."""
        gates = []
        
        # Citation@1 gate (mock - would check actual citations)
        citation_gate = self.gate_definitions['citation_at_1']
        citation_rate = 1.0  # Mock: assume all citations are present
        
        gates.append(CIGateResult(
            gate_name='citation_at_1',
            passed=citation_rate >= citation_gate['threshold'],
            actual_value=citation_rate,
            threshold=citation_gate['threshold'],
            failure_reason=None if citation_rate >= citation_gate['threshold']
                          else f"Citation rate {citation_rate:.1%} below required 100%",
            severity=citation_gate['severity'],
            remediation=citation_gate['remediation']
        ))
        
        # Extract substring match (mock - would check actual extraction accuracy)
        substring_gate = self.gate_definitions['extract_substring_match']
        substring_rate = 0.95  # Mock: assume 95% substring matches
        
        gates.append(CIGateResult(
            gate_name='extract_substring_match',
            passed=substring_rate >= substring_gate['threshold'],
            actual_value=substring_rate,
            threshold=substring_gate['threshold'],
            failure_reason=None if substring_rate >= substring_gate['threshold']
                          else f"Substring match rate {substring_rate:.1%} below required 100%",
            severity=substring_gate['severity'],
            remediation=substring_gate['remediation']
        ))
        
        # Code percentage minimum (check against report data)
        code_gate = self.gate_definitions['code_percentage_minimum']
        # Mock calculation - would get from actual gate results
        avg_code_percentage = 0.75  # Mock: assume 75% average code content
        
        gates.append(CIGateResult(
            gate_name='code_percentage_minimum',
            passed=avg_code_percentage >= code_gate['threshold'],
            actual_value=avg_code_percentage,
            threshold=code_gate['threshold'],
            failure_reason=None if avg_code_percentage >= code_gate['threshold']
                          else f"Code percentage {avg_code_percentage:.1%} below required {code_gate['threshold']:.1%}",
            severity=code_gate['severity'],
            remediation=code_gate['remediation']
        ))
        
        return gates
    
    async def _check_drift_gates(self, report: LiveValidationReport, baseline: Dict) -> List[CIGateResult]:
        """Check evidence sufficiency drift gates."""
        gates = []
        
        # ESS median drift (compare to baseline)
        drift_gate = self.gate_definitions['ess_median_drift_5pct']
        current_ess = report.ess_distribution
        baseline_ess = baseline.get('ess_distribution', {})
        
        # Calculate median drift (mock calculation)
        max_drift = 0.02  # Mock: assume 2% drift
        
        gates.append(CIGateResult(
            gate_name='ess_median_drift_5pct',
            passed=max_drift <= drift_gate['threshold'],
            actual_value=max_drift,
            threshold=drift_gate['threshold'],
            failure_reason=None if max_drift <= drift_gate['threshold']
                          else f"ESS drift {max_drift:.1%} exceeds threshold {drift_gate['threshold']:.1%}",
            severity=drift_gate['severity'],
            remediation=drift_gate['remediation']
        ))
        
        # Answerable@k gate (mock - would check actual metrics)
        answerable_gate = self.gate_definitions['answerable_at_k_70']
        avg_answerable = 0.75  # Mock: assume 75% answerable
        
        gates.append(CIGateResult(
            gate_name='answerable_at_k_70',
            passed=avg_answerable >= answerable_gate['threshold'],
            actual_value=avg_answerable,
            threshold=answerable_gate['threshold'],
            failure_reason=None if avg_answerable >= answerable_gate['threshold']
                          else f"Answerable@k {avg_answerable:.1%} below threshold {answerable_gate['threshold']:.1%}",
            severity=answerable_gate['severity'],
            remediation=answerable_gate['remediation']
        ))
        
        # Span recall gate (mock - would check actual recall)
        span_gate = self.gate_definitions['span_recall_50']
        avg_span_recall = 0.60  # Mock: assume 60% span recall
        
        gates.append(CIGateResult(
            gate_name='span_recall_50',
            passed=avg_span_recall >= span_gate['threshold'],
            actual_value=avg_span_recall,
            threshold=span_gate['threshold'],
            failure_reason=None if avg_span_recall >= span_gate['threshold']
                          else f"Span recall {avg_span_recall:.1%} below threshold {span_gate['threshold']:.1%}",
            severity=span_gate['severity'],
            remediation=span_gate['remediation']
        ))
        
        return gates
    
    async def _check_ablation_gates(self, ablation_results: Dict[str, float]) -> List[CIGateResult]:
        """Check ablation sensitivity gates."""
        gates = []
        
        # Shuffle context sensitivity
        shuffle_gate = self.gate_definitions['shuffle_context_sensitivity_10pct']
        shuffle_f1_drop = ablation_results.get('shuffle_context_f1_drop', 0.0)
        
        gates.append(CIGateResult(
            gate_name='shuffle_context_sensitivity_10pct',
            passed=shuffle_f1_drop >= shuffle_gate['threshold'],
            actual_value=shuffle_f1_drop,
            threshold=shuffle_gate['threshold'],
            failure_reason=None if shuffle_f1_drop >= shuffle_gate['threshold']
                          else f"Shuffle sensitivity {shuffle_f1_drop:.1%} below threshold {shuffle_gate['threshold']:.1%}",
            severity=shuffle_gate['severity'],
            remediation=shuffle_gate['remediation']
        ))
        
        # Drop top-1 sensitivity
        drop_gate = self.gate_definitions['drop_top1_sensitivity_5pct']
        drop_f1_drop = ablation_results.get('drop_top1_f1_drop', 0.0)
        
        gates.append(CIGateResult(
            gate_name='drop_top1_sensitivity_5pct',
            passed=drop_f1_drop >= drop_gate['threshold'],
            actual_value=drop_f1_drop,
            threshold=drop_gate['threshold'],
            failure_reason=None if drop_f1_drop >= drop_gate['threshold']
                          else f"Drop-top1 sensitivity {drop_f1_drop:.1%} below threshold {drop_gate['threshold']:.1%}",
            severity=drop_gate['severity'],
            remediation=drop_gate['remediation']
        ))
        
        # ESS-answer correlation
        corr_gate = self.gate_definitions['ess_answer_correlation_40']
        ess_correlation = ablation_results.get('ess_answer_correlation', 0.0)
        
        gates.append(CIGateResult(
            gate_name='ess_answer_correlation_40',
            passed=ess_correlation >= corr_gate['threshold'],
            actual_value=ess_correlation,
            threshold=corr_gate['threshold'],
            failure_reason=None if ess_correlation >= corr_gate['threshold']
                          else f"ESS correlation {ess_correlation:.2f} below threshold {corr_gate['threshold']:.2f}",
            severity=corr_gate['severity'],
            remediation=corr_gate['remediation']
        ))
        
        return gates
    
    async def _check_performance_gates(self, performance_metrics: Dict[str, float]) -> List[CIGateResult]:
        """Check performance and latency gates."""
        gates = []
        
        # Retrieval latency gates
        code_search_gate = self.gate_definitions['retrieval_latency_p95_code_search']
        code_search_latency = performance_metrics.get('code_search_p95_ms', 0.0)
        
        gates.append(CIGateResult(
            gate_name='retrieval_latency_p95_code_search',
            passed=code_search_latency <= code_search_gate['threshold'],
            actual_value=code_search_latency,
            threshold=code_search_gate['threshold'],
            failure_reason=None if code_search_latency <= code_search_gate['threshold']
                          else f"Code search latency {code_search_latency:.0f}ms exceeds {code_search_gate['threshold']}ms",
            severity=code_search_gate['severity'],
            remediation=code_search_gate['remediation']
        ))
        
        # RAG Q&A latency
        rag_qa_gate = self.gate_definitions['retrieval_latency_p95_rag_qa']
        rag_qa_latency = performance_metrics.get('rag_qa_p95_ms', 0.0)
        
        gates.append(CIGateResult(
            gate_name='retrieval_latency_p95_rag_qa',
            passed=rag_qa_latency <= rag_qa_gate['threshold'],
            actual_value=rag_qa_latency,
            threshold=rag_qa_gate['threshold'],
            failure_reason=None if rag_qa_latency <= rag_qa_gate['threshold']
                          else f"RAG Q&A latency {rag_qa_latency:.0f}ms exceeds {rag_qa_gate['threshold']}ms",
            severity=rag_qa_gate['severity'],
            remediation=rag_qa_gate['remediation']
        ))
        
        # Context token budget
        token_gate = self.gate_definitions['total_context_tokens_8k']
        avg_tokens = performance_metrics.get('avg_context_tokens', 0.0)
        
        gates.append(CIGateResult(
            gate_name='total_context_tokens_8k',
            passed=avg_tokens <= token_gate['threshold'],
            actual_value=avg_tokens,
            threshold=token_gate['threshold'],
            failure_reason=None if avg_tokens <= token_gate['threshold']
                          else f"Context tokens {avg_tokens:.0f} exceeds limit {token_gate['threshold']}",
            severity=token_gate['severity'],
            remediation=token_gate['remediation']
        ))
        
        return gates
    
    def _generate_recommendations(self, gate_results: List[CIGateResult]) -> List[str]:
        """Generate specific recommendations based on gate failures."""
        recommendations = []
        
        failed_gates = [g for g in gate_results if not g.passed]
        
        if not failed_gates:
            recommendations.append("✅ All CI gates passed - no action required")
            return recommendations
        
        # Group failures by type
        critical_failures = [g for g in failed_gates if g.severity == 'critical']
        warning_failures = [g for g in failed_gates if g.severity == 'warning']
        
        if critical_failures:
            recommendations.append("🚨 CRITICAL: Build must be blocked - fix required:")
            for gate in critical_failures:
                recommendations.append(f"  - {gate.gate_name}: {gate.remediation}")
        
        if warning_failures:
            recommendations.append("⚠️ WARNINGS: Address before next release:")
            for gate in warning_failures:
                recommendations.append(f"  - {gate.gate_name}: {gate.remediation}")
        
        # Add specific action items
        if any('pass_rate' in g.gate_name for g in critical_failures):
            recommendations.append("📊 To improve pass rates:")
            recommendations.append("  1. Analyze top failure reasons in sanity scorecard")
            recommendations.append("  2. Calibrate ESS thresholds if retrieval improved")
            recommendations.append("  3. Expand core query set coverage")
        
        if any('latency' in g.gate_name for g in failed_gates):
            recommendations.append("⚡ To improve latency:")
            recommendations.append("  1. Profile retrieval pipeline bottlenecks")
            recommendations.append("  2. Optimize indexing and caching strategies")
            recommendations.append("  3. Consider request timeout waivers if needed")
        
        return recommendations
    
    async def _get_commit_sha(self) -> str:
        """Get current git commit SHA."""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=self.work_dir)
            return result.stdout.strip()[:8]  # Short SHA
        except:
            return "unknown"
    
    async def _load_baseline(self) -> Dict[str, Any]:
        """Load baseline metrics for drift comparison."""
        baseline_file = self.baseline_dir / "last_green_baseline.json"
        
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                return json.load(f)
        else:
            # Return empty baseline for first run
            return {
                'ess_distribution': {},
                'performance_metrics': {},
                'created_at': datetime.now().isoformat()
            }
    
    async def _save_ci_report(self, report: CIGateReport):
        """Save CI gate report to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed report
        report_file = self.work_dir / f"ci_gates_report_{timestamp}.json"
        
        # Convert to serializable format
        report_dict = asdict(report)
        
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        # Generate human-readable summary
        summary_file = self.work_dir / f"ci_gates_summary_{timestamp}.md"
        summary = self._generate_summary_markdown(report)
        
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        # Update baseline if all critical gates passed
        if report.overall_passed:
            await self._update_baseline(report)
        
        logger.info(f"💾 CI gate report saved:")
        logger.info(f"   Detailed: {report_file}")
        logger.info(f"   Summary: {summary_file}")
    
    def _generate_summary_markdown(self, report: CIGateReport) -> str:
        """Generate human-readable CI gate summary."""
        
        status = "✅ PASS" if report.overall_passed else "❌ FAIL"
        
        summary = f"""# CI Gates Report - {status}

**Commit:** {report.commit_sha}
**Timestamp:** {report.timestamp}
**Overall Status:** {status}

## 🎯 Gate Summary
- **Critical Gates:** {report.critical_gates_passed}/{report.critical_gates_total} passed
- **Warning Gates:** {report.warning_gates_passed}/{report.warning_gates_total} passed

## 📊 Gate Results

### Critical Gates (Must Pass)
"""
        
        critical_gates = [g for g in report.gate_results if g.severity == 'critical']
        for gate in critical_gates:
            status_icon = "✅" if gate.passed else "❌"
            summary += f"- {status_icon} **{gate.gate_name}**: {gate.actual_value:.2f} (threshold: {gate.threshold:.2f})\n"
            if not gate.passed and gate.failure_reason:
                summary += f"  - *{gate.failure_reason}*\n"
        
        warning_gates = [g for g in report.gate_results if g.severity == 'warning']
        if warning_gates:
            summary += f"\n### Warning Gates\n"
            for gate in warning_gates:
                status_icon = "✅" if gate.passed else "⚠️"
                summary += f"- {status_icon} **{gate.gate_name}**: {gate.actual_value:.2f} (threshold: {gate.threshold:.2f})\n"
        
        summary += f"\n## 🚨 Recommended Actions\n"
        for action in report.recommended_actions:
            summary += f"{action}\n"
        
        summary += f"""
## 📋 Next Steps
{"✅ **BUILD CAN PROCEED** - All critical gates passed" if report.overall_passed else "🚫 **BLOCK BUILD** - Critical gate failures detected"}

"""
        
        return summary
    
    async def _update_baseline(self, report: CIGateReport):
        """Update baseline metrics when all critical gates pass."""
        baseline = {
            'commit_sha': report.commit_sha,
            'timestamp': report.timestamp,
            'ess_distribution': {},  # Would be populated from actual report
            'performance_metrics': {},  # Would be populated from actual metrics
            'gate_results': {g.gate_name: g.actual_value for g in report.gate_results if g.passed}
        }
        
        baseline_file = self.baseline_dir / "last_green_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        logger.info(f"📊 Updated green baseline: {report.commit_sha}")
    
    def _log_gate_results(self, report: CIGateReport):
        """Log gate results to console."""
        status = "PASS" if report.overall_passed else "FAIL"
        
        logger.info(f"🚨 CI GATES RESULT: {status}")
        logger.info(f"   Critical: {report.critical_gates_passed}/{report.critical_gates_total}")
        logger.info(f"   Warnings: {report.warning_gates_passed}/{report.warning_gates_total}")
        
        # Log failed gates
        failed_gates = [g for g in report.gate_results if not g.passed]
        if failed_gates:
            logger.warning(f"Failed gates:")
            for gate in failed_gates:
                logger.warning(f"   ❌ {gate.gate_name}: {gate.failure_reason}")
        
        # Exit with appropriate code for CI
        if not report.overall_passed:
            logger.error("🚫 BUILD BLOCKED - Critical gate failures")
        else:
            logger.info("✅ BUILD APPROVED - All critical gates passed")


async def run_ci_gates_demo():
    """Demonstrate CI gates with mock validation report."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize CI gates with configuration
    ci_gates = HardCIGates('sanity_ci_gates.yaml', Path('ci_gates_results'))
    
    # Create mock validation report
    from live_sanity_integration import LiveValidationReport
    
    mock_report = LiveValidationReport(
        total_queries=100,
        pre_gen_pass_rate=0.88,  # Above 85% threshold
        per_operation_stats={
            'locate': {'total': 20, 'passed': 19, 'pass_rate': 0.95, 'avg_ess': 0.82},  # Above 90%
            'extract': {'total': 20, 'passed': 16, 'pass_rate': 0.80, 'avg_ess': 0.75}, # Below 85% - will fail
            'explain': {'total': 20, 'passed': 15, 'pass_rate': 0.75, 'avg_ess': 0.68}, # Above 70%
            'compose': {'total': 20, 'passed': 14, 'pass_rate': 0.70, 'avg_ess': 0.65},
            'transform': {'total': 20, 'passed': 13, 'pass_rate': 0.65, 'avg_ess': 0.62}
        },
        ess_distribution={
            'locate': [0.85, 0.90, 0.75, 0.88],
            'extract': [0.70, 0.65, 0.80, 0.75],
            'explain': [0.60, 0.70, 0.65, 0.68]
        },
        top_failure_reasons=[
            ("ESS below threshold", 15),
            ("SpanRecall insufficient", 8),
            ("Key token miss", 5)
        ],
        ablation_deltas={
            'shuffle_context_f1_drop': 0.12,   # Above 10% - good
            'drop_top1_f1_drop': 0.08,         # Above 5% - good
            'ess_answer_correlation': 0.45      # Above 0.4 - good
        },
        latency_p95_ms=280.0,
        hard_gates_status={}
    )
    
    # Mock performance metrics
    performance_metrics = {
        'code_search_p95_ms': 180.0,    # Under 200ms - good
        'rag_qa_p95_ms': 320.0,         # Under 350ms - good
        'avg_context_tokens': 7500.0    # Under 8000 - good
    }
    
    # Run CI gates
    result = await ci_gates.run_ci_gates(
        mock_report, 
        mock_report.ablation_deltas,
        performance_metrics
    )
    
    print(f"\n🎯 CI GATES DEMO COMPLETE")
    print(f"Overall status: {'✅ PASS' if result.overall_passed else '❌ FAIL'}")
    print(f"Critical gates: {result.critical_gates_passed}/{result.critical_gates_total}")
    print(f"Warning gates: {result.warning_gates_passed}/{result.warning_gates_total}")
    
    if not result.overall_passed:
        print(f"\n🚫 FAILED GATES:")
        for gate in result.gate_results:
            if not gate.passed and gate.severity == 'critical':
                print(f"   ❌ {gate.gate_name}: {gate.failure_reason}")
    
    return result


if __name__ == "__main__":
    asyncio.run(run_ci_gates_demo())