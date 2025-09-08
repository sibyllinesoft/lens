#!/usr/bin/env node

/**
 * Phase 2 Quality Gate Validation
 * 
 * Validates the gates as specified in TODO.md:
 * - ΔnDCG@10 ≥ +2% (p<0.05) - PRIMARY GATE
 * - Recall@50 ≥ baseline (maintained from Phase 1)
 * - spans ≥98%
 * - hard-negative leakage into top-5 ≤ +1.5% abs
 */

const fs = require('fs');

function validatePhase2Gates() {
  console.log('🎯 Validating Phase 2 Quality Gates...\n');
  
  try {
    // Load baseline and phase2 results
    const baseline = JSON.parse(fs.readFileSync('results/baseline/baseline_metrics.json', 'utf8'));
    const phase2 = JSON.parse(fs.readFileSync('results/phase2/phase2_results.json', 'utf8'));
    
    const gates = [];
    let allPassed = true;
    
    // Gate 1: ΔnDCG@10 ≥ +2% (PRIMARY GATE)
    const ndcgDelta = ((phase2.metrics.ndcg_at_10 - baseline.metrics.ndcg_at_10) / baseline.metrics.ndcg_at_10) * 100;
    const ndcgGate = {
      name: 'ΔnDCG@10 ≥ +2%',
      required: '≥ +2%',
      actual: `+${ndcgDelta.toFixed(1)}%`,
      passed: ndcgDelta >= 2.0,
      critical: true,
      primary: true
    };
    gates.push(ndcgGate);
    if (!ndcgGate.passed) allPassed = false;
    
    // Gate 2: Recall@50 ≥ baseline (maintained)
    const recallMaintained = phase2.metrics.recall_at_50 >= baseline.metrics.recall_at_50;
    const recallGate = {
      name: 'Recall@50 ≥ baseline',
      required: `≥ ${baseline.metrics.recall_at_50.toFixed(3)}`,
      actual: phase2.metrics.recall_at_50.toFixed(3),
      passed: recallMaintained,
      critical: true
    };
    gates.push(recallGate);
    if (!recallGate.passed) allPassed = false;
    
    // Gate 3: Span coverage ≥98%
    const spanCoverage = phase2.span_coverage * 100;
    const spanGate = {
      name: 'spans ≥ 98%',
      required: '≥ 98%',
      actual: `${spanCoverage.toFixed(1)}%`,
      passed: spanCoverage >= 98.0,
      critical: true
    };
    gates.push(spanGate);
    if (!spanGate.passed) allPassed = false;
    
    // Gate 4: Hard-negative leakage ≤ +1.5% abs
    const leakage = phase2.hard_negative_leakage.top_5_leakage * 100;
    const leakageGate = {
      name: 'hard-negative leakage ≤ +1.5% abs',
      required: '≤ 1.5%',
      actual: `${leakage.toFixed(1)}%`,
      passed: leakage <= 1.5,
      critical: false
    };
    gates.push(leakageGate);
    if (!leakageGate.passed) allPassed = false;
    
    // Display results
    console.log('🎯 Phase 2 Quality Gate Results:');
    console.log('='.repeat(80));
    
    gates.forEach(gate => {
      const status = gate.passed ? '✅ PASS' : '❌ FAIL';
      const critical = gate.critical ? ' (CRITICAL)' : '';
      const primary = gate.primary ? ' (PRIMARY)' : '';
      console.log(`${status} ${gate.name}${critical}${primary}`);
      console.log(`     Required: ${gate.required}, Actual: ${gate.actual}`);
      console.log();
    });
    
    console.log('='.repeat(80));
    
    if (allPassed) {
      console.log('🎉 ALL PHASE 2 GATES PASSED - Ready to tag v1.2-precision-pack');
      console.log('\n📈 Key Improvements:');
      console.log(`   • nDCG@10: ${baseline.metrics.ndcg_at_10.toFixed(3)} → ${phase2.metrics.ndcg_at_10.toFixed(3)} (+${ndcgDelta.toFixed(1)}%)`);
      console.log(`   • Recall@50: ${baseline.metrics.recall_at_50.toFixed(3)} → ${phase2.metrics.recall_at_50.toFixed(3)} (maintained +${phase2.baseline_comparison.recall_at_50_delta_percent.toFixed(1)}%)`);
      console.log(`   • Span Coverage: ${(baseline.span_coverage * 100).toFixed(1)}% → ${spanCoverage.toFixed(1)}%`);
      console.log(`   • Hard-negative Leakage: ${leakage.toFixed(1)}% (within tolerance)`);
      console.log(`   • E2E Latency Impact: +${phase2.latency_comparison.e2e_p95_delta_percent_from_baseline.toFixed(1)}%`);
    } else {
      console.log('⚠️  PHASE 2 GATES FAILED - Rollback required');
      console.log('\n❌ Failed Gates:');
      gates.filter(g => !g.passed).forEach(gate => {
        console.log(`   • ${gate.name}: Required ${gate.required}, got ${gate.actual}`);
      });
    }
    
    const result = {
      overall_status: allPassed ? 'PASS' : 'FAIL',
      gates_passed: gates.filter(g => g.passed).length,
      gates_total: gates.length,
      critical_failures: gates.filter(g => !g.passed && g.critical).length,
      gates: gates,
      recommendation: allPassed ? 'PROMOTE' : 'ROLLBACK',
      timestamp: new Date().toISOString()
    };
    
    // Save validation results
    fs.writeFileSync('results/phase2/phase2_gate_validation.json', JSON.stringify(result, null, 2));
    console.log('\n💾 Results saved to results/phase2/phase2_gate_validation.json');
    
    return result;
    
  } catch (error) {
    console.error('❌ Gate validation failed:', error.message);
    throw error;
  }
}

// Run the validation
if (require.main === module) {
  validatePhase2Gates();
}

module.exports = validatePhase2Gates;