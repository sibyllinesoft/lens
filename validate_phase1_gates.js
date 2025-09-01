#!/usr/bin/env node

/**
 * Phase 1 Quality Gate Validation
 * 
 * Validates the gates as specified in TODO.md:
 * - ΔRecall@50 ≥ +5% (p<0.05)
 * - ΔnDCG@10 ≥ 0
 * - spans ≥98%
 * - E2E p95 ≤ +25%
 * - p99 ≤ 2× p95
 * - positives-in-candidates ≥ +6%
 */

const fs = require('fs');

function validatePhase1Gates() {
  console.log('🚦 Validating Phase 1 Quality Gates...\n');
  
  try {
    // Load baseline and phase1 results
    const baseline = JSON.parse(fs.readFileSync('baseline_metrics.json', 'utf8'));
    const phase1 = JSON.parse(fs.readFileSync('phase1_results.json', 'utf8'));
    
    const gates = [];
    let allPassed = true;
    
    // Gate 1: ΔRecall@50 ≥ +5%
    const recallDelta = ((phase1.metrics.recall_at_50 - baseline.metrics.recall_at_50) / baseline.metrics.recall_at_50) * 100;
    const recallGate = {
      name: 'ΔRecall@50 ≥ +5%',
      required: '≥ +5%',
      actual: `+${recallDelta.toFixed(1)}%`,
      passed: recallDelta >= 5.0,
      critical: true
    };
    gates.push(recallGate);
    if (!recallGate.passed) allPassed = false;
    
    // Gate 2: ΔnDCG@10 ≥ 0
    const ndcgDelta = ((phase1.metrics.ndcg_at_10 - baseline.metrics.ndcg_at_10) / baseline.metrics.ndcg_at_10) * 100;
    const ndcgGate = {
      name: 'ΔnDCG@10 ≥ 0',
      required: '≥ 0%',
      actual: `${ndcgDelta >= 0 ? '+' : ''}${ndcgDelta.toFixed(1)}%`,
      passed: ndcgDelta >= 0,
      critical: false
    };
    gates.push(ndcgGate);
    if (!ndcgGate.passed) allPassed = false;
    
    // Gate 3: Span coverage ≥98%
    const spanCoverage = phase1.span_coverage * 100;
    const spanGate = {
      name: 'spans ≥ 98%',
      required: '≥ 98%',
      actual: `${spanCoverage.toFixed(1)}%`,
      passed: spanCoverage >= 98.0,
      critical: true
    };
    gates.push(spanGate);
    if (!spanGate.passed) allPassed = false;
    
    // Gate 4: E2E p95 ≤ +25%
    const latencyDelta = ((phase1.stage_latencies.e2e_p95 - baseline.stage_latencies.e2e_p95) / baseline.stage_latencies.e2e_p95) * 100;
    const latencyGate = {
      name: 'E2E p95 ≤ +25%',
      required: '≤ +25%',
      actual: `+${latencyDelta.toFixed(1)}%`,
      passed: latencyDelta <= 25.0,
      critical: true
    };
    gates.push(latencyGate);
    if (!latencyGate.passed) allPassed = false;
    
    // Gate 5: p99 ≤ 2× p95
    const p99P95Ratio = phase1.stage_latencies.e2e_p99 / phase1.stage_latencies.e2e_p95;
    const ratioGate = {
      name: 'p99 ≤ 2× p95',
      required: '≤ 2.0×',
      actual: `${p99P95Ratio.toFixed(1)}×`,
      passed: p99P95Ratio <= 2.0,
      critical: true
    };
    gates.push(ratioGate);
    if (!ratioGate.passed) allPassed = false;
    
    // Gate 6: positives-in-candidates ≥ +6%
    const positivesDelta = ((phase1.fan_out_sizes.positives_in_candidates - baseline.fan_out_sizes.positives_in_candidates) / baseline.fan_out_sizes.positives_in_candidates) * 100;
    const positivesGate = {
      name: 'positives-in-candidates ≥ +6%',
      required: '≥ +6%',
      actual: `+${positivesDelta.toFixed(1)}%`,
      passed: positivesDelta >= 6.0,
      critical: false
    };
    gates.push(positivesGate);
    if (!positivesGate.passed) allPassed = false;
    
    // Display results
    console.log('📊 Phase 1 Quality Gate Results:');
    console.log('='.repeat(80));
    
    gates.forEach(gate => {
      const status = gate.passed ? '✅ PASS' : '❌ FAIL';
      const critical = gate.critical ? ' (CRITICAL)' : '';
      console.log(`${status} ${gate.name}${critical}`);
      console.log(`     Required: ${gate.required}, Actual: ${gate.actual}`);
      console.log();
    });
    
    console.log('='.repeat(80));
    
    if (allPassed) {
      console.log('🎉 ALL PHASE 1 GATES PASSED - Ready to tag v1.1-recall-pack');
      console.log('\n📈 Key Improvements:');
      console.log(`   • Recall@50: ${baseline.metrics.recall_at_50.toFixed(3)} → ${phase1.metrics.recall_at_50.toFixed(3)} (+${recallDelta.toFixed(1)}%)`);
      console.log(`   • Span Coverage: ${(baseline.span_coverage * 100).toFixed(1)}% → ${spanCoverage.toFixed(1)}% (+${((phase1.span_coverage - baseline.span_coverage) * 100).toFixed(1)}%)`);
      console.log(`   • Positives in Candidates: ${baseline.fan_out_sizes.positives_in_candidates} → ${phase1.fan_out_sizes.positives_in_candidates} (+${positivesDelta.toFixed(1)}%)`);
    } else {
      console.log('⚠️  PHASE 1 GATES FAILED - Rollback required');
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
    fs.writeFileSync('phase1_gate_validation.json', JSON.stringify(result, null, 2));
    console.log('\n💾 Results saved to phase1_gate_validation.json');
    
    return result;
    
  } catch (error) {
    console.error('❌ Gate validation failed:', error.message);
    throw error;
  }
}

// Run the validation
if (require.main === module) {
  validatePhase1Gates().catch(console.error);
}

module.exports = validatePhase1Gates;