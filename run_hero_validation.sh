#!/bin/bash
# Hero Configuration End-to-End Validation Test Runner
# 
# This script runs the comprehensive validation test to verify that the Rust
# implementation with hero defaults produces equivalent results to the production
# hero canary configuration.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Configuration
TEST_NAME="hero_validation_e2e"
REPORT_FILE="hero_validation_report.md"
RESULTS_DIR="hero_validation_results"

echo -e "${BLUE}🚀 Hero Configuration End-to-End Validation Test${NC}"
echo "=================================================="
echo ""

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

# Check if golden datasets exist
if [ ! -d "../lens-external-data/validation-data/" ]; then
    echo -e "${RED}❌ Error: Golden dataset directory not found at ../lens-external-data/validation-data/${NC}"
    echo "Please ensure the lens-external-data repository is cloned parallel to this directory."
    exit 1
fi

# Check for required files
required_files=(
    "../lens-external-data/validation-data/night-1-2025-09-09.json"
    "../lens-external-data/validation-data/night-2-2025-09-09.json"
    "../lens-external-data/validation-data/night-3-2025-09-09.json"
    "../lens-external-data/validation-data/three-night-state.json"
    "release/hero.lock.json"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}❌ Required file not found: $file${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✅ All prerequisites satisfied${NC}"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Display hero configuration
echo -e "${BLUE}🎯 Hero Configuration to Validate${NC}"
echo "--------------------------------"
if [ -f "release/hero.lock.json" ]; then
    echo "Config ID: $(jq -r '.config_id' release/hero.lock.json)"
    echo "Fusion: $(jq -r '.params.fusion' release/hero.lock.json)"
    echo "Chunk Policy: $(jq -r '.params.chunk_policy' release/hero.lock.json)"
    echo "Chunk Length: $(jq -r '.params.chunk_len' release/hero.lock.json)"
    echo "Retrieval K: $(jq -r '.params.retrieval_k' release/hero.lock.json)"
    echo "Symbol Boost: $(jq -r '.params.symbol_boost' release/hero.lock.json)"
    echo "Graph Expand Hops: $(jq -r '.params.graph_expand_hops' release/hero.lock.json)"
    echo ""
    echo "Production Baseline Metrics:"
    echo "- Pass Rate Core: $(jq -r '.metrics.pass_rate_core' release/hero.lock.json)"
    echo "- Answerable at K: $(jq -r '.metrics.answerable_at_k' release/hero.lock.json)"
    echo "- Span Recall: $(jq -r '.metrics.span_recall' release/hero.lock.json)"
    echo "- P95 Improvement: $(jq -r '.metrics.p95_improvement_pct' release/hero.lock.json)%"
    echo "- NDCG Improvement: $(jq -r '.metrics.ndcg_improvement_pct' release/hero.lock.json)%"
else
    echo -e "${RED}❌ Hero configuration file not found${NC}"
    exit 1
fi
echo ""

# Run validation test
echo -e "${BLUE}🧪 Running Hero Validation Tests${NC}"
echo "-------------------------------"

echo "Starting test execution..."
start_time=$(date +%s)

# Set environment variables for test
export RUST_LOG=info
export RUST_BACKTRACE=1

# Run the specific hero validation test
if cargo test --test hero_validation_e2e test_hero_configuration_equivalence -- --nocapture; then
    echo -e "${GREEN}✅ Main validation test PASSED${NC}"
    main_test_result="PASSED"
else
    echo -e "${RED}❌ Main validation test FAILED${NC}"
    main_test_result="FAILED"
fi

# Run hero configuration loading test
if cargo test --test hero_validation_e2e test_hero_config_loading -- --nocapture; then
    echo -e "${GREEN}✅ Config loading test PASSED${NC}"
    config_test_result="PASSED"
else
    echo -e "${RED}❌ Config loading test FAILED${NC}"
    config_test_result="FAILED"
fi

# Run performance benchmark test
if cargo test --test hero_validation_e2e test_hero_performance_benchmark -- --nocapture; then
    echo -e "${GREEN}✅ Performance benchmark test PASSED${NC}"
    perf_test_result="PASSED"
else
    echo -e "${RED}❌ Performance benchmark test FAILED${NC}"
    perf_test_result="FAILED"
fi

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo -e "${BLUE}📊 Test Execution Summary${NC}"
echo "========================="
echo "Execution Time: ${duration}s"
echo "Main Validation: $main_test_result"
echo "Config Loading: $config_test_result"
echo "Performance Benchmark: $perf_test_result"
echo ""

# Move generated report to results directory
if [ -f "$REPORT_FILE" ]; then
    mv "$REPORT_FILE" "$RESULTS_DIR/"
    echo -e "${GREEN}📋 Detailed report saved to: $RESULTS_DIR/$REPORT_FILE${NC}"
    echo ""
    
    # Display report summary
    echo -e "${BLUE}📄 Validation Report Summary${NC}"
    echo "============================="
    if grep -q "✅ VALIDATION PASSED" "$RESULTS_DIR/$REPORT_FILE"; then
        echo -e "${GREEN}✅ OVERALL RESULT: PASSED${NC}"
        echo "The Rust implementation with hero defaults produces equivalent results"
        echo "to the production hero canary configuration within ±2% tolerance."
    elif grep -q "❌ VALIDATION FAILED" "$RESULTS_DIR/$REPORT_FILE"; then
        echo -e "${RED}❌ OVERALL RESULT: FAILED${NC}"
        echo "The Rust implementation does not produce equivalent results within"
        echo "the specified tolerance. Review the detailed report for specifics."
    else
        echo -e "${YELLOW}⚠️  OVERALL RESULT: INCONCLUSIVE${NC}"
        echo "Unable to determine validation result from report."
    fi
    echo ""
    
    # Show first few lines of report for quick overview
    echo -e "${BLUE}Report Preview:${NC}"
    echo "---------------"
    head -20 "$RESULTS_DIR/$REPORT_FILE" 2>/dev/null || echo "Unable to preview report"
    echo ""
fi

# Final status
overall_result="UNKNOWN"
if [ "$main_test_result" == "PASSED" ] && [ "$config_test_result" == "PASSED" ] && [ "$perf_test_result" == "PASSED" ]; then
    overall_result="PASSED"
    echo -e "${GREEN}🎉 HERO VALIDATION COMPLETE: ALL TESTS PASSED${NC}"
    echo ""
    echo "The Rust implementation with hero defaults is validated as equivalent"
    echo "to the production hero canary configuration. The 22.1% P95 improvement"
    echo "can be expected from the Rust implementation."
    exit_code=0
else
    overall_result="FAILED"
    echo -e "${RED}❌ HERO VALIDATION COMPLETE: SOME TESTS FAILED${NC}"
    echo ""
    echo "The validation did not pass all requirements. Please review the"
    echo "detailed results and address any issues before deployment."
    exit_code=1
fi

# Create summary JSON for CI/automation
summary_file="$RESULTS_DIR/validation_summary.json"
cat > "$summary_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "overall_result": "$overall_result",
    "execution_duration_seconds": $duration,
    "tests": {
        "main_validation": "$main_test_result",
        "config_loading": "$config_test_result", 
        "performance_benchmark": "$perf_test_result"
    },
    "hero_config": {
        "config_id": "$(jq -r '.config_id' release/hero.lock.json 2>/dev/null || echo 'unknown')",
        "expected_p95_improvement_pct": $(jq -r '.metrics.p95_improvement_pct' release/hero.lock.json 2>/dev/null || echo '0'),
        "expected_pass_rate_core": $(jq -r '.metrics.pass_rate_core' release/hero.lock.json 2>/dev/null || echo '0'),
        "expected_answerable_at_k": $(jq -r '.metrics.answerable_at_k' release/hero.lock.json 2>/dev/null || echo '0'),
        "expected_span_recall": $(jq -r '.metrics.span_recall' release/hero.lock.json 2>/dev/null || echo '0')
    },
    "tolerance": 0.02,
    "results_directory": "$RESULTS_DIR"
}
EOF

echo -e "${BLUE}💾 Validation summary saved to: $summary_file${NC}"
echo ""

# Instructions for next steps
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}🚀 Next Steps:${NC}"
    echo "1. Review the detailed validation report"
    echo "2. Proceed with hero configuration deployment"
    echo "3. Monitor production metrics for confirmation"
else
    echo -e "${YELLOW}🔧 Next Steps:${NC}"
    echo "1. Review failed test details in the validation report"
    echo "2. Address any configuration or implementation issues"
    echo "3. Re-run validation after fixes"
    echo "4. Consider adjusting tolerance if differences are acceptable"
fi

echo ""
echo "Validation complete."
exit $exit_code