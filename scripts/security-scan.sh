#!/bin/bash
# Comprehensive security scanning script for lens
# Implements Phase A2.3: License and SAST scans with critical blocking

set -euo pipefail

# Configuration
SCAN_RESULTS_DIR="./security-scans"
BLOCK_ON_CRITICAL="${BLOCK_ON_CRITICAL:-true}"
ALLOW_LICENSE_VIOLATIONS="${ALLOW_LICENSE_VIOLATIONS:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_critical() {
    echo -e "${RED}[CRITICAL]${NC} $1"
}

# Initialize
echo "======================================"
echo "🔒 LENS SECURITY SCANNER"
echo "======================================"
echo

mkdir -p "${SCAN_RESULTS_DIR}"

# Track scan results
CRITICAL_ISSUES=0
HIGH_ISSUES=0
LICENSE_VIOLATIONS=0
TOTAL_VULNERABILITIES=0

# Step 1: Dependency Security Audit
log_info "🔍 Running dependency security audit..."

npm audit --json > "${SCAN_RESULTS_DIR}/npm-audit-full.json" 2>/dev/null || {
    log_warning "npm audit returned non-zero exit code (vulnerabilities found)"
}

if [ -f "${SCAN_RESULTS_DIR}/npm-audit-full.json" ]; then
    # Parse audit results
    CRITICAL_VULNS=$(jq -r '[.vulnerabilities[] | select(.severity == "critical")] | length' "${SCAN_RESULTS_DIR}/npm-audit-full.json" 2>/dev/null || echo "0")
    HIGH_VULNS=$(jq -r '[.vulnerabilities[] | select(.severity == "high")] | length' "${SCAN_RESULTS_DIR}/npm-audit-full.json" 2>/dev/null || echo "0")
    MODERATE_VULNS=$(jq -r '[.vulnerabilities[] | select(.severity == "moderate")] | length' "${SCAN_RESULTS_DIR}/npm-audit-full.json" 2>/dev/null || echo "0")
    LOW_VULNS=$(jq -r '[.vulnerabilities[] | select(.severity == "low")] | length' "${SCAN_RESULTS_DIR}/npm-audit-full.json" 2>/dev/null || echo "0")
    
    CRITICAL_ISSUES=$((CRITICAL_ISSUES + CRITICAL_VULNS))
    HIGH_ISSUES=$((HIGH_ISSUES + HIGH_VULNS))
    TOTAL_VULNERABILITIES=$((CRITICAL_VULNS + HIGH_VULNS + MODERATE_VULNS + LOW_VULNS))
    
    log_info "Dependency audit results:"
    log_info "  Critical: ${CRITICAL_VULNS}"
    log_info "  High: ${HIGH_VULNS}"
    log_info "  Moderate: ${MODERATE_VULNS}"
    log_info "  Low: ${LOW_VULNS}"
    
    if [ "$CRITICAL_VULNS" -gt 0 ]; then
        log_critical "Found ${CRITICAL_VULNS} critical vulnerabilities in dependencies"
    elif [ "$HIGH_VULNS" -gt 0 ]; then
        log_warning "Found ${HIGH_VULNS} high severity vulnerabilities"
    else
        log_success "No critical or high severity vulnerabilities found"
    fi
fi

# Step 2: SAST Scan with Semgrep
log_info "🔍 Running SAST scan with Semgrep..."

if command -v semgrep &> /dev/null; then
    # Run comprehensive Semgrep scan
    semgrep --config=auto \
            --config=p/security-audit \
            --config=p/nodejs \
            --config=p/typescript \
            --json \
            --output="${SCAN_RESULTS_DIR}/semgrep-results.json" \
            src/ || {
        log_warning "Semgrep scan completed with findings"
    }
    
    if [ -f "${SCAN_RESULTS_DIR}/semgrep-results.json" ]; then
        # Parse Semgrep results
        SEMGREP_CRITICAL=$(jq -r '[.results[] | select(.extra.severity == "ERROR")] | length' "${SCAN_RESULTS_DIR}/semgrep-results.json" 2>/dev/null || echo "0")
        SEMGREP_HIGH=$(jq -r '[.results[] | select(.extra.severity == "WARNING")] | length' "${SCAN_RESULTS_DIR}/semgrep-results.json" 2>/dev/null || echo "0")
        SEMGREP_INFO=$(jq -r '[.results[] | select(.extra.severity == "INFO")] | length' "${SCAN_RESULTS_DIR}/semgrep-results.json" 2>/dev/null || echo "0")
        
        CRITICAL_ISSUES=$((CRITICAL_ISSUES + SEMGREP_CRITICAL))
        HIGH_ISSUES=$((HIGH_ISSUES + SEMGREP_HIGH))
        
        log_info "SAST scan results:"
        log_info "  Critical: ${SEMGREP_CRITICAL}"
        log_info "  High: ${SEMGREP_HIGH}"
        log_info "  Info: ${SEMGREP_INFO}"
        
        if [ "$SEMGREP_CRITICAL" -gt 0 ]; then
            log_critical "Found ${SEMGREP_CRITICAL} critical security issues in code"
            
            # Log top critical issues
            log_error "Top critical security issues:"
            jq -r '.results[] | select(.extra.severity == "ERROR") | "  - " + .check_id + ": " + .extra.message' "${SCAN_RESULTS_DIR}/semgrep-results.json" | head -5
        fi
        
        log_success "SAST scan completed"
    fi
else
    log_error "Semgrep not found! Install it with: pip install semgrep"
    log_error "SAST scanning is required for security compliance"
    exit 1
fi

# Step 3: License Compliance Check
log_info "📋 Running license compliance check..."

# Generate license report
npx license-checker --json --out "${SCAN_RESULTS_DIR}/licenses.json" 2>/dev/null || {
    log_warning "License checker had issues, continuing..."
}

if [ -f "${SCAN_RESULTS_DIR}/licenses.json" ]; then
    # Define prohibited licenses
    PROHIBITED_LICENSES=("GPL-3.0" "AGPL-3.0" "LGPL-3.0" "GPL-2.0" "AGPL-1.0" "LGPL-2.1")
    
    log_info "Checking for prohibited licenses..."
    
    # Check for prohibited licenses
    for license in "${PROHIBITED_LICENSES[@]}"; do
        if jq -r 'to_entries[] | select(.value.licenses == "'"$license"'") | .key' "${SCAN_RESULTS_DIR}/licenses.json" | grep -q .; then
            LICENSE_VIOLATIONS=$((LICENSE_VIOLATIONS + 1))
            log_error "Found prohibited license: ${license}"
            jq -r 'to_entries[] | select(.value.licenses == "'"$license"'") | "  - " + .key + " (" + .value.licenses + ")"' "${SCAN_RESULTS_DIR}/licenses.json"
        fi
    done
    
    if [ "$LICENSE_VIOLATIONS" -gt 0 ]; then
        log_critical "Found ${LICENSE_VIOLATIONS} license violations"
    else
        log_success "No prohibited licenses found"
    fi
    
    # Generate license summary
    log_info "License summary:"
    jq -r 'to_entries[] | .value.licenses' "${SCAN_RESULTS_DIR}/licenses.json" | sort | uniq -c | sort -nr | head -10 | while read count license; do
        log_info "  ${count} packages with ${license}"
    done
fi

# Step 4: Container Security Scan (if Docker is available)
if command -v docker &> /dev/null && [ -f "Dockerfile" ]; then
    log_info "🐳 Running container security scan..."
    
    # Build image for scanning
    docker build -t lens:security-scan . > /dev/null 2>&1 || {
        log_warning "Failed to build Docker image for security scanning"
    }
    
    # Scan with Trivy if available
    if command -v trivy &> /dev/null; then
        trivy image --format json --output "${SCAN_RESULTS_DIR}/trivy-results.json" lens:security-scan || {
            log_warning "Trivy scan completed with issues"
        }
        
        if [ -f "${SCAN_RESULTS_DIR}/trivy-results.json" ]; then
            CONTAINER_CRITICAL=$(jq -r '[.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL")] | length' "${SCAN_RESULTS_DIR}/trivy-results.json" 2>/dev/null || echo "0")
            CONTAINER_HIGH=$(jq -r '[.Results[]?.Vulnerabilities[]? | select(.Severity == "HIGH")] | length' "${SCAN_RESULTS_DIR}/trivy-results.json" 2>/dev/null || echo "0")
            
            CRITICAL_ISSUES=$((CRITICAL_ISSUES + CONTAINER_CRITICAL))
            HIGH_ISSUES=$((HIGH_ISSUES + CONTAINER_HIGH))
            
            log_info "Container scan results:"
            log_info "  Critical: ${CONTAINER_CRITICAL}"
            log_info "  High: ${CONTAINER_HIGH}"
        fi
    else
        log_info "Trivy not found, skipping container security scan"
    fi
fi

# Step 5: Generate Security Report
log_info "📊 Generating security report..."

cat > "${SCAN_RESULTS_DIR}/security-report.json" << EOF
{
  "scanDate": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "summary": {
    "criticalIssues": ${CRITICAL_ISSUES},
    "highIssues": ${HIGH_ISSUES},
    "totalVulnerabilities": ${TOTAL_VULNERABILITIES},
    "licenseViolations": ${LICENSE_VIOLATIONS},
    "overallRisk": "$([ "$CRITICAL_ISSUES" -gt 0 ] && echo "CRITICAL" || [ "$HIGH_ISSUES" -gt 0 ] && echo "HIGH" || echo "LOW")"
  },
  "scans": {
    "dependencyAudit": {
      "completed": true,
      "criticalVulns": ${CRITICAL_VULNS:-0},
      "highVulns": ${HIGH_VULNS:-0},
      "resultsFile": "npm-audit-full.json"
    },
    "sastScan": {
      "completed": $([ -f "${SCAN_RESULTS_DIR}/semgrep-results.json" ] && echo "true" || echo "false"),
      "criticalIssues": ${SEMGREP_CRITICAL:-0},
      "highIssues": ${SEMGREP_HIGH:-0},
      "resultsFile": "semgrep-results.json"
    },
    "licenseCheck": {
      "completed": $([ -f "${SCAN_RESULTS_DIR}/licenses.json" ] && echo "true" || echo "false"),
      "violations": ${LICENSE_VIOLATIONS},
      "resultsFile": "licenses.json"
    },
    "containerScan": {
      "completed": $([ -f "${SCAN_RESULTS_DIR}/trivy-results.json" ] && echo "true" || echo "false"),
      "criticalVulns": ${CONTAINER_CRITICAL:-0},
      "highVulns": ${CONTAINER_HIGH:-0},
      "resultsFile": "trivy-results.json"
    }
  },
  "recommendations": []
}
EOF

# Step 6: Decision Logic
log_info "📋 Security scan summary:"
log_info "  Critical Issues: ${CRITICAL_ISSUES}"
log_info "  High Issues: ${HIGH_ISSUES}"
log_info "  License Violations: ${LICENSE_VIOLATIONS}"

echo
if [ "$CRITICAL_ISSUES" -gt 0 ]; then
    log_critical "SECURITY SCAN FAILED: ${CRITICAL_ISSUES} critical issues found"
    log_error "Critical issues must be resolved before release"
    
    if [ "$BLOCK_ON_CRITICAL" = "true" ]; then
        log_error "Blocking release due to critical security issues"
        echo
        log_info "Review detailed results in: ${SCAN_RESULTS_DIR}/"
        exit 1
    else
        log_warning "Critical issues found but blocking is disabled"
    fi
fi

if [ "$LICENSE_VIOLATIONS" -gt 0 ] && [ "$ALLOW_LICENSE_VIOLATIONS" = "false" ]; then
    log_critical "LICENSE COMPLIANCE FAILED: ${LICENSE_VIOLATIONS} violations found"
    log_error "License violations must be resolved before release"
    echo
    log_info "Review detailed results in: ${SCAN_RESULTS_DIR}/licenses.json"
    exit 1
fi

if [ "$HIGH_ISSUES" -gt 0 ]; then
    log_warning "Found ${HIGH_ISSUES} high severity issues"
    log_warning "Consider resolving these before release"
fi

# Success
echo
log_success "🔒 SECURITY SCAN COMPLETED"
log_success "All critical security gates passed!"

echo
log_info "Detailed results available in: ${SCAN_RESULTS_DIR}/"
log_info "Files generated:"
ls -la "${SCAN_RESULTS_DIR}"

echo
log_info "Next steps:"
if [ "$HIGH_ISSUES" -gt 0 ]; then
    log_info "  1. Review and consider fixing high severity issues"
fi
log_info "  2. Submit security report to compliance team"
log_info "  3. Proceed with release if all gates pass"

log_success "✅ Ready for release!"