#!/bin/bash
# Secure build script for lens with SBOM and security scanning
# Implements Phase A3 requirements for packaging and security

set -euo pipefail

# Configuration
BUILD_DIR="./dist"
ARTIFACTS_DIR="./build-artifacts"
VERSION="${LENS_VERSION:-1.0.0-rc.1}"
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_SHA="${GIT_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')}"

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

# Parse command line arguments
GENERATE_SBOM=false
ENABLE_SAST=false
LOCK_DEPS=false
BUILD_CONTAINER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --sbom)
            GENERATE_SBOM=true
            shift
            ;;
        --sast)
            ENABLE_SAST=true
            shift
            ;;
        --lock)
            LOCK_DEPS=true
            shift
            ;;
        --container)
            BUILD_CONTAINER=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Usage: $0 [--sbom] [--sast] [--lock] [--container]"
            exit 1
            ;;
    esac
done

log_info "Starting secure build for lens v${VERSION}"
log_info "Git SHA: ${GIT_SHA}"
log_info "Build date: ${BUILD_DATE}"

# Create artifacts directory
mkdir -p "${ARTIFACTS_DIR}"

# Step 1: Clean previous builds
log_info "Cleaning previous builds..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# Step 2: Lock dependencies if requested
if [ "$LOCK_DEPS" = true ]; then
    log_info "Locking dependencies..."
    if [ -f "package-lock.json" ]; then
        log_success "Found existing package-lock.json"
    else
        log_warning "No package-lock.json found, generating..."
        npm install --package-lock-only
    fi
    
    # Verify lock file integrity
    npm ci --dry-run > /dev/null 2>&1 || {
        log_error "Lock file integrity check failed"
        exit 1
    }
    log_success "Dependencies locked and verified"
fi

# Step 3: Install dependencies
log_info "Installing dependencies..."
if [ "$LOCK_DEPS" = true ]; then
    npm ci
else
    npm install
fi

# Step 4: Run linting and type checking
log_info "Running code quality checks..."
npm run lint || {
    log_error "Linting failed"
    exit 1
}

npx tsc --noEmit || {
    log_error "Type checking failed"
    exit 1
}

log_success "Code quality checks passed"

# Step 5: Run SAST scan if enabled
if [ "$ENABLE_SAST" = true ]; then
    log_info "Running SAST security scan..."
    
    # Check if semgrep is available
    if command -v semgrep &> /dev/null; then
        log_info "Running Semgrep SAST scan..."
        semgrep --config=auto --json --output="${ARTIFACTS_DIR}/sast-report.json" src/ || {
            log_warning "SAST scan found issues, check ${ARTIFACTS_DIR}/sast-report.json"
        }
        log_success "SAST scan completed"
    else
        log_warning "Semgrep not found, skipping SAST scan"
        log_info "Install semgrep: pip install semgrep"
    fi
fi

# Step 6: Run dependency audit
log_info "Running dependency security audit..."
npm audit --json > "${ARTIFACTS_DIR}/npm-audit.json" 2>/dev/null || {
    log_warning "npm audit found vulnerabilities, check ${ARTIFACTS_DIR}/npm-audit.json"
}

# Check for high/critical vulnerabilities
HIGH_VULNS=$(jq -r '.vulnerabilities | to_entries[] | select(.value.severity == "high" or .value.severity == "critical") | length' "${ARTIFACTS_DIR}/npm-audit.json" 2>/dev/null || echo "0")

if [ "$HIGH_VULNS" -gt 0 ]; then
    log_error "Found ${HIGH_VULNS} high/critical vulnerabilities"
    log_error "Review and fix vulnerabilities before release"
    exit 1
fi

log_success "Dependency audit completed - no critical vulnerabilities"

# Step 7: Build TypeScript
log_info "Building TypeScript..."
npm run build
log_success "TypeScript build completed"

# Step 8: Generate SBOM if requested
if [ "$GENERATE_SBOM" = true ]; then
    log_info "Generating Software Bill of Materials (SBOM)..."
    
    # Create SBOM using npm list
    npm list --all --json > "${ARTIFACTS_DIR}/npm-dependencies.json" 2>/dev/null || true
    
    # Generate CycloneDX SBOM if tool is available
    if command -v cyclonedx-npm &> /dev/null; then
        cyclonedx-npm --output-file "${ARTIFACTS_DIR}/sbom.json"
        log_success "CycloneDX SBOM generated"
    else
        log_info "CycloneDX tool not found, generating simple SBOM..."
        # Generate a simple SBOM format
        cat > "${ARTIFACTS_DIR}/sbom.json" << EOF
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.4",
  "version": 1,
  "metadata": {
    "timestamp": "${BUILD_DATE}",
    "component": {
      "type": "application",
      "name": "lens",
      "version": "${VERSION}",
      "description": "Local sharded code search system"
    }
  },
  "components": $(jq '.dependencies // {}' "${ARTIFACTS_DIR}/npm-dependencies.json" 2>/dev/null || echo '{}')
}
EOF
    fi
    
    log_success "SBOM generated at ${ARTIFACTS_DIR}/sbom.json"
fi

# Step 9: Create build manifest
log_info "Creating build manifest..."
cat > "${ARTIFACTS_DIR}/build-manifest.json" << EOF
{
  "version": "${VERSION}",
  "build_date": "${BUILD_DATE}",
  "git_sha": "${GIT_SHA}",
  "build_tools": {
    "node_version": "$(node --version)",
    "npm_version": "$(npm --version)",
    "typescript_version": "$(npx tsc --version | cut -d' ' -f2)"
  },
  "security": {
    "sbom_generated": ${GENERATE_SBOM},
    "sast_scan": ${ENABLE_SAST},
    "dependency_lock": ${LOCK_DEPS}
  },
  "checksums": {}
}
EOF

# Step 10: Generate checksums
log_info "Generating checksums..."
find "${BUILD_DIR}" -type f -name "*.js" -exec sha256sum {} \; > "${ARTIFACTS_DIR}/checksums.txt"

# Step 11: Build container if requested
if [ "$BUILD_CONTAINER" = true ]; then
    log_info "Building container image..."
    docker build -t "lens:${VERSION}" -t "lens:latest" .
    
    # Save container image
    docker save "lens:${VERSION}" | gzip > "${ARTIFACTS_DIR}/lens-${VERSION}.tar.gz"
    log_success "Container image built and saved"
fi

# Step 12: Create tarball artifact
log_info "Creating distribution tarball..."
tar -czf "${ARTIFACTS_DIR}/lens-${VERSION}.tar.gz" \
    --exclude="node_modules" \
    --exclude=".git" \
    --exclude="build-artifacts" \
    --exclude="coverage" \
    --exclude="logs" \
    .

log_success "Distribution tarball created"

# Step 13: Final validation
log_info "Performing final validation..."

# Check if all required files exist
REQUIRED_FILES=("${BUILD_DIR}/server.js" "${BUILD_DIR}/cli.js" "package.json")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        log_error "Required file missing: $file"
        exit 1
    fi
done

log_success "Final validation passed"

# Summary
echo
log_success "🚀 Secure build completed successfully!"
echo
log_info "Artifacts generated in ${ARTIFACTS_DIR}:"
ls -la "${ARTIFACTS_DIR}"

echo
log_info "Next steps:"
log_info "  1. Review security scan results"
log_info "  2. Test the built application"
log_info "  3. Deploy using the generated artifacts"

if [ "$GENERATE_SBOM" = true ]; then
    log_info "  4. Submit SBOM to vulnerability scanning services"
fi