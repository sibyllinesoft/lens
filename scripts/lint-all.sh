#!/bin/bash
# Comprehensive linting for the Lens monorepo

set -e

echo "🔍 Linting Lens monorepo..."

# Change to project root
cd "$(dirname "$0")/.."

# Error counter
ERRORS=0

# Rust linting
echo "📦 Linting Rust code..."
if ! cargo clippy --workspace --all-targets --all-features -- -D warnings; then
    echo "❌ Rust linting failed"
    ((ERRORS++))
fi

# Check Rust formatting
echo "🎨 Checking Rust formatting..."
if ! cargo fmt --all -- --check; then
    echo "❌ Rust formatting check failed"
    ((ERRORS++))
fi

# Run Rust tests
echo "🧪 Running Rust tests..."
if ! cargo test --workspace --lib; then
    echo "❌ Rust unit tests failed"
    ((ERRORS++))
fi

# Security audit
echo "🔒 Running security audit..."
if command -v cargo-audit &> /dev/null; then
    if ! cargo audit; then
        echo "❌ Security audit found issues"
        ((ERRORS++))
    fi
else
    echo "⚠️  cargo-audit not found, skipping security audit"
fi

# Check for typos in code (if typos is available)
if command -v typos &> /dev/null; then
    echo "📝 Checking for typos..."
    if ! typos; then
        echo "❌ Typos found in code"
        ((ERRORS++))
    fi
else
    echo "⚠️  typos not found, skipping typo check"
fi

# Check dependencies with cargo-machete (if available)
if command -v cargo-machete &> /dev/null; then
    echo "📦 Checking for unused dependencies..."
    if ! cargo machete; then
        echo "❌ Unused dependencies found"
        ((ERRORS++))
    fi
else
    echo "⚠️  cargo-machete not found, skipping unused dependency check"
fi

# JavaScript/TypeScript linting (if eslint is available)
if command -v eslint &> /dev/null && [ -f ".eslintrc.json" ]; then
    echo "🟨 Linting JavaScript/TypeScript..."
    if ! eslint "**/*.{js,ts,jsx,tsx}" --ignore-path .gitignore; then
        echo "❌ JavaScript/TypeScript linting failed"
        ((ERRORS++))
    fi
else
    echo "⚠️  ESLint not configured, skipping JS/TS linting"
fi

# Prettier formatting check
if command -v prettier &> /dev/null; then
    echo "💅 Checking Prettier formatting..."
    if ! prettier --check "**/*.{js,ts,jsx,tsx,json,yaml,yml,md}" --ignore-path .gitignore; then
        echo "❌ Prettier formatting check failed"
        ((ERRORS++))
    fi
fi

# YAML linting (if yamllint is available)
if command -v yamllint &> /dev/null; then
    echo "📄 Linting YAML files..."
    if ! yamllint .github/workflows/*.yml docker-compose.yml; then
        echo "❌ YAML linting failed"
        ((ERRORS++))
    fi
else
    echo "⚠️  yamllint not found, skipping YAML linting"
fi

# Dockerfile linting (if hadolint is available)
if command -v hadolint &> /dev/null; then
    echo "🐳 Linting Dockerfiles..."
    if ! hadolint Dockerfile*; then
        echo "❌ Dockerfile linting failed"
        ((ERRORS++))
    fi
else
    echo "⚠️  hadolint not found, skipping Dockerfile linting"
fi

# Shell script linting (if shellcheck is available)
if command -v shellcheck &> /dev/null; then
    echo "🐚 Linting shell scripts..."
    if ! find . -name "*.sh" -not -path "./target/*" -not -path "./.git/*" -exec shellcheck {} \;; then
        echo "❌ Shell script linting failed"
        ((ERRORS++))
    fi
else
    echo "⚠️  shellcheck not found, skipping shell script linting"
fi

# Check for TODO/FIXME comments that should be tracked
echo "📝 Checking for untracked TODO/FIXME comments..."
TODO_COUNT=$(grep -r "TODO\|FIXME" --include="*.rs" --include="*.ts" --include="*.js" --exclude-dir=target --exclude-dir=node_modules . | wc -l)
if [ "$TODO_COUNT" -gt 0 ]; then
    echo "⚠️  Found $TODO_COUNT TODO/FIXME comments. Consider creating issues for them."
    grep -r "TODO\|FIXME" --include="*.rs" --include="*.ts" --include="*.js" --exclude-dir=target --exclude-dir=node_modules . | head -10
fi

# Final report
echo ""
echo "===================="
if [ $ERRORS -eq 0 ]; then
    echo "✅ All linting checks passed!"
    exit 0
else
    echo "❌ Found $ERRORS linting issues"
    echo ""
    echo "To fix formatting issues, run:"
    echo "  ./scripts/format-all.sh"
    echo ""
    echo "To install missing tools, run:"
    echo "  ./scripts/setup-dev-tools.sh"
    exit 1
fi