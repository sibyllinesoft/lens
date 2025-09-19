#!/bin/bash
# Format all code in the Lens monorepo

set -e

echo "🎨 Formatting Lens monorepo..."

# Change to project root
cd "$(dirname "$0")/.."

# Format Rust code
echo "📦 Formatting Rust code..."
cargo fmt --all

# Format JavaScript/TypeScript code (if any)
if command -v prettier &> /dev/null; then
    echo "🟨 Formatting JavaScript/TypeScript/JSON/YAML files..."
    prettier --write "**/*.{js,ts,jsx,tsx,json,yaml,yml,md}" \
        --ignore-path .gitignore \
        --log-level warn
else
    echo "⚠️  Prettier not found, skipping JS/TS formatting"
fi

# Format TOML files with taplo (if available)
if command -v taplo &> /dev/null; then
    echo "⚙️  Formatting TOML files..."
    taplo format **/*.toml
else
    echo "⚠️  taplo not found, skipping TOML formatting"
fi

# Format shell scripts with shfmt (if available)
if command -v shfmt &> /dev/null; then
    echo "🐚 Formatting shell scripts..."
    find . -name "*.sh" -not -path "./target/*" -not -path "./.git/*" | xargs shfmt -w -i 2
else
    echo "⚠️  shfmt not found, skipping shell script formatting"
fi

echo "✅ Formatting complete!"

# Check for any remaining formatting issues
echo ""
echo "🔍 Checking for remaining formatting issues..."

# Check Rust formatting
if ! cargo fmt --all -- --check; then
    echo "❌ Rust code formatting issues found"
    exit 1
fi

# Check Prettier formatting (if available)
if command -v prettier &> /dev/null; then
    if ! prettier --check "**/*.{js,ts,jsx,tsx,json,yaml,yml,md}" --ignore-path .gitignore; then
        echo "❌ JavaScript/TypeScript/JSON/YAML formatting issues found"
        exit 1
    fi
fi

echo "✅ All code is properly formatted!"