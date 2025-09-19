#!/bin/bash
# Install development tools for the Lens project

set -e

echo "🔧 Setting up development tools for Lens..."

# Change to project root
cd "$(dirname "$0")/.."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install cargo tool if not present
install_cargo_tool() {
    local tool=$1
    local crate=${2:-$tool}
    
    if ! command_exists "$tool"; then
        echo "📦 Installing $tool..."
        cargo install "$crate"
    else
        echo "✅ $tool is already installed"
    fi
}

# Ensure Rust is installed
if ! command_exists cargo; then
    echo "❌ Rust/Cargo is not installed. Please install from https://rustup.rs/"
    exit 1
fi

echo "🦀 Rust toolchain setup..."

# Install Rust components
rustup component add clippy rustfmt rust-analyzer

# Install essential Cargo tools
install_cargo_tool cargo-audit
install_cargo_tool cargo-deny
install_cargo_tool cargo-watch
install_cargo_tool cargo-tarpaulin
install_cargo_tool cargo-machete
install_cargo_tool typos typos-cli

# Install formatting tools
install_cargo_tool taplo taplo-cli

echo "🟨 JavaScript/TypeScript tools setup..."

# Install Node.js tools (if Node.js is available)
if command_exists npm; then
    echo "📦 Installing Node.js development tools..."
    
    # Install prettier globally if not present
    if ! command_exists prettier; then
        npm install -g prettier
    fi
    
    # Install other useful tools
    npm install -g eslint @typescript-eslint/parser @typescript-eslint/eslint-plugin
    
    echo "✅ Node.js tools installed"
elif command_exists bun; then
    echo "📦 Installing Bun development tools..."
    
    # Install with Bun
    bun add -g prettier eslint @typescript-eslint/parser @typescript-eslint/eslint-plugin
    
    echo "✅ Bun tools installed"
else
    echo "⚠️  Node.js/Bun not found, skipping JS/TS tools"
fi

echo "🛠️  System tools setup..."

# Install system tools (Ubuntu/Debian)
if command_exists apt; then
    echo "📦 Installing system development tools..."
    
    # Update package list
    sudo apt update
    
    # Install tools
    sudo apt install -y \
        shellcheck \
        yamllint \
        hadolint \
        shfmt \
        jq \
        curl \
        git
        
    echo "✅ System tools installed"
elif command_exists brew; then
    echo "📦 Installing development tools with Homebrew..."
    
    brew install \
        shellcheck \
        yamllint \
        hadolint \
        shfmt \
        jq
        
    echo "✅ Homebrew tools installed"
else
    echo "⚠️  Package manager not found, skipping system tools"
    echo "   Please install manually: shellcheck, yamllint, hadolint, shfmt, jq"
fi

echo "🐳 Docker tools setup..."

# Check Docker setup
if command_exists docker; then
    echo "✅ Docker is available"
    
    # Check docker-compose
    if command_exists docker-compose; then
        echo "✅ docker-compose is available"
    else
        echo "⚠️  docker-compose not found. Install it from https://docs.docker.com/compose/install/"
    fi
else
    echo "⚠️  Docker not found. Install it from https://docker.com/"
fi

echo "🔍 Git hooks setup..."

# Setup git hooks directory
mkdir -p .git/hooks

# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for Lens

echo "🔍 Running pre-commit checks..."

# Run formatting check
if ! ./scripts/format-all.sh --check; then
    echo "❌ Code formatting issues found. Run './scripts/format-all.sh' to fix."
    exit 1
fi

# Run basic linting
if ! cargo clippy --workspace --all-targets --all-features -- -D warnings; then
    echo "❌ Clippy issues found. Fix them before committing."
    exit 1
fi

# Run tests
if ! cargo test --workspace --lib --quiet; then
    echo "❌ Tests failed. Fix them before committing."
    exit 1
fi

echo "✅ Pre-commit checks passed!"
EOF

# Make pre-commit hook executable
chmod +x .git/hooks/pre-commit

echo "🎯 VSCode setup..."

# Create VSCode settings directory
mkdir -p .vscode

# Create VSCode settings
cat > .vscode/settings.json << 'EOF'
{
  "rust-analyzer.checkOnSave.command": "clippy",
  "rust-analyzer.checkOnSave.allTargets": true,
  "rust-analyzer.cargo.allFeatures": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll": true
  },
  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer",
    "editor.tabSize": 4
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode",
    "editor.tabSize": 2
  },
  "[javascript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode",
    "editor.tabSize": 2
  },
  "[json]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode",
    "editor.tabSize": 2
  },
  "[yaml]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode",
    "editor.tabSize": 2
  },
  "[toml]": {
    "editor.defaultFormatter": "tamasfe.even-better-toml",
    "editor.tabSize": 2
  }
}
EOF

# Create VSCode extensions recommendations
cat > .vscode/extensions.json << 'EOF'
{
  "recommendations": [
    "rust-lang.rust-analyzer",
    "esbenp.prettier-vscode",
    "tamasfe.even-better-toml",
    "ms-vscode.vscode-json",
    "redhat.vscode-yaml",
    "timonwong.shellcheck",
    "streetsidesoftware.code-spell-checker"
  ]
}
EOF

echo "✅ VSCode configuration created"

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Available commands:"
echo "  make lint          # Run all linting checks"  
echo "  make format        # Format all code"
echo "  make test          # Run all tests"
echo "  make ci            # Run full CI pipeline"
echo ""
echo "  ./scripts/lint-all.sh     # Comprehensive linting"
echo "  ./scripts/format-all.sh   # Format all code"
echo ""
echo "Git hooks installed:"
echo "  pre-commit         # Runs formatting and basic checks"
echo ""
echo "Next steps:"
echo "  1. Run 'make format' to format all existing code"
echo "  2. Run 'make lint' to check for any issues"
echo "  3. Run 'make test' to verify everything works"
echo ""