#!/bin/bash
# Protocol Buffer Compiler Installation Script
# Generated on: 2025-09-06
# Review this script before execution

set -euo pipefail  # Exit on any error

echo "🔍 Checking system requirements..."
if ! command -v curl &> /dev/null; then
    echo "❌ curl is required but not installed"
    exit 1
fi

echo "📦 Installing Protocol Buffer Compiler (protoc)..."

# Check if we're on a supported system
if command -v apt-get &> /dev/null; then
    echo "🔧 Detected Debian/Ubuntu system"
    echo "Running: sudo apt-get update && sudo apt-get install -y protobuf-compiler"
    sudo apt-get update
    sudo apt-get install -y protobuf-compiler
elif command -v yum &> /dev/null; then
    echo "🔧 Detected RedHat/CentOS system"
    echo "Running: sudo yum install -y protobuf-compiler"
    sudo yum install -y protobuf-compiler
elif command -v dnf &> /dev/null; then
    echo "🔧 Detected Fedora system"
    echo "Running: sudo dnf install -y protobuf-compiler"
    sudo dnf install -y protobuf-compiler
elif command -v brew &> /dev/null; then
    echo "🔧 Detected macOS with Homebrew"
    echo "Running: brew install protobuf"
    brew install protobuf
else
    echo "❌ Unsupported package manager. Please install protobuf-compiler manually."
    echo "📥 Download from: https://github.com/protocolbuffers/protobuf/releases"
    exit 1
fi

echo "✅ Verifying installation..."
if command -v protoc &> /dev/null; then
    PROTOC_VERSION=$(protoc --version)
    echo "🎉 protoc installed successfully: $PROTOC_VERSION"
else
    echo "❌ protoc installation failed"
    exit 1
fi

echo "🎉 Installation complete!"
echo "You can now run: cargo build --release"