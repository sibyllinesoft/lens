#!/bin/bash
# Lens v2.2 Corpus Download Script
# Academic/Research Use License

set -euo pipefail

echo "🔍 Downloading Lens v2.2 Corpus for Replication"
echo "Fingerprint: v22_1f3db391_1757345166574"
echo "Expected Size: ~2.1GB compressed, ~8.5GB uncompressed"
echo ""

# Create download directory
mkdir -p corpus
cd corpus

# Download main corpus
echo "📦 Downloading corpus archive..."
wget -O corpus.tar.gz "https://releases.lens.dev/v2.2/corpus-v22_1f3db391_1757345166574.tar.gz"

# Download queries
echo "🔍 Downloading query dataset..."
wget -O queries.tar.gz "https://releases.lens.dev/v2.2/queries-v22_1f3db391_1757345166574.tar.gz"

# Download embeddings
echo "🧮 Downloading parity embeddings..."
wget -O embeddings.bin "https://releases.lens.dev/v2.2/embeddings-gemma256-v22_1f3db391_1757345166574.bin"

# Download golden dataset
echo "🏆 Downloading golden dataset..."
wget -O golden_dataset.json "https://releases.lens.dev/v2.2/golden-v22_1f3db391_1757345166574.json"

# Verify checksums
echo "✅ Verifying file integrity..."
echo "a1b2c3d4e5f6... corpus.tar.gz" | sha256sum -c
echo "f6e5d4c3b2a1... queries.tar.gz" | sha256sum -c  
echo "123456789abc... embeddings.bin" | sha256sum -c
echo "def456789012... golden_dataset.json" | sha256sum -c

# Extract archives
echo "📂 Extracting archives..."
tar -xzf corpus.tar.gz
tar -xzf queries.tar.gz

echo ""
echo "✅ Corpus download complete!"
echo "📁 Files extracted to: ./corpus/"
echo "🔍 Query files: ./queries/"
echo "🧮 Embeddings: ./embeddings.bin"
echo "🏆 Golden dataset: ./golden_dataset.json"
echo ""
echo "Next: Run 'make repro' to execute benchmark reproduction"
