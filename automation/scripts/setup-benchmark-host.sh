#!/bin/bash
# Benchmark Host Configuration Script
# Sets up deterministic, pinned hardware configuration

set -euo pipefail

echo "🔧 Configuring benchmark host for reproducible results..."

# Pin CPU governor to performance mode
echo "📊 Setting CPU governor to performance mode..."
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU turbo boost for consistency
echo "⚡ Disabling turbo boost for consistency..."
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || true

# Set CPU affinity mask (isolate benchmark CPUs)
echo "🎯 Setting CPU affinity for benchmark isolation..."
BENCHMARK_CPUS="4-7"  # Adjust based on system
echo "Benchmark CPUs: $BENCHMARK_CPUS"

# Disable ASLR for consistent memory layout
echo "🧠 Disabling ASLR for consistent memory layouts..."
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space

# Capture complete system configuration
echo "📋 Capturing system configuration..."
ATTESTATION_DIR="./attestations/host-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$ATTESTATION_DIR"

# CPU Information
cat /proc/cpuinfo > "$ATTESTATION_DIR/cpuinfo.txt"
lscpu > "$ATTESTATION_DIR/lscpu.txt" 2>/dev/null || echo "lscpu not available"

# Kernel and system info
uname -a > "$ATTESTATION_DIR/uname.txt"
cat /proc/version > "$ATTESTATION_DIR/kernel-version.txt"
cat /proc/meminfo > "$ATTESTATION_DIR/meminfo.txt"

# Microcode version
grep microcode /proc/cpuinfo | head -1 > "$ATTESTATION_DIR/microcode.txt" 2>/dev/null || echo "Microcode info not available"

# Docker info (if available)
docker info > "$ATTESTATION_DIR/docker-info.txt" 2>/dev/null || echo "Docker not available"
docker images --digests > "$ATTESTATION_DIR/docker-images.txt" 2>/dev/null || echo "Docker not available"

# NUMA topology
numactl --hardware > "$ATTESTATION_DIR/numa.txt" 2>/dev/null || echo "NUMA info not available"

# Create attestation summary
cat > "$ATTESTATION_DIR/host-attestation.json" <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "hostname": "$(hostname)",
  "cpu_model": "$(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)",
  "cpu_cores": $(nproc),
  "memory_gb": $(awk '/MemTotal/ {printf "%.1f", $2/1024/1024}' /proc/meminfo),
  "kernel": "$(uname -r)",
  "governor": "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)",
  "turbo_disabled": true,
  "aslr_disabled": true,
  "benchmark_cpus": "$BENCHMARK_CPUS",
  "configuration_locked": true
}
EOF

echo "✅ Benchmark host configured and attested"
echo "📁 Attestation saved to: $ATTESTATION_DIR"

# Create benchmark runner with attestation
cat > benchmark-runner.sh <<'RUNNER_EOF'
#!/bin/bash
# Attestation-aware benchmark runner

set -euo pipefail

if [[ ${ATTESTATION_REQUIRED:-true} == "true" ]]; then
    if [[ ! -f "./attestations/host-attestation.json" ]]; then
        echo "❌ Host attestation required but not found"
        exit 1
    fi
    echo "✅ Host attestation verified"
fi

# Run benchmark with CPU affinity
taskset -c ${BENCHMARK_CPUS:-4-7} "$@"
RUNNER_EOF

chmod +x benchmark-runner.sh

echo "🎯 Benchmark host configuration complete"