# Lens v2.2 Release Notes

## Headline Results

**Lens achieves 0.5234 nDCG@10 (±0.0045 CI) on span-only evaluation**

- **SLA**: 150ms hard timeout across all systems
- **Evaluation**: Parity embeddings (Gemma-256 baseline)  
- **Methodology**: Pooled qrels, bootstrap sampling (n=2000)
- **Hardware**: Standardized benchmark environment

## Key Improvements

✅ **Quality Gates Passed**
- Max-slice ECE: 0.0146 ≤ 0.02
- Tail ratio (p99/p95): 1.03 ≤ 2.0  
- CI width: 0.0045 ≤ 0.03
- File credit: 2.3% ≤ 5%

✅ **Reproducibility**
- Docker Compose setup with pinned digests
- Frozen artifact manifest with SHA256 hashes
- One-click reproduction: `docker compose up && make repro`

## Reproduction

```bash
# Clone and reproduce
git clone https://github.com/sibyllinesoft/lens.git
cd lens
git checkout v2.2
docker compose up -d
make repro

# Expected output: hero_span_v22.csv within ±0.1 pp
```

## Artifact Integrity

All artifacts verified with SHA256:
- Fingerprint: v22_1f3db391_1757345166574
- Full manifest: MANIFEST.json
- SBOM: SBOM.json  
- Attestation: ATTESTATION.json

Generated: 2025-09-08T16:04:56.638Z
