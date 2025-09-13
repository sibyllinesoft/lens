# Green CI Gates Validation Report

**Status**: 🔴 NOT READY for Production
**Validation Date**: 2025-09-13T01:15:22.164246
**Overall Passed**: False

## 🎯 Green Gate Results

### Core Requirements
- **Pass Rate Core**: 60.0% (≥85% required) ❌
- **Extract Substring Containment**: 100.0% (100% required) ✅
- **Pointer Extract Success**: ✅

### System Integrity
- **Manifest Valid**: ❌
- **No Configuration Drift**: ✅
- **Ablation Sensitivity**: ✅
- **Critical CI Gates**: ❌

## 📊 Detailed Metrics

### Operation Performance
- **Locate**: 0/20 (0.0%)
- **Extract**: 60/60 (100.0%)
- **Explain**: 0/20 (0.0%)

### Pointer Extract Performance
- **Pointer Extractions**: 60
- **Generative Fallbacks**: 0
- **Containment Violations**: 0
- **Normalization Fixes**: 60

### CI Gate Summary
- **Critical Gates**: 6/10
- **Warning Gates**: 6/6

## 🚫 Blocking Issues

- Pass rate core 60.0% < 85.0%
- Manifest validation failed
- Critical CI gates failed

## 🔧 Required Actions

Before production deployment:

1. **Address Blocking Issues**: Fix all critical validation failures
2. **Re-run Validation**: Confirm all gates pass after fixes
3. **Update Manifest**: Create new signed manifest with fixes
4. **Verify Drift**: Ensure no configuration drift remains
