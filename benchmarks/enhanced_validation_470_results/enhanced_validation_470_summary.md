# Enhanced Validation Report - 470 Query Core Set

**Status**: 🔴 NOT READY for Green CI Gates
**Total Queries**: 452
**Validation Date**: 51893.716702555

## 🎯 Key Results

### Overall Performance
- **Overall Pass Rate**: 57.5%
- **Extract Pass Rate**: 95.6%
- **Extract Substring Containment**: 95.6%

### Pointer-First Extract Performance  
- **Pointer Extractions**: 272
- **Generative Fallbacks**: 12
- **Containment Violations**: 0
- **Normalization Fixes**: 260

## 📊 Operation Breakdown

**Locate**: 0/90 (0.0%)
**Extract**: 260/272 (95.6%)
**Explain**: 0/90 (0.0%)

## 🚨 CI Gate Readiness

- ✅ **extract_pass_rate_ready**: True
- ❌ **substring_containment_ready**: False
- ❌ **overall_pass_rate_ready**: False

## 🔧 Recommendations

⚠️ NOT READY FOR GREEN CI GATES
🚫 Blocking issues:
  - substring_containment_ready
  - overall_pass_rate_ready

🔧 Required fixes:
  - Fix substring containment: 95.6% → 100%
    * Debug pointer-first extraction issues
    * Improve normalization and span matching
