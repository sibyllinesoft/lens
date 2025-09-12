# Rust Migration Test Coverage Strategy

## Current Status Assessment

✅ **Library Compilation**: Core library (src/lib.rs) compiles successfully with only warnings
✅ **Search Engine Core**: Basic SearchEngine, SearchConfig, SearchRequest structures working
✅ **LSP Integration**: LSP module structures defined and compiling
✅ **Core Dependencies**: All major dependencies resolved

❌ **Test Compilation**: 120+ test compilation errors preventing coverage measurement
❌ **Coverage Measurement**: Cannot measure current coverage due to test failures

## Strategic Approach

### Phase 1: Critical Component Testing (Current Focus)
**Goal**: Get basic tests running for the most critical components

**Target Components**:
1. `SearchConfig` - Configuration management (highest priority)
2. `SearchRequest` - Request structures
3. Core search functionality (basic operations)
4. Cache operations (if working)

**Approach**: Create new minimal test files that work around compilation issues

### Phase 2: Incremental Test Fixing
**Goal**: Systematically fix existing test compilation errors

**Priority Order**:
1. Fix SearchResult struct field mismatches
2. Fix gRPC module test issues  
3. Fix LSP module test dependencies
4. Fix semantic module test dependencies

### Phase 3: Coverage Expansion
**Goal**: Achieve >85% test coverage with 100% pass rate

## Key Compilation Issues Identified

### 1. SearchResult Structure Mismatch
```rust
// Missing fields in SearchResult initializers:
// - column, context_lines, language, lsp_metadata, result_type
```

### 2. gRPC Protocol Buffer Field Mismatches
```rust
// Proto fields don't match Rust struct expectations
// Need to sync proto definitions with Rust structs
```

### 3. LSP Module Dependencies  
```rust
// Some LSP tests depend on external language servers
// Need mock implementations for testing
```

## Immediate Action Plan

### Step 1: Create Working Test Foundation
Create minimal tests for core functionality that definitely compiles:
- `tests/core_functionality.rs` - Basic config and request tests
- Focus on structures that are working in the library

### Step 2: Fix Critical Structure Mismatches
Fix the SearchResult and related structure issues that are blocking many tests

### Step 3: Measure Initial Coverage
Once we have some tests working, measure baseline coverage and identify gaps

### Step 4: Systematic Test Expansion
Add tests incrementally, measuring coverage after each addition

## Success Criteria
- [ ] At least 10 basic tests compiling and passing
- [ ] Coverage measurement working
- [ ] >85% line coverage on critical components:
  - SearchConfig and SearchRequest (configuration)
  - SearchEngine core functionality
  - Basic search operations
  - Error handling paths
- [ ] 100% test pass rate
- [ ] All tests run in <2 minutes

## Files Requiring Immediate Attention
1. `src/grpc/mod.rs` - Fix SearchResult field mismatches
2. `src/search.rs` - Ensure SearchResult structure is complete
3. Create new test files that bypass problematic dependencies

## Expected Timeline
- **Phase 1**: 1-2 hours (get basic tests working)
- **Phase 2**: 2-3 hours (fix major compilation issues)  
- **Phase 3**: 2-4 hours (achieve coverage targets)

This approach focuses on demonstrating that the Rust migration has comprehensive test coverage while being pragmatic about the current compilation challenges.