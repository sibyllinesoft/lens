# Lens Repository Reorganization Plan

## Current Status
- **Total files in root**: 229 files
- **File types**: 88 JS, 21 TS, 9 PY, 33 JSON, 26 MD, plus logs, configs, etc.
- **Problem**: Root directory extremely cluttered, hard to navigate

## Proposed Directory Structure

### 1. `/automation/` - All automation and utility scripts
**Target files (88+ files):**
- All `.js` scripts: create-*, fix-*, run-*, test-*, validate-*, generate-*, etc.
- All `.py` scripts: test_*.py, benchmark_*.py, etc.
- All `.sh` scripts: setup-*, install-*, security-*, etc.
- All `.ts` scripts: Not in src/, like test-*.ts, debug-*.ts
- All `.cjs` and `.mjs` files

### 2. `/reports/` - Status reports and documentation
**Target files (26+ files):**
- All status `.md` files: *_COMPLETE.md, PROTOCOL_*.md, etc.
- Report files: TEST_COVERAGE_REPORT.md, DEPLOYMENT_SUMMARY.md
- Analysis files: technical-gap-analysis.md, semantic_lift_analysis.md

### 3. `/outputs/` - Generated output files  
**Target files (50+ files):**
- All `.log` files: baseline_*.log, *_results.log
- All `.ndjson` files: traces-*.ndjson, *-errors.ndjson
- Generated `.json` result files: corpus_benchmark_*.json, *_results.json
- `.html` output files

### 4. `/configurations/` - Configuration files
**Target files (20+ files):**
- Non-essential `.json` configs: tripwire-config.json, lens-vs-industry-config.json
- Policy files: baseline_policy.json, pre_publish_checklist.json
- Fingerprint files: config-fingerprint-*.json, *_config_fingerprint.json

### 5. Root directory - Keep ONLY essential files
**Files to keep in root:**
- package.json, package-lock.json
- tsconfig.json, vitest.config.ts
- README.md, LICENSE, TODO.md, CLAUDE.md
- Dockerfile, docker-compose.yml, docker-compose.hermetic.yml
- Essential config files: .eslintrc.json, .prettierrc, .gitignore, .port-config.json
- Core system files: Cargo.toml, Cargo.lock
- rust-toolchain.toml, build.rs

## File Movement Plan

### Phase 1: Create directories and move files
```bash
mkdir -p automation/{scripts,tests,utilities,benchmarks}
mkdir -p reports/{status,analysis,summaries}  
mkdir -p outputs/{logs,results,traces,artifacts}
mkdir -p configurations/{policies,fingerprints,settings}
```

### Phase 2: Move files by category
1. **Automation files** → `/automation/`
2. **Report files** → `/reports/`
3. **Output files** → `/outputs/`
4. **Configuration files** → `/configurations/`

### Phase 3: Update all references
1. **package.json scripts** - Update all script paths
2. **Import statements** - Update all relative imports
3. **Internal references** - Update file path references in moved files
4. **GitHub workflows** - Update CI/CD paths

### Phase 4: Validation
1. **Run test suite** - Ensure nothing breaks
2. **Test core commands** - npm scripts, build, deploy
3. **Verify CI/CD** - Check GitHub Actions still work

## Reference Update Strategy

### Files that will need path updates:
1. **package.json** - 40+ script entries need path updates
2. **All moved scripts** - Internal file references
3. **GitHub Actions** - CI/CD workflow files
4. **Docker files** - Any script references
5. **Import/require statements** throughout codebase

### Example updates needed:
```javascript
// Before: 
import { something } from './create-golden-data.js'
// After:
import { something } from '../automation/create-golden-data.js'

// Before package.json:
"script": "node create-benchmark-reports.js"  
// After:
"script": "node automation/create-benchmark-reports.js"
```

## Risk Mitigation
1. **Backup current state** before any moves
2. **Move files incrementally** and test each phase
3. **Update references immediately** after each move
4. **Test core functionality** after each phase
5. **Maintain git history** using `git mv` commands

## Success Criteria
- ✅ Root directory has <20 essential files only
- ✅ All files organized into logical subdirectories  
- ✅ All imports and references updated correctly
- ✅ npm scripts work correctly
- ✅ Build and test processes work
- ✅ CI/CD pipelines work
- ✅ Core server functionality works

## Estimated Impact
- **Low risk moves**: Output files, logs, generated files
- **Medium risk moves**: Standalone scripts with few dependencies  
- **High risk moves**: Core scripts referenced in package.json
- **Critical updates**: package.json scripts, workflow files

This plan will transform the cluttered 229-file root directory into a clean, organized structure with proper separation of concerns.