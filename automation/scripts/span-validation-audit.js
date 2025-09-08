#!/usr/bin/env node

/**
 * Span Validation Audit System
 * 
 * Per TODO.md: "Instrument & localize (one run): add span_validated and span_error_reason 
 * to every hit and compute coverage by {stage, lang, zone, why}."
 *
 * Implements the pseudocode:
 * for hit in results:
 *   blob = read(repo_sha, hit.file)            // exact snapshot
 *   canon = normalize(blob)                     // utf8, CRLF→LF, tabs=1
 *   ok = validate_span(canon, hit.line, hit.col, hit.span_len OR hit.byte_offset)
 *   log(hit.trace_id, validated=ok, reason=diagnose())
 */

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

// Span validation error categories per TODO.md
const SPAN_ERROR_REASONS = {
  OOB: 'Out of bounds - line/col exceeds file dimensions',
  BYTE_VS_CP: 'Byte vs codepoint indexing mismatch',
  CRLF: 'CRLF vs LF newline normalization issue',
  TAB: 'Tab width calculation disagreement', 
  AST_RANGE: 'AST range conversion error',
  LSIF_RANGE: 'LSIF range conversion error',
  CANDIDATE_MISSING: 'Stage-A scanner missing candidate',
  UNKNOWN: 'Unknown validation failure'
};

class SpanValidationAuditor {
  constructor() {
    this.corpusDir = './indexed-content';
    this.anchorDataset = null;
    this.auditResults = [];
    this.stats = {
      total_spans: 0,
      validated_spans: 0,
      failed_spans: 0,
      error_breakdown: {},
      coverage_by_stage: {},
      coverage_by_lang: {},
      coverage_by_zone: {}
    };
  }

  async runSpanAudit() {
    console.log('🔍 Starting Span Validation Audit...');
    
    try {
      // 1. Load AnchorSmoke dataset
      await this.loadAnchorDataset();
      
      // 2. Run validation audit on each query result
      await this.auditAllSpans();
      
      // 3. Analyze results and generate breakdown
      const analysis = this.analyzeSpanFailures();
      
      // 4. Generate audit report
      await this.generateAuditReport(analysis);
      
      console.log('✅ Span validation audit complete!');
      return analysis;
      
    } catch (error) {
      console.error('❌ Span audit failed:', error.message);
      throw error;
    }
  }

  async loadAnchorDataset() {
    console.log('📂 Loading AnchorSmoke dataset for span audit...');
    
    const anchorPath = './anchor-datasets/anchor_current.json';
    if (!fs.existsSync(anchorPath)) {
      throw new Error(`AnchorSmoke dataset not found: ${anchorPath}`);
    }
    
    this.anchorDataset = JSON.parse(fs.readFileSync(anchorPath, 'utf-8'));
    console.log(`   ✅ Loaded ${this.anchorDataset.total_queries} queries with golden spans`);
  }

  async auditAllSpans() {
    console.log('🔬 Auditing span validation for all queries...');
    
    const queries = this.anchorDataset.queries;
    
    for (let i = 0; i < queries.length; i++) {
      const query = queries[i];
      
      console.log(`   Processing query ${i + 1}/${queries.length}: "${query.query}"`);
      
      // Audit each golden span for this query
      for (const span of query.golden_spans) {
        const auditResult = await this.auditSpan(query, span);
        this.auditResults.push(auditResult);
        this.updateStats(auditResult);
      }
    }
    
    console.log(`   ✅ Audited ${this.stats.total_spans} spans total`);
    console.log(`   📊 Validation rate: ${((this.stats.validated_spans / this.stats.total_spans) * 100).toFixed(1)}%`);
  }

  async auditSpan(query, span) {
    const auditResult = {
      query_id: query.id,
      query: query.query,
      intent: query.intent,
      language: query.language,
      zone: query.zone,
      span_file: span.file,
      span_line: span.line,
      span_col: span.col,
      trace_id: this.generateTraceId(),
      timestamp: new Date().toISOString()
    };

    try {
      // Step 1: Read exact snapshot (blob)
      const blob = await this.readCorpusFile(span.file);
      if (!blob) {
        auditResult.span_validated = false;
        auditResult.span_error_reason = 'CANDIDATE_MISSING';
        auditResult.error_details = 'Corpus file not found';
        return auditResult;
      }

      // Step 2: Normalize blob (utf8, CRLF→LF, tabs=1)
      const canonicalBlob = this.normalizeBlob(blob);
      
      // Step 3: Validate span coordinates
      const validation = this.validateSpanCoordinates(
        canonicalBlob, 
        span.line, 
        span.col, 
        span.span_len || null,
        span.byte_offset || null
      );
      
      auditResult.span_validated = validation.valid;
      auditResult.span_error_reason = validation.error_reason;
      auditResult.error_details = validation.error_details;
      auditResult.normalized_content_length = canonicalBlob.length;
      auditResult.line_count = canonicalBlob.split('\n').length;
      
      // Additional diagnostics
      auditResult.diagnostics = this.generateSpanDiagnostics(blob, canonicalBlob, span);
      
    } catch (error) {
      auditResult.span_validated = false;
      auditResult.span_error_reason = 'UNKNOWN';
      auditResult.error_details = error.message;
    }

    return auditResult;
  }

  async readCorpusFile(filename) {
    const filePath = path.join(this.corpusDir, filename);
    
    if (!fs.existsSync(filePath)) {
      return null;
    }
    
    try {
      return fs.readFileSync(filePath, 'utf-8');
    } catch (error) {
      console.warn(`     ⚠️  Cannot read ${filename}: ${error.message}`);
      return null;
    }
  }

  normalizeBlob(blob) {
    // Per TODO.md: "utf8, CRLF→LF, tabs=1"
    let normalized = blob;
    
    // Convert CRLF to LF
    normalized = normalized.replace(/\r\n/g, '\n');
    
    // Convert standalone CR to LF (rare but possible)
    normalized = normalized.replace(/\r/g, '\n');
    
    // Note: tabs policy is "tabs advance by 1" - no conversion needed,
    // just consistent counting in validation
    
    return normalized;
  }

  validateSpanCoordinates(canonicalBlob, line, col, spanLen, byteOffset) {
    const lines = canonicalBlob.split('\n');
    
    // Validate line bounds (1-indexed)
    if (line < 1 || line > lines.length) {
      return {
        valid: false,
        error_reason: 'OOB',
        error_details: `Line ${line} out of bounds (1-${lines.length})`
      };
    }
    
    const targetLine = lines[line - 1]; // Convert to 0-indexed
    
    // Validate column bounds (1-indexed, codepoint-based)
    const lineCodepoints = Array.from(targetLine); // Split into Unicode codepoints
    if (col < 1 || col > lineCodepoints.length + 1) { // +1 allows end-of-line
      return {
        valid: false,
        error_reason: 'OOB',
        error_details: `Column ${col} out of bounds (1-${lineCodepoints.length + 1}) for line ${line}`
      };
    }
    
    // Check for common span validation issues
    const diagnostics = this.diagnoseCodingIssues(canonicalBlob, targetLine, line, col);
    
    if (diagnostics.likely_issue) {
      return {
        valid: false,
        error_reason: diagnostics.likely_issue,
        error_details: diagnostics.details
      };
    }
    
    // If we get here, span coordinates are valid
    return {
      valid: true,
      error_reason: null,
      error_details: 'Span coordinates validated successfully'
    };
  }

  diagnoseCodingIssues(canonicalBlob, targetLine, line, col) {
    const diagnostics = { likely_issue: null, details: '' };
    
    // Check for byte vs codepoint issues
    const lineBytes = Buffer.from(targetLine, 'utf-8');
    const lineCodepoints = Array.from(targetLine);
    
    if (lineBytes.length !== lineCodepoints.length) {
      // Multi-byte characters present - could be byte/codepoint confusion
      if (col > lineBytes.length && col <= lineCodepoints.length) {
        diagnostics.likely_issue = 'BYTE_VS_CP';
        diagnostics.details = `Column ${col} valid as codepoints (${lineCodepoints.length}) but not bytes (${lineBytes.length})`;
        return diagnostics;
      }
    }
    
    // Check for tab width calculation issues
    if (targetLine.includes('\t')) {
      const tabCount = (targetLine.match(/\t/g) || []).length;
      diagnostics.likely_issue = 'TAB';
      diagnostics.details = `Line contains ${tabCount} tabs, could cause width calculation issues`;
      return diagnostics;
    }
    
    // Check for original CRLF normalization issues
    const originalLine = targetLine.replace(/\n/g, '\r\n'); // Simulate original CRLF
    if (originalLine.length !== targetLine.length) {
      diagnostics.likely_issue = 'CRLF';
      diagnostics.details = `Line length differs after CRLF normalization: ${originalLine.length} → ${targetLine.length}`;
      return diagnostics;
    }
    
    return diagnostics;
  }

  generateSpanDiagnostics(originalBlob, canonicalBlob, span) {
    return {
      original_size_bytes: Buffer.from(originalBlob, 'utf-8').length,
      canonical_size_bytes: Buffer.from(canonicalBlob, 'utf-8').length,
      original_size_codepoints: Array.from(originalBlob).length,
      canonical_size_codepoints: Array.from(canonicalBlob).length,
      crlf_count: (originalBlob.match(/\r\n/g) || []).length,
      cr_count: (originalBlob.match(/\r(?!\n)/g) || []).length,
      tab_count: (originalBlob.match(/\t/g) || []).length,
      normalization_changed: originalBlob !== canonicalBlob
    };
  }

  generateTraceId() {
    return crypto.randomBytes(8).toString('hex');
  }

  updateStats(auditResult) {
    this.stats.total_spans++;
    
    if (auditResult.span_validated) {
      this.stats.validated_spans++;
    } else {
      this.stats.failed_spans++;
      
      // Update error breakdown
      const reason = auditResult.span_error_reason || 'UNKNOWN';
      this.stats.error_breakdown[reason] = (this.stats.error_breakdown[reason] || 0) + 1;
    }
    
    // Update coverage by language
    const lang = auditResult.language || 'unknown';
    if (!this.stats.coverage_by_lang[lang]) {
      this.stats.coverage_by_lang[lang] = { total: 0, validated: 0 };
    }
    this.stats.coverage_by_lang[lang].total++;
    if (auditResult.span_validated) {
      this.stats.coverage_by_lang[lang].validated++;
    }
    
    // Update coverage by zone
    const zone = auditResult.zone || 'unknown';
    if (!this.stats.coverage_by_zone[zone]) {
      this.stats.coverage_by_zone[zone] = { total: 0, validated: 0 };
    }
    this.stats.coverage_by_zone[zone].total++;
    if (auditResult.span_validated) {
      this.stats.coverage_by_zone[zone].validated++;
    }
  }

  analyzeSpanFailures() {
    console.log('📊 Analyzing span failure patterns...');
    
    const analysis = {
      summary: {
        total_spans: this.stats.total_spans,
        validation_rate: this.stats.total_spans > 0 ? 
          (this.stats.validated_spans / this.stats.total_spans) : 0,
        dominant_failure_reason: null,
        dominant_failure_percentage: 0
      },
      error_breakdown: {},
      coverage_by_dimension: {
        language: {},
        zone: {}
      },
      recommendations: []
    };
    
    // Calculate error breakdown percentages
    for (const [reason, count] of Object.entries(this.stats.error_breakdown)) {
      const percentage = (count / this.stats.failed_spans) * 100;
      analysis.error_breakdown[reason] = {
        count,
        percentage: percentage.toFixed(1)
      };
      
      // Track dominant failure reason
      if (percentage > analysis.summary.dominant_failure_percentage) {
        analysis.summary.dominant_failure_reason = reason;
        analysis.summary.dominant_failure_percentage = percentage;
      }
    }
    
    // Calculate coverage by dimensions
    for (const [lang, stats] of Object.entries(this.stats.coverage_by_lang)) {
      analysis.coverage_by_dimension.language[lang] = {
        total: stats.total,
        validated: stats.validated,
        coverage_rate: (stats.validated / stats.total) * 100
      };
    }
    
    for (const [zone, stats] of Object.entries(this.stats.coverage_by_zone)) {
      analysis.coverage_by_dimension.zone[zone] = {
        total: stats.total,
        validated: stats.validated,
        coverage_rate: (stats.validated / stats.total) * 100
      };
    }
    
    // Generate recommendations based on dominant failure
    analysis.recommendations = this.generateRecommendations(analysis);
    
    console.log(`   🎯 Dominant failure: ${analysis.summary.dominant_failure_reason} (${analysis.summary.dominant_failure_percentage.toFixed(1)}%)`);
    console.log(`   📈 Overall validation rate: ${(analysis.summary.validation_rate * 100).toFixed(1)}%`);
    
    return analysis;
  }

  generateRecommendations(analysis) {
    const recommendations = [];
    const dominant = analysis.summary.dominant_failure_reason;
    const dominantPct = analysis.summary.dominant_failure_percentage;
    
    if (dominantPct >= 60) {
      recommendations.push({
        priority: 'HIGH',
        action: `Fix ${dominant} issue - represents ${dominantPct.toFixed(1)}% of failures`,
        implementation: this.getImplementationGuidance(dominant)
      });
    }
    
    // Check for language-specific issues
    for (const [lang, stats] of Object.entries(analysis.coverage_by_dimension.language)) {
      if (stats.coverage_rate < 80) {
        recommendations.push({
          priority: 'MEDIUM',
          action: `Investigate ${lang} span issues`,
          implementation: `${lang} has only ${stats.coverage_rate.toFixed(1)}% span coverage`
        });
      }
    }
    
    // Check for zone-specific issues  
    for (const [zone, stats] of Object.entries(analysis.coverage_by_dimension.zone)) {
      if (stats.coverage_rate < 80) {
        recommendations.push({
          priority: 'MEDIUM',
          action: `Investigate ${zone} zone span issues`,
          implementation: `${zone} zone has only ${stats.coverage_rate.toFixed(1)}% span coverage`
        });
      }
    }
    
    return recommendations;
  }

  getImplementationGuidance(errorReason) {
    const guidance = {
      'BYTE_VS_CP': 'Regenerate byte↔cp maps using codepoint indexing (not grapheme). Ensure all stages use same conversion.',
      'CRLF': 'Ensure ingest replaces \\r\\n with \\n BEFORE tokenization and AST parsing.',
      'TAB': 'Enforce tabs advance by 1 policy. Store per-file line_starts_cp[] for O(1) conversion.',
      'AST_RANGE': 'Apply converter at emission time; never return tool-native ranges.',
      'LSIF_RANGE': 'Apply converter at emission time; never return tool-native ranges.',
      'CANDIDATE_MISSING': 'Stage-A scanner fallback bug: enable native scanner for py/js/ts temporarily.',
      'OOB': 'Check span coordinate calculation and file boundary detection.'
    };
    
    return guidance[errorReason] || 'Review span coordinate calculation logic';
  }

  async generateAuditReport(analysis) {
    const reportDir = './span-audit-results';
    if (!fs.existsSync(reportDir)) {
      fs.mkdirSync(reportDir, { recursive: true });
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5) + 'Z';
    
    // Save detailed audit results
    const detailedResults = {
      timestamp,
      analysis,
      audit_results: this.auditResults,
      configuration: {
        normalization_policy: 'CRLF→LF, tabs=1',
        coordinate_system: 'codepoint-based, 1-indexed',
        corpus_snapshot: 'indexed-content'
      }
    };
    
    fs.writeFileSync(
      path.join(reportDir, `span-audit-${timestamp}.json`),
      JSON.stringify(detailedResults, null, 2)
    );
    
    // Generate human-readable report
    const reportContent = this.generateMarkdownReport(analysis, timestamp);
    const reportPath = path.join(reportDir, `span-audit-report-${timestamp}.md`);
    fs.writeFileSync(reportPath, reportContent);
    
    console.log('📄 Span audit reports generated:');
    console.log(`   📊 Detailed results: ${reportDir}/span-audit-${timestamp}.json`);
    console.log(`   📋 Audit report: ${reportPath}`);
  }

  generateMarkdownReport(analysis, timestamp) {
    let report = `# Span Validation Audit Report\n\n`;
    report += `**Timestamp:** ${timestamp}  \n`;
    report += `**Total Spans Audited:** ${analysis.summary.total_spans}  \n`;
    report += `**Validation Rate:** ${(analysis.summary.validation_rate * 100).toFixed(1)}%  \n\n`;
    
    if (analysis.summary.dominant_failure_reason) {
      report += `## 🎯 Key Finding\n\n`;
      report += `**Dominant Failure Reason:** ${analysis.summary.dominant_failure_reason} (${analysis.summary.dominant_failure_percentage.toFixed(1)}% of failures)  \n\n`;
    }
    
    report += `## 📊 Error Breakdown\n\n`;
    report += `| Error Reason | Count | Percentage |\n`;
    report += `|--------------|-------|------------|\n`;
    for (const [reason, stats] of Object.entries(analysis.error_breakdown)) {
      report += `| ${reason} | ${stats.count} | ${stats.percentage}% |\n`;
    }
    report += '\n';
    
    report += `## 🔍 Coverage by Language\n\n`;
    report += `| Language | Total | Validated | Coverage |\n`;
    report += `|----------|-------|-----------|----------|\n`;
    for (const [lang, stats] of Object.entries(analysis.coverage_by_dimension.language)) {
      report += `| ${lang} | ${stats.total} | ${stats.validated} | ${stats.coverage_rate.toFixed(1)}% |\n`;
    }
    report += '\n';
    
    report += `## 🗂️ Coverage by Zone\n\n`;
    report += `| Zone | Total | Validated | Coverage |\n`;
    report += `|------|-------|-----------|----------|\n`;
    for (const [zone, stats] of Object.entries(analysis.coverage_by_dimension.zone)) {
      report += `| ${zone} | ${stats.total} | ${stats.validated} | ${stats.coverage_rate.toFixed(1)}% |\n`;
    }
    report += '\n';
    
    if (analysis.recommendations.length > 0) {
      report += `## 🔧 Recommendations\n\n`;
      for (const rec of analysis.recommendations) {
        report += `### ${rec.priority}: ${rec.action}\n\n`;
        report += `${rec.implementation}\n\n`;
      }
    }
    
    report += `## 🛠️ Next Steps\n\n`;
    report += `Per TODO.md requirements:\n\n`;
    if (analysis.summary.dominant_failure_percentage >= 60) {
      report += `1. **Immediate Fix**: Address ${analysis.summary.dominant_failure_reason} issue (≥60% dominant)\n`;
    } else {
      report += `1. **Multi-faceted Fix**: No single dominant issue - address top 2-3 failure reasons\n`;
    }
    report += `2. **Implement Targeted Patch**: Apply PATCH /policy/spans with canonical_v2 resolver\n`;
    report += `3. **Rerun Gates**: Target span ≥99% with quality flat\n`;
    report += `4. **Enable Precision Knobs**: Once span coverage clears ≥99%\n\n`;
    
    report += `---\n\n`;
    report += `*Generated by Span Validation Auditor*  \n`;
    report += `*Normalization: CRLF→LF, tabs=1, codepoint-indexed*\n`;
    
    return report;
  }
}

// Main execution
async function main() {
  try {
    console.log('🚀 Starting Span Validation Audit System...');
    
    const auditor = new SpanValidationAuditor();
    const analysis = await auditor.runSpanAudit();
    
    // Exit with appropriate code based on validation rate
    const validationRate = analysis.summary.validation_rate;
    if (validationRate >= 0.99) {
      console.log('\n✅ SPAN COVERAGE EXCELLENT (≥99%)');
      process.exit(0);
    } else if (validationRate >= 0.90) {
      console.log('\n⚠️ SPAN COVERAGE NEEDS IMPROVEMENT (90-99%)');
      process.exit(1);
    } else {
      console.log('\n❌ SPAN COVERAGE CRITICAL (<90%)');
      process.exit(2);
    }
    
  } catch (error) {
    console.error('❌ Span validation audit failed:', error.message);
    process.exit(3);
  }
}

console.log('Script loaded. import.meta.url:', import.meta.url);

main().catch(console.error);

export { SpanValidationAuditor };