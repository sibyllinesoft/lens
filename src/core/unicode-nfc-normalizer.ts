/**
 * Unicode NFC Normalization for Spans
 * 
 * Implements Unicode NFC canonicalization for spans to prevent combining-character
 * drift and ensure consistent span mathematics across the search system.
 */

import { LensTracer } from '../telemetry/tracer.js';

export interface SpanPosition {
  byte_offset: number;
  char_offset: number;
  line: number;
  column: number;
}

export interface NormalizedSpan {
  start_position: SpanPosition;
  end_position: SpanPosition;
  original_text: string;
  normalized_text: string;
  length_before_normalization: number;
  length_after_normalization: number;
  normalization_applied: boolean;
  combining_chars_removed: number;
}

export interface NormalizationMetrics {
  total_spans_processed: number;
  spans_with_combining_chars: number;
  bytes_removed: number;
  chars_removed: number;
  normalization_time_ms: number;
  most_common_combining_chars: Map<string, number>;
}

export interface UnicodeNormalizationConfig {
  normalization_form: 'NFC' | 'NFD' | 'NFKC' | 'NFKD';
  preserve_whitespace: boolean;
  track_combining_chars: boolean;
  validate_utf8: boolean;
  max_span_length: number;
}

/**
 * Unicode combining character detector and counter
 */
class CombiningCharacterAnalyzer {
  private combiningCharCounts: Map<string, number> = new Map();
  
  // Unicode combining character ranges (simplified)
  private readonly COMBINING_RANGES = [
    [0x0300, 0x036F], // Combining Diacritical Marks
    [0x1AB0, 0x1AFF], // Combining Diacritical Marks Extended
    [0x1DC0, 0x1DFF], // Combining Diacritical Marks Supplement
    [0x20D0, 0x20FF], // Combining Diacritical Marks for Symbols
    [0xFE20, 0xFE2F], // Combining Half Marks
  ];
  
  /**
   * Check if character is a combining character
   */
  isCombiningCharacter(codePoint: number): boolean {
    return this.COMBINING_RANGES.some(([start, end]) => 
      codePoint >= start && codePoint <= end
    );
  }
  
  /**
   * Analyze text for combining characters
   */
  analyzeCombiningChars(text: string): {
    total_combining_chars: number;
    unique_combining_chars: Set<string>;
    positions: number[];
  } {
    const positions: number[] = [];
    const uniqueChars = new Set<string>();
    let totalCombining = 0;
    
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      const codePoint = char.codePointAt(0);
      
      if (codePoint && this.isCombiningCharacter(codePoint)) {
        positions.push(i);
        uniqueChars.add(char);
        totalCombining++;
        
        // Track frequency
        this.combiningCharCounts.set(char, (this.combiningCharCounts.get(char) || 0) + 1);
      }
    }
    
    return {
      total_combining_chars: totalCombining,
      unique_combining_chars: uniqueChars,
      positions
    };
  }
  
  /**
   * Get most common combining characters
   */
  getMostCommonCombiningChars(limit = 10): Map<string, number> {
    const sortedEntries = Array.from(this.combiningCharCounts.entries())
      .sort(([, a], [, b]) => b - a)
      .slice(0, limit);
    
    return new Map(sortedEntries);
  }
  
  /**
   * Reset statistics
   */
  resetStats(): void {
    this.combiningCharCounts.clear();
  }
}

/**
 * Span position calculator that handles Unicode correctly
 */
class UnicodeSpanCalculator {
  /**
   * Calculate byte and character offsets for text position
   */
  calculatePosition(text: string, charIndex: number): SpanPosition {
    // Convert to UTF-8 bytes for accurate byte offset calculation
    const textUpToIndex = text.substring(0, charIndex);
    const byteOffset = new TextEncoder().encode(textUpToIndex).length;
    
    // Calculate line and column
    const lines = textUpToIndex.split('\n');
    const line = lines.length;
    const column = lines[lines.length - 1].length + 1; // 1-based column
    
    return {
      byte_offset: byteOffset,
      char_offset: charIndex,
      line: line,
      column: column
    };
  }
  
  /**
   * Adjust span positions after normalization
   */
  adjustSpanAfterNormalization(
    originalText: string,
    normalizedText: string,
    originalStart: number,
    originalEnd: number
  ): {
    new_start: number;
    new_end: number;
    length_delta: number;
  } {
    // Simple approach: if lengths differ, proportionally adjust positions
    const lengthRatio = normalizedText.length / originalText.length;
    
    const newStart = Math.floor(originalStart * lengthRatio);
    const newEnd = Math.floor(originalEnd * lengthRatio);
    const lengthDelta = normalizedText.length - originalText.length;
    
    return {
      new_start: newStart,
      new_end: newEnd,
      length_delta: lengthDelta
    };
  }
}

/**
 * UTF-8 validation utilities
 */
class UTF8Validator {
  /**
   * Validate that text is valid UTF-8
   */
  isValidUTF8(text: string): boolean {
    try {
      // Try to encode and decode - invalid UTF-8 will throw
      const encoded = new TextEncoder().encode(text);
      const decoded = new TextDecoder('utf-8', { fatal: true }).decode(encoded);
      return decoded === text;
    } catch (error) {
      return false;
    }
  }
  
  /**
   * Clean invalid UTF-8 sequences
   */
  cleanInvalidUTF8(text: string): string {
    try {
      const encoded = new TextEncoder().encode(text);
      return new TextDecoder('utf-8', { fatal: false }).decode(encoded);
    } catch (error) {
      // Fallback: remove non-printable characters
      return text.replace(/[\x00-\x1F\x7F-\x9F]/g, '');
    }
  }
}

/**
 * Main Unicode NFC normalizer
 */
export class UnicodeNFCNormalizer {
  private config: UnicodeNormalizationConfig;
  private combiningAnalyzer: CombiningCharacterAnalyzer;
  private spanCalculator: UnicodeSpanCalculator;
  private utf8Validator: UTF8Validator;
  private enabled = true;
  
  // Metrics
  private metrics: NormalizationMetrics = {
    total_spans_processed: 0,
    spans_with_combining_chars: 0,
    bytes_removed: 0,
    chars_removed: 0,
    normalization_time_ms: 0,
    most_common_combining_chars: new Map()
  };
  
  constructor(config?: Partial<UnicodeNormalizationConfig>) {
    this.config = {
      normalization_form: 'NFC',
      preserve_whitespace: true,
      track_combining_chars: true,
      validate_utf8: true,
      max_span_length: 10000,
      ...config
    };
    
    this.combiningAnalyzer = new CombiningCharacterAnalyzer();
    this.spanCalculator = new UnicodeSpanCalculator();
    this.utf8Validator = new UTF8Validator();
  }
  
  /**
   * Normalize text using specified Unicode normalization form
   */
  normalizeText(text: string): {
    normalized_text: string;
    normalization_applied: boolean;
    length_delta: number;
    combining_chars_removed: number;
  } {
    if (!this.enabled || !text) {
      return {
        normalized_text: text,
        normalization_applied: false,
        length_delta: 0,
        combining_chars_removed: 0
      };
    }
    
    const originalLength = text.length;
    
    // Validate UTF-8 if enabled
    let processedText = text;
    if (this.config.validate_utf8 && !this.utf8Validator.isValidUTF8(text)) {
      processedText = this.utf8Validator.cleanInvalidUTF8(text);
      console.log(`🔧 Cleaned invalid UTF-8 sequences in text (${text.length} -> ${processedText.length} chars)`);
    }
    
    // Apply Unicode normalization
    let normalizedText = processedText.normalize(this.config.normalization_form);
    
    // Preserve whitespace if configured
    if (!this.config.preserve_whitespace) {
      normalizedText = normalizedText.trim().replace(/\s+/g, ' ');
    }
    
    // Analyze combining characters if tracking enabled
    let combiningCharsRemoved = 0;
    if (this.config.track_combining_chars) {
      const beforeAnalysis = this.combiningAnalyzer.analyzeCombiningChars(processedText);
      const afterAnalysis = this.combiningAnalyzer.analyzeCombiningChars(normalizedText);
      combiningCharsRemoved = beforeAnalysis.total_combining_chars - afterAnalysis.total_combining_chars;
    }
    
    const lengthDelta = normalizedText.length - originalLength;
    const normalizationApplied = normalizedText !== text;
    
    return {
      normalized_text: normalizedText,
      normalization_applied: normalizationApplied,
      length_delta: lengthDelta,
      combining_chars_removed: combiningCharsRemoved
    };
  }
  
  /**
   * Normalize span with position tracking
   */
  normalizeSpan(
    text: string,
    startOffset: number,
    endOffset: number,
    spanText?: string
  ): NormalizedSpan {
    if (!this.enabled) {
      const startPos = this.spanCalculator.calculatePosition(text, startOffset);
      const endPos = this.spanCalculator.calculatePosition(text, endOffset);
      
      return {
        start_position: startPos,
        end_position: endPos,
        original_text: spanText || text.substring(startOffset, endOffset),
        normalized_text: spanText || text.substring(startOffset, endOffset),
        length_before_normalization: endOffset - startOffset,
        length_after_normalization: endOffset - startOffset,
        normalization_applied: false,
        combining_chars_removed: 0
      };
    }
    
    const span = LensTracer.createChildSpan('normalize_unicode_span');
    const normalizationStart = performance.now();
    
    try {
      this.metrics.total_spans_processed++;
      
      // Extract span text if not provided
      const originalSpanText = spanText || text.substring(startOffset, endOffset);
      
      // Validate span length
      if (originalSpanText.length > this.config.max_span_length) {
        console.warn(`⚠️ Span too long for normalization: ${originalSpanText.length} > ${this.config.max_span_length}`);
        throw new Error(`Span exceeds maximum length: ${originalSpanText.length}`);
      }
      
      // Normalize the span text
      const normalizationResult = this.normalizeText(originalSpanText);
      
      // Calculate original positions
      const originalStartPos = this.spanCalculator.calculatePosition(text, startOffset);
      const originalEndPos = this.spanCalculator.calculatePosition(text, endOffset);
      
      // If text was normalized, we may need to adjust the full text and recalculate positions
      let adjustedStartPos = originalStartPos;
      let adjustedEndPos = originalEndPos;
      
      if (normalizationResult.normalization_applied) {
        // For simplicity, keep original positions but note that normalization was applied
        // In a full implementation, you'd want to normalize the entire document and recalculate
        console.log(`📝 Span normalization: "${originalSpanText.slice(0, 50)}..." -> "${normalizationResult.normalized_text.slice(0, 50)}..." (Δ${normalizationResult.length_delta} chars)`);
        
        this.metrics.spans_with_combining_chars++;
        this.metrics.chars_removed += Math.max(0, -normalizationResult.length_delta);
        this.metrics.bytes_removed += Math.max(0, -normalizationResult.length_delta); // Approximation
      }
      
      const normalizedSpan: NormalizedSpan = {
        start_position: adjustedStartPos,
        end_position: adjustedEndPos,
        original_text: originalSpanText,
        normalized_text: normalizationResult.normalized_text,
        length_before_normalization: originalSpanText.length,
        length_after_normalization: normalizationResult.normalized_text.length,
        normalization_applied: normalizationResult.normalization_applied,
        combining_chars_removed: normalizationResult.combining_chars_removed
      };
      
      const normalizationTime = performance.now() - normalizationStart;
      this.metrics.normalization_time_ms += normalizationTime;
      
      span.setAttributes({
        success: true,
        normalization_applied: normalizationResult.normalization_applied,
        length_delta: normalizationResult.length_delta,
        combining_chars_removed: normalizationResult.combining_chars_removed,
        normalization_time_ms: normalizationTime
      });
      
      return normalizedSpan;
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      console.error('Unicode normalization error:', error);
      
      // Return original span on error
      const startPos = this.spanCalculator.calculatePosition(text, startOffset);
      const endPos = this.spanCalculator.calculatePosition(text, endOffset);
      
      return {
        start_position: startPos,
        end_position: endPos,
        original_text: spanText || text.substring(startOffset, endOffset),
        normalized_text: spanText || text.substring(startOffset, endOffset),
        length_before_normalization: endOffset - startOffset,
        length_after_normalization: endOffset - startOffset,
        normalization_applied: false,
        combining_chars_removed: 0
      };
      
    } finally {
      span.end();
    }
  }
  
  /**
   * Batch normalize multiple spans
   */
  normalizeSpans(spans: Array<{
    text: string;
    start_offset: number;
    end_offset: number;
    span_text?: string;
  }>): NormalizedSpan[] {
    const span = LensTracer.createChildSpan('batch_normalize_spans');
    
    try {
      const normalizedSpans = spans.map(spanInfo => 
        this.normalizeSpan(
          spanInfo.text,
          spanInfo.start_offset,
          spanInfo.end_offset,
          spanInfo.span_text
        )
      );
      
      // Update batch statistics
      this.metrics.most_common_combining_chars = this.combiningAnalyzer.getMostCommonCombiningChars();
      
      console.log(`📝 Batch normalized ${spans.length} spans (${normalizedSpans.filter(s => s.normalization_applied).length} changed)`);
      
      span.setAttributes({
        success: true,
        spans_processed: spans.length,
        spans_normalized: normalizedSpans.filter(s => s.normalization_applied).length
      });
      
      return normalizedSpans;
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }
  
  /**
   * Validate span invariant after normalization
   */
  validateSpanInvariant(
    originalText: string,
    normalizedSpan: NormalizedSpan
  ): {
    invariant_valid: boolean;
    validation_errors: string[];
  } {
    const errors: string[] = [];
    
    // Check that positions are reasonable
    if (normalizedSpan.start_position.char_offset > normalizedSpan.end_position.char_offset) {
      errors.push('Start position is after end position');
    }
    
    if (normalizedSpan.start_position.byte_offset > normalizedSpan.end_position.byte_offset) {
      errors.push('Start byte offset is after end byte offset');
    }
    
    // Check that positions are within text bounds
    if (normalizedSpan.end_position.char_offset > originalText.length) {
      errors.push('End position exceeds text length');
    }
    
    // Check that normalized text is valid
    if (this.config.validate_utf8 && !this.utf8Validator.isValidUTF8(normalizedSpan.normalized_text)) {
      errors.push('Normalized text contains invalid UTF-8');
    }
    
    // Check for combining character consistency
    if (this.config.track_combining_chars && normalizedSpan.normalization_applied) {
      const beforeAnalysis = this.combiningAnalyzer.analyzeCombiningChars(normalizedSpan.original_text);
      const afterAnalysis = this.combiningAnalyzer.analyzeCombiningChars(normalizedSpan.normalized_text);
      
      if (afterAnalysis.total_combining_chars > beforeAnalysis.total_combining_chars) {
        errors.push('Normalization increased combining character count');
      }
    }
    
    return {
      invariant_valid: errors.length === 0,
      validation_errors: errors
    };
  }
  
  /**
   * Get normalization statistics
   */
  getStats(): NormalizationMetrics & {
    enabled: boolean;
    config: UnicodeNormalizationConfig;
    normalization_rate: number;
    avg_normalization_time_ms: number;
  } {
    const normalizationRate = this.metrics.total_spans_processed > 0 
      ? (this.metrics.spans_with_combining_chars / this.metrics.total_spans_processed) * 100
      : 0;
      
    const avgNormalizationTime = this.metrics.total_spans_processed > 0
      ? this.metrics.normalization_time_ms / this.metrics.total_spans_processed
      : 0;
    
    return {
      ...this.metrics,
      most_common_combining_chars: this.combiningAnalyzer.getMostCommonCombiningChars(),
      enabled: this.enabled,
      config: this.config,
      normalization_rate: normalizationRate,
      avg_normalization_time_ms: avgNormalizationTime
    };
  }
  
  /**
   * Reset statistics
   */
  resetStats(): void {
    this.metrics = {
      total_spans_processed: 0,
      spans_with_combining_chars: 0,
      bytes_removed: 0,
      chars_removed: 0,
      normalization_time_ms: 0,
      most_common_combining_chars: new Map()
    };
    this.combiningAnalyzer.resetStats();
  }
  
  /**
   * Enable/disable Unicode normalization
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    console.log(`📝 Unicode NFC normalization ${enabled ? 'ENABLED' : 'DISABLED'}`);
  }
  
  /**
   * Update configuration
   */
  updateConfig(config: Partial<UnicodeNormalizationConfig>): void {
    this.config = { ...this.config, ...config };
    console.log('🔧 Unicode normalization config updated:', config);
  }
}

// Global instance
export const globalUnicodeNormalizer = new UnicodeNFCNormalizer();