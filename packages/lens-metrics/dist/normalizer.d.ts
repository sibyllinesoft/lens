/**
 * Path and Data Normalization for Consistent Joins
 */
import { SearchResult, ExpectedSpan, ExpectedSymbol, ExpectedFile, MetricsConfig } from './types.js';
export declare class DataNormalizer {
    private config;
    constructor(config: MetricsConfig);
    /**
     * Normalize path to canonical form: repo-relative, NFC, / separators
     */
    normalizePath(repo: string, rawPath: string): string;
    /**
     * Generate snippet hash for robust joining when spans don't match
     */
    generateSnippetHash(snippet: string): string;
    /**
     * Normalize search result for consistent matching
     */
    normalizeSearchResult(result: SearchResult): SearchResult;
    /**
     * Create join keys for hierarchical matching
     */
    createJoinKeys(result: SearchResult): {
        spanKey: string | null;
        symbolKey: string | null;
        fileKey: string;
        snippetKey: string | null;
    };
    /**
     * Create expected keys from canonical query
     */
    createExpectedKeys(query: {
        expected_spans?: ExpectedSpan[];
        expected_symbols?: ExpectedSymbol[];
        expected_files: ExpectedFile[];
    }): {
        spanKeys: Set<string>;
        symbolKeys: Set<string>;
        fileKeys: Set<string>;
    };
}
