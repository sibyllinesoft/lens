"use strict";
/**
 * Layer 2: Symbol/AST Search Implementation
 * Tree-sitter parsing + Symbol indexing using LSP-like structures
 * Target: 3-10ms (Stage-B) - ctags/LSIF/tree-sitter patterns
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.SymbolSearchEngine = void 0;
const tracer_js_1 = require("../telemetry/tracer.js");
/**
 * Symbol/AST search engine implementing Layer 2 of the pipeline
 */
class SymbolSearchEngine {
    symbolIndex = new Map(); // symbol_name -> definitions
    referenceIndex = new Map(); // symbol_name -> references
    astIndex = new Map(); // file_path -> ast_nodes
    segmentStorage;
    // Language-specific patterns for symbol extraction
    languagePatterns = {
        typescript: {
            functionPattern: /(?:function|const|let|var)\s+(\w+)\s*(?:\(|=)/g,
            classPattern: /(?:class|interface|type)\s+(\w+)/g,
            variablePattern: /(?:const|let|var)\s+(\w+)/g,
            importPattern: /import\s+.*?from\s+['"]([^'"]*)['"]/g,
        },
        python: {
            functionPattern: /def\s+(\w+)\s*\(/g,
            classPattern: /class\s+(\w+)/g,
            variablePattern: /^\s*(\w+)\s*=/gm,
            importPattern: /(?:from\s+[\w.]+\s+)?import\s+([\w\s,]+)/g,
        },
        rust: {
            functionPattern: /fn\s+(\w+)\s*\(/g,
            classPattern: /(?:struct|enum|trait|impl)\s+(\w+)/g,
            variablePattern: /let\s+(?:mut\s+)?(\w+)/g,
            importPattern: /use\s+([\w:]+)/g,
        },
        go: {
            functionPattern: /func\s+(\w*\s*)?(\w+)\s*\(/g,
            classPattern: /type\s+(\w+)\s+(?:struct|interface)/g,
            variablePattern: /(?:var|:=)\s*(\w+)/g,
            importPattern: /import\s+(?:[\w\s]*\s+)?"([^"]+)"/g,
        },
        java: {
            functionPattern: /(?:public|private|protected|static)\s+[\w<>\[\]]+\s+(\w+)\s*\(/g,
            classPattern: /(?:public\s+)?(?:class|interface|enum)\s+(\w+)/g,
            variablePattern: /(?:public|private|protected|static)\s+[\w<>\[\]]+\s+(\w+)\s*[=;]/g,
            importPattern: /import\s+([\w.]+)/g,
        },
        bash: {
            functionPattern: /function\s+(\w+)|(\w+)\s*\(\s*\)/g,
            classPattern: /(?:)/g, // Bash doesn't have classes
            variablePattern: /(\w+)=/g,
            importPattern: /(?:source|\.)\s+([^\s]+)/g,
        },
    };
    constructor(segmentStorage) {
        this.segmentStorage = segmentStorage;
    }
    /**
     * Initialize the symbol search engine
     */
    async initialize() {
        const span = tracer_js_1.LensTracer.createChildSpan('symbol_engine_init');
        try {
            // Load existing symbol segments
            const segments = this.segmentStorage.listSegments();
            const symbolSegments = segments.filter(id => id.includes('symbols'));
            for (const segmentId of symbolSegments) {
                await this.loadSymbolSegment(segmentId);
            }
            span.setAttributes({
                success: true,
                segments_loaded: symbolSegments.length
            });
        }
        catch (error) {
            span.recordException(error);
            span.setAttributes({ success: false });
            throw error;
        }
        finally {
            span.end();
        }
    }
    /**
     * Index a file's symbols and AST
     */
    async indexFile(filePath, content, language) {
        const span = tracer_js_1.LensTracer.createChildSpan('index_file', {
            'file.path': filePath,
            'file.language': language,
            'file.size': content.length,
        });
        try {
            // Extract symbols using language-specific patterns
            const symbols = this.extractSymbols(content, language, filePath);
            const references = this.extractReferences(content, language, filePath);
            const astNodes = this.parseAST(content, language, filePath);
            // Store in indices
            symbols.forEach(symbol => {
                const existing = this.symbolIndex.get(symbol.name) || [];
                existing.push(symbol);
                this.symbolIndex.set(symbol.name, existing);
            });
            references.forEach(ref => {
                const existing = this.referenceIndex.get(ref.symbol_name) || [];
                existing.push(ref);
                this.referenceIndex.set(ref.symbol_name, existing);
            });
            this.astIndex.set(filePath, astNodes);
            span.setAttributes({
                success: true,
                symbols_found: symbols.length,
                references_found: references.length,
                ast_nodes: astNodes.length,
            });
        }
        catch (error) {
            span.recordException(error);
            span.setAttributes({ success: false });
            throw error;
        }
        finally {
            span.end();
        }
    }
    /**
     * Search for symbols matching a query
     */
    async searchSymbols(query, context, maxResults = 50) {
        const span = tracer_js_1.LensTracer.createChildSpan('search_symbols', {
            'search.query': query,
            'search.max_results': maxResults,
        });
        try {
            const candidates = [];
            const queryLower = query.toLowerCase();
            // Search symbol definitions
            for (const [symbolName, definitions] of this.symbolIndex) {
                if (symbolName.toLowerCase().includes(queryLower)) {
                    for (const def of definitions) {
                        candidates.push({
                            doc_id: `${def.file_path}:${def.line}:${def.col}`,
                            file_path: def.file_path,
                            line: def.line,
                            col: def.col,
                            score: this.calculateSymbolScore(symbolName, query, def.kind),
                            match_reasons: ['symbol'],
                            symbol_kind: def.kind,
                            ast_path: def.scope,
                            context: def.signature || `${def.kind} ${def.name}`,
                        });
                    }
                }
            }
            // Sort by relevance score
            candidates.sort((a, b) => b.score - a.score);
            const results = candidates.slice(0, maxResults);
            span.setAttributes({
                success: true,
                candidates_found: candidates.length,
                results_returned: results.length,
            });
            return results;
        }
        catch (error) {
            span.recordException(error);
            span.setAttributes({ success: false });
            throw error;
        }
        finally {
            span.end();
        }
    }
    /**
     * Find symbols near a specific location
     */
    async findSymbolsNear(filePath, line, radius = 10) {
        const span = tracer_js_1.LensTracer.createChildSpan('symbols_near', {
            'file.path': filePath,
            'location.line': line,
            'search.radius': radius,
        });
        try {
            const candidates = [];
            const astNodes = this.astIndex.get(filePath) || [];
            // Find AST nodes within radius
            const nearNodes = astNodes.filter(node => Math.abs(node.start_line - line) <= radius ||
                (node.start_line <= line && node.end_line >= line));
            for (const node of nearNodes) {
                // Find corresponding symbol definition
                const symbols = this.symbolIndex.get(node.text) || [];
                const symbol = symbols.find(s => s.file_path === filePath);
                if (symbol) {
                    candidates.push({
                        doc_id: `${filePath}:${node.start_line}:${node.start_col}`,
                        file_path: filePath,
                        line: node.start_line,
                        col: node.start_col,
                        score: 1.0 - Math.abs(node.start_line - line) / radius,
                        match_reasons: ['struct'],
                        symbol_kind: symbol.kind,
                        ast_path: node.parent_id || 'root',
                        context: node.text,
                    });
                }
            }
            // Sort by proximity and relevance
            candidates.sort((a, b) => b.score - a.score);
            span.setAttributes({
                success: true,
                ast_nodes_found: nearNodes.length,
                candidates_returned: candidates.length,
            });
            return candidates;
        }
        catch (error) {
            span.recordException(error);
            span.setAttributes({ success: false });
            throw error;
        }
        finally {
            span.end();
        }
    }
    /**
     * Extract symbols from code using regex patterns
     */
    extractSymbols(content, language, filePath) {
        const symbols = [];
        const patterns = this.languagePatterns[language];
        const lines = content.split('\n');
        // Extract functions
        const funcMatches = Array.from(content.matchAll(patterns.functionPattern));
        for (const match of funcMatches) {
            const name = match[1] || match[2]; // Handle different capture groups
            if (name) {
                const line = this.getLineNumber(content, match.index);
                symbols.push({
                    name,
                    kind: 'function',
                    file_path: filePath,
                    line,
                    col: match.index - content.lastIndexOf('\n', match.index) - 1,
                    scope: this.determineScope(lines, line),
                    signature: match[0],
                });
            }
        }
        // Extract classes/types
        const classMatches = Array.from(content.matchAll(patterns.classPattern));
        for (const match of classMatches) {
            const name = match[1];
            if (name) {
                const line = this.getLineNumber(content, match.index);
                symbols.push({
                    name,
                    kind: this.determineClassKind(match[0]),
                    file_path: filePath,
                    line,
                    col: match.index - content.lastIndexOf('\n', match.index) - 1,
                    scope: 'global',
                    signature: match[0],
                });
            }
        }
        // Extract variables
        const varMatches = Array.from(content.matchAll(patterns.variablePattern));
        for (const match of varMatches) {
            const name = match[1];
            if (name) {
                const line = this.getLineNumber(content, match.index);
                symbols.push({
                    name,
                    kind: 'variable',
                    file_path: filePath,
                    line,
                    col: match.index - content.lastIndexOf('\n', match.index) - 1,
                    scope: this.determineScope(lines, line),
                });
            }
        }
        return symbols;
    }
    /**
     * Extract symbol references from code
     */
    extractReferences(content, language, filePath) {
        const references = [];
        const lines = content.split('\n');
        // Simple reference extraction - could be enhanced with proper AST parsing
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const words = line.match(/\b[a-zA-Z_][a-zA-Z0-9_]*\b/g) || [];
            for (const word of words) {
                if (this.symbolIndex.has(word)) {
                    references.push({
                        symbol_name: word,
                        file_path: filePath,
                        line: i + 1,
                        col: line.indexOf(word),
                        context: line.trim(),
                    });
                }
            }
        }
        return references;
    }
    /**
     * Parse AST structure (simplified implementation)
     */
    parseAST(content, language, filePath) {
        const nodes = [];
        const lines = content.split('\n');
        // Simplified AST parsing - in a real implementation, use tree-sitter
        let nodeId = 0;
        const brackets = [];
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            // Track opening braces/brackets
            for (let j = 0; j < line.length; j++) {
                const char = line[j];
                if (char === '{' || char === '(' || char === '[') {
                    const id = `node_${nodeId++}`;
                    const parentId = brackets.length > 0 ? brackets[brackets.length - 1].id : undefined;
                    brackets.push({ line: i + 1, col: j, id, parentId });
                    nodes.push({
                        id,
                        type: char === '{' ? 'block' : char === '(' ? 'expression' : 'array',
                        file_path: filePath,
                        start_line: i + 1,
                        start_col: j,
                        end_line: i + 1, // Will be updated when closing bracket is found
                        end_col: j,
                        parent_id: parentId,
                        children_ids: [],
                        text: line.trim(),
                    });
                }
                else if (char === '}' || char === ')' || char === ']') {
                    const opening = brackets.pop();
                    if (opening) {
                        const node = nodes.find(n => n.id === opening.id);
                        if (node) {
                            node.end_line = i + 1;
                            node.end_col = j;
                        }
                    }
                }
            }
        }
        return nodes;
    }
    /**
     * Calculate relevance score for symbol matches
     */
    calculateSymbolScore(symbolName, query, kind) {
        const queryLower = query.toLowerCase();
        const nameLower = symbolName.toLowerCase();
        // Exact match gets highest score
        if (nameLower === queryLower) {
            return 1.0;
        }
        // Prefix match
        if (nameLower.startsWith(queryLower)) {
            return 0.9;
        }
        // Contains match
        if (nameLower.includes(queryLower)) {
            return 0.7;
        }
        // Kind-based scoring
        const kindBonus = this.getKindBonus(kind);
        return 0.5 + kindBonus;
    }
    /**
     * Get scoring bonus based on symbol kind
     */
    getKindBonus(kind) {
        const bonuses = {
            function: 0.3,
            class: 0.25,
            interface: 0.2,
            type: 0.2,
            method: 0.15,
            variable: 0.1,
            property: 0.1,
            constant: 0.05,
            enum: 0.05,
        };
        return bonuses[kind] || 0;
    }
    /**
     * Determine the kind of class-like symbol
     */
    determineClassKind(match) {
        if (match.includes('interface'))
            return 'interface';
        if (match.includes('type'))
            return 'type';
        if (match.includes('enum'))
            return 'enum';
        return 'class';
    }
    /**
     * Determine the scope of a symbol based on indentation
     */
    determineScope(lines, lineIndex) {
        const line = lines[lineIndex - 1]; // lineIndex is 1-based
        const indent = line.match(/^(\s*)/)?.[1]?.length || 0;
        if (indent === 0)
            return 'global';
        // Look backwards for enclosing scope
        for (let i = lineIndex - 2; i >= 0; i--) {
            const prevLine = lines[i];
            const prevIndent = prevLine.match(/^(\s*)/)?.[1]?.length || 0;
            if (prevIndent < indent && prevLine.trim()) {
                const match = prevLine.match(/(?:function|class|interface|namespace)\s+(\w+)/);
                if (match) {
                    return match[1];
                }
            }
        }
        return 'local';
    }
    /**
     * Get line number from character index
     */
    getLineNumber(content, index) {
        return content.substring(0, index).split('\n').length;
    }
    /**
     * Load symbols from a segment file
     */
    async loadSymbolSegment(segmentId) {
        const span = tracer_js_1.LensTracer.createChildSpan('load_symbol_segment', {
            'segment.id': segmentId,
        });
        try {
            const segment = await this.segmentStorage.openSegment(segmentId, true);
            const data = await this.segmentStorage.readFromSegment(segmentId, 0, segment.size);
            // Parse symbol data (simplified - would be binary format in production)
            const symbolData = JSON.parse(data.toString('utf8'));
            if (symbolData.symbols) {
                for (const [name, definitions] of Object.entries(symbolData.symbols)) {
                    this.symbolIndex.set(name, definitions);
                }
            }
            if (symbolData.references) {
                for (const [name, refs] of Object.entries(symbolData.references)) {
                    this.referenceIndex.set(name, refs);
                }
            }
            span.setAttributes({ success: true });
        }
        catch (error) {
            span.recordException(error);
            span.setAttributes({ success: false });
            throw error;
        }
        finally {
            span.end();
        }
    }
    /**
     * Get symbol statistics
     */
    getStats() {
        const symbolCount = Array.from(this.symbolIndex.values()).reduce((sum, defs) => sum + defs.length, 0);
        const referenceCount = Array.from(this.referenceIndex.values()).reduce((sum, refs) => sum + refs.length, 0);
        const astNodeCount = Array.from(this.astIndex.values()).reduce((sum, nodes) => sum + nodes.length, 0);
        return {
            symbols: symbolCount,
            references: referenceCount,
            ast_nodes: astNodeCount,
        };
    }
    /**
     * Cleanup resources
     */
    async shutdown() {
        this.symbolIndex.clear();
        this.referenceIndex.clear();
        this.astIndex.clear();
        console.log('Symbol search engine shut down');
    }
}
exports.SymbolSearchEngine = SymbolSearchEngine;
