#!/usr/bin/env python3
"""
Sanity Pyramid: Contract-Based RAG Validation Framework

Validates the contract between query → evidence → expected operation → output
before any RAG tuning. Only optimizes where the contract is met.

Core Concepts:
- 5 primitive operations: Locate, Extract, Explain, Compose, Transform
- Evidence Sufficiency Score (ESS) with operation-specific thresholds
- Operation Feasibility Matrix to prevent invalid LLM tasks
- Always-on sanity tracer for continuous validation
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """5 primitive operations for code search/RAG."""
    LOCATE = "locate"      # Find path/line (file location)
    EXTRACT = "extract"    # Verbatim span extraction  
    EXPLAIN = "explain"    # Natural language over retrieved code
    COMPOSE = "compose"    # Join multiple spans/files
    TRANSFORM = "transform"  # Convert snippet to usage/example


@dataclass
class EvidenceSpan:
    """Evidence span with location and content metadata."""
    path: str
    char_start: int
    char_end: int
    content: str
    symbol_hits: List[str]
    contains_gold_span: bool
    key_tokens: Set[str]


@dataclass
class EvidenceMap:
    """Complete evidence mapping for a query."""
    query_id: str
    spans: List[EvidenceSpan]
    gold_paths: List[str]
    gold_spans: List[Tuple[str, int, int]]  # (path, start, end)
    answerable_at_k: float  # Gold file in top-k
    span_recall: float      # Fraction of gold tokens present
    key_token_hit: float    # Key tokens found in context
    context_budget_ok: bool # Within token limits with proper code/comment ratio
    

@dataclass
class OperationContract:
    """Contract definition for each operation type."""
    operation: OperationType
    ess_threshold: float    # Evidence Sufficiency Score threshold
    requires_gold_span: bool
    requires_file_level_gold: bool
    min_spans_required: int
    max_context_tokens: int
    min_code_ratio: float   # Minimum fraction of code vs commentary
    output_validators: List[str]  # List of validation functions


@dataclass
class SanityResult:
    """Result of sanity pyramid validation."""
    query_id: str
    operation: OperationType
    evidence_map: EvidenceMap
    ess_score: float
    contract_met: bool
    failure_reason: Optional[str]
    answerable: bool
    ready_for_generation: bool


class SanityPyramid:
    """Main sanity pyramid validator."""
    
    def __init__(self):
        self.operation_contracts = self._init_operation_contracts()
        self.query_classifier = QueryClassifier()
        self.evidence_analyzer = EvidenceAnalyzer()
        self.output_validator = OutputValidator()
        
    def _init_operation_contracts(self) -> Dict[OperationType, OperationContract]:
        """Initialize operation contracts with thresholds."""
        return {
            OperationType.LOCATE: OperationContract(
                operation=OperationType.LOCATE,
                ess_threshold=0.8,  # High threshold - must find the file
                requires_gold_span=False,
                requires_file_level_gold=True,
                min_spans_required=1,
                max_context_tokens=2048,
                min_code_ratio=0.7,
                output_validators=["validate_path_format", "validate_resolvable_location"]
            ),
            OperationType.EXTRACT: OperationContract(
                operation=OperationType.EXTRACT,
                ess_threshold=0.8,  # High threshold - must have exact span
                requires_gold_span=True,
                requires_file_level_gold=True,
                min_spans_required=1,
                max_context_tokens=1024,
                min_code_ratio=0.8,
                output_validators=["validate_substring_match", "validate_verbatim_extraction"]
            ),
            OperationType.EXPLAIN: OperationContract(
                operation=OperationType.EXPLAIN,
                ess_threshold=0.6,  # Lower threshold - can explain with context
                requires_gold_span=False,
                requires_file_level_gold=True,
                min_spans_required=1,
                max_context_tokens=4096,
                min_code_ratio=0.5,
                output_validators=["validate_citations", "validate_coherent_explanation"]
            ),
            OperationType.COMPOSE: OperationContract(
                operation=OperationType.COMPOSE,
                ess_threshold=0.7,  # Medium-high threshold - needs multiple sources
                requires_gold_span=False,
                requires_file_level_gold=True,
                min_spans_required=2,
                max_context_tokens=6144,
                min_code_ratio=0.6,
                output_validators=["validate_multi_file_citations", "validate_composition_coherence"]
            ),
            OperationType.TRANSFORM: OperationContract(
                operation=OperationType.TRANSFORM,
                ess_threshold=0.7,  # Medium-high threshold - needs good examples
                requires_gold_span=False,
                requires_file_level_gold=True,
                min_spans_required=1,
                max_context_tokens=3072,
                min_code_ratio=0.6,
                output_validators=["validate_transformation_accuracy", "validate_runnable_code"]
            )
        }
    
    def validate_query(self, query: str, query_id: str, retrieved_chunks: List[Dict], 
                      gold_data: Dict) -> SanityResult:
        """Main validation pipeline for a single query."""
        
        # Step 1: Classify operation type
        operation = self.query_classifier.classify(query)
        contract = self.operation_contracts[operation]
        
        # Step 2: Build evidence map
        evidence_map = self.evidence_analyzer.build_evidence_map(
            query_id, retrieved_chunks, gold_data
        )
        
        # Step 3: Calculate Evidence Sufficiency Score
        ess_score = self._calculate_ess(evidence_map)
        
        # Step 4: Check operation feasibility
        contract_met, failure_reason = self._check_operation_feasibility(
            evidence_map, contract, ess_score
        )
        
        # Step 5: Determine if ready for generation
        ready_for_generation = contract_met and ess_score >= contract.ess_threshold
        
        return SanityResult(
            query_id=query_id,
            operation=operation,
            evidence_map=evidence_map,
            ess_score=ess_score,
            contract_met=contract_met,
            failure_reason=failure_reason,
            answerable=evidence_map.answerable_at_k > 0,
            ready_for_generation=ready_for_generation
        )
    
    def _calculate_ess(self, evidence_map: EvidenceMap) -> float:
        """Calculate Evidence Sufficiency Score: 0.5·Answerable + 0.3·SpanRecall + 0.2·KeyTokenHit"""
        return (0.5 * evidence_map.answerable_at_k + 
                0.3 * evidence_map.span_recall + 
                0.2 * evidence_map.key_token_hit)
    
    def _check_operation_feasibility(self, evidence_map: EvidenceMap, 
                                   contract: OperationContract, 
                                   ess_score: float) -> Tuple[bool, Optional[str]]:
        """Check if operation is feasible given evidence and contract."""
        
        # Check ESS threshold
        if ess_score < contract.ess_threshold:
            return False, f"ESS score {ess_score:.3f} below threshold {contract.ess_threshold}"
        
        # Check span requirements
        if contract.requires_gold_span and evidence_map.span_recall == 0:
            return False, f"Operation {contract.operation.value} requires gold span but SpanRecall=0"
        
        # Check minimum spans
        if len(evidence_map.spans) < contract.min_spans_required:
            return False, f"Operation requires {contract.min_spans_required} spans, got {len(evidence_map.spans)}"
        
        # Check file-level gold requirement
        if contract.requires_file_level_gold and evidence_map.answerable_at_k == 0:
            return False, f"Operation requires file-level gold but Answerable@k=0"
        
        # Check context budget
        if not evidence_map.context_budget_ok:
            return False, "Context budget exceeded or code ratio too low"
        
        return True, None
    
    def validate_output(self, query_result: SanityResult, generated_output: str) -> Dict[str, Any]:
        """Validate generated output against operation contract."""
        if not query_result.ready_for_generation:
            return {"valid": False, "reason": "Query not ready for generation"}
        
        contract = self.operation_contracts[query_result.operation]
        
        # Run operation-specific validators
        validation_results = {}
        for validator_name in contract.output_validators:
            validator_func = getattr(self.output_validator, validator_name, None)
            if validator_func:
                validation_results[validator_name] = validator_func(
                    generated_output, query_result.evidence_map, query_result.operation
                )
        
        # Overall validation
        all_valid = all(validation_results.values())
        
        return {
            "valid": all_valid,
            "operation": query_result.operation.value,
            "individual_validations": validation_results,
            "ess_score": query_result.ess_score
        }


class QueryClassifier:
    """Classifies queries into operation types using ruleset."""
    
    def classify(self, query: str) -> OperationType:
        """Classify query into one of 5 operation types."""
        query_lower = query.lower()
        
        # Locate patterns: find, where is, locate
        if any(pattern in query_lower for pattern in [
            "find function", "find class", "find method", "where is", 
            "locate", "which file", "path to", "file containing"
        ]):
            return OperationType.LOCATE
        
        # Extract patterns: show, get code, extract
        if any(pattern in query_lower for pattern in [
            "show me the", "get the code", "extract", "what does",
            "full definition", "complete implementation"
        ]) and not any(explain_word in query_lower for explain_word in ["why", "how", "explain"]):
            return OperationType.EXTRACT
        
        # Explain patterns: why, how, explain
        if any(pattern in query_lower for pattern in [
            "why", "how", "explain", "what is the purpose", "what does this do",
            "how does", "can you explain"
        ]):
            return OperationType.EXPLAIN
        
        # Compose patterns: multiple files, combine, relationship
        if any(pattern in query_lower for pattern in [
            "how do", "work together", "relationship", "combine", "integrate",
            "across", "multiple", "both", "all files", "entire"
        ]):
            return OperationType.COMPOSE
        
        # Transform patterns: convert, example, usage, how to use
        if any(pattern in query_lower for pattern in [
            "how to use", "example", "convert", "transform", "usage",
            "implement", "create using", "write code"
        ]):
            return OperationType.TRANSFORM
        
        # Default: if contains code symbols or specific references, likely EXTRACT
        if re.search(r'\b[A-Z][a-zA-Z]*\b|\b[a-z_]+\([^)]*\)', query):
            return OperationType.EXTRACT
        
        # Final fallback: EXPLAIN (most general)
        return OperationType.EXPLAIN


class EvidenceAnalyzer:
    """Analyzes retrieved evidence and builds evidence maps."""
    
    def build_evidence_map(self, query_id: str, retrieved_chunks: List[Dict], 
                          gold_data: Dict) -> EvidenceMap:
        """Build complete evidence map for validation."""
        
        # Extract gold information
        gold_paths = gold_data.get("gold_paths", [])
        gold_spans = gold_data.get("gold_spans", [])
        
        # Process retrieved chunks into evidence spans
        spans = []
        total_context_tokens = 0
        code_tokens = 0
        
        for i, chunk in enumerate(retrieved_chunks):
            # Extract span information
            span = EvidenceSpan(
                path=chunk.get("file_path", f"chunk_{i}"),
                char_start=chunk.get("char_start", 0),
                char_end=chunk.get("char_end", len(chunk.get("content", ""))),
                content=chunk.get("content", ""),
                symbol_hits=self._extract_symbols(chunk.get("content", "")),
                contains_gold_span=self._check_gold_span_overlap(chunk, gold_spans),
                key_tokens=self._extract_key_tokens(chunk.get("content", ""))
            )
            spans.append(span)
            
            # Token counting for budget validation
            chunk_tokens = len(span.content.split())
            total_context_tokens += chunk_tokens
            
            # Estimate code vs comment ratio
            if self._is_code_content(span.content):
                code_tokens += chunk_tokens
        
        # Calculate metrics
        answerable_at_k = self._calculate_answerable_at_k(spans, gold_paths)
        span_recall = self._calculate_span_recall(spans, gold_spans)
        key_token_hit = self._calculate_key_token_hit(spans, gold_data.get("query", ""))
        
        # Context budget validation
        code_ratio = code_tokens / max(total_context_tokens, 1)
        context_budget_ok = total_context_tokens <= 4096 and code_ratio >= 0.5
        
        return EvidenceMap(
            query_id=query_id,
            spans=spans,
            gold_paths=gold_paths,
            gold_spans=gold_spans,
            answerable_at_k=answerable_at_k,
            span_recall=span_recall,
            key_token_hit=key_token_hit,
            context_budget_ok=context_budget_ok
        )
    
    def _extract_symbols(self, content: str) -> List[str]:
        """Extract programming symbols from content."""
        # Find function definitions, class names, method calls
        symbols = []
        
        # Python patterns
        symbols.extend(re.findall(r'def\s+(\w+)', content))
        symbols.extend(re.findall(r'class\s+(\w+)', content))
        symbols.extend(re.findall(r'(\w+)\s*\(', content))
        
        # General patterns
        symbols.extend(re.findall(r'\b[A-Z][a-zA-Z]*\b', content))  # CamelCase
        symbols.extend(re.findall(r'\b[a-z_][a-z0-9_]*\b', content))  # snake_case
        
        return list(set(symbols))
    
    def _check_gold_span_overlap(self, chunk: Dict, gold_spans: List[Tuple[str, int, int]]) -> bool:
        """Check if chunk overlaps with any gold span."""
        chunk_path = chunk.get("file_path", "")
        chunk_start = chunk.get("char_start", 0)
        chunk_end = chunk.get("char_end", 0)
        
        for gold_path, gold_start, gold_end in gold_spans:
            if (chunk_path == gold_path and 
                not (chunk_end <= gold_start or chunk_start >= gold_end)):
                return True
        return False
    
    def _extract_key_tokens(self, content: str) -> Set[str]:
        """Extract key tokens that might be important for the query."""
        # Simple tokenization - can be enhanced
        tokens = set(re.findall(r'\b\w+\b', content.lower()))
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        return tokens - stop_words
    
    def _is_code_content(self, content: str) -> bool:
        """Determine if content is primarily code vs comments/docs."""
        # Heuristic: check for code patterns
        code_indicators = [
            r'def\s+\w+',     # function definitions
            r'class\s+\w+',   # class definitions
            r'import\s+\w+',  # imports
            r'=\s*[^=]',      # assignments
            r'\w+\([^)]*\)',  # function calls
            r'{\s*\w+',       # object/dict literals
            r'if\s+\w+',      # conditionals
        ]
        
        code_matches = sum(len(re.findall(pattern, content)) for pattern in code_indicators)
        total_lines = len(content.split('\n'))
        
        return code_matches > total_lines * 0.3  # At least 30% code indicators
    
    def _calculate_answerable_at_k(self, spans: List[EvidenceSpan], gold_paths: List[str]) -> float:
        """Calculate if gold file is in top-k retrieved spans."""
        if not gold_paths:
            return 0.0
        
        retrieved_paths = {span.path for span in spans}
        gold_paths_set = set(gold_paths)
        
        return 1.0 if gold_paths_set & retrieved_paths else 0.0
    
    def _calculate_span_recall(self, spans: List[EvidenceSpan], gold_spans: List[Tuple[str, int, int]]) -> float:
        """Calculate fraction of gold tokens present in retrieved spans."""
        if not gold_spans:
            return 0.0
        
        # For simplicity, check if any span contains gold span
        gold_spans_found = sum(1 for span in spans if span.contains_gold_span)
        return min(1.0, gold_spans_found / len(gold_spans))
    
    def _calculate_key_token_hit(self, spans: List[EvidenceSpan], query: str) -> float:
        """Calculate fraction of query key tokens found in spans."""
        query_tokens = self._extract_key_tokens(query)
        if not query_tokens:
            return 0.0
        
        all_span_tokens = set()
        for span in spans:
            all_span_tokens.update(span.key_tokens)
        
        hits = len(query_tokens & all_span_tokens)
        return hits / len(query_tokens)


class OutputValidator:
    """Validates generated outputs against operation contracts."""
    
    def validate_path_format(self, output: str, evidence_map: EvidenceMap, 
                           operation: OperationType) -> bool:
        """Validate that output contains properly formatted paths."""
        # Look for path-like patterns: file.py:123 or /path/to/file.py
        path_patterns = [
            r'\b\w+\.py:\d+',           # file.py:123
            r'/[\w/]+\.py',             # /path/to/file.py
            r'\b[\w/]+\.py\b',          # relative/path/file.py
        ]
        
        return any(re.search(pattern, output) for pattern in path_patterns)
    
    def validate_resolvable_location(self, output: str, evidence_map: EvidenceMap, 
                                   operation: OperationType) -> bool:
        """Validate that mentioned locations exist in evidence."""
        # Extract mentioned paths from output
        mentioned_paths = re.findall(r'([\w/]+\.py)', output)
        evidence_paths = {span.path for span in evidence_map.spans}
        
        # At least one mentioned path should be in evidence
        return any(path in evidence_paths for path in mentioned_paths)
    
    def validate_substring_match(self, output: str, evidence_map: EvidenceMap, 
                                operation: OperationType) -> bool:
        """Validate that extracted content is substring of provided context."""
        # For extract operation, output should be verbatim from one of the spans
        for span in evidence_map.spans:
            if output.strip() in span.content:
                return True
        return False
    
    def validate_verbatim_extraction(self, output: str, evidence_map: EvidenceMap, 
                                   operation: OperationType) -> bool:
        """Validate that extraction is truly verbatim (no hallucination)."""
        # Similar to substring match but stricter
        output_clean = re.sub(r'\s+', ' ', output.strip())
        
        for span in evidence_map.spans:
            span_clean = re.sub(r'\s+', ' ', span.content.strip())
            if output_clean in span_clean:
                return True
        return False
    
    def validate_citations(self, output: str, evidence_map: EvidenceMap, 
                          operation: OperationType) -> bool:
        """Validate that output cites at least one retrieved path."""
        evidence_paths = {span.path for span in evidence_map.spans}
        
        # Look for any evidence path mentioned in output
        for path in evidence_paths:
            if path in output or Path(path).name in output:
                return True
        return False
    
    def validate_coherent_explanation(self, output: str, evidence_map: EvidenceMap, 
                                    operation: OperationType) -> bool:
        """Validate that explanation is coherent and substantial."""
        # Basic coherence checks
        if len(output.strip()) < 50:  # Too short
            return False
        
        # Should contain some technical terms from the evidence
        evidence_symbols = set()
        for span in evidence_map.spans:
            evidence_symbols.update(span.symbol_hits)
        
        output_lower = output.lower()
        symbol_mentions = sum(1 for symbol in evidence_symbols if symbol.lower() in output_lower)
        
        return symbol_mentions > 0  # At least mention some symbols from evidence
    
    def validate_multi_file_citations(self, output: str, evidence_map: EvidenceMap, 
                                     operation: OperationType) -> bool:
        """Validate that composition cites multiple distinct paths."""
        evidence_paths = {span.path for span in evidence_map.spans}
        cited_paths = 0
        
        for path in evidence_paths:
            if path in output or Path(path).name in output:
                cited_paths += 1
        
        return cited_paths >= 2  # Must cite at least 2 different files
    
    def validate_composition_coherence(self, output: str, evidence_map: EvidenceMap, 
                                     operation: OperationType) -> bool:
        """Validate that composition shows relationships between parts."""
        # Look for connecting words/phrases that indicate composition
        composition_indicators = [
            "together", "combined", "works with", "calls", "imports", 
            "inherits", "implements", "extends", "uses", "interacts"
        ]
        
        output_lower = output.lower()
        return any(indicator in output_lower for indicator in composition_indicators)
    
    def validate_transformation_accuracy(self, output: str, evidence_map: EvidenceMap, 
                                       operation: OperationType) -> bool:
        """Validate that transformation maintains semantic accuracy."""
        # Basic check: output should contain some code structure
        code_patterns = [
            r'def\s+\w+',     # function definitions
            r'class\s+\w+',   # class definitions
            r'\w+\([^)]*\)',  # function calls
            r'=\s*[^=]',      # assignments
        ]
        
        return any(re.search(pattern, output) for pattern in code_patterns)
    
    def validate_runnable_code(self, output: str, evidence_map: EvidenceMap, 
                              operation: OperationType) -> bool:
        """Validate that generated code appears syntactically correct."""
        # Basic syntax checks - can be enhanced with actual parsing
        
        # Check balanced parentheses
        paren_count = output.count('(') - output.count(')')
        bracket_count = output.count('[') - output.count(']')
        brace_count = output.count('{') - output.count('}')
        
        if paren_count != 0 or bracket_count != 0 or brace_count != 0:
            return False
        
        # Check for obvious syntax issues
        if re.search(r'def\s*\(\)', output):  # malformed function def
            return False
        
        return True


class SanityTracer:
    """Always-on instrumentation for sanity pyramid."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.trace_file = output_dir / "sanity_trace.jsonl"
    
    def log_validation(self, result: SanityResult):
        """Log validation result to trace file."""
        trace_entry = {
            "timestamp": "2025-09-13T00:00:00Z",  # Use actual timestamp
            "query_id": result.query_id,
            "operation": result.operation.value,
            "ess_score": result.ess_score,
            "contract_met": result.contract_met,
            "failure_reason": result.failure_reason,
            "answerable": result.answerable,
            "ready_for_generation": result.ready_for_generation,
            "evidence_stats": {
                "num_spans": len(result.evidence_map.spans),
                "answerable_at_k": result.evidence_map.answerable_at_k,
                "span_recall": result.evidence_map.span_recall,
                "key_token_hit": result.evidence_map.key_token_hit,
                "context_budget_ok": result.evidence_map.context_budget_ok
            }
        }
        
        with open(self.trace_file, 'a') as f:
            f.write(json.dumps(trace_entry) + '\n')
    
    def generate_sanity_report(self) -> Dict[str, Any]:
        """Generate summary report from trace data."""
        if not self.trace_file.exists():
            return {"error": "No trace data found"}
        
        entries = []
        with open(self.trace_file, 'r') as f:
            for line in f:
                entries.append(json.loads(line.strip()))
        
        if not entries:
            return {"error": "No trace entries found"}
        
        # Aggregate statistics by operation
        by_operation = {}
        for entry in entries:
            op = entry["operation"]
            if op not in by_operation:
                by_operation[op] = {
                    "total_queries": 0,
                    "contract_met": 0,
                    "ready_for_generation": 0,
                    "ess_scores": [],
                    "failure_reasons": []
                }
            
            stats = by_operation[op]
            stats["total_queries"] += 1
            if entry["contract_met"]:
                stats["contract_met"] += 1
            if entry["ready_for_generation"]:
                stats["ready_for_generation"] += 1
            stats["ess_scores"].append(entry["ess_score"])
            if entry["failure_reason"]:
                stats["failure_reasons"].append(entry["failure_reason"])
        
        # Calculate pass rates and statistics
        report = {
            "total_queries": len(entries),
            "by_operation": {}
        }
        
        for op, stats in by_operation.items():
            total = stats["total_queries"]
            report["by_operation"][op] = {
                "total_queries": total,
                "pass_rate": stats["contract_met"] / total if total > 0 else 0,
                "generation_ready_rate": stats["ready_for_generation"] / total if total > 0 else 0,
                "avg_ess_score": sum(stats["ess_scores"]) / len(stats["ess_scores"]) if stats["ess_scores"] else 0,
                "top_failure_reasons": self._get_top_failures(stats["failure_reasons"])
            }
        
        return report
    
    def _get_top_failures(self, failures: List[str]) -> List[Tuple[str, int]]:
        """Get top failure reasons with counts."""
        from collections import Counter
        counter = Counter(failures)
        return counter.most_common(5)


def run_sanity_pyramid_validation(queries_file: Path, results_dir: Path) -> Dict[str, Any]:
    """Run sanity pyramid validation on a set of queries."""
    
    pyramid = SanityPyramid()
    tracer = SanityTracer(results_dir)
    
    # Load queries (assume JSONL format)
    queries = []
    with open(queries_file, 'r') as f:
        for line in f:
            queries.append(json.loads(line.strip()))
    
    validation_results = []
    
    for query_data in queries:
        query_id = query_data["query_id"]
        query_text = query_data["query"]
        retrieved_chunks = query_data.get("retrieved_chunks", [])
        gold_data = query_data.get("gold_data", {})
        
        # Run validation
        result = pyramid.validate_query(query_text, query_id, retrieved_chunks, gold_data)
        
        # Log to tracer
        tracer.log_validation(result)
        
        validation_results.append(result)
    
    # Generate summary report
    summary_report = tracer.generate_sanity_report()
    
    # Save detailed results
    detailed_results_file = results_dir / "sanity_pyramid_results.json"
    with open(detailed_results_file, 'w') as f:
        json.dump([asdict(result) for result in validation_results], f, indent=2)
    
    # Save summary report
    summary_file = results_dir / "sanity_pyramid_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    return summary_report


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python sanity_pyramid.py <queries_file.jsonl> <results_dir>")
        sys.exit(1)
    
    queries_file = Path(sys.argv[1])
    results_dir = Path(sys.argv[2])
    results_dir.mkdir(exist_ok=True)
    
    report = run_sanity_pyramid_validation(queries_file, results_dir)
    
    print("🏗️  SANITY PYRAMID VALIDATION COMPLETE")
    print("=" * 50)
    print(f"Total queries processed: {report['total_queries']}")
    
    for op, stats in report["by_operation"].items():
        print(f"\n📊 Operation: {op.upper()}")
        print(f"  Pass rate: {stats['pass_rate']:.1%}")
        print(f"  Ready for generation: {stats['generation_ready_rate']:.1%}")
        print(f"  Average ESS score: {stats['avg_ess_score']:.3f}")
        if stats['top_failure_reasons']:
            print(f"  Top failures: {stats['top_failure_reasons'][0][0]}")