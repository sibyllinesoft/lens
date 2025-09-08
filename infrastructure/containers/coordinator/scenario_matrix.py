#!/usr/bin/env python3
"""
Scenario Matrix Generator - Protocol v2.0 Benchmark Scenarios
Generates comprehensive test scenarios across all competitor systems
"""

import json
import logging
import uuid
from typing import Dict, List, Any
import itertools

logger = logging.getLogger(__name__)

class ScenarioMatrix:
    """
    Generates the complete benchmark scenario matrix as defined in TODO.md:
    {Regex, Substring, Symbol, Structural-pattern, NL→Span, Cross-repo, Time-travel, Clone-heavy, Noisy/bloat}
    """
    
    def __init__(self):
        self.datasets = ["coir", "swebench", "codesearchnet", "cosqa"]
        self.languages = ["python", "javascript", "java", "go", "typescript", "rust"]
        
        logger.info("Scenario matrix generator initialized")
    
    def generate_scenarios(self) -> List[Dict[str, Any]]:
        """Generate comprehensive scenario matrix"""
        logger.info("Generating Protocol v2.0 scenario matrix...")
        
        scenarios = []
        
        # Generate scenarios for each scenario type
        scenarios.extend(self._generate_regex_scenarios())
        scenarios.extend(self._generate_substring_scenarios())
        scenarios.extend(self._generate_symbol_scenarios())
        scenarios.extend(self._generate_structural_scenarios())
        scenarios.extend(self._generate_nl_span_scenarios())
        scenarios.extend(self._generate_cross_repo_scenarios())
        scenarios.extend(self._generate_time_travel_scenarios())
        scenarios.extend(self._generate_clone_heavy_scenarios())
        scenarios.extend(self._generate_noisy_bloat_scenarios())
        
        logger.info(f"Generated {len(scenarios)} total scenarios")
        return scenarios
    
    def _generate_regex_scenarios(self) -> List[Dict[str, Any]]:
        """Generate regex search scenarios"""
        logger.info("Generating regex scenarios...")
        
        scenarios = []
        
        regex_patterns = [
            # Function definitions
            r"def\s+\w+\s*\(",
            r"function\s+\w+\s*\(",
            r"async\s+def\s+\w+",
            # Variable assignments
            r"\w+\s*=\s*\[.*\]",
            r"const\s+\w+\s*=",
            # Import patterns
            r"import\s+\w+",
            r"from\s+\w+\s+import",
            # Class patterns
            r"class\s+\w+",
            r"interface\s+\w+",
            # Error patterns
            r"except\s+\w+:",
            r"catch\s*\(",
            # API patterns
            r"@\w+",
            r"\.get\s*\(",
            r"\.post\s*\("
        ]
        
        for i, pattern in enumerate(regex_patterns):
            for dataset in self.datasets:
                for language in self.languages:
                    scenario = {
                        "scenario_id": f"regex_{i+1}_{dataset}_{language}",
                        "scenario_type": "Regex",
                        "suite": "Protocol_v2.0",
                        "query": pattern,
                        "corpus": dataset,
                        "language": language,
                        "expected_complexity": "medium",
                        "query_class": "regex_pattern",
                        "description": f"Regex search for pattern: {pattern}"
                    }
                    scenarios.append(scenario)
        
        return scenarios
    
    def _generate_substring_scenarios(self) -> List[Dict[str, Any]]:
        """Generate substring search scenarios"""
        logger.info("Generating substring scenarios...")
        
        scenarios = []
        
        substring_queries = [
            "addEventListener",
            "querySelectorAll", 
            "fetch",
            "async/await",
            "try/catch",
            "console.log",
            "JSON.parse",
            "JSON.stringify",
            "map",
            "filter",
            "reduce",
            "forEach",
            "__init__",
            "self.",
            "import numpy",
            "pandas",
            "matplotlib",
            "requests.get",
            "requests.post",
            "def main(",
            "if __name__",
            "print(",
            "len(",
            "str(",
            "int("
        ]
        
        for i, query in enumerate(substring_queries):
            for dataset in self.datasets:
                for language in self.languages:
                    scenario = {
                        "scenario_id": f"substring_{i+1}_{dataset}_{language}",
                        "scenario_type": "Substring",
                        "suite": "Protocol_v2.0",
                        "query": query,
                        "corpus": dataset,
                        "language": language,
                        "expected_complexity": "low",
                        "query_class": "literal_string",
                        "description": f"Substring search for: {query}"
                    }
                    scenarios.append(scenario)
        
        return scenarios
    
    def _generate_symbol_scenarios(self) -> List[Dict[str, Any]]:
        """Generate symbol-based search scenarios"""
        logger.info("Generating symbol scenarios...")
        
        scenarios = []
        
        symbol_queries = [
            # Function symbols
            "fetchUserData",
            "processPayment", 
            "validateEmail",
            "parseJSON",
            "createConnection",
            "handleError",
            "generateHash",
            "encryptData",
            "formatDate",
            "calculateTotal",
            # Class symbols
            "UserManager",
            "DatabaseConnection",
            "PaymentProcessor", 
            "EmailValidator",
            "ConfigParser",
            "LoggerService",
            "CacheManager",
            "ApiClient",
            "DataProcessor",
            "EventHandler"
        ]
        
        for i, symbol in enumerate(symbol_queries):
            for dataset in self.datasets:
                for language in self.languages:
                    scenario = {
                        "scenario_id": f"symbol_{i+1}_{dataset}_{language}",
                        "scenario_type": "Symbol",
                        "suite": "Protocol_v2.0",
                        "query": symbol,
                        "corpus": dataset,
                        "language": language,
                        "expected_complexity": "medium",
                        "query_class": "symbol_identifier",
                        "description": f"Symbol search for identifier: {symbol}"
                    }
                    scenarios.append(scenario)
        
        return scenarios
    
    def _generate_structural_scenarios(self) -> List[Dict[str, Any]]:
        """Generate structural pattern search scenarios"""
        logger.info("Generating structural scenarios...")
        
        scenarios = []
        
        structural_patterns = [
            # Control flow patterns
            "if (...) { ... } else { ... }",
            "for (...) { ... }",
            "while (...) { ... }",
            "try { ... } catch (...) { ... }",
            # Function patterns
            "function $NAME($ARGS) { $BODY }",
            "def $NAME($ARGS): $BODY",
            "async function $NAME($ARGS) { $BODY }",
            # Class patterns
            "class $NAME extends $BASE { $BODY }",
            "class $NAME: $BODY",
            # Object patterns
            "{ $KEY: $VALUE }",
            "[$ITEMS]",
            # Import patterns
            "import $MODULE from '$PATH'",
            "from $MODULE import $ITEMS",
            # React patterns
            "<$TAG $PROPS>$CHILDREN</$TAG>",
            "useState($INITIAL)",
            "useEffect(() => { $EFFECT }, [$DEPS])"
        ]
        
        for i, pattern in enumerate(structural_patterns):
            for dataset in self.datasets:
                for language in self.languages:
                    scenario = {
                        "scenario_id": f"structural_{i+1}_{dataset}_{language}",
                        "scenario_type": "Structural-pattern",
                        "suite": "Protocol_v2.0", 
                        "query": pattern,
                        "corpus": dataset,
                        "language": language,
                        "expected_complexity": "high",
                        "query_class": "structural_template",
                        "description": f"Structural pattern search: {pattern}"
                    }
                    scenarios.append(scenario)
        
        return scenarios
    
    def _generate_nl_span_scenarios(self) -> List[Dict[str, Any]]:
        """Generate natural language to code span scenarios"""
        logger.info("Generating NL→Span scenarios...")
        
        scenarios = []
        
        nl_queries = [
            # Function behavior descriptions
            "function that validates email addresses",
            "method that processes credit card payments",
            "function to hash passwords securely",
            "code that connects to database",
            "function that sends HTTP requests",
            "method to parse JSON data",
            "code that handles file uploads",
            "function that generates random numbers",
            "method to format dates",
            "code that validates input forms",
            # Algorithm descriptions
            "binary search implementation",
            "sort array in ascending order",
            "find duplicates in list",
            "calculate Fibonacci sequence",
            "implement breadth-first search",
            "merge two sorted arrays",
            "find maximum element in array",
            "reverse a linked list",
            "implement quick sort algorithm",
            "check if string is palindrome",
            # API descriptions
            "REST API endpoint for user authentication",
            "GraphQL resolver for fetching posts",
            "middleware for request logging",
            "endpoint that returns user profile",
            "API route for file download",
            "webhook handler for payments",
            "controller for user registration",
            "service for sending emails",
            "handler for real-time messages",
            "API for search functionality"
        ]
        
        for i, query in enumerate(nl_queries):
            for dataset in self.datasets:
                for language in self.languages:
                    scenario = {
                        "scenario_id": f"nl_span_{i+1}_{dataset}_{language}",
                        "scenario_type": "NL→Span",
                        "suite": "Protocol_v2.0",
                        "query": query,
                        "corpus": dataset,
                        "language": language,
                        "expected_complexity": "high",
                        "query_class": "natural_language",
                        "description": f"Natural language query: {query}"
                    }
                    scenarios.append(scenario)
        
        return scenarios
    
    def _generate_cross_repo_scenarios(self) -> List[Dict[str, Any]]:
        """Generate cross-repository search scenarios"""
        logger.info("Generating cross-repo scenarios...")
        
        scenarios = []
        
        cross_repo_queries = [
            "shared utility functions",
            "common configuration patterns",
            "similar error handling code",
            "repeated API patterns",
            "duplicate validation logic",
            "common database queries",
            "shared styling patterns",
            "similar test patterns",
            "repeated authentication code",
            "common logging patterns"
        ]
        
        # Cross-repo scenarios test across multiple corpora
        for i, query in enumerate(cross_repo_queries):
            for lang in self.languages:
                scenario = {
                    "scenario_id": f"cross_repo_{i+1}_{lang}",
                    "scenario_type": "Cross-repo",
                    "suite": "Protocol_v2.0",
                    "query": query,
                    "corpus": "all",  # Search across all datasets
                    "language": lang,
                    "expected_complexity": "very_high",
                    "query_class": "cross_repository",
                    "description": f"Cross-repo search: {query}"
                }
                scenarios.append(scenario)
        
        return scenarios
    
    def _generate_time_travel_scenarios(self) -> List[Dict[str, Any]]:
        """Generate time-travel/version-aware scenarios"""
        logger.info("Generating time-travel scenarios...")
        
        scenarios = []
        
        time_travel_queries = [
            "functions that were deprecated",
            "code that was refactored recently",
            "methods that changed signatures",
            "APIs that were removed",
            "functions with performance improvements",
            "code that fixed security issues",
            "methods that added error handling",
            "APIs that changed return types",
            "code that was optimized",
            "functions that added validation"
        ]
        
        for i, query in enumerate(time_travel_queries):
            # Time-travel scenarios primarily use SWE-bench (has version history)
            for language in self.languages:
                scenario = {
                    "scenario_id": f"time_travel_{i+1}_{language}",
                    "scenario_type": "Time-travel",
                    "suite": "Protocol_v2.0",
                    "query": query,
                    "corpus": "swebench",  # SWE-bench has temporal data
                    "language": language,
                    "expected_complexity": "very_high",
                    "query_class": "temporal_evolution",
                    "description": f"Time-travel query: {query}"
                }
                scenarios.append(scenario)
        
        return scenarios
    
    def _generate_clone_heavy_scenarios(self) -> List[Dict[str, Any]]:
        """Generate scenarios for clone-heavy codebases"""
        logger.info("Generating clone-heavy scenarios...")
        
        scenarios = []
        
        clone_queries = [
            "similar function implementations",
            "repeated code blocks",
            "duplicate class definitions",
            "copied utility functions",
            "similar error messages",
            "repeated validation patterns",
            "duplicate API endpoints",
            "similar database queries",
            "repeated test patterns",
            "copied configuration code"
        ]
        
        for i, query in enumerate(clone_queries):
            for dataset in self.datasets:
                for language in self.languages:
                    scenario = {
                        "scenario_id": f"clone_heavy_{i+1}_{dataset}_{language}",
                        "scenario_type": "Clone-heavy",
                        "suite": "Protocol_v2.0",
                        "query": query,
                        "corpus": dataset,
                        "language": language,
                        "expected_complexity": "high",
                        "query_class": "similarity_detection",
                        "description": f"Clone detection query: {query}"
                    }
                    scenarios.append(scenario)
        
        return scenarios
    
    def _generate_noisy_bloat_scenarios(self) -> List[Dict[str, Any]]:
        """Generate scenarios for noisy/bloated codebases"""
        logger.info("Generating noisy/bloat scenarios...")
        
        scenarios = []
        
        noise_queries = [
            "actual implementation among generated code",
            "meaningful functions in test files", 
            "core logic in large files",
            "important methods with many parameters",
            "key algorithms in complex modules",
            "essential APIs among deprecated ones",
            "critical functions with long names",
            "main logic in files with many imports",
            "core features among utility functions",
            "primary methods in classes with many fields"
        ]
        
        for i, query in enumerate(noise_queries):
            for dataset in self.datasets:
                for language in self.languages:
                    scenario = {
                        "scenario_id": f"noisy_bloat_{i+1}_{dataset}_{language}",
                        "scenario_type": "Noisy/bloat",
                        "suite": "Protocol_v2.0",
                        "query": query,
                        "corpus": dataset,
                        "language": language,
                        "expected_complexity": "very_high",
                        "query_class": "signal_extraction",
                        "description": f"Noise-resistant query: {query}"
                    }
                    scenarios.append(scenario)
        
        return scenarios
    
    def get_warmup_queries(self, count: int = 5) -> List[str]:
        """Get simple warmup queries for system initialization"""
        warmup_queries = [
            "function",
            "import",
            "class",
            "return",
            "const"
        ]
        return warmup_queries[:count]
    
    def filter_scenarios_by_system(self, scenarios: List[Dict[str, Any]], 
                                   system_name: str) -> List[Dict[str, Any]]:
        """Filter scenarios appropriate for a specific system"""
        
        # System-specific filtering logic
        if system_name in ["ripgrep", "livegrep"]:
            # Regex-focused systems work best with regex and substring queries
            return [s for s in scenarios if s["scenario_type"] in ["Regex", "Substring"]]
            
        elif system_name in ["comby", "ast-grep"]:
            # Structural search systems work best with structural patterns
            return [s for s in scenarios if s["scenario_type"] in ["Structural-pattern", "Symbol"]]
            
        elif system_name in ["opensearch", "qdrant", "faiss"]:
            # Vector-based systems work best with semantic queries
            return [s for s in scenarios if s["scenario_type"] in ["NL→Span", "Symbol"]]
            
        elif system_name == "zoekt":
            # Code search engine works well with most query types
            return [s for s in scenarios if s["scenario_type"] in ["Regex", "Substring", "Symbol"]]
            
        else:
            # Lens and unknown systems get all scenarios
            return scenarios
    
    def get_scenario_statistics(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the scenario matrix"""
        
        stats = {
            "total_scenarios": len(scenarios),
            "by_type": {},
            "by_corpus": {},
            "by_language": {},
            "by_complexity": {}
        }
        
        for scenario in scenarios:
            # Count by type
            scenario_type = scenario["scenario_type"]
            stats["by_type"][scenario_type] = stats["by_type"].get(scenario_type, 0) + 1
            
            # Count by corpus
            corpus = scenario["corpus"]
            stats["by_corpus"][corpus] = stats["by_corpus"].get(corpus, 0) + 1
            
            # Count by language
            language = scenario["language"]
            stats["by_language"][language] = stats["by_language"].get(language, 0) + 1
            
            # Count by complexity
            complexity = scenario["expected_complexity"]
            stats["by_complexity"][complexity] = stats["by_complexity"].get(complexity, 0) + 1
        
        return stats