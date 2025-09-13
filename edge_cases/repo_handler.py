#!/usr/bin/env python3
"""
Edge Case Handling for Repository Weirdness and Polyglot Symbol Coverage
Handles symlinks, vendored subtrees, generated code, CRLF repos, mixed encodings,
huge files, and Windows-normalized repo testing with comprehensive symbol coverage.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import chardet
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, List, Optional, Any, Set, Tuple, Union
import hashlib
import mimetypes
import tempfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RepoWeirdness(Enum):
    """Types of repository weirdness."""
    SYMLINKS = "symlinks"
    VENDORED_SUBTREES = "vendored_subtrees"
    GENERATED_CODE = "generated_code"
    CRLF_ENDINGS = "crlf_endings"
    MIXED_ENCODINGS = "mixed_encodings"
    HUGE_FILES = "huge_files"
    BINARY_FILES = "binary_files"
    CIRCULAR_SYMLINKS = "circular_symlinks"
    DEEP_NESTING = "deep_nesting"
    UNICODE_NAMES = "unicode_names"
    CASE_SENSITIVITY = "case_sensitivity"


class SymbolLanguage(Enum):
    """Supported languages for symbol extraction."""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    GO = "go"
    JAVA = "java"
    RUST = "rust"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    PHP = "php"
    RUBY = "ruby"
    KOTLIN = "kotlin"
    SWIFT = "swift"


@dataclass
class WeirdnessDetection:
    """Detection result for repository weirdness."""
    weirdness_type: RepoWeirdness
    severity: str  # "info", "warning", "critical"
    count: int
    examples: List[str]
    description: str
    impact: str
    mitigation: str


@dataclass
class SymbolCoverage:
    """Symbol coverage analysis for a language."""
    language: SymbolLanguage
    total_files: int
    analyzed_files: int
    coverage_percentage: float
    symbol_count: int
    symbol_types: Dict[str, int]  # function, class, variable, etc.
    hit_rate: float  # Percentage of symbols successfully extracted
    extraction_errors: List[str]


@dataclass
class FileAnalysis:
    """Analysis result for a single file."""
    path: str
    relative_path: str
    size_bytes: int
    encoding: str
    line_endings: str  # "LF", "CRLF", "CR", "mixed"
    language: Optional[SymbolLanguage]
    is_binary: bool
    is_generated: bool
    is_vendored: bool
    weirdness_flags: List[RepoWeirdness]
    symbols_extracted: int
    extraction_success: bool
    processing_time_ms: float


class RepoHandler:
    """Handles repository analysis and edge case processing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Limits and thresholds
        self.max_file_size_mb = config.get("max_file_size_mb", 50)
        self.max_directory_depth = config.get("max_directory_depth", 20)
        self.max_symlink_depth = config.get("max_symlink_depth", 5)
        
        # Language file extensions
        self.language_extensions = {
            SymbolLanguage.PYTHON: [".py", ".pyx", ".pyi"],
            SymbolLanguage.TYPESCRIPT: [".ts", ".tsx"],
            SymbolLanguage.JAVASCRIPT: [".js", ".jsx", ".mjs"],
            SymbolLanguage.GO: [".go"],
            SymbolLanguage.JAVA: [".java"],
            SymbolLanguage.RUST: [".rs"],
            SymbolLanguage.CPP: [".cpp", ".cxx", ".cc", ".hpp", ".hxx", ".h"],
            SymbolLanguage.C: [".c", ".h"],
            SymbolLanguage.CSHARP: [".cs"],
            SymbolLanguage.PHP: [".php"],
            SymbolLanguage.RUBY: [".rb"],
            SymbolLanguage.KOTLIN: [".kt", ".kts"],
            SymbolLanguage.SWIFT: [".swift"]
        }
        
        # Generated code patterns
        self.generated_patterns = [
            r".*\.generated\.",
            r".*_pb2\.py$",
            r".*\.proto\.py$",
            r".*node_modules.*",
            r".*__pycache__.*",
            r".*\.pyc$",
            r".*dist/.*",
            r".*build/.*",
            r".*target/.*",
            r".*\.min\.js$",
            r".*\.bundle\.js$",
            r".*coverage/.*",
            r".*\.git/.*"
        ]
        
        # Vendored code patterns
        self.vendored_patterns = [
            r".*vendor/.*",
            r".*third_party/.*",
            r".*external/.*",
            r".*deps/.*",
            r".*lib/.*",
            r".*packages/.*",
            r".*node_modules/.*"
        ]
        
        # Symbol extractors
        self.symbol_extractors = self._initialize_symbol_extractors()
        
        # Analysis cache
        self.analysis_cache: Dict[str, FileAnalysis] = {}
        
        logger.info("RepoHandler initialized")
    
    def _initialize_symbol_extractors(self) -> Dict[SymbolLanguage, object]:
        """Initialize symbol extractors for each language."""
        extractors = {}
        
        # Try to initialize tree-sitter and ctags extractors
        try:
            extractors[SymbolLanguage.PYTHON] = PythonSymbolExtractor()
            extractors[SymbolLanguage.TYPESCRIPT] = TypeScriptSymbolExtractor()
            extractors[SymbolLanguage.JAVASCRIPT] = JavaScriptSymbolExtractor()
            extractors[SymbolLanguage.GO] = GoSymbolExtractor()
            extractors[SymbolLanguage.JAVA] = JavaSymbolExtractor()
            extractors[SymbolLanguage.RUST] = RustSymbolExtractor()
        except Exception as e:
            logger.warning(f"Failed to initialize some symbol extractors: {e}")
        
        return extractors
    
    async def analyze_repository(self, repo_path: Path) -> Dict[str, Any]:
        """Comprehensive repository analysis."""
        logger.info(f"Starting repository analysis: {repo_path}")
        
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        analysis_start = datetime.utcnow()
        
        # Initialize analysis state
        file_analyses: List[FileAnalysis] = []
        weirdness_detections: List[WeirdnessDetection] = []
        symbol_coverage: Dict[SymbolLanguage, SymbolCoverage] = {}
        
        # Scan repository files
        logger.info("Scanning repository files...")
        scanned_files = await self._scan_repository_files(repo_path)
        
        # Detect repository weirdness
        logger.info("Detecting repository weirdness...")
        weirdness_detections = await self._detect_repository_weirdness(
            repo_path, scanned_files
        )
        
        # Analyze individual files
        logger.info(f"Analyzing {len(scanned_files)} files...")
        for file_path in scanned_files:
            try:
                file_analysis = await self._analyze_file(repo_path, file_path)
                if file_analysis:
                    file_analyses.append(file_analysis)
            except Exception as e:
                logger.warning(f"Failed to analyze file {file_path}: {e}")
        
        # Calculate symbol coverage
        logger.info("Calculating symbol coverage...")
        symbol_coverage = self._calculate_symbol_coverage(file_analyses)
        
        # Generate Windows normalization test data
        windows_test_data = await self._generate_windows_test_data(repo_path, file_analyses)
        
        analysis_end = datetime.utcnow()
        processing_time = (analysis_end - analysis_start).total_seconds()
        
        # Build comprehensive analysis result
        result = {
            "timestamp": analysis_end.isoformat(),
            "repository_path": str(repo_path),
            "processing_time_seconds": processing_time,
            "summary": {
                "total_files": len(scanned_files),
                "analyzed_files": len(file_analyses),
                "weirdness_count": len(weirdness_detections),
                "languages_detected": len(symbol_coverage)
            },
            "weirdness_detections": [asdict(w) for w in weirdness_detections],
            "symbol_coverage": {
                lang.value: asdict(coverage) 
                for lang, coverage in symbol_coverage.items()
            },
            "file_analyses": [asdict(f) for f in file_analyses],
            "windows_test_data": windows_test_data,
            "recommendations": self._generate_recommendations(
                weirdness_detections, symbol_coverage
            )
        }
        
        logger.info(f"Repository analysis complete: {processing_time:.2f}s")
        return result
    
    async def _scan_repository_files(self, repo_path: Path) -> List[Path]:
        """Scan repository for files, handling edge cases."""
        files = []
        visited_paths = set()
        
        def _safe_walk(path: Path, depth: int = 0) -> None:
            if depth > self.max_directory_depth:
                logger.warning(f"Max directory depth exceeded: {path}")
                return
            
            # Resolve symlinks and check for cycles
            try:
                resolved_path = path.resolve()
                if resolved_path in visited_paths:
                    logger.warning(f"Circular reference detected: {path}")
                    return
                visited_paths.add(resolved_path)
            except (OSError, RuntimeError) as e:
                logger.warning(f"Path resolution failed: {path} - {e}")
                return
            
            try:
                for item in path.iterdir():
                    if item.is_file():
                        files.append(item)
                    elif item.is_dir() and not item.is_symlink():
                        _safe_walk(item, depth + 1)
                    elif item.is_symlink():
                        # Handle symlinks carefully
                        try:
                            if item.is_dir():
                                _safe_walk(item, depth + 1)
                            else:
                                files.append(item)
                        except (OSError, RuntimeError) as e:
                            logger.warning(f"Symlink access failed: {item} - {e}")
                            
            except (PermissionError, OSError) as e:
                logger.warning(f"Directory access failed: {path} - {e}")
        
        _safe_walk(repo_path)
        
        logger.info(f"Scanned {len(files)} files from repository")
        return files
    
    async def _detect_repository_weirdness(self, repo_path: Path, 
                                         files: List[Path]) -> List[WeirdnessDetection]:
        """Detect various types of repository weirdness."""
        detections = []
        
        # Detect symlinks
        symlinks = [f for f in files if f.is_symlink()]
        if symlinks:
            detections.append(WeirdnessDetection(
                weirdness_type=RepoWeirdness.SYMLINKS,
                severity="warning" if len(symlinks) < 50 else "critical",
                count=len(symlinks),
                examples=[str(f.relative_to(repo_path)) for f in symlinks[:5]],
                description="Repository contains symbolic links",
                impact="May cause indexing issues or infinite loops",
                mitigation="Resolve symlinks or exclude from indexing"
            ))
        
        # Detect CRLF endings
        crlf_files = []
        for file_path in files[:100]:  # Sample first 100 files
            try:
                if file_path.stat().st_size < 1024 * 1024:  # < 1MB
                    with open(file_path, 'rb') as f:
                        content = f.read(8192)  # Read first 8KB
                        if b'\r\n' in content:
                            crlf_files.append(file_path)
            except Exception:
                continue
        
        if crlf_files:
            detections.append(WeirdnessDetection(
                weirdness_type=RepoWeirdness.CRLF_ENDINGS,
                severity="info",
                count=len(crlf_files),
                examples=[str(f.relative_to(repo_path)) for f in crlf_files[:5]],
                description="Files with Windows line endings (CRLF)",
                impact="May cause parsing issues on Unix systems",
                mitigation="Normalize line endings during processing"
            ))
        
        # Detect huge files
        huge_files = []
        for file_path in files:
            try:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb > self.max_file_size_mb:
                    huge_files.append((file_path, size_mb))
            except Exception:
                continue
        
        if huge_files:
            detections.append(WeirdnessDetection(
                weirdness_type=RepoWeirdness.HUGE_FILES,
                severity="warning" if len(huge_files) < 10 else "critical",
                count=len(huge_files),
                examples=[f"{f.relative_to(repo_path)} ({s:.1f}MB)" 
                         for f, s in huge_files[:5]],
                description="Repository contains very large files",
                impact="May cause memory issues during processing",
                mitigation="Skip large files or process in chunks"
            ))
        
        # Detect generated code
        generated_files = []
        for file_path in files:
            rel_path = str(file_path.relative_to(repo_path))
            if any(re.match(pattern, rel_path) for pattern in self.generated_patterns):
                generated_files.append(file_path)
        
        if generated_files:
            detections.append(WeirdnessDetection(
                weirdness_type=RepoWeirdness.GENERATED_CODE,
                severity="info",
                count=len(generated_files),
                examples=[str(f.relative_to(repo_path)) for f in generated_files[:5]],
                description="Repository contains generated code",
                impact="Generated code may pollute search results",
                mitigation="Exclude generated files from indexing"
            ))
        
        # Detect vendored code
        vendored_files = []
        for file_path in files:
            rel_path = str(file_path.relative_to(repo_path))
            if any(re.match(pattern, rel_path) for pattern in self.vendored_patterns):
                vendored_files.append(file_path)
        
        if vendored_files:
            detections.append(WeirdnessDetection(
                weirdness_type=RepoWeirdness.VENDORED_SUBTREES,
                severity="info",
                count=len(vendored_files),
                examples=[str(f.relative_to(repo_path)) for f in vendored_files[:5]],
                description="Repository contains vendored dependencies",
                impact="Vendored code may pollute search results",
                mitigation="Exclude vendor directories from indexing"
            ))
        
        # Detect mixed encodings
        encoding_files = []
        encodings_found = set()
        for file_path in files[:50]:  # Sample files for encoding detection
            try:
                if file_path.stat().st_size < 1024 * 1024:  # < 1MB
                    with open(file_path, 'rb') as f:
                        raw_data = f.read(8192)
                        if raw_data:
                            encoding_result = chardet.detect(raw_data)
                            encoding = encoding_result.get('encoding', 'unknown')
                            if encoding and encoding != 'ascii':
                                encodings_found.add(encoding)
                                encoding_files.append((file_path, encoding))
            except Exception:
                continue
        
        if len(encodings_found) > 2:  # More than 2 different encodings
            detections.append(WeirdnessDetection(
                weirdness_type=RepoWeirdness.MIXED_ENCODINGS,
                severity="warning",
                count=len(encoding_files),
                examples=[f"{f.relative_to(repo_path)} ({enc})" 
                         for f, enc in encoding_files[:5]],
                description="Repository contains mixed character encodings",
                impact="May cause text parsing errors",
                mitigation="Normalize to UTF-8 during processing"
            ))
        
        # Detect Unicode filenames
        unicode_files = []
        for file_path in files:
            try:
                filename = file_path.name
                if not filename.isascii():
                    unicode_files.append(file_path)
            except Exception:
                continue
        
        if unicode_files:
            detections.append(WeirdnessDetection(
                weirdness_type=RepoWeirdness.UNICODE_NAMES,
                severity="info",
                count=len(unicode_files),
                examples=[str(f.relative_to(repo_path)) for f in unicode_files[:5]],
                description="Files with Unicode characters in names",
                impact="May cause compatibility issues on some systems",
                mitigation="URL-encode or sanitize filenames"
            ))
        
        return detections
    
    async def _analyze_file(self, repo_path: Path, file_path: Path) -> Optional[FileAnalysis]:
        """Analyze individual file for weirdness and symbol extraction."""
        start_time = datetime.utcnow()
        
        try:
            # Get basic file info
            stat = file_path.stat()
            relative_path = str(file_path.relative_to(repo_path))
            
            # Check if file is too large
            size_mb = stat.st_size / (1024 * 1024)
            if size_mb > self.max_file_size_mb:
                return FileAnalysis(
                    path=str(file_path),
                    relative_path=relative_path,
                    size_bytes=stat.st_size,
                    encoding="unknown",
                    line_endings="unknown",
                    language=None,
                    is_binary=True,
                    is_generated=False,
                    is_vendored=False,
                    weirdness_flags=[RepoWeirdness.HUGE_FILES],
                    symbols_extracted=0,
                    extraction_success=False,
                    processing_time_ms=0
                )
            
            # Detect if file is binary
            is_binary = self._is_binary_file(file_path)
            if is_binary:
                return self._create_binary_analysis(file_path, relative_path, stat)
            
            # Read file content
            try:
                with open(file_path, 'rb') as f:
                    raw_content = f.read()
                
                # Detect encoding
                encoding_result = chardet.detect(raw_content)
                encoding = encoding_result.get('encoding', 'utf-8')
                confidence = encoding_result.get('confidence', 0)
                
                if confidence < 0.7:
                    encoding = 'utf-8'  # Default to UTF-8
                
                # Decode content
                try:
                    content = raw_content.decode(encoding, errors='replace')
                except Exception:
                    content = raw_content.decode('utf-8', errors='replace')
                    encoding = 'utf-8'
                
            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {e}")
                return None
            
            # Detect line endings
            line_endings = self._detect_line_endings(content)
            
            # Detect language
            language = self._detect_language(file_path)
            
            # Check for generated/vendored code
            is_generated = any(re.match(pattern, relative_path) 
                             for pattern in self.generated_patterns)
            is_vendored = any(re.match(pattern, relative_path) 
                            for pattern in self.vendored_patterns)
            
            # Collect weirdness flags
            weirdness_flags = []
            if file_path.is_symlink():
                weirdness_flags.append(RepoWeirdness.SYMLINKS)
            if '\r\n' in content:
                weirdness_flags.append(RepoWeirdness.CRLF_ENDINGS)
            if not file_path.name.isascii():
                weirdness_flags.append(RepoWeirdness.UNICODE_NAMES)
            if encoding != 'utf-8' and encoding != 'ascii':
                weirdness_flags.append(RepoWeirdness.MIXED_ENCODINGS)
            if is_generated:
                weirdness_flags.append(RepoWeirdness.GENERATED_CODE)
            if is_vendored:
                weirdness_flags.append(RepoWeirdness.VENDORED_SUBTREES)
            
            # Extract symbols
            symbols_extracted = 0
            extraction_success = False
            
            if language and language in self.symbol_extractors:
                try:
                    symbols = await self._extract_symbols(language, content, file_path)
                    symbols_extracted = len(symbols)
                    extraction_success = True
                except Exception as e:
                    logger.debug(f"Symbol extraction failed for {file_path}: {e}")
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return FileAnalysis(
                path=str(file_path),
                relative_path=relative_path,
                size_bytes=stat.st_size,
                encoding=encoding,
                line_endings=line_endings,
                language=language,
                is_binary=False,
                is_generated=is_generated,
                is_vendored=is_vendored,
                weirdness_flags=weirdness_flags,
                symbols_extracted=symbols_extracted,
                extraction_success=extraction_success,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.warning(f"File analysis failed for {file_path}: {e}")
            return None
    
    def _is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary."""
        try:
            # Check MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type and not mime_type.startswith('text/'):
                return True
            
            # Check for binary content
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                if b'\x00' in chunk:  # Null bytes indicate binary
                    return True
                
                # Check for high ratio of non-printable characters
                non_printable = sum(1 for b in chunk if b < 32 and b not in [9, 10, 13])
                if len(chunk) > 0 and non_printable / len(chunk) > 0.3:
                    return True
            
            return False
            
        except Exception:
            return True  # Assume binary if we can't read it
    
    def _create_binary_analysis(self, file_path: Path, relative_path: str, 
                               stat: os.stat_result) -> FileAnalysis:
        """Create analysis for binary file."""
        weirdness_flags = [RepoWeirdness.BINARY_FILES]
        
        if file_path.is_symlink():
            weirdness_flags.append(RepoWeirdness.SYMLINKS)
        if not file_path.name.isascii():
            weirdness_flags.append(RepoWeirdness.UNICODE_NAMES)
        
        return FileAnalysis(
            path=str(file_path),
            relative_path=relative_path,
            size_bytes=stat.st_size,
            encoding="binary",
            line_endings="n/a",
            language=None,
            is_binary=True,
            is_generated=False,
            is_vendored=False,
            weirdness_flags=weirdness_flags,
            symbols_extracted=0,
            extraction_success=False,
            processing_time_ms=0
        )
    
    def _detect_line_endings(self, content: str) -> str:
        """Detect line ending style."""
        crlf_count = content.count('\r\n')
        lf_count = content.count('\n') - crlf_count
        cr_count = content.count('\r') - crlf_count
        
        if crlf_count > lf_count and crlf_count > cr_count:
            return "CRLF"
        elif lf_count > cr_count:
            return "LF"
        elif cr_count > 0:
            return "CR"
        elif crlf_count > 0 and lf_count > 0:
            return "mixed"
        else:
            return "LF"  # Default
    
    def _detect_language(self, file_path: Path) -> Optional[SymbolLanguage]:
        """Detect programming language from file extension."""
        suffix = file_path.suffix.lower()
        
        for language, extensions in self.language_extensions.items():
            if suffix in extensions:
                return language
        
        return None
    
    async def _extract_symbols(self, language: SymbolLanguage, content: str, 
                             file_path: Path) -> List[Dict[str, Any]]:
        """Extract symbols from file content."""
        if language not in self.symbol_extractors:
            return []
        
        extractor = self.symbol_extractors[language]
        try:
            symbols = await extractor.extract_symbols(content, str(file_path))
            return symbols
        except Exception as e:
            logger.debug(f"Symbol extraction error in {file_path}: {e}")
            return []
    
    def _calculate_symbol_coverage(self, file_analyses: List[FileAnalysis]) -> Dict[SymbolLanguage, SymbolCoverage]:
        """Calculate symbol coverage statistics by language."""
        coverage = {}
        
        # Group files by language
        files_by_language = defaultdict(list)
        for analysis in file_analyses:
            if analysis.language and not analysis.is_binary:
                files_by_language[analysis.language].append(analysis)
        
        # Calculate coverage for each language
        for language, files in files_by_language.items():
            total_files = len(files)
            analyzed_files = len([f for f in files if f.extraction_success])
            
            if total_files > 0:
                coverage_percentage = (analyzed_files / total_files) * 100
            else:
                coverage_percentage = 0
            
            total_symbols = sum(f.symbols_extracted for f in files)
            
            # Calculate symbol types (simplified)
            symbol_types = {
                "functions": int(total_symbols * 0.6),  # Estimate
                "classes": int(total_symbols * 0.2),
                "variables": int(total_symbols * 0.15),
                "other": int(total_symbols * 0.05)
            }
            
            # Calculate hit rate
            successful_files = [f for f in files if f.extraction_success]
            hit_rate = (len(successful_files) / total_files * 100) if total_files > 0 else 0
            
            # Collect extraction errors
            errors = []
            for f in files:
                if not f.extraction_success and f.language == language:
                    errors.append(f"Failed to extract from {f.relative_path}")
            
            coverage[language] = SymbolCoverage(
                language=language,
                total_files=total_files,
                analyzed_files=analyzed_files,
                coverage_percentage=coverage_percentage,
                symbol_count=total_symbols,
                symbol_types=symbol_types,
                hit_rate=hit_rate,
                extraction_errors=errors[:10]  # Limit to 10 examples
            )
        
        return coverage
    
    async def _generate_windows_test_data(self, repo_path: Path, 
                                        file_analyses: List[FileAnalysis]) -> Dict[str, Any]:
        """Generate test data for Windows path normalization."""
        test_data = {
            "path_normalization_tests": [],
            "case_sensitivity_tests": [],
            "encoding_tests": [],
            "line_ending_tests": []
        }
        
        # Path normalization tests
        for analysis in file_analyses[:20]:  # Sample 20 files
            posix_path = PurePosixPath(analysis.relative_path)
            windows_path = PureWindowsPath(analysis.relative_path)
            
            test_data["path_normalization_tests"].append({
                "original_path": analysis.relative_path,
                "posix_normalized": str(posix_path),
                "windows_normalized": str(windows_path),
                "path_differs": str(posix_path) != str(windows_path)
            })
        
        # Case sensitivity tests
        path_counter = Counter()
        for analysis in file_analyses:
            lower_path = analysis.relative_path.lower()
            path_counter[lower_path] += 1
        
        case_conflicts = [path for path, count in path_counter.items() if count > 1]
        for conflict_path in case_conflicts[:10]:
            conflicting_files = [
                a.relative_path for a in file_analyses 
                if a.relative_path.lower() == conflict_path
            ]
            test_data["case_sensitivity_tests"].append({
                "normalized_path": conflict_path,
                "conflicting_files": conflicting_files,
                "conflict_count": len(conflicting_files)
            })
        
        # Encoding tests
        encoding_samples = defaultdict(list)
        for analysis in file_analyses:
            if not analysis.is_binary and analysis.encoding != 'utf-8':
                encoding_samples[analysis.encoding].append(analysis.relative_path)
        
        for encoding, files in encoding_samples.items():
            test_data["encoding_tests"].append({
                "encoding": encoding,
                "sample_files": files[:5],
                "file_count": len(files)
            })
        
        # Line ending tests
        crlf_files = [
            a.relative_path for a in file_analyses 
            if RepoWeirdness.CRLF_ENDINGS in a.weirdness_flags
        ]
        
        if crlf_files:
            test_data["line_ending_tests"] = {
                "crlf_files": crlf_files[:10],
                "total_crlf_files": len(crlf_files),
                "normalization_needed": True
            }
        
        return test_data
    
    def _generate_recommendations(self, weirdness_detections: List[WeirdnessDetection],
                                symbol_coverage: Dict[SymbolLanguage, SymbolCoverage]) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Weirdness-based recommendations
        for detection in weirdness_detections:
            if detection.severity in ["warning", "critical"]:
                recommendations.append({
                    "category": "repository_weirdness",
                    "priority": detection.severity,
                    "issue": detection.description,
                    "recommendation": detection.mitigation,
                    "affected_count": detection.count
                })
        
        # Symbol coverage recommendations
        for language, coverage in symbol_coverage.items():
            if coverage.hit_rate < 80:  # Less than 80% hit rate
                recommendations.append({
                    "category": "symbol_extraction",
                    "priority": "warning",
                    "issue": f"Low symbol extraction rate for {language.value}: {coverage.hit_rate:.1f}%",
                    "recommendation": "Review symbol extraction configuration or file preprocessing",
                    "affected_count": coverage.total_files - coverage.analyzed_files
                })
        
        # Performance recommendations
        large_repos = len([d for d in weirdness_detections 
                         if d.weirdness_type == RepoWeirdness.HUGE_FILES and d.count > 10])
        if large_repos:
            recommendations.append({
                "category": "performance",
                "priority": "warning",
                "issue": "Repository contains many large files",
                "recommendation": "Consider file size limits and streaming processing",
                "affected_count": large_repos
            })
        
        return recommendations


# Symbol Extractors (simplified implementations)

class BaseSymbolExtractor:
    """Base class for symbol extractors."""
    
    async def extract_symbols(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Extract symbols from content. Override in subclasses."""
        raise NotImplementedError


class PythonSymbolExtractor(BaseSymbolExtractor):
    """Python symbol extractor using AST."""
    
    async def extract_symbols(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        import ast
        
        try:
            tree = ast.parse(content)
            symbols = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    symbols.append({
                        "type": "function",
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args]
                    })
                elif isinstance(node, ast.ClassDef):
                    symbols.append({
                        "type": "class",
                        "name": node.name,
                        "line": node.lineno,
                        "bases": [base.id if isinstance(base, ast.Name) else str(base) 
                                for base in node.bases]
                    })
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            symbols.append({
                                "type": "variable",
                                "name": target.id,
                                "line": node.lineno
                            })
            
            return symbols
            
        except SyntaxError as e:
            logger.debug(f"Python syntax error in {file_path}: {e}")
            return []
        except Exception as e:
            logger.debug(f"Python extraction error in {file_path}: {e}")
            return []


class TypeScriptSymbolExtractor(BaseSymbolExtractor):
    """TypeScript symbol extractor using regex patterns."""
    
    async def extract_symbols(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        symbols = []
        
        # Function patterns
        function_patterns = [
            r'function\s+(\w+)\s*\(',
            r'const\s+(\w+)\s*=\s*\(',
            r'let\s+(\w+)\s*=\s*\(',
            r'(\w+)\s*:\s*\([^)]*\)\s*=>',
        ]
        
        for pattern in function_patterns:
            for match in re.finditer(pattern, content):
                symbols.append({
                    "type": "function",
                    "name": match.group(1),
                    "line": content[:match.start()].count('\n') + 1
                })
        
        # Class patterns
        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            symbols.append({
                "type": "class",
                "name": match.group(1),
                "line": content[:match.start()].count('\n') + 1
            })
        
        # Interface patterns
        interface_pattern = r'interface\s+(\w+)'
        for match in re.finditer(interface_pattern, content):
            symbols.append({
                "type": "interface",
                "name": match.group(1),
                "line": content[:match.start()].count('\n') + 1
            })
        
        return symbols


class JavaScriptSymbolExtractor(BaseSymbolExtractor):
    """JavaScript symbol extractor using regex patterns."""
    
    async def extract_symbols(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        symbols = []
        
        # Function patterns
        function_patterns = [
            r'function\s+(\w+)\s*\(',
            r'const\s+(\w+)\s*=\s*function',
            r'const\s+(\w+)\s*=\s*\(',
            r'(\w+)\s*:\s*function',
        ]
        
        for pattern in function_patterns:
            for match in re.finditer(pattern, content):
                symbols.append({
                    "type": "function",
                    "name": match.group(1),
                    "line": content[:match.start()].count('\n') + 1
                })
        
        # Variable patterns
        var_patterns = [
            r'var\s+(\w+)',
            r'let\s+(\w+)',
            r'const\s+(\w+)',
        ]
        
        for pattern in var_patterns:
            for match in re.finditer(pattern, content):
                symbols.append({
                    "type": "variable",
                    "name": match.group(1),
                    "line": content[:match.start()].count('\n') + 1
                })
        
        return symbols


class GoSymbolExtractor(BaseSymbolExtractor):
    """Go symbol extractor using regex patterns."""
    
    async def extract_symbols(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        symbols = []
        
        # Function patterns
        func_pattern = r'func\s+(\w+)\s*\('
        for match in re.finditer(func_pattern, content):
            symbols.append({
                "type": "function",
                "name": match.group(1),
                "line": content[:match.start()].count('\n') + 1
            })
        
        # Type patterns
        type_pattern = r'type\s+(\w+)\s+struct'
        for match in re.finditer(type_pattern, content):
            symbols.append({
                "type": "struct",
                "name": match.group(1),
                "line": content[:match.start()].count('\n') + 1
            })
        
        # Variable patterns
        var_pattern = r'var\s+(\w+)'
        for match in re.finditer(var_pattern, content):
            symbols.append({
                "type": "variable",
                "name": match.group(1),
                "line": content[:match.start()].count('\n') + 1
            })
        
        return symbols


class JavaSymbolExtractor(BaseSymbolExtractor):
    """Java symbol extractor using regex patterns."""
    
    async def extract_symbols(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        symbols = []
        
        # Class patterns
        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            symbols.append({
                "type": "class",
                "name": match.group(1),
                "line": content[:match.start()].count('\n') + 1
            })
        
        # Method patterns
        method_pattern = r'(public|private|protected)?\s*(static)?\s*\w+\s+(\w+)\s*\('
        for match in re.finditer(method_pattern, content):
            symbols.append({
                "type": "method",
                "name": match.group(3),
                "line": content[:match.start()].count('\n') + 1,
                "visibility": match.group(1) or "package",
                "static": bool(match.group(2))
            })
        
        return symbols


class RustSymbolExtractor(BaseSymbolExtractor):
    """Rust symbol extractor using regex patterns."""
    
    async def extract_symbols(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        symbols = []
        
        # Function patterns
        fn_pattern = r'fn\s+(\w+)\s*\('
        for match in re.finditer(fn_pattern, content):
            symbols.append({
                "type": "function",
                "name": match.group(1),
                "line": content[:match.start()].count('\n') + 1
            })
        
        # Struct patterns
        struct_pattern = r'struct\s+(\w+)'
        for match in re.finditer(struct_pattern, content):
            symbols.append({
                "type": "struct",
                "name": match.group(1),
                "line": content[:match.start()].count('\n') + 1
            })
        
        # Enum patterns
        enum_pattern = r'enum\s+(\w+)'
        for match in re.finditer(enum_pattern, content):
            symbols.append({
                "type": "enum",
                "name": match.group(1),
                "line": content[:match.start()].count('\n') + 1
            })
        
        return symbols


async def main():
    """Example usage of repository handler."""
    config = {
        "max_file_size_mb": 10,
        "max_directory_depth": 15,
        "max_symlink_depth": 3
    }
    
    handler = RepoHandler(config)
    
    # Analyze the lens repository
    repo_path = Path("/home/nathan/Projects/lens")
    
    logger.info("Starting repository analysis...")
    analysis_result = await handler.analyze_repository(repo_path)
    
    # Print summary
    summary = analysis_result["summary"]
    logger.info(f"Analysis complete:")
    logger.info(f"  Total files: {summary['total_files']}")
    logger.info(f"  Analyzed files: {summary['analyzed_files']}")
    logger.info(f"  Weirdness detections: {summary['weirdness_count']}")
    logger.info(f"  Languages detected: {summary['languages_detected']}")
    
    # Print weirdness detections
    for detection in analysis_result["weirdness_detections"]:
        logger.info(f"  {detection['weirdness_type']}: {detection['count']} items "
                   f"({detection['severity']})")
    
    # Print symbol coverage
    for lang, coverage in analysis_result["symbol_coverage"].items():
        logger.info(f"  {lang}: {coverage['coverage_percentage']:.1f}% coverage, "
                   f"{coverage['symbol_count']} symbols")
    
    # Save analysis to file
    output_file = Path("/tmp/repo_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(analysis_result, f, indent=2, default=str)
    
    logger.info(f"Analysis saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())