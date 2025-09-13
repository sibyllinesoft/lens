#!/usr/bin/env python3
"""
Adversarial & Drift Sentinels for Quality Assurance
Implements shuffle-context/top-1-drop ablation on prod trickle,
requiring ≥10% quality drop to maintain evidence dependency.
Includes configuration drift detection and alerts.
"""

import asyncio
import json
import logging
import random
import statistics
import time
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
import threading
import hashlib
import yaml

from telemetry_system import TelemetryCollector, QueryTelemetry, QueryOperation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AblationTestType(Enum):
    """Types of adversarial ablation tests."""
    SHUFFLE_CONTEXT = "shuffle_context"
    TOP_1_DROP = "top_1_drop"
    RANDOM_DROP = "random_drop"
    CONTEXT_TRUNCATION = "context_truncation"
    NOISE_INJECTION = "noise_injection"


class DriftType(Enum):
    """Types of configuration drift."""
    CORPUS_DRIFT = "corpus_drift"
    PROMPT_DRIFT = "prompt_drift"
    MODEL_DRIFT = "model_drift"
    THRESHOLD_DRIFT = "threshold_drift"
    INFRASTRUCTURE_DRIFT = "infrastructure_drift"


class SentinelStatus(Enum):
    """Status of a sentinel."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class AblationTest:
    """Configuration for an ablation test."""
    test_id: str
    test_type: AblationTestType
    description: str
    traffic_percentage: float  # Percentage of queries to test
    expected_quality_drop: float  # Expected drop in quality metrics
    max_quality_drop: float  # Maximum allowed drop (>10% triggers alert)
    test_duration_hours: int = 24
    enabled: bool = True


@dataclass
class AblationResult:
    """Result of an ablation test."""
    test_id: str
    test_type: AblationTestType
    timestamp: datetime
    
    # Original query metrics
    original_ess: float
    original_answerable: float
    original_span_recall: float
    original_latency_ms: float
    
    # Ablated query metrics
    ablated_ess: float
    ablated_answerable: float
    ablated_span_recall: float
    ablated_latency_ms: float
    
    # Quality drop analysis
    ess_drop_pct: float
    answerable_drop_pct: float
    span_recall_drop_pct: float
    overall_quality_drop: float
    
    # Context
    query_id: str
    scenario: str
    repo: str
    ablation_details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.ablation_details is None:
            self.ablation_details = {}


@dataclass
class DriftAlert:
    """Configuration drift alert."""
    drift_type: DriftType
    component: str
    description: str
    severity: str  # "info", "warning", "critical"
    timestamp: datetime
    
    # Change details
    previous_value: Any
    current_value: Any
    change_magnitude: float  # Normalized change magnitude (0-1)
    
    # Context
    detection_method: str
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


class AdversarialSentinel:
    """Runs adversarial ablation tests on production traffic."""
    
    def __init__(self, config: Dict[str, Any], telemetry_collector: TelemetryCollector):
        self.config = config
        self.telemetry = telemetry_collector
        
        # Ablation tests
        self.ablation_tests = self._initialize_ablation_tests()
        self.test_results: Dict[str, deque] = {}
        
        # Initialize result queues
        for test_id in self.ablation_tests:
            self.test_results[test_id] = deque(maxlen=1000)
        
        # Traffic sampling
        self.traffic_sampler = ProductionTrafficSampler(
            config.get("traffic_sampling", {})
        )
        
        # Quality thresholds
        self.quality_drop_threshold = config.get("quality_drop_threshold", 0.1)  # 10%
        self.evidence_dependency_threshold = config.get("evidence_dependency_threshold", 0.1)
        
        # Alert management
        self.alert_callbacks: List[Callable] = []
        self.test_failures: Dict[str, int] = defaultdict(int)
        
        # Storage
        self.storage_dir = Path(config.get("storage_dir", "/tmp/adversarial_sentinels"))
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Background testing
        self.testing_enabled = True
        self.testing_interval_seconds = config.get("testing_interval_seconds", 300)  # 5 minutes
        self.testing_thread = None
        self.stop_testing = threading.Event()
        
        logger.info(f"AdversarialSentinel initialized with {len(self.ablation_tests)} tests")
    
    def _initialize_ablation_tests(self) -> Dict[str, AblationTest]:
        """Initialize ablation test configurations."""
        tests = {}
        
        # Shuffle context test
        tests["shuffle_context"] = AblationTest(
            test_id="shuffle_context",
            test_type=AblationTestType.SHUFFLE_CONTEXT,
            description="Shuffle retrieved context order to test position bias",
            traffic_percentage=0.1,  # 0.1% of traffic
            expected_quality_drop=0.05,  # Expect 5% drop
            max_quality_drop=0.1,  # Alert if >10% drop
            test_duration_hours=24,
            enabled=True
        )
        
        # Top-1 drop test
        tests["top_1_drop"] = AblationTest(
            test_id="top_1_drop",
            test_type=AblationTestType.TOP_1_DROP,
            description="Remove top-ranked context item to test ranking dependency",
            traffic_percentage=0.05,  # 0.05% of traffic
            expected_quality_drop=0.15,  # Expect 15% drop
            max_quality_drop=0.25,  # Alert if >25% drop
            test_duration_hours=24,
            enabled=True
        )
        
        # Random drop test
        tests["random_drop"] = AblationTest(
            test_id="random_drop",
            test_type=AblationTestType.RANDOM_DROP,
            description="Remove random context items to test robustness",
            traffic_percentage=0.05,  # 0.05% of traffic
            expected_quality_drop=0.08,  # Expect 8% drop
            max_quality_drop=0.15,  # Alert if >15% drop
            test_duration_hours=24,
            enabled=True
        )
        
        # Context truncation test
        tests["context_truncation"] = AblationTest(
            test_id="context_truncation",
            test_type=AblationTestType.CONTEXT_TRUNCATION,
            description="Truncate context to test length dependency",
            traffic_percentage=0.03,  # 0.03% of traffic
            expected_quality_drop=0.12,  # Expect 12% drop
            max_quality_drop=0.20,  # Alert if >20% drop
            test_duration_hours=24,
            enabled=True
        )
        
        # Noise injection test
        tests["noise_injection"] = AblationTest(
            test_id="noise_injection",
            test_type=AblationTestType.NOISE_INJECTION,
            description="Inject irrelevant context to test noise robustness",
            traffic_percentage=0.02,  # 0.02% of traffic
            expected_quality_drop=0.06,  # Expect 6% drop
            max_quality_drop=0.12,  # Alert if >12% drop
            test_duration_hours=24,
            enabled=False  # Disabled by default
        )
        
        # Override with config
        config_tests = self.config.get("ablation_tests", {})
        for test_id, test_config in config_tests.items():
            if test_id in tests:
                for key, value in test_config.items():
                    if hasattr(tests[test_id], key):
                        setattr(tests[test_id], key, value)
        
        return tests
    
    def start_testing(self):
        """Start adversarial testing in background thread."""
        if self.testing_thread and self.testing_thread.is_alive():
            return
        
        self.stop_testing.clear()
        self.testing_thread = threading.Thread(target=self._testing_loop, daemon=True)
        self.testing_thread.start()
        
        logger.info("Adversarial testing started")
    
    def stop_testing(self):
        """Stop adversarial testing thread."""
        self.testing_enabled = False
        self.stop_testing.set()
        if self.testing_thread:
            self.testing_thread.join(timeout=10)
        
        logger.info("Adversarial testing stopped")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for adversarial test alerts."""
        self.alert_callbacks.append(callback)
    
    async def process_query_for_ablation(self, query_telemetry: QueryTelemetry) -> Optional[str]:
        """Check if query should be subject to ablation testing."""
        if not self.testing_enabled:
            return None
        
        # Sample query for testing
        selected_test = self.traffic_sampler.sample_for_ablation(
            query_telemetry, self.ablation_tests
        )
        
        if selected_test:
            logger.debug(f"Query {query_telemetry.query_id} selected for ablation test: {selected_test}")
            return selected_test
        
        return None
    
    async def run_ablation_test(self, test_id: str, original_query: QueryTelemetry,
                               original_context: List[Dict[str, Any]]) -> AblationResult:
        """Run a specific ablation test on a query."""
        if test_id not in self.ablation_tests:
            raise ValueError(f"Unknown ablation test: {test_id}")
        
        ablation_test = self.ablation_tests[test_id]
        
        # Apply ablation to context
        ablated_context, ablation_details = self._apply_ablation(
            ablation_test.test_type, original_context
        )
        
        # Run query with ablated context
        ablated_query = await self._run_ablated_query(
            original_query, ablated_context
        )
        
        # Calculate quality drops
        ess_drop_pct = self._calculate_percentage_drop(
            original_query.ess_score, ablated_query.ess_score
        )
        answerable_drop_pct = self._calculate_percentage_drop(
            original_query.answerable_at_k, ablated_query.answerable_at_k
        )
        span_recall_drop_pct = self._calculate_percentage_drop(
            original_query.span_recall, ablated_query.span_recall
        )
        
        # Overall quality drop (weighted average)
        overall_quality_drop = (
            ess_drop_pct * 0.4 + 
            answerable_drop_pct * 0.4 + 
            span_recall_drop_pct * 0.2
        )
        
        # Create result
        result = AblationResult(
            test_id=test_id,
            test_type=ablation_test.test_type,
            timestamp=datetime.utcnow(),
            original_ess=original_query.ess_score,
            original_answerable=original_query.answerable_at_k,
            original_span_recall=original_query.span_recall,
            original_latency_ms=original_query.latency_ms,
            ablated_ess=ablated_query.ess_score,
            ablated_answerable=ablated_query.answerable_at_k,
            ablated_span_recall=ablated_query.span_recall,
            ablated_latency_ms=ablated_query.latency_ms,
            ess_drop_pct=ess_drop_pct,
            answerable_drop_pct=answerable_drop_pct,
            span_recall_drop_pct=span_recall_drop_pct,
            overall_quality_drop=overall_quality_drop,
            query_id=original_query.query_id,
            scenario=original_query.scenario,
            repo=original_query.repo,
            ablation_details=ablation_details
        )
        
        # Store result
        self.test_results[test_id].append(result)
        
        # Check for quality drop threshold
        if overall_quality_drop < self.evidence_dependency_threshold:
            self._alert_insufficient_evidence_dependency(test_id, result)
        elif overall_quality_drop > ablation_test.max_quality_drop:
            self._alert_excessive_quality_drop(test_id, result)
        
        logger.debug(f"Ablation test {test_id} completed: {overall_quality_drop:.1%} quality drop")
        
        return result
    
    def _apply_ablation(self, test_type: AblationTestType, 
                       context: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Apply ablation transformation to context."""
        if not context:
            return context, {}
        
        ablated_context = context.copy()
        details = {"original_length": len(context)}
        
        if test_type == AblationTestType.SHUFFLE_CONTEXT:
            # Shuffle the order of context items
            random.shuffle(ablated_context)
            details["shuffled"] = True
            
        elif test_type == AblationTestType.TOP_1_DROP:
            # Remove the top-ranked item (first item)
            if ablated_context:
                removed_item = ablated_context.pop(0)
                details["removed_top_item"] = removed_item.get("rank", 0)
                
        elif test_type == AblationTestType.RANDOM_DROP:
            # Remove 20% of items randomly
            num_to_remove = max(1, len(ablated_context) // 5)
            for _ in range(num_to_remove):
                if ablated_context:
                    removed_idx = random.randint(0, len(ablated_context) - 1)
                    ablated_context.pop(removed_idx)
            details["removed_count"] = num_to_remove
            
        elif test_type == AblationTestType.CONTEXT_TRUNCATION:
            # Truncate to 50% of original length
            truncate_length = len(ablated_context) // 2
            ablated_context = ablated_context[:truncate_length]
            details["truncated_to"] = truncate_length
            
        elif test_type == AblationTestType.NOISE_INJECTION:
            # Inject noise items
            noise_items = self._generate_noise_context(len(context) // 3)
            # Insert noise items at random positions
            for noise_item in noise_items:
                insert_pos = random.randint(0, len(ablated_context))
                ablated_context.insert(insert_pos, noise_item)
            details["noise_items_added"] = len(noise_items)
        
        details["final_length"] = len(ablated_context)
        return ablated_context, details
    
    def _generate_noise_context(self, count: int) -> List[Dict[str, Any]]:
        """Generate noise context items."""
        noise_items = []
        noise_content = [
            "This is irrelevant content that should not match the query",
            "Random function definition that has nothing to do with the search",
            "Unrelated configuration file content",
            "Lorem ipsum dolor sit amet consectetur adipiscing elit"
        ]
        
        for i in range(count):
            noise_item = {
                "content": random.choice(noise_content),
                "file": f"noise_file_{i}.txt",
                "score": random.uniform(0.1, 0.3),  # Low but not zero score
                "rank": 999 + i,
                "is_noise": True
            }
            noise_items.append(noise_item)
        
        return noise_items
    
    async def _run_ablated_query(self, original_query: QueryTelemetry,
                                ablated_context: List[Dict[str, Any]]) -> QueryTelemetry:
        """Run query with ablated context (simulated)."""
        # TODO: Integrate with actual query processing system
        # For now, simulate the impact of ablation on metrics
        
        ablated_query = QueryTelemetry(
            query_id=f"{original_query.query_id}_ablated",
            operation=original_query.operation,
            scenario=original_query.scenario,
            repo=original_query.repo,
            query_text=original_query.query_text
        )
        
        # Simulate realistic impact based on ablation
        context_quality_factor = len(ablated_context) / max(1, len(original_query.citations))
        
        # Add some randomness to simulate real-world variance
        noise_factor = random.uniform(0.95, 1.05)
        
        ablated_query.ess_score = original_query.ess_score * context_quality_factor * noise_factor
        ablated_query.answerable_at_k = original_query.answerable_at_k * context_quality_factor * noise_factor
        ablated_query.span_recall = original_query.span_recall * context_quality_factor * noise_factor
        
        # Latency might be slightly different due to less context to process
        ablated_query.latency_ms = original_query.latency_ms * (0.8 + 0.4 * context_quality_factor)
        
        # Simulate citations based on remaining context
        ablated_query.citations = ablated_context[:min(3, len(ablated_context))]
        
        return ablated_query
    
    def _calculate_percentage_drop(self, original: float, ablated: float) -> float:
        """Calculate percentage drop from original to ablated value."""
        if original == 0:
            return 0.0
        return max(0.0, (original - ablated) / original)
    
    def _alert_insufficient_evidence_dependency(self, test_id: str, result: AblationResult):
        """Alert when evidence dependency is insufficient."""
        alert_msg = (f"Insufficient evidence dependency in {test_id}: "
                    f"only {result.overall_quality_drop:.1%} quality drop "
                    f"(threshold: {self.evidence_dependency_threshold:.1%})")
        
        self._fire_alert("evidence_dependency", "warning", alert_msg, result)
        
        logger.warning(alert_msg)
    
    def _alert_excessive_quality_drop(self, test_id: str, result: AblationResult):
        """Alert when quality drop is excessive."""
        max_drop = self.ablation_tests[test_id].max_quality_drop
        alert_msg = (f"Excessive quality drop in {test_id}: "
                    f"{result.overall_quality_drop:.1%} drop "
                    f"(max allowed: {max_drop:.1%})")
        
        self._fire_alert("excessive_drop", "critical", alert_msg, result)
        
        logger.error(alert_msg)
    
    def _fire_alert(self, alert_type: str, severity: str, message: str, result: AblationResult):
        """Fire alert to registered callbacks."""
        alert_data = {
            "type": alert_type,
            "severity": severity,
            "message": message,
            "test_id": result.test_id,
            "timestamp": result.timestamp.isoformat(),
            "result": asdict(result)
        }
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _testing_loop(self):
        """Background testing loop."""
        while not self.stop_testing.wait(self.testing_interval_seconds):
            try:
                # Analyze recent test results
                self._analyze_test_results()
                
                # Export results
                self._export_test_results()
                
            except Exception as e:
                logger.error(f"Adversarial testing loop error: {e}")
    
    def _analyze_test_results(self):
        """Analyze recent test results for trends."""
        cutoff = datetime.utcnow() - timedelta(hours=1)
        
        for test_id, results in self.test_results.items():
            recent_results = [r for r in results if r.timestamp >= cutoff]
            
            if len(recent_results) >= 3:
                avg_quality_drop = statistics.mean(r.overall_quality_drop for r in recent_results)
                expected_drop = self.ablation_tests[test_id].expected_quality_drop
                
                # Check if quality drop is trending away from expected
                if abs(avg_quality_drop - expected_drop) > 0.05:
                    logger.info(f"Test {test_id} trending: avg drop {avg_quality_drop:.1%} "
                               f"vs expected {expected_drop:.1%}")
    
    def _export_test_results(self):
        """Export test results to storage."""
        export_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "test_results": {}
        }
        
        # Export recent results (last 6 hours)
        cutoff = datetime.utcnow() - timedelta(hours=6)
        for test_id, results in self.test_results.items():
            recent_results = [
                asdict(r) for r in results 
                if r.timestamp >= cutoff
            ]
            export_data["test_results"][test_id] = recent_results
        
        # Save to file
        export_file = self.storage_dir / f"ablation-results-{datetime.utcnow().strftime('%Y%m%d%H')}.json"
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all ablation tests."""
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "testing_enabled": self.testing_enabled,
            "test_summaries": {}
        }
        
        for test_id, test_config in self.ablation_tests.items():
            results = list(self.test_results[test_id])
            
            if results:
                recent_results = [r for r in results 
                                if r.timestamp >= datetime.utcnow() - timedelta(hours=24)]
                
                avg_quality_drop = statistics.mean(r.overall_quality_drop for r in recent_results) if recent_results else 0
                
                test_summary = {
                    "config": asdict(test_config),
                    "total_results": len(results),
                    "recent_results_24h": len(recent_results),
                    "avg_quality_drop_24h": avg_quality_drop,
                    "expected_quality_drop": test_config.expected_quality_drop,
                    "within_expected_range": abs(avg_quality_drop - test_config.expected_quality_drop) <= 0.05
                }
            else:
                test_summary = {
                    "config": asdict(test_config),
                    "total_results": 0,
                    "recent_results_24h": 0,
                    "avg_quality_drop_24h": 0,
                    "expected_quality_drop": test_config.expected_quality_drop,
                    "within_expected_range": True
                }
            
            summary["test_summaries"][test_id] = test_summary
        
        return summary


class ProductionTrafficSampler:
    """Samples production traffic for ablation testing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sampling_rates = config.get("sampling_rates", {})
        
        # Query tracking to ensure fair sampling
        self.query_counts = defaultdict(int)
        self.last_reset = datetime.utcnow()
        self.reset_interval_hours = config.get("reset_interval_hours", 24)
    
    def sample_for_ablation(self, query: QueryTelemetry, 
                           ablation_tests: Dict[str, AblationTest]) -> Optional[str]:
        """Sample query for ablation testing."""
        self._maybe_reset_counters()
        
        # Only sample from core scenarios for more reliable results
        if query.scenario != "core":
            return None
        
        # Check each test for sampling
        for test_id, test_config in ablation_tests.items():
            if not test_config.enabled:
                continue
            
            # Sample based on traffic percentage
            if random.random() < test_config.traffic_percentage / 100:
                self.query_counts[test_id] += 1
                return test_id
        
        return None
    
    def _maybe_reset_counters(self):
        """Reset counters periodically to ensure fresh sampling."""
        if datetime.utcnow() - self.last_reset > timedelta(hours=self.reset_interval_hours):
            self.query_counts.clear()
            self.last_reset = datetime.utcnow()


class DriftSentinel:
    """Monitors configuration drift and infrastructure changes."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitoring_enabled = True
        
        # Configuration tracking
        self.config_checksums: Dict[str, str] = {}
        self.config_snapshots: Dict[str, Any] = {}
        
        # Drift detection settings
        self.check_interval_minutes = config.get("check_interval_minutes", 60)
        self.drift_thresholds = config.get("drift_thresholds", {
            "corpus": 0.05,  # 5% change
            "prompt": 0.01,  # 1% change
            "model": 0.0,    # Any change
            "threshold": 0.1  # 10% change
        })
        
        # Alert management
        self.drift_alerts: List[DriftAlert] = []
        self.alert_callbacks: List[Callable] = []
        
        # Storage
        self.storage_dir = Path(config.get("storage_dir", "/tmp/drift_monitoring"))
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Background monitoring
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        logger.info("DriftSentinel initialized")
    
    def start_monitoring(self):
        """Start drift monitoring in background thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        # Take initial snapshots
        self._take_initial_snapshots()
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Drift monitoring started")
    
    def stop_monitoring(self):
        """Stop drift monitoring thread."""
        self.monitoring_enabled = False
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Drift monitoring stopped")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for drift alerts."""
        self.alert_callbacks.append(callback)
    
    def _take_initial_snapshots(self):
        """Take initial snapshots of all monitored configurations."""
        try:
            # Corpus snapshot
            corpus_config = self._snapshot_corpus_config()
            self.config_snapshots["corpus"] = corpus_config
            self.config_checksums["corpus"] = self._calculate_checksum(corpus_config)
            
            # Prompt snapshot
            prompt_config = self._snapshot_prompt_config()
            self.config_snapshots["prompt"] = prompt_config
            self.config_checksums["prompt"] = self._calculate_checksum(prompt_config)
            
            # Model snapshot
            model_config = self._snapshot_model_config()
            self.config_snapshots["model"] = model_config
            self.config_checksums["model"] = self._calculate_checksum(model_config)
            
            # Threshold snapshot
            threshold_config = self._snapshot_threshold_config()
            self.config_snapshots["threshold"] = threshold_config
            self.config_checksums["threshold"] = self._calculate_checksum(threshold_config)
            
            logger.info("Initial configuration snapshots taken")
            
        except Exception as e:
            logger.error(f"Failed to take initial snapshots: {e}")
    
    def _monitoring_loop(self):
        """Main drift monitoring loop."""
        while not self.stop_monitoring.wait(self.check_interval_minutes * 60):
            try:
                self._check_all_drift()
            except Exception as e:
                logger.error(f"Drift monitoring loop error: {e}")
    
    def _check_all_drift(self):
        """Check all configuration components for drift."""
        for component in ["corpus", "prompt", "model", "threshold"]:
            try:
                drift_alert = self._check_component_drift(component)
                if drift_alert:
                    self.drift_alerts.append(drift_alert)
                    self._fire_drift_alert(drift_alert)
            except Exception as e:
                logger.error(f"Error checking {component} drift: {e}")
    
    def _check_component_drift(self, component: str) -> Optional[DriftAlert]:
        """Check specific component for configuration drift."""
        if component not in self.config_snapshots:
            logger.warning(f"No baseline snapshot for component: {component}")
            return None
        
        # Take current snapshot
        if component == "corpus":
            current_config = self._snapshot_corpus_config()
        elif component == "prompt":
            current_config = self._snapshot_prompt_config()
        elif component == "model":
            current_config = self._snapshot_model_config()
        elif component == "threshold":
            current_config = self._snapshot_threshold_config()
        else:
            return None
        
        # Calculate current checksum
        current_checksum = self._calculate_checksum(current_config)
        previous_checksum = self.config_checksums[component]
        
        # Check for changes
        if current_checksum != previous_checksum:
            # Calculate change magnitude
            change_magnitude = self._calculate_change_magnitude(
                self.config_snapshots[component], current_config
            )
            
            # Determine severity
            severity = self._determine_drift_severity(component, change_magnitude)
            
            # Create drift alert
            alert = DriftAlert(
                drift_type=getattr(DriftType, f"{component.upper()}_DRIFT"),
                component=component,
                description=f"{component.title()} configuration has changed",
                severity=severity,
                timestamp=datetime.utcnow(),
                previous_value=self.config_snapshots[component],
                current_value=current_config,
                change_magnitude=change_magnitude,
                detection_method="checksum_comparison"
            )
            
            # Update stored values
            self.config_snapshots[component] = current_config
            self.config_checksums[component] = current_checksum
            
            return alert
        
        return None
    
    def _snapshot_corpus_config(self) -> Dict[str, Any]:
        """Take snapshot of corpus configuration."""
        # TODO: Integrate with actual corpus management system
        # For now, simulate corpus config
        return {
            "version": "2.1.0",
            "sha": "abc123def456",
            "size_mb": 1024,
            "file_count": 50000,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def _snapshot_prompt_config(self) -> Dict[str, Any]:
        """Take snapshot of prompt configurations."""
        # TODO: Integrate with actual prompt management system
        prompts_dir = Path("/home/nathan/Projects/lens/prompts")
        prompt_config = {}
        
        if prompts_dir.exists():
            for prompt_file in prompts_dir.glob("*.json"):
                try:
                    with open(prompt_file) as f:
                        content = json.load(f)
                    prompt_config[prompt_file.stem] = {
                        "checksum": hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest(),
                        "size": prompt_file.stat().st_size,
                        "modified": datetime.fromtimestamp(prompt_file.stat().st_mtime).isoformat()
                    }
                except Exception:
                    continue
        
        return prompt_config
    
    def _snapshot_model_config(self) -> Dict[str, Any]:
        """Take snapshot of model configurations."""
        # TODO: Integrate with actual model management system
        return {
            "search_model": "gpt-4-turbo-2024-04-09",
            "extract_model": "gpt-3.5-turbo-0125",
            "rag_model": "gpt-4-turbo-2024-04-09",
            "embedding_model": "text-embedding-3-large",
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def _snapshot_threshold_config(self) -> Dict[str, Any]:
        """Take snapshot of threshold configurations."""
        # TODO: Integrate with actual threshold management system
        return {
            "ess_thresholds": {
                "core": 0.85,
                "extended": 0.75,
                "experimental": 0.65
            },
            "latency_thresholds": {
                "search_ms": 200,
                "rag_ms": 350
            },
            "quality_thresholds": {
                "answerable_at_k": 0.7,
                "span_recall": 0.5
            },
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def _calculate_checksum(self, config: Dict[str, Any]) -> str:
        """Calculate checksum for configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def _calculate_change_magnitude(self, previous: Dict[str, Any], 
                                   current: Dict[str, Any]) -> float:
        """Calculate magnitude of change between configurations."""
        # Simple implementation: ratio of changed keys
        if not previous and not current:
            return 0.0
        
        if not previous or not current:
            return 1.0
        
        all_keys = set(previous.keys()) | set(current.keys())
        if not all_keys:
            return 0.0
        
        changed_keys = 0
        for key in all_keys:
            if previous.get(key) != current.get(key):
                changed_keys += 1
        
        return changed_keys / len(all_keys)
    
    def _determine_drift_severity(self, component: str, change_magnitude: float) -> str:
        """Determine severity of configuration drift."""
        threshold = self.drift_thresholds.get(component, 0.1)
        
        if change_magnitude == 0:
            return "info"
        elif change_magnitude < threshold:
            return "info"
        elif change_magnitude < threshold * 2:
            return "warning"
        else:
            return "critical"
    
    def _fire_drift_alert(self, alert: DriftAlert):
        """Fire drift alert to registered callbacks."""
        alert_data = asdict(alert)
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Drift alert callback failed: {e}")
        
        logger.warning(f"DRIFT ALERT [{alert.severity.upper()}]: {alert.description}")
    
    def get_drift_status(self) -> Dict[str, Any]:
        """Get current drift monitoring status."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "monitoring_enabled": self.monitoring_enabled,
            "active_alerts": len([a for a in self.drift_alerts 
                                if a.timestamp >= datetime.utcnow() - timedelta(hours=24)]),
            "recent_alerts_24h": [asdict(a) for a in self.drift_alerts 
                                if a.timestamp >= datetime.utcnow() - timedelta(hours=24)],
            "monitored_components": list(self.config_snapshots.keys()),
            "last_check_checksums": self.config_checksums
        }


async def main():
    """Example usage of adversarial and drift sentinels."""
    from telemetry_system import create_telemetry_system, QueryOperation
    
    # Create telemetry system
    telemetry_config = {"storage_dir": "/tmp/sentinel_demo_telemetry"}
    collector, _ = create_telemetry_system(telemetry_config)
    
    # Create adversarial sentinel
    adversarial_config = {
        "storage_dir": "/tmp/sentinel_demo_adversarial",
        "testing_interval_seconds": 30
    }
    adversarial_sentinel = AdversarialSentinel(adversarial_config, collector)
    
    # Create drift sentinel
    drift_config = {
        "storage_dir": "/tmp/sentinel_demo_drift",
        "check_interval_minutes": 1
    }
    drift_sentinel = DriftSentinel(drift_config)
    
    # Add alert callbacks
    def adversarial_alert_callback(alert_data):
        logger.warning(f"ADVERSARIAL ALERT: {alert_data['type']} - {alert_data['message']}")
    
    def drift_alert_callback(alert_data):
        logger.warning(f"DRIFT ALERT: {alert_data['component']} - {alert_data['description']}")
    
    adversarial_sentinel.add_alert_callback(adversarial_alert_callback)
    drift_sentinel.add_alert_callback(drift_alert_callback)
    
    # Start monitoring
    adversarial_sentinel.start_testing()
    drift_sentinel.start_monitoring()
    
    # Simulate queries and ablation testing
    for i in range(20):
        query_id = f"query_{i}"
        
        # Create query
        telemetry = collector.record_query_start(
            query_id=query_id,
            operation=QueryOperation.SEARCH,
            scenario="core",
            repo=f"repo-{i % 2 + 1}",
            query_text=f"Test query {i}"
        )
        
        # Update with metrics
        collector.update_query_metrics(
            query_id=query_id,
            ess_score=0.8 + random.uniform(-0.1, 0.1),
            answerable_at_k=0.75 + random.uniform(-0.1, 0.1),
            span_recall=0.6 + random.uniform(-0.1, 0.1),
            citations=[{"file": f"file_{i}.py", "line": i * 10}]
        )
        
        # Complete query
        collector.record_query_complete(query_id)
        
        # Check for ablation testing
        selected_test = await adversarial_sentinel.process_query_for_ablation(telemetry)
        if selected_test:
            # Simulate context
            original_context = [
                {"content": f"Context item {j}", "rank": j, "score": 0.9 - j * 0.1}
                for j in range(5)
            ]
            
            # Run ablation test
            result = await adversarial_sentinel.run_ablation_test(
                selected_test, telemetry, original_context
            )
            logger.info(f"Ablation test {selected_test}: {result.overall_quality_drop:.1%} drop")
        
        await asyncio.sleep(0.5)
    
    # Get status summaries
    await asyncio.sleep(5)  # Wait for processing
    
    adversarial_summary = adversarial_sentinel.get_test_summary()
    logger.info(f"Adversarial testing summary: {len(adversarial_summary['test_summaries'])} tests")
    
    drift_status = drift_sentinel.get_drift_status()
    logger.info(f"Drift monitoring status: {drift_status['active_alerts']} active alerts")
    
    # Cleanup
    adversarial_sentinel.stop_testing()
    drift_sentinel.stop_monitoring()
    collector.stop_processing()
    
    logger.info("Sentinel demonstration complete")


if __name__ == "__main__":
    asyncio.run(main())