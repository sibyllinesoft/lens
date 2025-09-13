#!/usr/bin/env python3
"""
Per-Query Telemetry System for Lens Search
Emits: {op, ESS, Answerable@k, SpanRecall, token_budget, latency_ms, citations[]}
Aggregates by scenario×repo with real-time dashboards and alerts.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
import threading
import numpy as np
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryOperation(Enum):
    """Types of query operations."""
    SEARCH = "search"
    EXTRACT = "extract"
    RAG = "rag"
    COMPOSITE = "composite"


class TelemetryEvent(Enum):
    """Types of telemetry events."""
    QUERY_START = "query_start"
    QUERY_COMPLETE = "query_complete"
    QUERY_ERROR = "query_error"
    STAGE_COMPLETE = "stage_complete"


@dataclass
class QueryTelemetry:
    """Per-query telemetry data."""
    query_id: str
    operation: QueryOperation
    scenario: str
    repo: str
    user_id: Optional[str] = None
    
    # Core metrics
    ess_score: float = 0.0
    answerable_at_k: float = 0.0
    span_recall: float = 0.0
    token_budget: int = 0
    tokens_used: int = 0
    latency_ms: float = 0.0
    
    # Citations and sources
    citations: List[Dict[str, Any]] = None
    source_files: List[str] = None
    extracted_spans: List[Dict[str, Any]] = None
    
    # Timing breakdown
    search_latency_ms: float = 0.0
    extract_latency_ms: float = 0.0
    rag_latency_ms: float = 0.0
    
    # Quality metrics
    relevance_score: float = 0.0
    confidence_score: float = 0.0
    completeness_score: float = 0.0
    
    # Error tracking
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    warning_count: int = 0
    
    # Context
    query_text: str = ""
    response_length: int = 0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.citations is None:
            self.citations = []
        if self.source_files is None:
            self.source_files = []
        if self.extracted_spans is None:
            self.extracted_spans = []
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for scenario×repo combination."""
    scenario: str
    repo: str
    window_start: datetime
    window_end: datetime
    
    # Query counts
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    
    # Performance metrics
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Quality metrics
    avg_ess_score: float = 0.0
    avg_answerable_at_k: float = 0.0
    avg_span_recall: float = 0.0
    avg_relevance_score: float = 0.0
    
    # Token usage
    total_tokens_used: int = 0
    avg_tokens_per_query: float = 0.0
    token_efficiency: float = 0.0  # quality per token
    
    # Citation analysis
    avg_citations_per_query: float = 0.0
    unique_source_files: int = 0
    citation_diversity: float = 0.0
    
    # Error analysis
    error_rate: float = 0.0
    top_errors: List[Tuple[str, int]] = None
    warning_rate: float = 0.0
    
    def __post_init__(self):
        if self.top_errors is None:
            self.top_errors = []


class TelemetryCollector:
    """Collects and processes query telemetry in real-time."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_queries: Dict[str, QueryTelemetry] = {}
        self.completed_queries: deque = deque(maxlen=config.get("max_history", 10000))
        
        # Aggregation windows
        self.aggregation_window_minutes = config.get("aggregation_window_minutes", 5)
        self.aggregated_metrics: Dict[Tuple[str, str], AggregatedMetrics] = {}
        
        # Real-time monitoring
        self.alert_thresholds = config.get("alert_thresholds", {})
        self.alert_callbacks: List[callable] = []
        
        # Storage
        self.storage_dir = Path(config.get("storage_dir", "/tmp/lens_telemetry"))
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Background processing
        self.processing_interval_seconds = config.get("processing_interval_seconds", 30)
        self.stop_processing = threading.Event()
        self.processing_thread = None
        
        # Metrics cache for fast access
        self.metrics_cache = {}
        self.cache_ttl_seconds = config.get("cache_ttl_seconds", 60)
        self.last_cache_update = 0
        
        logger.info("TelemetryCollector initialized")
    
    def start_processing(self):
        """Start background processing thread."""
        if self.processing_thread and self.processing_thread.is_alive():
            return
        
        self.stop_processing.clear()
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Telemetry processing started")
    
    def stop_processing(self):
        """Stop background processing thread."""
        self.stop_processing.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        logger.info("Telemetry processing stopped")
    
    def record_query_start(self, query_id: str, operation: QueryOperation, 
                          scenario: str, repo: str, query_text: str,
                          user_id: Optional[str] = None) -> QueryTelemetry:
        """Record the start of a query."""
        telemetry = QueryTelemetry(
            query_id=query_id,
            operation=operation,
            scenario=scenario,
            repo=repo,
            query_text=query_text,
            user_id=user_id,
            timestamp=datetime.utcnow()
        )
        
        self.active_queries[query_id] = telemetry
        
        # Emit telemetry event
        self._emit_event(TelemetryEvent.QUERY_START, telemetry)
        
        return telemetry
    
    def update_query_metrics(self, query_id: str, **metrics):
        """Update metrics for an active query."""
        if query_id not in self.active_queries:
            logger.warning(f"Query {query_id} not found in active queries")
            return
        
        telemetry = self.active_queries[query_id]
        
        # Update metrics
        for key, value in metrics.items():
            if hasattr(telemetry, key):
                setattr(telemetry, key, value)
            else:
                logger.warning(f"Unknown metric: {key}")
    
    def record_query_complete(self, query_id: str, final_metrics: Dict[str, Any] = None):
        """Record query completion with final metrics."""
        if query_id not in self.active_queries:
            logger.warning(f"Query {query_id} not found in active queries")
            return
        
        telemetry = self.active_queries.pop(query_id)
        
        # Update final metrics
        if final_metrics:
            for key, value in final_metrics.items():
                if hasattr(telemetry, key):
                    setattr(telemetry, key, value)
        
        # Calculate total latency
        if telemetry.latency_ms == 0.0:
            elapsed = (datetime.utcnow() - telemetry.timestamp).total_seconds() * 1000
            telemetry.latency_ms = elapsed
        
        # Add to completed queries
        self.completed_queries.append(telemetry)
        
        # Emit completion event
        self._emit_event(TelemetryEvent.QUERY_COMPLETE, telemetry)
        
        # Check real-time alerts
        self._check_real_time_alerts(telemetry)
        
        logger.debug(f"Query {query_id} completed: {telemetry.latency_ms:.1f}ms, ESS={telemetry.ess_score:.3f}")
    
    def record_query_error(self, query_id: str, error_type: str, error_message: str):
        """Record query error."""
        if query_id not in self.active_queries:
            logger.warning(f"Query {query_id} not found in active queries")
            return
        
        telemetry = self.active_queries.pop(query_id)
        telemetry.error_type = error_type
        telemetry.error_message = error_message
        
        # Calculate latency up to error
        elapsed = (datetime.utcnow() - telemetry.timestamp).total_seconds() * 1000
        telemetry.latency_ms = elapsed
        
        # Add to completed queries
        self.completed_queries.append(telemetry)
        
        # Emit error event
        self._emit_event(TelemetryEvent.QUERY_ERROR, telemetry)
        
        logger.warning(f"Query {query_id} failed: {error_type} - {error_message}")
    
    def get_real_time_metrics(self, scenario: str = None, repo: str = None) -> Dict[str, Any]:
        """Get real-time metrics for dashboards."""
        now = time.time()
        
        # Check cache
        cache_key = f"{scenario}:{repo}"
        if (cache_key in self.metrics_cache and 
            now - self.last_cache_update < self.cache_ttl_seconds):
            return self.metrics_cache[cache_key]
        
        # Calculate metrics
        metrics = self._calculate_real_time_metrics(scenario, repo)
        
        # Update cache
        self.metrics_cache[cache_key] = metrics
        self.last_cache_update = now
        
        return metrics
    
    def _calculate_real_time_metrics(self, scenario: str = None, repo: str = None) -> Dict[str, Any]:
        """Calculate real-time metrics from recent queries."""
        # Get recent queries (last 15 minutes)
        cutoff = datetime.utcnow() - timedelta(minutes=15)
        recent_queries = [
            q for q in self.completed_queries 
            if q.timestamp > cutoff and
            (not scenario or q.scenario == scenario) and
            (not repo or q.repo == repo)
        ]
        
        if not recent_queries:
            return {
                "total_queries": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Calculate metrics
        successful_queries = [q for q in recent_queries if not q.error_type]
        failed_queries = [q for q in recent_queries if q.error_type]
        
        latencies = [q.latency_ms for q in successful_queries if q.latency_ms > 0]
        ess_scores = [q.ess_score for q in successful_queries if q.ess_score > 0]
        answerable_scores = [q.answerable_at_k for q in successful_queries if q.answerable_at_k > 0]
        span_recalls = [q.span_recall for q in successful_queries if q.span_recall > 0]
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "window_minutes": 15,
            "total_queries": len(recent_queries),
            "successful_queries": len(successful_queries),
            "failed_queries": len(failed_queries),
            "error_rate": len(failed_queries) / len(recent_queries) if recent_queries else 0,
            
            # Performance
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
            "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0,
            
            # Quality
            "avg_ess_score": statistics.mean(ess_scores) if ess_scores else 0,
            "avg_answerable_at_k": statistics.mean(answerable_scores) if answerable_scores else 0,
            "avg_span_recall": statistics.mean(span_recalls) if span_recalls else 0,
            
            # Token usage
            "total_tokens": sum(q.tokens_used for q in successful_queries),
            "avg_tokens_per_query": statistics.mean([q.tokens_used for q in successful_queries if q.tokens_used > 0]) if successful_queries else 0,
            
            # Citations
            "avg_citations_per_query": statistics.mean([len(q.citations) for q in successful_queries]) if successful_queries else 0,
            
            # Error breakdown
            "top_errors": self._get_top_errors(failed_queries)
        }
        
        return metrics
    
    def _get_top_errors(self, failed_queries: List[QueryTelemetry]) -> List[Dict[str, Any]]:
        """Get top error types from failed queries."""
        error_counts = defaultdict(int)
        for query in failed_queries:
            if query.error_type:
                error_counts[query.error_type] += 1
        
        return [
            {"error_type": error_type, "count": count}
            for error_type, count in sorted(error_counts.items(), 
                                           key=lambda x: x[1], reverse=True)[:5]
        ]
    
    def get_aggregated_metrics(self, scenario: str, repo: str,
                              start_time: datetime = None) -> Optional[AggregatedMetrics]:
        """Get aggregated metrics for scenario×repo."""
        key = (scenario, repo)
        
        if start_time:
            # Find specific window
            for metrics in self.aggregated_metrics.values():
                if (metrics.scenario == scenario and metrics.repo == repo and
                    metrics.window_start <= start_time < metrics.window_end):
                    return metrics
            return None
        else:
            # Get latest
            return self.aggregated_metrics.get(key)
    
    def add_alert_callback(self, callback: callable):
        """Add callback for real-time alerts."""
        self.alert_callbacks.append(callback)
    
    def _emit_event(self, event_type: TelemetryEvent, telemetry: QueryTelemetry):
        """Emit telemetry event to configured outputs."""
        event = {
            "event_type": event_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "query_id": telemetry.query_id,
            "operation": telemetry.operation.value,
            "scenario": telemetry.scenario,
            "repo": telemetry.repo,
            "telemetry": asdict(telemetry)
        }
        
        # Write to storage (NDJSON format for streaming)
        event_file = self.storage_dir / f"events-{datetime.utcnow().strftime('%Y%m%d')}.ndjson"
        with open(event_file, 'a') as f:
            f.write(json.dumps(event, default=str) + '\n')
    
    def _check_real_time_alerts(self, telemetry: QueryTelemetry):
        """Check if query triggers any real-time alerts."""
        alerts = []
        
        # Check latency threshold
        if ("max_latency_ms" in self.alert_thresholds and 
            telemetry.latency_ms > self.alert_thresholds["max_latency_ms"]):
            alerts.append({
                "type": "high_latency",
                "message": f"Query {telemetry.query_id} took {telemetry.latency_ms:.1f}ms",
                "threshold": self.alert_thresholds["max_latency_ms"],
                "actual": telemetry.latency_ms
            })
        
        # Check ESS score threshold
        if ("min_ess_score" in self.alert_thresholds and 
            telemetry.ess_score < self.alert_thresholds["min_ess_score"]):
            alerts.append({
                "type": "low_ess_score",
                "message": f"Query {telemetry.query_id} ESS score {telemetry.ess_score:.3f}",
                "threshold": self.alert_thresholds["min_ess_score"],
                "actual": telemetry.ess_score
            })
        
        # Check error condition
        if telemetry.error_type:
            alerts.append({
                "type": "query_error",
                "message": f"Query {telemetry.query_id} failed: {telemetry.error_type}",
                "error_message": telemetry.error_message
            })
        
        # Fire alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert, telemetry)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def _processing_loop(self):
        """Background processing loop for aggregation and storage."""
        while not self.stop_processing.wait(self.processing_interval_seconds):
            try:
                # Aggregate metrics
                self._aggregate_metrics()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                # Export metrics
                self._export_metrics()
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
    
    def _aggregate_metrics(self):
        """Aggregate metrics by scenario×repo for current window."""
        window_start = datetime.utcnow().replace(second=0, microsecond=0)
        window_start = window_start - timedelta(
            minutes=window_start.minute % self.aggregation_window_minutes
        )
        window_end = window_start + timedelta(minutes=self.aggregation_window_minutes)
        
        # Group queries by scenario×repo
        groups = defaultdict(list)
        for query in self.completed_queries:
            if window_start <= query.timestamp < window_end:
                key = (query.scenario, query.repo)
                groups[key].append(query)
        
        # Calculate aggregated metrics for each group
        for (scenario, repo), queries in groups.items():
            if not queries:
                continue
            
            successful_queries = [q for q in queries if not q.error_type]
            failed_queries = [q for q in queries if q.error_type]
            
            # Calculate metrics
            metrics = AggregatedMetrics(
                scenario=scenario,
                repo=repo,
                window_start=window_start,
                window_end=window_end,
                total_queries=len(queries),
                successful_queries=len(successful_queries),
                failed_queries=len(failed_queries)
            )
            
            if successful_queries:
                latencies = [q.latency_ms for q in successful_queries if q.latency_ms > 0]
                if latencies:
                    metrics.avg_latency_ms = statistics.mean(latencies)
                    metrics.p95_latency_ms = np.percentile(latencies, 95)
                    metrics.p99_latency_ms = np.percentile(latencies, 99)
                
                # Quality metrics
                ess_scores = [q.ess_score for q in successful_queries if q.ess_score > 0]
                if ess_scores:
                    metrics.avg_ess_score = statistics.mean(ess_scores)
                
                answerable_scores = [q.answerable_at_k for q in successful_queries if q.answerable_at_k > 0]
                if answerable_scores:
                    metrics.avg_answerable_at_k = statistics.mean(answerable_scores)
                
                span_recalls = [q.span_recall for q in successful_queries if q.span_recall > 0]
                if span_recalls:
                    metrics.avg_span_recall = statistics.mean(span_recalls)
                
                relevance_scores = [q.relevance_score for q in successful_queries if q.relevance_score > 0]
                if relevance_scores:
                    metrics.avg_relevance_score = statistics.mean(relevance_scores)
                
                # Token metrics
                tokens_used = [q.tokens_used for q in successful_queries if q.tokens_used > 0]
                if tokens_used:
                    metrics.total_tokens_used = sum(tokens_used)
                    metrics.avg_tokens_per_query = statistics.mean(tokens_used)
                    
                    # Token efficiency (quality per token)
                    if metrics.avg_ess_score > 0 and metrics.avg_tokens_per_query > 0:
                        metrics.token_efficiency = metrics.avg_ess_score / metrics.avg_tokens_per_query
                
                # Citation metrics
                citation_counts = [len(q.citations) for q in successful_queries]
                if citation_counts:
                    metrics.avg_citations_per_query = statistics.mean(citation_counts)
                
                all_source_files = set()
                for q in successful_queries:
                    all_source_files.update(q.source_files)
                metrics.unique_source_files = len(all_source_files)
                
                # Citation diversity (unique files per query)
                if metrics.total_queries > 0 and metrics.unique_source_files > 0:
                    metrics.citation_diversity = metrics.unique_source_files / metrics.total_queries
            
            # Error metrics
            if queries:
                metrics.error_rate = len(failed_queries) / len(queries)
                
                # Top errors
                error_counts = defaultdict(int)
                for q in failed_queries:
                    if q.error_type:
                        error_counts[q.error_type] += 1
                metrics.top_errors = sorted(error_counts.items(), 
                                          key=lambda x: x[1], reverse=True)[:5]
                
                # Warning rate
                warning_count = sum(q.warning_count for q in queries)
                metrics.warning_rate = warning_count / len(queries)
            
            # Store aggregated metrics
            key = (scenario, repo)
            self.aggregated_metrics[key] = metrics
            
            logger.debug(f"Aggregated metrics for {scenario}×{repo}: "
                        f"{metrics.total_queries} queries, "
                        f"ESS={metrics.avg_ess_score:.3f}, "
                        f"latency={metrics.avg_latency_ms:.1f}ms")
    
    def _cleanup_old_data(self):
        """Cleanup old telemetry data to prevent memory growth."""
        # Remove old aggregated metrics (keep last 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        to_remove = []
        
        for key, metrics in self.aggregated_metrics.items():
            if metrics.window_end < cutoff:
                to_remove.append(key)
        
        for key in to_remove:
            del self.aggregated_metrics[key]
        
        # Clear metrics cache periodically
        if time.time() - self.last_cache_update > self.cache_ttl_seconds * 10:
            self.metrics_cache.clear()
    
    def _export_metrics(self):
        """Export aggregated metrics to storage."""
        metrics_file = self.storage_dir / f"metrics-{datetime.utcnow().strftime('%Y%m%d%H')}.json"
        
        export_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "aggregated_metrics": [
                asdict(metrics) for metrics in self.aggregated_metrics.values()
            ]
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)


class TelemetryDashboard:
    """Real-time dashboard for telemetry visualization."""
    
    def __init__(self, collector: TelemetryCollector):
        self.collector = collector
        self.dashboard_config = {
            "refresh_interval_seconds": 30,
            "display_scenarios": ["core", "extended", "experimental"],
            "display_repos": ["repo-1", "repo-2", "repo-3"]
        }
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for real-time dashboard."""
        dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "global_metrics": self.collector.get_real_time_metrics(),
            "scenario_metrics": {},
            "repo_metrics": {},
            "scenario_repo_matrix": {}
        }
        
        # Scenario-specific metrics
        for scenario in self.dashboard_config["display_scenarios"]:
            dashboard_data["scenario_metrics"][scenario] = \
                self.collector.get_real_time_metrics(scenario=scenario)
        
        # Repo-specific metrics
        for repo in self.dashboard_config["display_repos"]:
            dashboard_data["repo_metrics"][repo] = \
                self.collector.get_real_time_metrics(repo=repo)
        
        # Scenario×Repo matrix
        for scenario in self.dashboard_config["display_scenarios"]:
            for repo in self.dashboard_config["display_repos"]:
                key = f"{scenario}×{repo}"
                dashboard_data["scenario_repo_matrix"][key] = \
                    self.collector.get_real_time_metrics(scenario=scenario, repo=repo)
        
        return dashboard_data
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts."""
        # TODO: Implement alert tracking and summarization
        return {
            "active_alerts": 0,
            "alerts_last_hour": 0,
            "critical_alerts": 0,
            "alert_types": []
        }


def create_telemetry_system(config: Dict[str, Any]) -> Tuple[TelemetryCollector, TelemetryDashboard]:
    """Factory function to create telemetry system."""
    collector = TelemetryCollector(config)
    dashboard = TelemetryDashboard(collector)
    
    # Add default alert callback
    def default_alert_callback(alert: Dict[str, Any], telemetry: QueryTelemetry):
        logger.warning(f"TELEMETRY ALERT: {alert['type']} - {alert['message']}")
    
    collector.add_alert_callback(default_alert_callback)
    collector.start_processing()
    
    return collector, dashboard


async def main():
    """Example usage of telemetry system."""
    config = {
        "max_history": 10000,
        "aggregation_window_minutes": 5,
        "processing_interval_seconds": 30,
        "storage_dir": "/tmp/lens_telemetry",
        "cache_ttl_seconds": 60,
        "alert_thresholds": {
            "max_latency_ms": 1000,
            "min_ess_score": 0.7,
            "max_error_rate": 0.1
        }
    }
    
    # Create telemetry system
    collector, dashboard = create_telemetry_system(config)
    
    # Simulate some queries
    for i in range(20):
        query_id = f"query_{i}"
        operation = QueryOperation.SEARCH if i % 2 == 0 else QueryOperation.RAG
        scenario = "core" if i % 3 == 0 else "extended"
        repo = f"repo-{i % 3 + 1}"
        
        # Start query
        telemetry = collector.record_query_start(
            query_id=query_id,
            operation=operation,
            scenario=scenario,
            repo=repo,
            query_text=f"Test query {i}",
            user_id=f"user_{i % 5}"
        )
        
        # Simulate processing
        await asyncio.sleep(0.1)
        
        # Update metrics during processing
        collector.update_query_metrics(
            query_id=query_id,
            search_latency_ms=50 + i * 2,
            tokens_used=100 + i * 5,
            ess_score=0.8 + (i % 5) * 0.04,
            answerable_at_k=0.7 + (i % 3) * 0.1
        )
        
        # Complete query
        if i % 10 == 9:  # Simulate some failures
            collector.record_query_error(query_id, "timeout", "Request timed out")
        else:
            collector.record_query_complete(
                query_id=query_id,
                final_metrics={
                    "span_recall": 0.6 + (i % 4) * 0.1,
                    "citations": [{"file": f"file_{i}.py", "line": i * 10}],
                    "source_files": [f"file_{i}.py"],
                    "relevance_score": 0.85 + (i % 2) * 0.1
                }
            )
        
        if i % 5 == 0:
            # Show real-time metrics
            metrics = collector.get_real_time_metrics()
            logger.info(f"Real-time metrics: {metrics['total_queries']} queries, "
                       f"error_rate={metrics['error_rate']:.3f}, "
                       f"avg_latency={metrics['avg_latency_ms']:.1f}ms")
    
    # Generate dashboard data
    dashboard_data = dashboard.generate_dashboard_data()
    logger.info(f"Dashboard data: {len(dashboard_data['scenario_repo_matrix'])} matrix entries")
    
    # Cleanup
    collector.stop_processing()
    logger.info("Telemetry system demonstration complete")


if __name__ == "__main__":
    asyncio.run(main())