#!/usr/bin/env python3
"""
SLO Monitoring System with Automatic Rollback Triggers
Implements the four critical SLOs with 24h deltas and automatic rollback on violations:
- Pass-rate_core ≥85% (rollback if ↓ >5% from baseline)
- Answerable@k ≥0.7 on core slice
- SpanRecall ≥0.5
- P95 latency budgets: <200ms code search, <350ms RAG
"""

import asyncio
import json
import logging
import statistics
import time
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import threading
import numpy as np

from telemetry_system import TelemetryCollector, QueryTelemetry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SLOStatus(Enum):
    """Status of an SLO."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    VIOLATED = "violated"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SLODefinition:
    """Definition of an SLO with thresholds and rollback triggers."""
    name: str
    description: str
    target_value: float
    warning_threshold: float
    critical_threshold: float
    rollback_threshold: float
    comparison: str  # "gte" (>=) or "lte" (<=)
    measurement_window_minutes: int = 15
    baseline_window_hours: int = 24
    rollback_enabled: bool = True


@dataclass
class SLOMeasurement:
    """A single SLO measurement."""
    slo_name: str
    timestamp: datetime
    value: float
    baseline_value: Optional[float] = None
    delta_from_baseline: Optional[float] = None
    status: SLOStatus = SLOStatus.HEALTHY
    sample_size: int = 0
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class SLOAlert:
    """SLO violation alert."""
    slo_name: str
    severity: AlertSeverity
    message: str
    current_value: float
    target_value: float
    baseline_value: Optional[float]
    delta_from_baseline: Optional[float]
    timestamp: datetime
    measurement: SLOMeasurement
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


class SLOMonitor:
    """Monitors SLOs and triggers rollbacks on violations."""
    
    def __init__(self, config: Dict[str, Any], telemetry_collector: TelemetryCollector):
        self.config = config
        self.telemetry = telemetry_collector
        
        # SLO definitions
        self.slos = self._initialize_slos()
        
        # Measurements and baselines
        self.measurements: Dict[str, deque] = {}
        self.baselines: Dict[str, float] = {}
        self.baseline_measurements: Dict[str, deque] = {}
        
        # Initialize measurement queues
        for slo_name in self.slos:
            self.measurements[slo_name] = deque(maxlen=1000)
            self.baseline_measurements[slo_name] = deque(maxlen=10000)
        
        # Alert management
        self.active_alerts: Dict[str, SLOAlert] = {}
        self.alert_callbacks: List[Callable] = []
        self.rollback_callbacks: List[Callable] = []
        
        # Monitoring control
        self.monitoring_enabled = True
        self.rollback_triggered = False
        self.monitoring_interval_seconds = config.get("monitoring_interval_seconds", 60)
        
        # Storage
        self.storage_dir = Path(config.get("storage_dir", "/tmp/slo_monitoring"))
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Background monitoring
        self.monitor_thread = None
        self.stop_monitoring = threading.Event()
        
        logger.info(f"SLOMonitor initialized with {len(self.slos)} SLOs")
    
    def _initialize_slos(self) -> Dict[str, SLODefinition]:
        """Initialize SLO definitions."""
        slos = {}
        
        # Pass-rate_core ≥85% (rollback if ↓ >5% from baseline)
        slos["pass_rate_core"] = SLODefinition(
            name="pass_rate_core",
            description="Core pass rate - queries that return relevant results",
            target_value=0.85,
            warning_threshold=0.80,
            critical_threshold=0.75,
            rollback_threshold=0.05,  # 5% drop from baseline
            comparison="gte",
            measurement_window_minutes=15,
            baseline_window_hours=24,
            rollback_enabled=True
        )
        
        # Answerable@k ≥0.7 on core slice
        slos["answerable_at_k"] = SLODefinition(
            name="answerable_at_k",
            description="Fraction of queries answerable from retrieved context",
            target_value=0.7,
            warning_threshold=0.65,
            critical_threshold=0.6,
            rollback_threshold=0.1,  # 10% drop from baseline
            comparison="gte",
            measurement_window_minutes=15,
            baseline_window_hours=24,
            rollback_enabled=True
        )
        
        # SpanRecall ≥0.5
        slos["span_recall"] = SLODefinition(
            name="span_recall",
            description="Recall of relevant spans in extracted context",
            target_value=0.5,
            warning_threshold=0.45,
            critical_threshold=0.4,
            rollback_threshold=0.1,  # 10% drop from baseline
            comparison="gte",
            measurement_window_minutes=15,
            baseline_window_hours=24,
            rollback_enabled=True
        )
        
        # P95 latency budget: <200ms code search
        slos["p95_latency_search"] = SLODefinition(
            name="p95_latency_search",
            description="P95 latency for code search operations",
            target_value=200.0,  # milliseconds
            warning_threshold=250.0,
            critical_threshold=300.0,
            rollback_threshold=100.0,  # 100ms increase from baseline
            comparison="lte",
            measurement_window_minutes=15,
            baseline_window_hours=24,
            rollback_enabled=True
        )
        
        # P95 latency budget: <350ms RAG
        slos["p95_latency_rag"] = SLODefinition(
            name="p95_latency_rag",
            description="P95 latency for RAG operations",
            target_value=350.0,  # milliseconds
            warning_threshold=400.0,
            critical_threshold=500.0,
            rollback_threshold=150.0,  # 150ms increase from baseline
            comparison="lte",
            measurement_window_minutes=15,
            baseline_window_hours=24,
            rollback_enabled=True
        )
        
        # Override with config values
        config_slos = self.config.get("slos", {})
        for slo_name, slo_config in config_slos.items():
            if slo_name in slos:
                for key, value in slo_config.items():
                    if hasattr(slos[slo_name], key):
                        setattr(slos[slo_name], key, value)
        
        return slos
    
    def start_monitoring(self):
        """Start SLO monitoring in background thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return
        
        self.stop_monitoring.clear()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("SLO monitoring started")
    
    def stop_monitoring(self):
        """Stop SLO monitoring thread."""
        self.monitoring_enabled = False
        self.stop_monitoring.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        logger.info("SLO monitoring stopped")
    
    def add_alert_callback(self, callback: Callable[[SLOAlert], None]):
        """Add callback for SLO alerts."""
        self.alert_callbacks.append(callback)
    
    def add_rollback_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for rollback triggers."""
        self.rollback_callbacks.append(callback)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring.wait(self.monitoring_interval_seconds):
            if not self.monitoring_enabled or self.rollback_triggered:
                continue
            
            try:
                # Measure all SLOs
                asyncio.run(self._measure_all_slos())
                
                # Update baselines
                self._update_baselines()
                
                # Check for violations and alerts
                self._check_violations()
                
                # Export measurements
                self._export_measurements()
                
            except Exception as e:
                logger.error(f"SLO monitoring loop error: {e}")
    
    async def _measure_all_slos(self):
        """Measure all SLOs from current telemetry."""
        for slo_name, slo_def in self.slos.items():
            try:
                measurement = await self._measure_slo(slo_def)
                if measurement:
                    self.measurements[slo_name].append(measurement)
                    self.baseline_measurements[slo_name].append(measurement)
                    
                    logger.debug(f"SLO {slo_name}: {measurement.value:.3f} "
                               f"(target: {slo_def.target_value:.3f})")
                    
            except Exception as e:
                logger.error(f"Failed to measure SLO {slo_name}: {e}")
    
    async def _measure_slo(self, slo_def: SLODefinition) -> Optional[SLOMeasurement]:
        """Measure a specific SLO from telemetry data."""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=slo_def.measurement_window_minutes)
        
        # Get recent telemetry data
        recent_queries = []
        for query in self.telemetry.completed_queries:
            if query.timestamp >= window_start and not query.error_type:
                recent_queries.append(query)
        
        if not recent_queries:
            logger.debug(f"No recent queries for SLO {slo_def.name}")
            return None
        
        # Calculate SLO value based on type
        value = None
        context = {"sample_size": len(recent_queries)}
        
        if slo_def.name == "pass_rate_core":
            # Pass rate for core scenario only
            core_queries = [q for q in recent_queries if q.scenario == "core"]
            if core_queries:
                passed_queries = [q for q in core_queries if q.ess_score >= 0.7]
                value = len(passed_queries) / len(core_queries)
                context["core_queries"] = len(core_queries)
                context["passed_queries"] = len(passed_queries)
        
        elif slo_def.name == "answerable_at_k":
            # Answerable@k for core scenario only
            core_queries = [q for q in recent_queries if q.scenario == "core"]
            answerable_scores = [q.answerable_at_k for q in core_queries if q.answerable_at_k > 0]
            if answerable_scores:
                value = statistics.mean(answerable_scores)
                context["core_queries"] = len(core_queries)
                context["avg_answerable"] = value
        
        elif slo_def.name == "span_recall":
            # Span recall across all scenarios
            span_recalls = [q.span_recall for q in recent_queries if q.span_recall > 0]
            if span_recalls:
                value = statistics.mean(span_recalls)
                context["recall_samples"] = len(span_recalls)
        
        elif slo_def.name == "p95_latency_search":
            # P95 latency for search operations
            search_queries = [q for q in recent_queries if q.operation.value == "search"]
            latencies = [q.latency_ms for q in search_queries if q.latency_ms > 0]
            if latencies:
                value = np.percentile(latencies, 95)
                context["search_queries"] = len(search_queries)
                context["p50_latency"] = np.percentile(latencies, 50)
                context["p90_latency"] = np.percentile(latencies, 90)
        
        elif slo_def.name == "p95_latency_rag":
            # P95 latency for RAG operations
            rag_queries = [q for q in recent_queries if q.operation.value == "rag"]
            latencies = [q.latency_ms for q in rag_queries if q.latency_ms > 0]
            if latencies:
                value = np.percentile(latencies, 95)
                context["rag_queries"] = len(rag_queries)
                context["p50_latency"] = np.percentile(latencies, 50)
                context["p90_latency"] = np.percentile(latencies, 90)
        
        if value is None:
            return None
        
        # Calculate baseline and delta
        baseline_value = self.baselines.get(slo_def.name)
        delta_from_baseline = None
        if baseline_value is not None:
            delta_from_baseline = value - baseline_value
        
        # Determine status
        status = self._determine_slo_status(slo_def, value, baseline_value, delta_from_baseline)
        
        return SLOMeasurement(
            slo_name=slo_def.name,
            timestamp=now,
            value=value,
            baseline_value=baseline_value,
            delta_from_baseline=delta_from_baseline,
            status=status,
            sample_size=len(recent_queries),
            context=context
        )
    
    def _determine_slo_status(self, slo_def: SLODefinition, value: float,
                            baseline_value: Optional[float], 
                            delta_from_baseline: Optional[float]) -> SLOStatus:
        """Determine SLO status based on value and thresholds."""
        
        # Check for rollback condition (delta from baseline)
        if (baseline_value is not None and delta_from_baseline is not None and 
            slo_def.rollback_enabled):
            
            if slo_def.comparison == "gte":
                # Higher is better - rollback if drop is too large
                if delta_from_baseline < -slo_def.rollback_threshold:
                    return SLOStatus.VIOLATED
            else:
                # Lower is better - rollback if increase is too large
                if delta_from_baseline > slo_def.rollback_threshold:
                    return SLOStatus.VIOLATED
        
        # Check absolute thresholds
        if slo_def.comparison == "gte":
            # Higher is better
            if value < slo_def.critical_threshold:
                return SLOStatus.CRITICAL
            elif value < slo_def.warning_threshold:
                return SLOStatus.WARNING
            elif value >= slo_def.target_value:
                return SLOStatus.HEALTHY
            else:
                return SLOStatus.WARNING
        else:
            # Lower is better
            if value > slo_def.critical_threshold:
                return SLOStatus.CRITICAL
            elif value > slo_def.warning_threshold:
                return SLOStatus.WARNING
            elif value <= slo_def.target_value:
                return SLOStatus.HEALTHY
            else:
                return SLOStatus.WARNING
    
    def _update_baselines(self):
        """Update baseline values from historical data."""
        for slo_name, slo_def in self.slos.items():
            baseline_window = datetime.utcnow() - timedelta(hours=slo_def.baseline_window_hours)
            
            # Get measurements from baseline window (excluding recent ones)
            recent_cutoff = datetime.utcnow() - timedelta(hours=1)  # Exclude last hour
            baseline_measurements = [
                m for m in self.baseline_measurements[slo_name]
                if baseline_window <= m.timestamp < recent_cutoff
            ]
            
            if len(baseline_measurements) >= 10:  # Need minimum samples
                baseline_values = [m.value for m in baseline_measurements]
                self.baselines[slo_name] = statistics.median(baseline_values)
                
                logger.debug(f"Updated baseline for {slo_name}: "
                           f"{self.baselines[slo_name]:.3f} "
                           f"(from {len(baseline_measurements)} samples)")
    
    def _check_violations(self):
        """Check for SLO violations and trigger alerts/rollbacks."""
        for slo_name, slo_def in self.slos.items():
            if not self.measurements[slo_name]:
                continue
            
            latest_measurement = self.measurements[slo_name][-1]
            
            # Check if alert conditions have changed
            current_alert = self.active_alerts.get(slo_name)
            new_severity = self._get_alert_severity(latest_measurement.status)
            
            if new_severity == AlertSeverity.INFO and current_alert:
                # SLO recovered
                self._resolve_alert(slo_name)
                
            elif new_severity != AlertSeverity.INFO:
                # SLO violation or escalation
                if not current_alert or current_alert.severity != new_severity:
                    self._trigger_alert(slo_def, latest_measurement, new_severity)
            
            # Check for rollback condition
            if (latest_measurement.status == SLOStatus.VIOLATED and 
                slo_def.rollback_enabled and not self.rollback_triggered):
                self._trigger_rollback(slo_def, latest_measurement)
    
    def _get_alert_severity(self, status: SLOStatus) -> AlertSeverity:
        """Get alert severity from SLO status."""
        if status == SLOStatus.VIOLATED:
            return AlertSeverity.EMERGENCY
        elif status == SLOStatus.CRITICAL:
            return AlertSeverity.CRITICAL
        elif status == SLOStatus.WARNING:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    def _trigger_alert(self, slo_def: SLODefinition, measurement: SLOMeasurement,
                      severity: AlertSeverity):
        """Trigger SLO alert."""
        alert = SLOAlert(
            slo_name=slo_def.name,
            severity=severity,
            message=self._generate_alert_message(slo_def, measurement, severity),
            current_value=measurement.value,
            target_value=slo_def.target_value,
            baseline_value=measurement.baseline_value,
            delta_from_baseline=measurement.delta_from_baseline,
            timestamp=measurement.timestamp,
            measurement=measurement,
            context=measurement.context
        )
        
        self.active_alerts[slo_def.name] = alert
        
        # Fire alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        logger.warning(f"SLO ALERT [{severity.value.upper()}]: {alert.message}")
    
    def _resolve_alert(self, slo_name: str):
        """Resolve active SLO alert."""
        if slo_name in self.active_alerts:
            alert = self.active_alerts.pop(slo_name)
            logger.info(f"SLO RECOVERED: {slo_name} - {alert.message}")
    
    def _generate_alert_message(self, slo_def: SLODefinition, measurement: SLOMeasurement,
                              severity: AlertSeverity) -> str:
        """Generate human-readable alert message."""
        if measurement.baseline_value is not None and measurement.delta_from_baseline is not None:
            delta_str = f" (Δ{measurement.delta_from_baseline:+.3f} from baseline {measurement.baseline_value:.3f})"
        else:
            delta_str = ""
        
        return (f"{slo_def.description}: {measurement.value:.3f} "
               f"{'<' if slo_def.comparison == 'lte' else '>'} "
               f"{slo_def.target_value:.3f}{delta_str}")
    
    def _trigger_rollback(self, slo_def: SLODefinition, measurement: SLOMeasurement):
        """Trigger system rollback due to SLO violation."""
        if self.rollback_triggered:
            return
        
        self.rollback_triggered = True
        logger.error(f"AUTOMATIC ROLLBACK TRIGGERED by SLO violation: {slo_def.name}")
        
        rollback_context = {
            "trigger_slo": slo_def.name,
            "trigger_measurement": asdict(measurement),
            "all_active_alerts": [asdict(alert) for alert in self.active_alerts.values()],
            "timestamp": datetime.utcnow().isoformat(),
            "rollback_reason": f"SLO {slo_def.name} violated: {measurement.value:.3f} "
                              f"(baseline: {measurement.baseline_value:.3f}, "
                              f"delta: {measurement.delta_from_baseline:+.3f})"
        }
        
        # Fire rollback callbacks
        for callback in self.rollback_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(rollback_context))
                else:
                    callback(rollback_context)
            except Exception as e:
                logger.error(f"Rollback callback failed: {e}")
        
        # Save rollback event
        rollback_file = self.storage_dir / f"rollback-{int(time.time())}.json"
        with open(rollback_file, 'w') as f:
            json.dump(rollback_context, f, indent=2, default=str)
    
    def get_slo_status(self) -> Dict[str, Any]:
        """Get current SLO status summary."""
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "monitoring_enabled": self.monitoring_enabled,
            "rollback_triggered": self.rollback_triggered,
            "slos": {},
            "active_alerts": len(self.active_alerts),
            "alert_breakdown": defaultdict(int)
        }
        
        for slo_name, slo_def in self.slos.items():
            latest_measurement = None
            if self.measurements[slo_name]:
                latest_measurement = self.measurements[slo_name][-1]
            
            baseline_value = self.baselines.get(slo_name)
            
            slo_status = {
                "definition": asdict(slo_def),
                "latest_measurement": asdict(latest_measurement) if latest_measurement else None,
                "baseline_value": baseline_value,
                "measurement_count": len(self.measurements[slo_name]),
                "baseline_measurement_count": len(self.baseline_measurements[slo_name])
            }
            
            if slo_name in self.active_alerts:
                slo_status["active_alert"] = asdict(self.active_alerts[slo_name])
                status["alert_breakdown"][self.active_alerts[slo_name].severity.value] += 1
            
            status["slos"][slo_name] = slo_status
        
        return status
    
    def get_slo_history(self, slo_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get SLO measurement history."""
        if slo_name not in self.measurements:
            return []
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        history = [
            asdict(measurement) for measurement in self.measurements[slo_name]
            if measurement.timestamp >= cutoff
        ]
        
        return history
    
    def _export_measurements(self):
        """Export SLO measurements to storage."""
        export_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "measurements": {},
            "baselines": self.baselines,
            "active_alerts": [asdict(alert) for alert in self.active_alerts.values()]
        }
        
        # Export recent measurements (last hour)
        cutoff = datetime.utcnow() - timedelta(hours=1)
        for slo_name, measurements in self.measurements.items():
            recent = [
                asdict(m) for m in measurements 
                if m.timestamp >= cutoff
            ]
            export_data["measurements"][slo_name] = recent
        
        # Save to file
        export_file = self.storage_dir / f"slo-measurements-{datetime.utcnow().strftime('%Y%m%d%H')}.json"
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def reset_rollback_state(self):
        """Reset rollback state (for testing or manual recovery)."""
        self.rollback_triggered = False
        logger.info("Rollback state reset - monitoring resumed")
    
    def manual_rollback(self, reason: str):
        """Manually trigger rollback."""
        if self.rollback_triggered:
            logger.warning("Rollback already triggered")
            return
        
        self.rollback_triggered = True
        logger.error(f"MANUAL ROLLBACK TRIGGERED: {reason}")
        
        rollback_context = {
            "trigger_type": "manual",
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
            "all_active_alerts": [asdict(alert) for alert in self.active_alerts.values()]
        }
        
        # Fire rollback callbacks
        for callback in self.rollback_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(rollback_context))
                else:
                    callback(rollback_context)
            except Exception as e:
                logger.error(f"Rollback callback failed: {e}")


async def main():
    """Example usage of SLO monitoring system."""
    from telemetry_system import create_telemetry_system, QueryOperation
    
    # Create telemetry system
    telemetry_config = {
        "max_history": 1000,
        "storage_dir": "/tmp/slo_demo_telemetry"
    }
    collector, _ = create_telemetry_system(telemetry_config)
    
    # Create SLO monitor
    slo_config = {
        "monitoring_interval_seconds": 10,
        "storage_dir": "/tmp/slo_demo_monitoring"
    }
    slo_monitor = SLOMonitor(slo_config, collector)
    
    # Add alert callback
    def alert_callback(alert: SLOAlert):
        logger.error(f"SLO ALERT: {alert.severity.value} - {alert.message}")
    
    # Add rollback callback
    def rollback_callback(context: Dict[str, Any]):
        logger.error(f"ROLLBACK TRIGGERED: {context['rollback_reason']}")
    
    slo_monitor.add_alert_callback(alert_callback)
    slo_monitor.add_rollback_callback(rollback_callback)
    
    # Start monitoring
    slo_monitor.start_monitoring()
    
    # Simulate queries with varying quality
    for i in range(50):
        query_id = f"query_{i}"
        operation = QueryOperation.SEARCH if i % 2 == 0 else QueryOperation.RAG
        scenario = "core" if i % 3 == 0 else "extended"
        
        # Start query
        telemetry = collector.record_query_start(
            query_id=query_id,
            operation=operation,
            scenario=scenario,
            repo=f"repo-{i % 2 + 1}",
            query_text=f"Test query {i}"
        )
        
        # Simulate processing with degrading quality over time
        quality_factor = max(0.5, 1.0 - (i / 100))  # Gradual degradation
        
        collector.update_query_metrics(
            query_id=query_id,
            ess_score=0.9 * quality_factor,
            answerable_at_k=0.8 * quality_factor,
            span_recall=0.7 * quality_factor,
            tokens_used=150,
            latency_ms=150 + (i * 5)  # Increasing latency
        )
        
        # Complete query
        collector.record_query_complete(query_id)
        
        await asyncio.sleep(0.2)
        
        # Check status every 10 queries
        if i % 10 == 9:
            status = slo_monitor.get_slo_status()
            logger.info(f"SLO Status: {status['active_alerts']} active alerts, "
                       f"rollback_triggered={status['rollback_triggered']}")
    
    # Wait for final monitoring cycle
    await asyncio.sleep(15)
    
    # Get final status
    final_status = slo_monitor.get_slo_status()
    logger.info(f"Final SLO status: {json.dumps(final_status, indent=2, default=str)}")
    
    # Cleanup
    slo_monitor.stop_monitoring()
    collector.stop_processing()
    
    logger.info("SLO monitoring demonstration complete")


if __name__ == "__main__":
    asyncio.run(main())