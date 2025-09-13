#!/usr/bin/env python3
"""
Runtime Gates & Kill Switches for Lens Search
Implements pointer-first Extract path enforcement, generative fallback kill-switch,
and hard rollback triggers for SLO violations.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import threading
import weakref

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GateStatus(Enum):
    """Status of a runtime gate."""
    OPEN = "open"           # Gate allows traffic
    CLOSED = "closed"       # Gate blocks traffic
    DEGRADED = "degraded"   # Gate allows limited traffic
    BYPASSED = "bypassed"   # Gate is bypassed (emergency)


class KillSwitchType(Enum):
    """Types of kill switches."""
    GENERATIVE_FALLBACK = "generative_fallback"
    POINTER_EXTRACT = "pointer_extract"
    RAG_PIPELINE = "rag_pipeline"
    SEARCH_INDEX = "search_index"
    FULL_SYSTEM = "full_system"


@dataclass
class GateMetrics:
    """Metrics for a runtime gate."""
    gate_name: str
    status: GateStatus
    requests_allowed: int = 0
    requests_blocked: int = 0
    last_check: Optional[datetime] = None
    last_violation: Optional[datetime] = None
    violation_count: int = 0
    bypass_reason: Optional[str] = None


@dataclass
class SLOViolation:
    """SLO violation event."""
    metric_name: str
    current_value: float
    threshold: float
    severity: str  # "warning", "critical"
    timestamp: datetime
    context: Dict[str, Any]


class PointerExtractGate:
    """Enforces pointer-first Extract path with fallback controls."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.status = GateStatus.OPEN
        self.metrics = GateMetrics("pointer_extract", self.status)
        
        # Pointer-first enforcement settings
        self.require_pointer_first = config.get("require_pointer_first", True)
        self.max_generative_ratio = config.get("max_generative_ratio", 0.1)  # 10% max
        self.pointer_timeout_ms = config.get("pointer_timeout_ms", 100)
        
        # Tracking
        self.pointer_attempts = 0
        self.generative_fallbacks = 0
        self.total_extracts = 0
        
        # Circuit breaker
        self.failure_threshold = config.get("failure_threshold", 0.5)
        self.failure_window_minutes = config.get("failure_window_minutes", 5)
        self.recent_failures = []
        
        logger.info(f"PointerExtractGate initialized: pointer_first={self.require_pointer_first}")
    
    async def check_extract_request(self, extract_context: Dict[str, Any]) -> Dict[str, Any]:
        """Check extract request and enforce pointer-first path."""
        self.total_extracts += 1
        
        # Check if gate is open
        if self.status == GateStatus.CLOSED:
            return {
                "allowed": False,
                "reason": "pointer_extract_gate_closed",
                "fallback": "none"
            }
        
        # Check pointer-first requirement
        if self.require_pointer_first:
            pointer_result = await self._attempt_pointer_extract(extract_context)
            
            if pointer_result["success"]:
                self.pointer_attempts += 1
                self.metrics.requests_allowed += 1
                return {
                    "allowed": True,
                    "method": "pointer",
                    "result": pointer_result["data"]
                }
            else:
                # Pointer failed, check if generative fallback is allowed
                current_ratio = self.generative_fallbacks / max(self.total_extracts, 1)
                
                if current_ratio >= self.max_generative_ratio:
                    self.metrics.requests_blocked += 1
                    await self._record_failure("generative_ratio_exceeded")
                    return {
                        "allowed": False,
                        "reason": "generative_ratio_exceeded",
                        "current_ratio": current_ratio,
                        "max_ratio": self.max_generative_ratio
                    }
                
                # Allow generative fallback
                self.generative_fallbacks += 1
                self.metrics.requests_allowed += 1
                return {
                    "allowed": True,
                    "method": "generative_fallback",
                    "warning": "pointer_extract_failed"
                }
        else:
            # Pointer-first not required, allow any method
            self.metrics.requests_allowed += 1
            return {
                "allowed": True,
                "method": "any"
            }
    
    async def _attempt_pointer_extract(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt pointer-based extraction with timeout."""
        try:
            # Simulate pointer extraction with timeout
            start_time = time.time()
            
            # TODO: Implement actual pointer extraction logic
            # For now, simulate success/failure based on context
            await asyncio.sleep(0.05)  # Simulate processing time
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            if elapsed_ms > self.pointer_timeout_ms:
                return {
                    "success": False,
                    "reason": "timeout",
                    "elapsed_ms": elapsed_ms
                }
            
            # Simulate pointer extraction success/failure
            # In real implementation, this would be actual pointer logic
            success_rate = context.get("pointer_success_rate", 0.85)
            import random
            success = random.random() < success_rate
            
            if success:
                return {
                    "success": True,
                    "data": {"extracted_spans": ["span1", "span2"], "confidence": 0.9},
                    "elapsed_ms": elapsed_ms
                }
            else:
                return {
                    "success": False,
                    "reason": "no_pointer_match",
                    "elapsed_ms": elapsed_ms
                }
                
        except Exception as e:
            logger.error(f"Pointer extraction error: {e}")
            return {
                "success": False,
                "reason": "exception",
                "error": str(e)
            }
    
    async def _record_failure(self, reason: str):
        """Record extraction failure for circuit breaker."""
        now = datetime.utcnow()
        self.recent_failures.append((now, reason))
        
        # Clean old failures outside window
        cutoff = now - timedelta(minutes=self.failure_window_minutes)
        self.recent_failures = [(ts, r) for ts, r in self.recent_failures if ts > cutoff]
        
        # Check if we should trip circuit breaker
        failure_rate = len(self.recent_failures) / max(self.total_extracts, 1)
        if failure_rate >= self.failure_threshold:
            logger.warning(f"Pointer extract gate failure threshold exceeded: {failure_rate:.3f}")
            self.status = GateStatus.DEGRADED
            self.metrics.violation_count += 1
            self.metrics.last_violation = now
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current gate metrics."""
        return {
            **asdict(self.metrics),
            "pointer_attempts": self.pointer_attempts,
            "generative_fallbacks": self.generative_fallbacks,
            "total_extracts": self.total_extracts,
            "generative_ratio": self.generative_fallbacks / max(self.total_extracts, 1),
            "recent_failures": len(self.recent_failures)
        }


class GenerativeFallbackKillSwitch:
    """Kill switch for generative fallback functionality."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = True
        self.kill_switch_active = False
        self.metrics = GateMetrics("generative_fallback", GateStatus.OPEN)
        
        # Kill switch triggers
        self.max_error_rate = config.get("max_error_rate", 0.2)  # 20%
        self.max_latency_p95 = config.get("max_latency_p95", 5000)  # 5s
        self.min_quality_score = config.get("min_quality_score", 0.6)
        
        # Monitoring window
        self.monitoring_window_minutes = config.get("monitoring_window_minutes", 10)
        self.recent_requests = []
        
        logger.info("GenerativeFallbackKillSwitch initialized")
    
    async def check_fallback_request(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if generative fallback is allowed."""
        if self.kill_switch_active:
            self.metrics.requests_blocked += 1
            return {
                "allowed": False,
                "reason": "generative_fallback_killed",
                "kill_switch_active": True
            }
        
        if not self.enabled:
            self.metrics.requests_blocked += 1
            return {
                "allowed": False,
                "reason": "generative_fallback_disabled"
            }
        
        self.metrics.requests_allowed += 1
        return {
            "allowed": True,
            "monitoring": True
        }
    
    async def record_fallback_result(self, result: Dict[str, Any]):
        """Record generative fallback result for monitoring."""
        now = datetime.utcnow()
        self.recent_requests.append({
            "timestamp": now,
            "success": result.get("success", False),
            "error": result.get("error"),
            "latency_ms": result.get("latency_ms", 0),
            "quality_score": result.get("quality_score", 0.0)
        })
        
        # Clean old requests outside monitoring window
        cutoff = now - timedelta(minutes=self.monitoring_window_minutes)
        self.recent_requests = [r for r in self.recent_requests if r["timestamp"] > cutoff]
        
        # Check kill switch triggers
        await self._check_kill_triggers()
    
    async def _check_kill_triggers(self):
        """Check if kill switch should be activated."""
        if not self.recent_requests:
            return
        
        # Calculate metrics
        total_requests = len(self.recent_requests)
        error_count = sum(1 for r in self.recent_requests if not r["success"])
        error_rate = error_count / total_requests
        
        latencies = [r["latency_ms"] for r in self.recent_requests if r["latency_ms"] > 0]
        if latencies:
            latencies.sort()
            p95_latency = latencies[int(0.95 * len(latencies))]
        else:
            p95_latency = 0
        
        quality_scores = [r["quality_score"] for r in self.recent_requests if r["quality_score"] > 0]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 1.0
        
        # Check triggers
        triggers = []
        
        if error_rate > self.max_error_rate:
            triggers.append(f"error_rate_{error_rate:.3f}")
        
        if p95_latency > self.max_latency_p95:
            triggers.append(f"p95_latency_{p95_latency:.1f}ms")
        
        if avg_quality < self.min_quality_score:
            triggers.append(f"quality_score_{avg_quality:.3f}")
        
        if triggers:
            logger.error(f"Generative fallback kill switch ACTIVATED: {triggers}")
            self.kill_switch_active = True
            self.metrics.status = GateStatus.CLOSED
            self.metrics.last_violation = datetime.utcnow()
            self.metrics.violation_count += 1
    
    def manual_kill(self, reason: str):
        """Manually activate kill switch."""
        logger.warning(f"Generative fallback MANUALLY KILLED: {reason}")
        self.kill_switch_active = True
        self.metrics.status = GateStatus.CLOSED
        self.metrics.bypass_reason = f"manual_kill: {reason}"
    
    def revive_switch(self, reason: str):
        """Manually reactivate generative fallback."""
        logger.info(f"Generative fallback REVIVED: {reason}")
        self.kill_switch_active = False
        self.metrics.status = GateStatus.OPEN
        self.metrics.bypass_reason = None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current kill switch metrics."""
        recent_count = len(self.recent_requests)
        if recent_count > 0:
            error_count = sum(1 for r in self.recent_requests if not r["success"])
            error_rate = error_count / recent_count
            
            latencies = [r["latency_ms"] for r in self.recent_requests if r["latency_ms"] > 0]
            p95_latency = sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0
            
            quality_scores = [r["quality_score"] for r in self.recent_requests if r["quality_score"] > 0]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        else:
            error_rate = 0
            p95_latency = 0
            avg_quality = 0
        
        return {
            **asdict(self.metrics),
            "kill_switch_active": self.kill_switch_active,
            "enabled": self.enabled,
            "recent_requests": recent_count,
            "error_rate": error_rate,
            "p95_latency_ms": p95_latency,
            "avg_quality_score": avg_quality
        }


class SLORollbackTrigger:
    """Monitors SLOs and triggers hard rollbacks on violations."""
    
    def __init__(self, config: Dict[str, Any], rollback_callback: Callable):
        self.config = config
        self.rollback_callback = rollback_callback
        self.monitoring = True
        self.rollback_triggered = False
        
        # SLO thresholds
        self.slo_thresholds = {
            "pass_rate_core": config.get("pass_rate_core_min", 0.85),
            "answerable_at_k": config.get("answerable_at_k_min", 0.7),
            "span_recall": config.get("span_recall_min", 0.5),
            "p95_latency_ms": config.get("p95_latency_max", 350),
            "error_rate": config.get("error_rate_max", 0.05)
        }
        
        # Violation tracking
        self.violation_window_minutes = config.get("violation_window_minutes", 15)
        self.max_violations_per_window = config.get("max_violations_per_window", 3)
        self.recent_violations = []
        
        # Monitoring thread
        self.monitor_thread = None
        self.stop_monitoring = threading.Event()
        
        logger.info(f"SLORollbackTrigger initialized with thresholds: {self.slo_thresholds}")
    
    def start_monitoring(self):
        """Start SLO monitoring in background thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return
        
        self.stop_monitoring.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("SLO monitoring started")
    
    def stop_monitoring_thread(self):
        """Stop SLO monitoring thread."""
        self.stop_monitoring.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("SLO monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring.wait(30):  # Check every 30 seconds
            try:
                asyncio.run(self._check_slos())
            except Exception as e:
                logger.error(f"SLO monitoring error: {e}")
    
    async def _check_slos(self):
        """Check current SLO metrics against thresholds."""
        if self.rollback_triggered or not self.monitoring:
            return
        
        # TODO: Integrate with actual metrics collection
        # For now, simulate SLO metrics
        current_metrics = await self._collect_current_metrics()
        
        violations = []
        
        for metric_name, threshold in self.slo_thresholds.items():
            current_value = current_metrics.get(metric_name, 0)
            
            if metric_name in ["pass_rate_core", "answerable_at_k", "span_recall"]:
                # Higher is better
                if current_value < threshold:
                    severity = "critical" if current_value < threshold * 0.9 else "warning"
                    violations.append(SLOViolation(
                        metric_name=metric_name,
                        current_value=current_value,
                        threshold=threshold,
                        severity=severity,
                        timestamp=datetime.utcnow(),
                        context=current_metrics
                    ))
            else:
                # Lower is better (latency, error rate)
                if current_value > threshold:
                    severity = "critical" if current_value > threshold * 1.5 else "warning"
                    violations.append(SLOViolation(
                        metric_name=metric_name,
                        current_value=current_value,
                        threshold=threshold,
                        severity=severity,
                        timestamp=datetime.utcnow(),
                        context=current_metrics
                    ))
        
        # Process violations
        for violation in violations:
            await self._handle_violation(violation)
    
    async def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current SLO metrics."""
        # TODO: Integrate with actual telemetry system
        # For now, simulate metrics with some variation
        import random
        
        base_metrics = {
            "pass_rate_core": 0.87,
            "answerable_at_k": 0.75,
            "span_recall": 0.68,
            "p95_latency_ms": 180,
            "error_rate": 0.02
        }
        
        # Add some realistic variation
        return {
            metric: max(0, value + random.uniform(-0.1 * value, 0.1 * value))
            for metric, value in base_metrics.items()
        }
    
    async def _handle_violation(self, violation: SLOViolation):
        """Handle SLO violation."""
        now = datetime.utcnow()
        self.recent_violations.append(violation)
        
        # Clean old violations
        cutoff = now - timedelta(minutes=self.violation_window_minutes)
        self.recent_violations = [v for v in self.recent_violations if v.timestamp > cutoff]
        
        # Log violation
        logger.warning(f"SLO violation: {violation.metric_name}={violation.current_value:.3f}, "
                      f"threshold={violation.threshold:.3f}, severity={violation.severity}")
        
        # Check if rollback should be triggered
        critical_violations = [v for v in self.recent_violations if v.severity == "critical"]
        total_violations = len(self.recent_violations)
        
        if (len(critical_violations) >= 1 or 
            total_violations >= self.max_violations_per_window):
            
            await self._trigger_rollback(violation)
    
    async def _trigger_rollback(self, triggering_violation: SLOViolation):
        """Trigger hard rollback due to SLO violations."""
        if self.rollback_triggered:
            return
        
        self.rollback_triggered = True
        logger.error(f"HARD ROLLBACK TRIGGERED by SLO violation: {triggering_violation.metric_name}")
        
        # Build rollback context
        rollback_context = {
            "trigger": "slo_violation",
            "triggering_violation": asdict(triggering_violation),
            "recent_violations": [asdict(v) for v in self.recent_violations],
            "violation_count": len(self.recent_violations),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Execute rollback callback
        try:
            if asyncio.iscoroutinefunction(self.rollback_callback):
                await self.rollback_callback(rollback_context)
            else:
                self.rollback_callback(rollback_context)
        except Exception as e:
            logger.error(f"Rollback callback failed: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current SLO monitoring metrics."""
        return {
            "monitoring": self.monitoring,
            "rollback_triggered": self.rollback_triggered,
            "slo_thresholds": self.slo_thresholds,
            "recent_violations": len(self.recent_violations),
            "violation_window_minutes": self.violation_window_minutes,
            "max_violations_per_window": self.max_violations_per_window
        }


class RuntimeGateOrchestrator:
    """Orchestrates all runtime gates and kill switches."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gates = {}
        self.kill_switches = {}
        self.slo_monitor = None
        
        # Initialize gates
        self._initialize_gates()
        
        # Start monitoring
        self._start_monitoring()
        
        logger.info("RuntimeGateOrchestrator initialized")
    
    def _initialize_gates(self):
        """Initialize all runtime gates and kill switches."""
        # Pointer Extract Gate
        self.gates["pointer_extract"] = PointerExtractGate(
            self.config.get("pointer_extract", {})
        )
        
        # Generative Fallback Kill Switch
        self.kill_switches["generative_fallback"] = GenerativeFallbackKillSwitch(
            self.config.get("generative_fallback", {})
        )
        
        # SLO Rollback Trigger
        self.slo_monitor = SLORollbackTrigger(
            self.config.get("slo_monitoring", {}),
            rollback_callback=self._handle_rollback
        )
    
    def _start_monitoring(self):
        """Start all monitoring components."""
        if self.slo_monitor:
            self.slo_monitor.start_monitoring()
    
    async def check_extract_request(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check extract request against all gates."""
        # Check pointer extract gate
        pointer_result = await self.gates["pointer_extract"].check_extract_request(context)
        
        if not pointer_result["allowed"]:
            return pointer_result
        
        # If using generative fallback, check kill switch
        if pointer_result.get("method") == "generative_fallback":
            fallback_result = await self.kill_switches["generative_fallback"].check_fallback_request(context)
            if not fallback_result["allowed"]:
                return fallback_result
        
        return pointer_result
    
    async def record_extract_result(self, result: Dict[str, Any]):
        """Record extract result for monitoring."""
        if result.get("method") == "generative_fallback":
            await self.kill_switches["generative_fallback"].record_fallback_result(result)
    
    async def _handle_rollback(self, context: Dict[str, Any]):
        """Handle rollback triggered by SLO violations."""
        logger.error(f"SYSTEM ROLLBACK INITIATED: {context}")
        
        # TODO: Integrate with actual rollback system
        # This would typically:
        # 1. Close all gates
        # 2. Activate kill switches
        # 3. Route traffic to previous version
        # 4. Alert operations team
        
        # For now, close all gates
        for gate in self.gates.values():
            if hasattr(gate, 'status'):
                gate.status = GateStatus.CLOSED
        
        for kill_switch in self.kill_switches.values():
            if hasattr(kill_switch, 'kill_switch_active'):
                kill_switch.kill_switch_active = True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "gates": {},
            "kill_switches": {},
            "slo_monitoring": {}
        }
        
        # Gate metrics
        for name, gate in self.gates.items():
            status["gates"][name] = gate.get_metrics()
        
        # Kill switch metrics
        for name, kill_switch in self.kill_switches.items():
            status["kill_switches"][name] = kill_switch.get_metrics()
        
        # SLO monitoring
        if self.slo_monitor:
            status["slo_monitoring"] = self.slo_monitor.get_metrics()
        
        return status
    
    def emergency_shutdown(self, reason: str):
        """Emergency shutdown of all systems."""
        logger.error(f"EMERGENCY SHUTDOWN: {reason}")
        
        # Close all gates
        for gate in self.gates.values():
            if hasattr(gate, 'status'):
                gate.status = GateStatus.CLOSED
        
        # Activate all kill switches
        for kill_switch in self.kill_switches.values():
            if hasattr(kill_switch, 'manual_kill'):
                kill_switch.manual_kill(f"emergency_shutdown: {reason}")
        
        # Stop monitoring
        if self.slo_monitor:
            self.slo_monitor.stop_monitoring_thread()
    
    def __del__(self):
        """Cleanup on destruction."""
        if self.slo_monitor:
            self.slo_monitor.stop_monitoring_thread()


async def main():
    """Example usage of runtime gates system."""
    config = {
        "pointer_extract": {
            "require_pointer_first": True,
            "max_generative_ratio": 0.1,
            "pointer_timeout_ms": 100
        },
        "generative_fallback": {
            "max_error_rate": 0.2,
            "max_latency_p95": 5000,
            "min_quality_score": 0.6
        },
        "slo_monitoring": {
            "pass_rate_core_min": 0.85,
            "answerable_at_k_min": 0.7,
            "span_recall_min": 0.5,
            "p95_latency_max": 350,
            "error_rate_max": 0.05
        }
    }
    
    # Initialize orchestrator
    orchestrator = RuntimeGateOrchestrator(config)
    
    # Simulate some requests
    for i in range(10):
        extract_context = {
            "query": f"test query {i}",
            "pointer_success_rate": 0.8  # Simulate 80% pointer success
        }
        
        # Check request
        result = await orchestrator.check_extract_request(extract_context)
        logger.info(f"Request {i}: {result}")
        
        # Record result (simulate)
        if result["allowed"]:
            extract_result = {
                "method": result.get("method", "pointer"),
                "success": True,
                "latency_ms": 150,
                "quality_score": 0.85
            }
            await orchestrator.record_extract_result(extract_result)
        
        await asyncio.sleep(1)
    
    # Get system status
    status = orchestrator.get_system_status()
    logger.info(f"System status: {json.dumps(status, indent=2, default=str)}")
    
    # Cleanup
    orchestrator.emergency_shutdown("test_complete")


if __name__ == "__main__":
    asyncio.run(main())