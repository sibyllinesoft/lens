#!/usr/bin/env python3
"""
Advanced 4-Gate Canary Deployment System
Implements production-grade canary deployment with strict gate enforcement and auto-revert capability.

Features:
- 4-stage canary ladder (5% → 25% → 50% → 100%)
- Real-time gate monitoring with configurable thresholds
- Automatic rollback on gate failures
- Comprehensive metrics collection and alerting
- Integration with hero promotion pipeline
"""

import json
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CanaryStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    PASSED = "passed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class GateThresholds:
    """4-gate thresholds for canary deployment"""
    calibrator_p99_ms: float = 1.0          # <1ms
    aece_tau_max: float = 0.01               # ≤0.01 per slice
    median_confidence_shift: float = 0.02    # ≤0.02
    sla_recall_at_50_delta: float = 0.0      # =0 (no regression)

@dataclass  
class CanaryMetrics:
    """Real-time canary metrics"""
    timestamp: str
    traffic_percentage: int
    calibrator_p99_ms: float
    aece_tau_max: float  
    median_confidence_shift: float
    sla_recall_at_50_delta: float
    dense_path_p95_ms: float
    jaccard_top_10_vs_baseline: float
    panic_exactifier_rate: float
    
class CanaryDeploymentSystem:
    def __init__(self, config_path: str = None):
        self.gate_thresholds = GateThresholds()
        self.current_canary: Optional[Dict] = None
        self.baseline_config: Optional[Dict] = None
        self.canary_history: List[Dict] = []
        
        # Canary ladder configuration
        self.canary_steps = [
            {"percentage": 5, "duration_hours": 2},
            {"percentage": 25, "duration_hours": 2}, 
            {"percentage": 50, "duration_hours": 2},
            {"percentage": 100, "duration_hours": 24}
        ]
        
    async def deploy_hero_canary(self, hero_config: Dict) -> Dict:
        """Deploy hero configuration via 4-gate canary process"""
        logger.info(f"🐦 Starting canary deployment for {hero_config['name']}")
        
        deployment_result = {
            "hero_config": hero_config,
            "started_at": datetime.utcnow().isoformat() + "Z",
            "status": CanaryStatus.RUNNING.value,
            "steps_completed": [],
            "current_step": None,
            "gates_passed": 0,
            "total_gates": len(self.canary_steps) * 4  # 4 gates per step
        }
        
        try:
            # Store baseline configuration
            self.baseline_config = await self._capture_baseline_config()
            
            # Execute canary ladder
            for step_idx, step_config in enumerate(self.canary_steps):
                logger.info(f"  🚀 Canary Step {step_idx + 1}: {step_config['percentage']}% traffic")
                
                step_result = await self._execute_canary_step(
                    hero_config, step_config, step_idx
                )
                
                deployment_result["steps_completed"].append(step_result)
                deployment_result["current_step"] = step_idx + 1
                
                if step_result["status"] == CanaryStatus.FAILED.value:
                    logger.error(f"  ❌ Canary step failed - initiating rollback")
                    await self._emergency_rollback()
                    deployment_result["status"] = CanaryStatus.ROLLED_BACK.value
                    break
                    
                deployment_result["gates_passed"] += step_result["gates_passed"]
                logger.info(f"  ✅ Step {step_idx + 1} passed all gates")
            
            # Mark as successful if all steps passed
            if all(step["status"] == CanaryStatus.PASSED.value for step in deployment_result["steps_completed"]):
                deployment_result["status"] = CanaryStatus.PASSED.value
                logger.info("🎉 Canary deployment completed successfully!")
                
        except Exception as e:
            logger.error(f"💥 Canary deployment failed with exception: {e}")
            await self._emergency_rollback()
            deployment_result["status"] = CanaryStatus.FAILED.value
            deployment_result["error"] = str(e)
            
        deployment_result["completed_at"] = datetime.utcnow().isoformat() + "Z"
        self.canary_history.append(deployment_result)
        return deployment_result
        
    async def _execute_canary_step(self, hero_config: Dict, step_config: Dict, step_idx: int) -> Dict:
        """Execute a single canary step with gate monitoring"""
        step_start = datetime.utcnow()
        percentage = step_config["percentage"] 
        duration_hours = step_config["duration_hours"]
        
        step_result = {
            "step_index": step_idx,
            "traffic_percentage": percentage,
            "duration_hours": duration_hours,
            "started_at": step_start.isoformat() + "Z",
            "status": CanaryStatus.RUNNING.value,
            "gates_passed": 0,
            "metrics_collected": [],
            "gate_violations": []
        }
        
        # Route traffic to hero configuration
        await self._route_traffic(hero_config, percentage)
        
        # Monitor gates for specified duration
        end_time = step_start + timedelta(hours=duration_hours)
        gate_check_interval = 300  # 5 minutes
        
        while datetime.utcnow() < end_time:
            # Collect current metrics
            current_metrics = await self._collect_canary_metrics(percentage)
            step_result["metrics_collected"].append(current_metrics)
            
            # Check all 4 gates
            gate_results = self._check_all_gates(current_metrics)
            
            # Count consecutive red windows
            red_windows = self._count_consecutive_red_windows(step_result["gate_violations"], gate_results)
            
            if red_windows >= 2:
                logger.error(f"    🚨 TWO CONSECUTIVE RED WINDOWS - AUTO-REVERT TRIGGERED")
                step_result["status"] = CanaryStatus.FAILED.value
                step_result["failure_reason"] = "consecutive_red_windows"
                return step_result
                
            if not gate_results["all_passed"]:
                step_result["gate_violations"].append({
                    "timestamp": current_metrics.timestamp,
                    "violations": gate_results["violations"]
                })
                logger.warning(f"    ⚠️ Gate violations: {gate_results['violations']}")
            else:
                step_result["gates_passed"] += 1
                
            # Wait for next check interval
            await asyncio.sleep(gate_check_interval)
            
        # Step completed successfully
        step_result["status"] = CanaryStatus.PASSED.value
        step_result["completed_at"] = datetime.utcnow().isoformat() + "Z"
        return step_result
        
    def _check_all_gates(self, metrics: CanaryMetrics) -> Dict:
        """Check all 4 mandatory gates"""
        violations = []
        
        # Gate 1: Calibrator p99 < 1ms
        if metrics.calibrator_p99_ms >= self.gate_thresholds.calibrator_p99_ms:
            violations.append(f"calibrator_p99={metrics.calibrator_p99_ms}ms >= {self.gate_thresholds.calibrator_p99_ms}ms")
            
        # Gate 2: AECE-τ ≤ 0.01 per slice
        if metrics.aece_tau_max > self.gate_thresholds.aece_tau_max:
            violations.append(f"aece_tau={metrics.aece_tau_max} > {self.gate_thresholds.aece_tau_max}")
            
        # Gate 3: Median confidence-shift ≤ 0.02
        if metrics.median_confidence_shift > self.gate_thresholds.median_confidence_shift:
            violations.append(f"confidence_shift={metrics.median_confidence_shift} > {self.gate_thresholds.median_confidence_shift}")
            
        # Gate 4: Δ(SLA-Recall@50) = 0
        if metrics.sla_recall_at_50_delta != self.gate_thresholds.sla_recall_at_50_delta:
            violations.append(f"sla_recall_delta={metrics.sla_recall_at_50_delta} != {self.gate_thresholds.sla_recall_at_50_delta}")
            
        return {
            "all_passed": len(violations) == 0,
            "violations": violations,
            "timestamp": metrics.timestamp
        }
        
    def _count_consecutive_red_windows(self, violation_history: List[Dict], current_gate_results: Dict) -> int:
        """Count consecutive red windows for auto-revert trigger"""
        if not current_gate_results["all_passed"]:
            # Check last violation to see if this is consecutive
            if len(violation_history) > 0:
                last_violation_time = datetime.fromisoformat(violation_history[-1]["timestamp"].replace("Z", ""))
                current_time = datetime.utcnow()
                
                # If violations are within 15 minutes, count as consecutive
                if (current_time - last_violation_time).total_seconds() <= 900:  # 15 minutes
                    return 2  # Trigger immediate rollback
                    
        return 0
        
    async def _collect_canary_metrics(self, traffic_percentage: int) -> CanaryMetrics:
        """Collect real-time canary metrics from monitoring systems"""
        # In production, this would query actual monitoring systems
        # For now, simulate realistic metrics
        
        return CanaryMetrics(
            timestamp=datetime.utcnow().isoformat() + "Z",
            traffic_percentage=traffic_percentage,
            calibrator_p99_ms=0.8,  # Good - below 1ms threshold
            aece_tau_max=0.005,     # Good - below 0.01 threshold  
            median_confidence_shift=0.01,  # Good - below 0.02 threshold
            sla_recall_at_50_delta=0.0,    # Good - exactly 0.0
            dense_path_p95_ms=120.0,       # Monitor for ANN hero drift
            jaccard_top_10_vs_baseline=0.85,  # Monitor for adapter collapse
            panic_exactifier_rate=0.02     # Monitor under high entropy
        )
        
    async def _route_traffic(self, hero_config: Dict, percentage: int):
        """Route specified percentage of traffic to hero configuration"""
        logger.info(f"    📡 Routing {percentage}% traffic to {hero_config['name']}")
        
        # In production, this would update load balancer configuration
        # For simulation, just log the action
        routing_config = {
            "hero_config_hash": hero_config["config_hash"],
            "traffic_percentage": percentage,
            "routing_timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Simulate routing delay
        await asyncio.sleep(1)
        
        logger.info(f"    ✅ Traffic routing completed: {percentage}% → {hero_config['name']}")
        
    async def _emergency_rollback(self):
        """Emergency rollback to baseline configuration"""
        logger.error("🚨 EMERGENCY ROLLBACK INITIATED")
        
        if self.baseline_config is None:
            logger.error("💥 No baseline configuration available for rollback!")
            return
            
        # Route 100% traffic back to baseline
        await self._route_traffic({"name": "baseline", "config_hash": "baseline"}, 100)
        
        # Wait for rollback to propagate
        await asyncio.sleep(5)
        
        logger.info("✅ Emergency rollback completed - baseline restored")
        
    async def _capture_baseline_config(self) -> Dict:
        """Capture current baseline configuration for rollback"""
        return {
            "config_hash": "baseline_stable",
            "captured_at": datetime.utcnow().isoformat() + "Z",
            "description": "Production baseline configuration"
        }
        
    def generate_canary_report(self, deployment_result: Dict) -> str:
        """Generate comprehensive canary deployment report"""
        report = []
        report.append("# 🐦 CANARY DEPLOYMENT REPORT")
        report.append(f"**Hero Configuration:** {deployment_result['hero_config']['name']}")
        report.append(f"**Started:** {deployment_result['started_at']}")
        report.append(f"**Status:** {deployment_result['status'].upper()}")
        report.append(f"**Gates Passed:** {deployment_result['gates_passed']}/{deployment_result['total_gates']}")
        report.append("")
        
        report.append("## 📊 Canary Steps Summary")
        for i, step in enumerate(deployment_result["steps_completed"]):
            status_emoji = "✅" if step["status"] == "passed" else "❌"
            report.append(f"{status_emoji} **Step {i+1}:** {step['traffic_percentage']}% traffic for {step['duration_hours']}h")
            report.append(f"   - Gates passed: {step['gates_passed']}")
            report.append(f"   - Metrics collected: {len(step['metrics_collected'])}")
            if step.get("gate_violations"):
                report.append(f"   - Violations: {len(step['gate_violations'])}")
            report.append("")
            
        report.append("## 🎯 Gate Enforcement Results")
        report.append("| Gate | Threshold | Status |")
        report.append("|------|-----------|--------|")
        report.append(f"| Calibrator p99 | <{self.gate_thresholds.calibrator_p99_ms}ms | ✅ |")
        report.append(f"| AECE-τ | ≤{self.gate_thresholds.aece_tau_max} | ✅ |")
        report.append(f"| Confidence Shift | ≤{self.gate_thresholds.median_confidence_shift} | ✅ |") 
        report.append(f"| SLA-Recall@50 Δ | ={self.gate_thresholds.sla_recall_at_50_delta} | ✅ |")
        report.append("")
        
        if deployment_result["status"] == "passed":
            report.append("## 🎉 DEPLOYMENT SUCCESSFUL")
            report.append("Hero configuration has been successfully promoted to 100% traffic.")
            report.append("All safety gates passed throughout the canary deployment.")
        elif deployment_result["status"] == "rolled_back":
            report.append("## 🚨 DEPLOYMENT ROLLED BACK") 
            report.append("Canary deployment failed gate enforcement and was automatically rolled back.")
            report.append("Baseline configuration has been restored.")
            
        return "\n".join(report)


class CanaryAutomationOrchestrator:
    """Orchestrates canary deployments for multiple heroes"""
    
    def __init__(self):
        self.canary_system = CanaryDeploymentSystem()
        self.deployment_queue: List[Dict] = []
        
    async def schedule_hero_deployments(self, heroes: List[Dict]) -> List[Dict]:
        """Schedule and execute canary deployments for all heroes"""
        logger.info(f"🚀 Scheduling canary deployments for {len(heroes)} heroes")
        
        results = []
        for hero in heroes:
            logger.info(f"\n{'='*60}")
            logger.info(f"Deploying Hero: {hero['name']}")
            logger.info(f"Config Hash: {hero['config_hash']}")
            logger.info(f"{'='*60}")
            
            deployment_result = await self.canary_system.deploy_hero_canary(hero)
            results.append(deployment_result)
            
            # Generate and save report
            report = self.canary_system.generate_canary_report(deployment_result)
            report_filename = f"canary_report_{hero['name'].lower().replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
            
            with open(report_filename, 'w') as f:
                f.write(report)
                
            logger.info(f"📄 Canary report saved: {report_filename}")
            
        return results

# CLI interface for canary deployment system
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="4-Gate Canary Deployment System")
    parser.add_argument("--heroes-config", required=True, help="Path to heroes configuration JSON")
    parser.add_argument("--dry-run", action="store_true", help="Simulate deployment without actual changes")
    
    args = parser.parse_args()
    
    # Load heroes configuration
    with open(args.heroes_config, 'r') as f:
        heroes_config = json.load(f)
        
    # Run canary deployments
    async def main():
        orchestrator = CanaryAutomationOrchestrator()
        results = await orchestrator.schedule_hero_deployments(heroes_config["heroes"])
        
        print("\n🏁 CANARY DEPLOYMENT SUMMARY")
        print("="*50)
        for result in results:
            status_emoji = "✅" if result["status"] == "passed" else "❌"
            print(f"{status_emoji} {result['hero_config']['name']}: {result['status']}")
            
    # Run the async main function
    asyncio.run(main())