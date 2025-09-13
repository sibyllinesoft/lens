#!/usr/bin/env python3
"""
Staged Rollout System for Lens Search Deployment
Implements shadow → canary → ramp deployment with automatic rollback triggers.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import hashlib
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RolloutStage(Enum):
    """Deployment stages for staged rollout."""
    SHADOW = "shadow"
    CANARY = "canary" 
    RAMP = "ramp"
    FULL = "full"
    ROLLBACK = "rollback"


class RolloutStatus(Enum):
    """Status of a rollout stage."""
    PENDING = "pending"
    RUNNING = "running"
    VALIDATING = "validating"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class RolloutConfig:
    """Configuration for staged rollout."""
    shadow_duration_hours: int = 24
    canary_percentage: float = 5.0
    canary_duration_hours: int = 48
    ramp_percentage_steps: List[float] = None
    ramp_step_duration_hours: int = 24
    slo_validation_window_hours: int = 24
    auto_rollback_enabled: bool = True
    rollback_triggers: Dict[str, float] = None
    
    def __post_init__(self):
        if self.ramp_percentage_steps is None:
            self.ramp_percentage_steps = [10.0, 25.0, 50.0, 75.0, 100.0]
        if self.rollback_triggers is None:
            self.rollback_triggers = {
                "pass_rate_core_drop": 0.05,  # 5% drop triggers rollback
                "answerable_at_k_floor": 0.7,
                "span_recall_floor": 0.5,
                "p95_latency_ceiling_ms": 350,
                "error_rate_ceiling": 0.1
            }


@dataclass
class StageMetrics:
    """Metrics collected during a rollout stage."""
    stage: RolloutStage
    start_time: datetime
    end_time: Optional[datetime] = None
    status: RolloutStatus = RolloutStatus.PENDING
    
    # Core SLO metrics
    pass_rate_core: float = 0.0
    answerable_at_k: float = 0.0
    span_recall: float = 0.0
    p95_latency_ms: float = 0.0
    error_rate: float = 0.0
    
    # Traffic metrics
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    traffic_percentage: float = 0.0
    
    # Rollback triggers
    triggered_rollbacks: List[str] = None
    
    def __post_init__(self):
        if self.triggered_rollbacks is None:
            self.triggered_rollbacks = []


@dataclass
class RolloutManifest:
    """Immutable deployment manifest with signed artifacts."""
    version: str
    release_id: str
    corpus_sha: str
    ess_thresholds: Dict[str, float]
    prompt_ids: Dict[str, str]
    model_versions: Dict[str, str]
    created_at: datetime
    signed_by: str
    green_fingerprint: str
    
    @classmethod
    def create(cls, version: str, corpus_sha: str, ess_thresholds: Dict[str, float],
               prompt_ids: Dict[str, str], model_versions: Dict[str, str], 
               signed_by: str) -> 'RolloutManifest':
        """Create a new signed manifest with green fingerprint."""
        release_id = f"lens-{version}-{int(time.time())}"
        created_at = datetime.utcnow()
        
        # Generate green fingerprint from critical components
        manifest_data = {
            "version": version,
            "corpus_sha": corpus_sha,
            "ess_thresholds": ess_thresholds,
            "prompt_ids": prompt_ids,
            "model_versions": model_versions
        }
        manifest_hash = hashlib.sha256(
            json.dumps(manifest_data, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        green_fingerprint = f"green-{manifest_hash}"
        
        return cls(
            version=version,
            release_id=release_id,
            corpus_sha=corpus_sha,
            ess_thresholds=ess_thresholds,
            prompt_ids=prompt_ids,
            model_versions=model_versions,
            created_at=created_at,
            signed_by=signed_by,
            green_fingerprint=green_fingerprint
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary for serialization."""
        return {
            **asdict(self),
            "created_at": self.created_at.isoformat()
        }
    
    def save(self, path: Path):
        """Save manifest to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class StagedRolloutOrchestrator:
    """Orchestrates staged deployment with monitoring and rollback."""
    
    def __init__(self, config: RolloutConfig, manifest: RolloutManifest,
                 deployment_dir: Path):
        self.config = config
        self.manifest = manifest
        self.deployment_dir = deployment_dir
        self.current_stage = RolloutStage.SHADOW
        self.stage_metrics: Dict[RolloutStage, StageMetrics] = {}
        self.baseline_metrics: Optional[StageMetrics] = None
        self.rollback_triggered = False
        
        # Initialize deployment directory
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.deployment_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
    async def start_rollout(self):
        """Start the staged rollout process."""
        logger.info(f"Starting staged rollout for {self.manifest.release_id}")
        logger.info(f"Green fingerprint: {self.manifest.green_fingerprint}")
        
        # Save manifest
        manifest_path = self.deployment_dir / f"manifest-{self.manifest.version}.json"
        self.manifest.save(manifest_path)
        
        try:
            # Stage 1: Shadow mode
            await self._run_shadow_stage()
            
            if not self.rollback_triggered:
                # Stage 2: Canary deployment
                await self._run_canary_stage()
            
            if not self.rollback_triggered:
                # Stage 3: Ramp deployment
                await self._run_ramp_stage()
                
            if not self.rollback_triggered:
                # Stage 4: Full deployment
                await self._run_full_stage()
                
        except Exception as e:
            logger.error(f"Rollout failed: {e}")
            await self._trigger_rollback("deployment_exception")
            raise
        
        # Generate final report
        await self._generate_rollout_report()
    
    async def _run_shadow_stage(self):
        """Run shadow mode deployment - read-only telemetry on prod traffic."""
        logger.info("Starting shadow stage")
        self.current_stage = RolloutStage.SHADOW
        
        metrics = StageMetrics(
            stage=RolloutStage.SHADOW,
            start_time=datetime.utcnow(),
            status=RolloutStatus.RUNNING,
            traffic_percentage=0.0  # No user-facing traffic
        )
        self.stage_metrics[RolloutStage.SHADOW] = metrics
        
        # Deploy to shadow environment
        await self._deploy_to_environment("shadow", 0.0)
        
        # Collect baseline metrics for comparison
        await self._collect_baseline_metrics()
        
        # Monitor shadow traffic
        await self._monitor_stage(
            duration_hours=self.config.shadow_duration_hours,
            metrics=metrics
        )
        
        if metrics.status == RolloutStatus.SUCCESS:
            logger.info("Shadow stage completed successfully")
        else:
            await self._trigger_rollback("shadow_stage_failed")
    
    async def _run_canary_stage(self):
        """Run canary deployment - small percentage of orgs/repos."""
        logger.info(f"Starting canary stage ({self.config.canary_percentage}% traffic)")
        self.current_stage = RolloutStage.CANARY
        
        metrics = StageMetrics(
            stage=RolloutStage.CANARY,
            start_time=datetime.utcnow(),
            status=RolloutStatus.RUNNING,
            traffic_percentage=self.config.canary_percentage
        )
        self.stage_metrics[RolloutStage.CANARY] = metrics
        
        # Deploy to canary environment
        await self._deploy_to_environment("canary", self.config.canary_percentage)
        
        # Monitor canary traffic with SLO validation
        await self._monitor_stage(
            duration_hours=self.config.canary_duration_hours,
            metrics=metrics,
            validate_slos=True
        )
        
        if metrics.status == RolloutStatus.SUCCESS:
            logger.info("Canary stage completed successfully")
        else:
            await self._trigger_rollback("canary_stage_failed")
    
    async def _run_ramp_stage(self):
        """Run ramp deployment - gradually scale to 100%."""
        logger.info("Starting ramp stage")
        self.current_stage = RolloutStage.RAMP
        
        for step, percentage in enumerate(self.config.ramp_percentage_steps):
            if self.rollback_triggered:
                break
                
            logger.info(f"Ramp step {step + 1}: {percentage}% traffic")
            
            metrics = StageMetrics(
                stage=RolloutStage.RAMP,
                start_time=datetime.utcnow(),
                status=RolloutStatus.RUNNING,
                traffic_percentage=percentage
            )
            
            # Deploy to production with traffic percentage
            await self._deploy_to_environment("production", percentage)
            
            # Monitor step with SLO validation
            await self._monitor_stage(
                duration_hours=self.config.ramp_step_duration_hours,
                metrics=metrics,
                validate_slos=True
            )
            
            if metrics.status != RolloutStatus.SUCCESS:
                await self._trigger_rollback(f"ramp_step_{step + 1}_failed")
                break
                
            # Store step metrics
            step_key = f"ramp_step_{step + 1}"
            self.stage_metrics[step_key] = metrics
        
        if not self.rollback_triggered:
            logger.info("Ramp stage completed successfully")
    
    async def _run_full_stage(self):
        """Run full deployment - 100% traffic."""
        logger.info("Starting full deployment stage")
        self.current_stage = RolloutStage.FULL
        
        metrics = StageMetrics(
            stage=RolloutStage.FULL,
            start_time=datetime.utcnow(),
            status=RolloutStatus.RUNNING,
            traffic_percentage=100.0
        )
        self.stage_metrics[RolloutStage.FULL] = metrics
        
        # Deploy to full production
        await self._deploy_to_environment("production", 100.0)
        
        # Monitor full deployment with extended SLO validation
        await self._monitor_stage(
            duration_hours=self.config.slo_validation_window_hours,
            metrics=metrics,
            validate_slos=True
        )
        
        if metrics.status == RolloutStatus.SUCCESS:
            logger.info("Full deployment completed successfully")
        else:
            await self._trigger_rollback("full_deployment_failed")
    
    async def _deploy_to_environment(self, environment: str, traffic_percentage: float):
        """Deploy manifest to specified environment with traffic routing."""
        logger.info(f"Deploying to {environment} with {traffic_percentage}% traffic")
        
        # Create deployment configuration
        deployment_config = {
            "environment": environment,
            "manifest": self.manifest.to_dict(),
            "traffic_percentage": traffic_percentage,
            "deployment_time": datetime.utcnow().isoformat(),
            "routing_rules": self._generate_routing_rules(traffic_percentage)
        }
        
        # Save deployment config
        config_path = self.deployment_dir / f"deployment-{environment}-{int(time.time())}.json"
        with open(config_path, 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        # TODO: Integrate with actual deployment system (Kubernetes, etc.)
        # For now, simulate deployment delay
        await asyncio.sleep(5)
        
        logger.info(f"Deployment to {environment} completed")
    
    def _generate_routing_rules(self, traffic_percentage: float) -> Dict[str, Any]:
        """Generate traffic routing rules for deployment."""
        if traffic_percentage == 0.0:
            # Shadow mode - no user traffic
            return {
                "user_traffic": 0.0,
                "shadow_traffic": 100.0,
                "canary_orgs": [],
                "canary_repos": []
            }
        elif traffic_percentage <= 10.0:
            # Canary mode - specific orgs/repos
            return {
                "user_traffic": traffic_percentage,
                "shadow_traffic": 0.0,
                "canary_orgs": self._select_canary_orgs(traffic_percentage),
                "canary_repos": self._select_canary_repos(traffic_percentage)
            }
        else:
            # Ramp/Full mode - percentage-based routing
            return {
                "user_traffic": traffic_percentage,
                "shadow_traffic": 0.0,
                "percentage_routing": True,
                "sticky_sessions": True
            }
    
    def _select_canary_orgs(self, percentage: float) -> List[str]:
        """Select organizations for canary deployment."""
        # TODO: Integrate with actual org selection logic
        # For now, return mock canary orgs
        return ["org-alpha", "org-beta"] if percentage >= 5.0 else ["org-alpha"]
    
    def _select_canary_repos(self, percentage: float) -> List[str]:
        """Select repositories for canary deployment."""
        # TODO: Integrate with actual repo selection logic
        # For now, return mock canary repos
        return ["repo-1", "repo-2", "repo-3"] if percentage >= 5.0 else ["repo-1"]
    
    async def _monitor_stage(self, duration_hours: int, metrics: StageMetrics,
                           validate_slos: bool = False):
        """Monitor a deployment stage for specified duration."""
        end_time = datetime.utcnow() + timedelta(hours=duration_hours)
        check_interval_minutes = 5
        
        logger.info(f"Monitoring {metrics.stage.value} stage for {duration_hours} hours")
        
        while datetime.utcnow() < end_time and not self.rollback_triggered:
            # Collect current metrics
            await self._collect_stage_metrics(metrics)
            
            # Validate SLOs if required
            if validate_slos:
                rollback_triggers = self._check_rollback_triggers(metrics)
                if rollback_triggers:
                    metrics.triggered_rollbacks.extend(rollback_triggers)
                    await self._trigger_rollback(f"slo_violation: {rollback_triggers}")
                    return
            
            # Log progress
            elapsed_hours = (datetime.utcnow() - metrics.start_time).total_seconds() / 3600
            logger.info(f"Stage {metrics.stage.value} progress: {elapsed_hours:.1f}/{duration_hours}h, "
                       f"queries: {metrics.total_queries}, success_rate: {metrics.pass_rate_core:.3f}")
            
            # Wait for next check
            await asyncio.sleep(check_interval_minutes * 60)
        
        # Mark stage as complete
        metrics.end_time = datetime.utcnow()
        if not self.rollback_triggered:
            metrics.status = RolloutStatus.SUCCESS
            logger.info(f"Stage {metrics.stage.value} monitoring completed successfully")
    
    async def _collect_baseline_metrics(self):
        """Collect baseline metrics from production for comparison."""
        logger.info("Collecting baseline metrics from production")
        
        # TODO: Integrate with actual metrics collection system
        # For now, simulate baseline metrics collection
        await asyncio.sleep(2)
        
        self.baseline_metrics = StageMetrics(
            stage=RolloutStage.SHADOW,  # Using shadow as baseline reference
            start_time=datetime.utcnow(),
            status=RolloutStatus.SUCCESS,
            pass_rate_core=0.87,
            answerable_at_k=0.75,
            span_recall=0.68,
            p95_latency_ms=180,
            error_rate=0.02,
            total_queries=10000,
            successful_queries=8700,
            failed_queries=200
        )
        
        logger.info(f"Baseline metrics collected: pass_rate={self.baseline_metrics.pass_rate_core:.3f}")
    
    async def _collect_stage_metrics(self, metrics: StageMetrics):
        """Collect current metrics for a stage."""
        # TODO: Integrate with actual telemetry system
        # For now, simulate metrics collection with slight variations
        import random
        
        base_pass_rate = 0.87 if self.baseline_metrics else 0.85
        base_answerable = 0.75 if self.baseline_metrics else 0.73
        base_span_recall = 0.68 if self.baseline_metrics else 0.65
        base_latency = 180 if self.baseline_metrics else 200
        
        # Add some realistic variation
        variation = random.uniform(-0.02, 0.02)
        
        metrics.pass_rate_core = max(0.0, base_pass_rate + variation)
        metrics.answerable_at_k = max(0.0, base_answerable + variation)
        metrics.span_recall = max(0.0, base_span_recall + variation)
        metrics.p95_latency_ms = max(50, base_latency + random.uniform(-20, 30))
        metrics.error_rate = max(0.0, 0.02 + random.uniform(-0.01, 0.02))
        
        # Update query counts
        metrics.total_queries += random.randint(100, 500)
        metrics.successful_queries = int(metrics.total_queries * metrics.pass_rate_core)
        metrics.failed_queries = metrics.total_queries - metrics.successful_queries
    
    def _check_rollback_triggers(self, metrics: StageMetrics) -> List[str]:
        """Check if any rollback triggers are activated."""
        triggers = []
        
        if not self.baseline_metrics:
            return triggers  # No baseline to compare against
        
        # Check pass rate drop
        pass_rate_drop = self.baseline_metrics.pass_rate_core - metrics.pass_rate_core
        if pass_rate_drop > self.config.rollback_triggers["pass_rate_core_drop"]:
            triggers.append(f"pass_rate_drop_{pass_rate_drop:.3f}")
        
        # Check answerable@k floor
        if metrics.answerable_at_k < self.config.rollback_triggers["answerable_at_k_floor"]:
            triggers.append(f"answerable_at_k_below_{metrics.answerable_at_k:.3f}")
        
        # Check span recall floor
        if metrics.span_recall < self.config.rollback_triggers["span_recall_floor"]:
            triggers.append(f"span_recall_below_{metrics.span_recall:.3f}")
        
        # Check latency ceiling
        if metrics.p95_latency_ms > self.config.rollback_triggers["p95_latency_ceiling_ms"]:
            triggers.append(f"p95_latency_above_{metrics.p95_latency_ms:.1f}ms")
        
        # Check error rate ceiling
        if metrics.error_rate > self.config.rollback_triggers["error_rate_ceiling"]:
            triggers.append(f"error_rate_above_{metrics.error_rate:.3f}")
        
        return triggers
    
    async def _trigger_rollback(self, reason: str):
        """Trigger automatic rollback due to SLO violation or failure."""
        if not self.config.auto_rollback_enabled:
            logger.warning(f"Rollback triggered but disabled: {reason}")
            return
        
        logger.error(f"ROLLBACK TRIGGERED: {reason}")
        self.rollback_triggered = True
        self.current_stage = RolloutStage.ROLLBACK
        
        # Create rollback metrics
        rollback_metrics = StageMetrics(
            stage=RolloutStage.ROLLBACK,
            start_time=datetime.utcnow(),
            status=RolloutStatus.RUNNING,
            triggered_rollbacks=[reason]
        )
        self.stage_metrics[RolloutStage.ROLLBACK] = rollback_metrics
        
        # Execute rollback
        await self._execute_rollback()
        
        rollback_metrics.end_time = datetime.utcnow()
        rollback_metrics.status = RolloutStatus.SUCCESS
        
        logger.info("Rollback completed")
    
    async def _execute_rollback(self):
        """Execute the actual rollback process."""
        logger.info("Executing rollback to previous stable version")
        
        # TODO: Integrate with actual rollback system
        # This would typically:
        # 1. Route all traffic back to previous version
        # 2. Scale down new deployment
        # 3. Verify previous version is healthy
        # 4. Clean up new deployment resources
        
        # Simulate rollback operations
        await asyncio.sleep(10)
        
        logger.info("Rollback execution completed")
    
    async def _generate_rollout_report(self):
        """Generate comprehensive rollout report."""
        logger.info("Generating rollout report")
        
        report = {
            "manifest": self.manifest.to_dict(),
            "rollout_config": asdict(self.config),
            "rollout_summary": {
                "started_at": min(m.start_time for m in self.stage_metrics.values()).isoformat(),
                "completed_at": datetime.utcnow().isoformat(),
                "rollback_triggered": self.rollback_triggered,
                "final_stage": self.current_stage.value,
                "total_stages": len(self.stage_metrics)
            },
            "stage_metrics": {
                stage: asdict(metrics) for stage, metrics in self.stage_metrics.items()
            },
            "baseline_metrics": asdict(self.baseline_metrics) if self.baseline_metrics else None
        }
        
        # Save report
        report_path = self.deployment_dir / f"rollout-report-{self.manifest.version}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate human-readable summary
        await self._generate_summary_report(report, report_path.with_suffix('.md'))
        
        logger.info(f"Rollout report saved to {report_path}")
    
    async def _generate_summary_report(self, report: Dict[str, Any], output_path: Path):
        """Generate human-readable summary report."""
        summary = []
        summary.append(f"# Rollout Report: {report['manifest']['version']}")
        summary.append(f"**Release ID:** {report['manifest']['release_id']}")
        summary.append(f"**Green Fingerprint:** {report['manifest']['green_fingerprint']}")
        summary.append(f"**Status:** {'ROLLED BACK' if report['rollout_summary']['rollback_triggered'] else 'SUCCESS'}")
        summary.append("")
        
        summary.append("## Stage Summary")
        for stage_name, metrics in report['stage_metrics'].items():
            summary.append(f"- **{stage_name}**: {metrics['status']} "
                          f"({metrics['traffic_percentage']}% traffic)")
            if metrics['triggered_rollbacks']:
                summary.append(f"  - Rollback triggers: {metrics['triggered_rollbacks']}")
        summary.append("")
        
        summary.append("## Key Metrics")
        if report['baseline_metrics']:
            baseline = report['baseline_metrics']
            summary.append(f"- **Baseline Pass Rate:** {baseline['pass_rate_core']:.3f}")
            summary.append(f"- **Baseline Answerable@k:** {baseline['answerable_at_k']:.3f}")
            summary.append(f"- **Baseline Span Recall:** {baseline['span_recall']:.3f}")
            summary.append(f"- **Baseline P95 Latency:** {baseline['p95_latency_ms']:.1f}ms")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(summary))


async def main():
    """Example usage of staged rollout system."""
    # Create deployment manifest
    manifest = RolloutManifest.create(
        version="2.1.0",
        corpus_sha="abc123def456",
        ess_thresholds={"core": 0.8, "extended": 0.7},
        prompt_ids={"search": "prompt-v3", "rag": "prompt-v2"},
        model_versions={"search": "gpt-4", "extract": "gpt-3.5-turbo"},
        signed_by="deployment-system"
    )
    
    # Configure rollout
    config = RolloutConfig(
        shadow_duration_hours=2,  # Reduced for testing
        canary_duration_hours=4,
        ramp_step_duration_hours=2,
        auto_rollback_enabled=True
    )
    
    # Create deployment directory
    deployment_dir = Path("/home/nathan/Projects/lens/deployment/rollouts") / manifest.version
    
    # Start rollout
    orchestrator = StagedRolloutOrchestrator(config, manifest, deployment_dir)
    await orchestrator.start_rollout()


if __name__ == "__main__":
    asyncio.run(main())