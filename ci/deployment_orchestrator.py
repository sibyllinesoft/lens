#!/usr/bin/env python3
"""
Deployment Configuration and CI Integration for Lens Search
Integrates all monitoring systems with CI/CD to ensure "works in CI equals stays green in prod".
Implements comprehensive validation gates and automatic green fingerprint management.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import yaml
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import threading

# Import our monitoring systems
sys.path.append(str(Path(__file__).parent.parent))
from deployment.staged_rollout import StagedRolloutOrchestrator, RolloutConfig, RolloutManifest
from deployment.release_manifest import ReleaseManager, SignedManifest
from deployment.runtime_gates import RuntimeGateOrchestrator
from monitoring.telemetry_system import TelemetryCollector, create_telemetry_system
from monitoring.slo_monitor import SLOMonitor
from monitoring.adversarial_sentinels import AdversarialSentinel, DriftSentinel
from monitoring.sanity_scorecard import SanityScorecard
from security.abuse_protection import SecurityOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentStage(Enum):
    """Stages in the deployment pipeline."""
    VALIDATION = "validation"
    BUILD = "build"
    TEST = "test"
    SECURITY_SCAN = "security_scan"
    INTEGRATION_TEST = "integration_test"
    STAGING_DEPLOY = "staging_deploy"
    STAGING_VALIDATE = "staging_validate"
    PRODUCTION_DEPLOY = "production_deploy"
    PRODUCTION_VALIDATE = "production_validate"
    MONITORING_SETUP = "monitoring_setup"
    ROLLBACK = "rollback"


class GateStatus(Enum):
    """Status of deployment gates."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DeploymentGate:
    """Individual deployment gate configuration."""
    name: str
    stage: DeploymentStage
    description: str
    required: bool = True
    timeout_minutes: int = 30
    retry_count: int = 2
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class GateResult:
    """Result of a deployment gate execution."""
    gate_name: str
    status: GateStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = {}


@dataclass
class DeploymentConfig:
    """Complete deployment configuration."""
    environment: str
    release_version: str
    green_fingerprint: str
    
    # Component configurations
    monitoring_config: Dict[str, Any]
    security_config: Dict[str, Any]
    rollout_config: Dict[str, Any]
    
    # Gate configurations
    deployment_gates: List[DeploymentGate]
    
    # Environment-specific settings
    environment_config: Dict[str, Any]
    
    # CI/CD integration
    ci_config: Dict[str, Any]


class DeploymentOrchestrator:
    """Main orchestrator for CI/CD and deployment pipeline."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_configuration()
        
        # Deployment state
        self.current_deployment: Optional[str] = None
        self.gate_results: Dict[str, GateResult] = {}
        self.deployment_start_time: Optional[datetime] = None
        
        # Component systems
        self.telemetry_collector: Optional[TelemetryCollector] = None
        self.slo_monitor: Optional[SLOMonitor] = None
        self.security_orchestrator: Optional[SecurityOrchestrator] = None
        self.sanity_scorecard: Optional[SanityScorecard] = None
        self.rollout_orchestrator: Optional[StagedRolloutOrchestrator] = None
        
        # CI/CD integration
        self.ci_environment = self._detect_ci_environment()
        self.is_ci = bool(self.ci_environment)
        
        # Artifacts and reporting
        self.artifacts_dir = Path(self.config.get("artifacts_dir", "./deployment-artifacts"))
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DeploymentOrchestrator initialized for {self.config.get('environment', 'unknown')}")
        logger.info(f"CI Environment: {self.ci_environment or 'local'}")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load deployment configuration from file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            if self.config_path.suffix.lower() in ['.yml', '.yaml']:
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        # Validate required configuration
        required_keys = ["environment", "deployment_gates", "monitoring", "security"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        return config
    
    def _detect_ci_environment(self) -> Optional[str]:
        """Detect CI/CD environment."""
        ci_indicators = {
            "github": "GITHUB_ACTIONS",
            "gitlab": "GITLAB_CI",
            "jenkins": "JENKINS_URL",
            "circleci": "CIRCLECI",
            "travis": "TRAVIS",
            "azure": "AZURE_PIPELINES",
            "buildkite": "BUILDKITE"
        }
        
        for ci_name, env_var in ci_indicators.items():
            if os.getenv(env_var):
                return ci_name
        
        return None
    
    async def run_deployment_pipeline(self) -> bool:
        """Run complete deployment pipeline."""
        deployment_id = f"deploy-{int(time.time())}"
        self.current_deployment = deployment_id
        self.deployment_start_time = datetime.utcnow()
        
        logger.info(f"Starting deployment pipeline: {deployment_id}")
        
        try:
            # Initialize all monitoring systems
            await self._initialize_monitoring_systems()
            
            # Parse deployment gates
            gates = self._parse_deployment_gates()
            
            # Execute deployment gates in dependency order
            success = await self._execute_deployment_gates(gates)
            
            if success:
                # Generate deployment report
                await self._generate_deployment_report(True)
                
                # Set CI status to green
                if self.is_ci:
                    self._set_ci_status("success", "Deployment pipeline passed")
                
                logger.info(f"Deployment pipeline completed successfully: {deployment_id}")
                return True
            else:
                # Generate failure report
                await self._generate_deployment_report(False)
                
                # Set CI status to red
                if self.is_ci:
                    self._set_ci_status("failure", "Deployment pipeline failed")
                
                logger.error(f"Deployment pipeline failed: {deployment_id}")
                return False
                
        except Exception as e:
            logger.error(f"Deployment pipeline exception: {e}")
            
            # Generate error report
            await self._generate_deployment_report(False, str(e))
            
            # Set CI status to error
            if self.is_ci:
                self._set_ci_status("error", f"Deployment pipeline error: {e}")
            
            return False
        
        finally:
            # Cleanup monitoring systems
            await self._cleanup_monitoring_systems()
    
    async def _initialize_monitoring_systems(self):
        """Initialize all monitoring and security systems."""
        logger.info("Initializing monitoring systems...")
        
        # Initialize telemetry
        telemetry_config = self.config["monitoring"].get("telemetry", {})
        self.telemetry_collector, _ = create_telemetry_system(telemetry_config)
        
        # Initialize SLO monitoring
        slo_config = self.config["monitoring"].get("slo", {})
        self.slo_monitor = SLOMonitor(slo_config, self.telemetry_collector)
        self.slo_monitor.start_monitoring()
        
        # Initialize security
        security_config = self.config["security"]
        self.security_orchestrator = SecurityOrchestrator(security_config)
        self.security_orchestrator.start_protection()
        
        # Initialize adversarial sentinels
        adversarial_config = self.config["monitoring"].get("adversarial", {})
        adversarial_sentinel = AdversarialSentinel(adversarial_config, self.telemetry_collector)
        adversarial_sentinel.start_testing()
        
        # Initialize drift sentinel
        drift_config = self.config["monitoring"].get("drift", {})
        drift_sentinel = DriftSentinel(drift_config)
        drift_sentinel.start_monitoring()
        
        # Initialize sanity scorecard
        scorecard_config = self.config["monitoring"].get("scorecard", {})
        self.sanity_scorecard = SanityScorecard(
            scorecard_config, self.telemetry_collector, self.slo_monitor,
            adversarial_sentinel, drift_sentinel
        )
        self.sanity_scorecard.start_monitoring()
        
        # Store references for cleanup
        self.adversarial_sentinel = adversarial_sentinel
        self.drift_sentinel = drift_sentinel
        
        logger.info("Monitoring systems initialized")
    
    async def _cleanup_monitoring_systems(self):
        """Cleanup monitoring systems."""
        logger.info("Cleaning up monitoring systems...")
        
        try:
            if self.sanity_scorecard:
                self.sanity_scorecard.stop_monitoring()
            if self.slo_monitor:
                self.slo_monitor.stop_monitoring()
            if self.security_orchestrator:
                self.security_orchestrator.stop_protection()
            if hasattr(self, 'adversarial_sentinel'):
                self.adversarial_sentinel.stop_testing()
            if hasattr(self, 'drift_sentinel'):
                self.drift_sentinel.stop_monitoring()
            if self.telemetry_collector:
                self.telemetry_collector.stop_processing()
        except Exception as e:
            logger.warning(f"Error during monitoring cleanup: {e}")
    
    def _parse_deployment_gates(self) -> List[DeploymentGate]:
        """Parse deployment gates from configuration."""
        gates = []
        
        for gate_config in self.config["deployment_gates"]:
            gate = DeploymentGate(
                name=gate_config["name"],
                stage=DeploymentStage(gate_config["stage"]),
                description=gate_config["description"],
                required=gate_config.get("required", True),
                timeout_minutes=gate_config.get("timeout_minutes", 30),
                retry_count=gate_config.get("retry_count", 2),
                dependencies=gate_config.get("dependencies", [])
            )
            gates.append(gate)
        
        return gates
    
    async def _execute_deployment_gates(self, gates: List[DeploymentGate]) -> bool:
        """Execute deployment gates in dependency order."""
        # Sort gates by dependencies (topological sort)
        sorted_gates = self._topological_sort_gates(gates)
        
        for gate in sorted_gates:
            logger.info(f"Executing gate: {gate.name}")
            
            # Check dependencies
            for dependency in gate.dependencies:
                if dependency not in self.gate_results:
                    logger.error(f"Gate {gate.name} dependency {dependency} not found")
                    return False
                
                dep_result = self.gate_results[dependency]
                if dep_result.status != GateStatus.PASSED:
                    logger.error(f"Gate {gate.name} dependency {dependency} failed")
                    return False
            
            # Execute gate
            result = await self._execute_single_gate(gate)
            self.gate_results[gate.name] = result
            
            # Check if gate failed
            if result.status == GateStatus.FAILED and gate.required:
                logger.error(f"Required gate {gate.name} failed")
                return False
            
            logger.info(f"Gate {gate.name} completed: {result.status.value}")
        
        return True
    
    def _topological_sort_gates(self, gates: List[DeploymentGate]) -> List[DeploymentGate]:
        """Sort gates by dependencies using topological sort."""
        # Build dependency graph
        graph = {gate.name: gate.dependencies for gate in gates}
        gate_map = {gate.name: gate for gate in gates}
        
        # Kahn's algorithm
        in_degree = {gate: 0 for gate in graph}
        for dependencies in graph.values():
            for dep in dependencies:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        queue = [gate for gate, degree in in_degree.items() if degree == 0]
        sorted_gates = []
        
        while queue:
            current = queue.pop(0)
            sorted_gates.append(gate_map[current])
            
            for dependencies in graph.values():
                if current in dependencies:
                    for neighbor in graph:
                        if current in graph[neighbor]:
                            in_degree[neighbor] -= 1
                            if in_degree[neighbor] == 0:
                                queue.append(neighbor)
        
        return sorted_gates
    
    async def _execute_single_gate(self, gate: DeploymentGate) -> GateResult:
        """Execute a single deployment gate."""
        result = GateResult(
            gate_name=gate.name,
            status=GateStatus.RUNNING,
            start_time=datetime.utcnow()
        )
        
        try:
            # Execute gate based on stage type
            if gate.stage == DeploymentStage.VALIDATION:
                success = await self._run_validation_gate(gate, result)
            elif gate.stage == DeploymentStage.BUILD:
                success = await self._run_build_gate(gate, result)
            elif gate.stage == DeploymentStage.TEST:
                success = await self._run_test_gate(gate, result)
            elif gate.stage == DeploymentStage.SECURITY_SCAN:
                success = await self._run_security_scan_gate(gate, result)
            elif gate.stage == DeploymentStage.INTEGRATION_TEST:
                success = await self._run_integration_test_gate(gate, result)
            elif gate.stage == DeploymentStage.STAGING_DEPLOY:
                success = await self._run_staging_deploy_gate(gate, result)
            elif gate.stage == DeploymentStage.STAGING_VALIDATE:
                success = await self._run_staging_validate_gate(gate, result)
            elif gate.stage == DeploymentStage.PRODUCTION_DEPLOY:
                success = await self._run_production_deploy_gate(gate, result)
            elif gate.stage == DeploymentStage.PRODUCTION_VALIDATE:
                success = await self._run_production_validate_gate(gate, result)
            elif gate.stage == DeploymentStage.MONITORING_SETUP:
                success = await self._run_monitoring_setup_gate(gate, result)
            else:
                logger.warning(f"Unknown gate stage: {gate.stage}")
                success = False
            
            result.status = GateStatus.PASSED if success else GateStatus.FAILED
            
        except Exception as e:
            logger.error(f"Gate {gate.name} execution error: {e}")
            result.status = GateStatus.FAILED
            result.error_message = str(e)
        
        finally:
            result.end_time = datetime.utcnow()
            if result.end_time:
                result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        
        return result
    
    async def _run_validation_gate(self, gate: DeploymentGate, result: GateResult) -> bool:
        """Run validation gate (linting, static analysis, etc.)."""
        commands = [
            ["python", "-m", "flake8", ".", "--max-line-length=100"],
            ["python", "-m", "mypy", ".", "--ignore-missing-imports"],
            ["python", "-m", "black", ".", "--check"]
        ]
        
        for cmd in commands:
            success, stdout, stderr = await self._run_command(cmd, timeout=300)
            result.stdout += f"Command: {' '.join(cmd)}\n{stdout}\n"
            result.stderr += stderr
            
            if not success:
                result.error_message = f"Validation failed: {' '.join(cmd)}"
                return False
        
        return True
    
    async def _run_build_gate(self, gate: DeploymentGate, result: GateResult) -> bool:
        """Run build gate."""
        # Install dependencies
        success, stdout, stderr = await self._run_command(
            ["pip", "install", "-r", "requirements.txt"], timeout=600
        )
        result.stdout += stdout
        result.stderr += stderr
        
        if not success:
            result.error_message = "Failed to install dependencies"
            return False
        
        # Build artifacts (if applicable)
        if Path("setup.py").exists():
            success, stdout, stderr = await self._run_command(
                ["python", "setup.py", "build"], timeout=300
            )
            result.stdout += stdout
            result.stderr += stderr
            
            if not success:
                result.error_message = "Build failed"
                return False
        
        return True
    
    async def _run_test_gate(self, gate: DeploymentGate, result: GateResult) -> bool:
        """Run test gate."""
        # Run unit tests
        success, stdout, stderr = await self._run_command(
            ["python", "-m", "pytest", "tests/", "-v", "--cov=.", "--cov-report=xml"],
            timeout=900
        )
        result.stdout += stdout
        result.stderr += stderr
        
        if not success:
            result.error_message = "Unit tests failed"
            return False
        
        # Check coverage
        if Path("coverage.xml").exists():
            # Parse coverage and store in artifacts
            result.artifacts["coverage_report"] = "coverage.xml"
        
        return True
    
    async def _run_security_scan_gate(self, gate: DeploymentGate, result: GateResult) -> bool:
        """Run security scanning gate."""
        commands = [
            ["bandit", "-r", ".", "-f", "json", "-o", "bandit-report.json"],
            ["safety", "check", "--json", "--output", "safety-report.json"]
        ]
        
        for cmd in commands:
            success, stdout, stderr = await self._run_command(cmd, timeout=300)
            result.stdout += f"Command: {' '.join(cmd)}\n{stdout}\n"
            result.stderr += stderr
            
            # Security tools may return non-zero for findings, check output
            if "bandit" in cmd:
                if Path("bandit-report.json").exists():
                    result.artifacts["bandit_report"] = "bandit-report.json"
            elif "safety" in cmd:
                if Path("safety-report.json").exists():
                    result.artifacts["safety_report"] = "safety-report.json"
        
        return True  # Security scans produce reports but don't fail deployment
    
    async def _run_integration_test_gate(self, gate: DeploymentGate, result: GateResult) -> bool:
        """Run integration test gate."""
        # Start test services
        await self._start_test_services()
        
        try:
            # Run integration tests
            success, stdout, stderr = await self._run_command(
                ["python", "-m", "pytest", "tests/integration/", "-v"],
                timeout=1200
            )
            result.stdout += stdout
            result.stderr += stderr
            
            return success
            
        finally:
            # Stop test services
            await self._stop_test_services()
    
    async def _run_staging_deploy_gate(self, gate: DeploymentGate, result: GateResult) -> bool:
        """Run staging deployment gate."""
        # Create release manifest
        repo_root = Path.cwd()
        release_manager = ReleaseManager(repo_root)
        
        manifest = release_manager.create_release(
            base_version="1.0.0",
            changes=["threshold"],  # Example change
            corpus_path=str(repo_root / "indexed-content"),
            signed_by="ci-system"
        )
        
        # Deploy to staging
        staging_config = RolloutConfig(
            shadow_duration_hours=1,
            canary_duration_hours=2,
            auto_rollback_enabled=True
        )
        
        deployment_dir = self.artifacts_dir / "staging"
        self.rollout_orchestrator = StagedRolloutOrchestrator(
            staging_config, manifest, deployment_dir
        )
        
        # Run shadow and canary stages only for staging
        await self.rollout_orchestrator._run_shadow_stage()
        if not self.rollout_orchestrator.rollback_triggered:
            await self.rollout_orchestrator._run_canary_stage()
        
        return not self.rollout_orchestrator.rollback_triggered
    
    async def _run_staging_validate_gate(self, gate: DeploymentGate, result: GateResult) -> bool:
        """Run staging validation gate."""
        if not self.sanity_scorecard:
            result.error_message = "Sanity scorecard not initialized"
            return False
        
        # Wait for metrics to stabilize
        await asyncio.sleep(30)
        
        # Get sanity scorecard data
        dashboard_data = self.sanity_scorecard.get_dashboard_data()
        if not dashboard_data:
            result.error_message = "No dashboard data available"
            return False
        
        # Check health score
        health_score = dashboard_data.health_score
        if health_score < 80:  # Minimum health score threshold
            result.error_message = f"Health score too low: {health_score:.1f}%"
            return False
        
        # Check SLO status
        slo_status = self.slo_monitor.get_slo_status()
        if slo_status["rollback_triggered"]:
            result.error_message = "SLO rollback triggered"
            return False
        
        result.artifacts["health_score"] = health_score
        result.artifacts["slo_status"] = slo_status
        
        return True
    
    async def _run_production_deploy_gate(self, gate: DeploymentGate, result: GateResult) -> bool:
        """Run production deployment gate."""
        if not self.rollout_orchestrator:
            result.error_message = "No rollout orchestrator available"
            return False
        
        # Continue with ramp and full deployment
        await self.rollout_orchestrator._run_ramp_stage()
        if not self.rollout_orchestrator.rollback_triggered:
            await self.rollout_orchestrator._run_full_stage()
        
        return not self.rollout_orchestrator.rollback_triggered
    
    async def _run_production_validate_gate(self, gate: DeploymentGate, result: GateResult) -> bool:
        """Run production validation gate."""
        # Extended validation period for production
        await asyncio.sleep(120)  # 2 minutes
        
        # Check all monitoring systems
        security_status = self.security_orchestrator.get_security_status()
        if security_status["blocked_requests_1h"] > 10:  # Too many blocked requests
            result.error_message = f"High security violations: {security_status['blocked_requests_1h']}"
            return False
        
        slo_status = self.slo_monitor.get_slo_status()
        if slo_status["rollback_triggered"]:
            result.error_message = "Production SLO rollback triggered"
            return False
        
        dashboard_data = self.sanity_scorecard.get_dashboard_data()
        if dashboard_data and dashboard_data.health_score < 85:  # Higher threshold for prod
            result.error_message = f"Production health score too low: {dashboard_data.health_score:.1f}%"
            return False
        
        result.artifacts["security_status"] = security_status
        result.artifacts["final_health_score"] = dashboard_data.health_score if dashboard_data else 0
        
        return True
    
    async def _run_monitoring_setup_gate(self, gate: DeploymentGate, result: GateResult) -> bool:
        """Run monitoring setup gate."""
        # Verify all monitoring systems are healthy
        systems_status = {
            "telemetry": self.telemetry_collector is not None,
            "slo_monitor": self.slo_monitor is not None,
            "security": self.security_orchestrator is not None,
            "scorecard": self.sanity_scorecard is not None
        }
        
        result.artifacts["monitoring_systems"] = systems_status
        
        # All systems should be running
        return all(systems_status.values())
    
    async def _start_test_services(self):
        """Start services needed for integration testing."""
        # This would typically start docker containers, test databases, etc.
        logger.info("Starting test services...")
        await asyncio.sleep(5)  # Simulate service startup
    
    async def _stop_test_services(self):
        """Stop test services."""
        logger.info("Stopping test services...")
        await asyncio.sleep(2)  # Simulate service shutdown
    
    async def _run_command(self, cmd: List[str], timeout: int = 300) -> Tuple[bool, str, str]:
        """Run shell command with timeout."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            
            return process.returncode == 0, stdout.decode(), stderr.decode()
            
        except asyncio.TimeoutError:
            logger.error(f"Command timed out: {' '.join(cmd)}")
            return False, "", f"Command timed out after {timeout}s"
        except Exception as e:
            logger.error(f"Command failed: {' '.join(cmd)} - {e}")
            return False, "", str(e)
    
    def _set_ci_status(self, status: str, description: str):
        """Set CI system status."""
        if self.ci_environment == "github":
            # Set GitHub Actions status
            if status == "success":
                print("::notice::Deployment pipeline passed")
            elif status == "failure":
                print(f"::error::Deployment pipeline failed: {description}")
            elif status == "error":
                print(f"::error::Deployment pipeline error: {description}")
        
        # Could add support for other CI systems here
        logger.info(f"CI Status: {status} - {description}")
    
    async def _generate_deployment_report(self, success: bool, error_message: str = None):
        """Generate comprehensive deployment report."""
        report = {
            "deployment_id": self.current_deployment,
            "timestamp": datetime.utcnow().isoformat(),
            "success": success,
            "environment": self.config["environment"],
            "ci_environment": self.ci_environment,
            "error_message": error_message,
            "duration_seconds": (
                (datetime.utcnow() - self.deployment_start_time).total_seconds()
                if self.deployment_start_time else 0
            ),
            "gate_results": {
                name: asdict(result) for name, result in self.gate_results.items()
            }
        }
        
        # Add monitoring data if available
        if self.sanity_scorecard:
            dashboard_data = self.sanity_scorecard.get_dashboard_data()
            if dashboard_data:
                report["final_health_score"] = dashboard_data.health_score
                report["dashboard_status"] = dashboard_data.overall_health
        
        if self.slo_monitor:
            report["slo_status"] = self.slo_monitor.get_slo_status()
        
        if self.security_orchestrator:
            report["security_status"] = self.security_orchestrator.get_security_status()
        
        # Save report
        report_file = self.artifacts_dir / f"deployment-report-{self.current_deployment}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate human-readable summary
        summary_file = self.artifacts_dir / f"deployment-summary-{self.current_deployment}.md"
        await self._generate_summary_report(report, summary_file)
        
        logger.info(f"Deployment report saved: {report_file}")
    
    async def _generate_summary_report(self, report: Dict[str, Any], output_path: Path):
        """Generate human-readable deployment summary."""
        summary = []
        summary.append(f"# Deployment Report: {report['deployment_id']}")
        summary.append(f"**Status:** {'✅ SUCCESS' if report['success'] else '❌ FAILED'}")
        summary.append(f"**Environment:** {report['environment']}")
        summary.append(f"**Duration:** {report['duration_seconds']:.1f}s")
        
        if report.get('error_message'):
            summary.append(f"**Error:** {report['error_message']}")
        
        summary.append("")
        summary.append("## Gate Results")
        
        for gate_name, result in report['gate_results'].items():
            status_emoji = {
                'passed': '✅',
                'failed': '❌', 
                'skipped': '⏭️',
                'running': '🔄'
            }.get(result['status'], '❓')
            
            summary.append(f"- **{gate_name}**: {status_emoji} {result['status']} "
                          f"({result['duration_seconds']:.1f}s)")
            
            if result.get('error_message'):
                summary.append(f"  - Error: {result['error_message']}")
        
        summary.append("")
        summary.append("## System Health")
        
        if 'final_health_score' in report:
            summary.append(f"- **Health Score:** {report['final_health_score']:.1f}%")
        
        if 'slo_status' in report:
            slo = report['slo_status']
            summary.append(f"- **SLO Status:** {slo.get('active_alerts', 0)} active alerts")
            summary.append(f"- **Rollback Triggered:** {'Yes' if slo.get('rollback_triggered') else 'No'}")
        
        if 'security_status' in report:
            sec = report['security_status']
            summary.append(f"- **Security Violations (1h):** {sec.get('violations_1h', 0)}")
            summary.append(f"- **Blocked Requests (1h):** {sec.get('blocked_requests_1h', 0)}")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(summary))


def create_default_deployment_config() -> Dict[str, Any]:
    """Create default deployment configuration."""
    return {
        "environment": "staging",
        "deployment_gates": [
            {
                "name": "validation",
                "stage": "validation",
                "description": "Code linting and static analysis",
                "required": True,
                "timeout_minutes": 10
            },
            {
                "name": "build",
                "stage": "build", 
                "description": "Build application and install dependencies",
                "required": True,
                "timeout_minutes": 15,
                "dependencies": ["validation"]
            },
            {
                "name": "unit_tests",
                "stage": "test",
                "description": "Run unit tests and coverage analysis",
                "required": True,
                "timeout_minutes": 20,
                "dependencies": ["build"]
            },
            {
                "name": "security_scan",
                "stage": "security_scan",
                "description": "Security vulnerability scanning",
                "required": False,
                "timeout_minutes": 10,
                "dependencies": ["build"]
            },
            {
                "name": "integration_tests",
                "stage": "integration_test",
                "description": "Integration and end-to-end tests",
                "required": True,
                "timeout_minutes": 30,
                "dependencies": ["unit_tests"]
            },
            {
                "name": "staging_deploy",
                "stage": "staging_deploy",
                "description": "Deploy to staging environment",
                "required": True,
                "timeout_minutes": 20,
                "dependencies": ["integration_tests"]
            },
            {
                "name": "staging_validate",
                "stage": "staging_validate",
                "description": "Validate staging deployment",
                "required": True,
                "timeout_minutes": 15,
                "dependencies": ["staging_deploy", "monitoring_setup"]
            },
            {
                "name": "monitoring_setup",
                "stage": "monitoring_setup",
                "description": "Initialize monitoring systems",
                "required": True,
                "timeout_minutes": 5,
                "dependencies": ["staging_deploy"]
            },
            {
                "name": "production_deploy",
                "stage": "production_deploy",
                "description": "Deploy to production environment",
                "required": True,
                "timeout_minutes": 30,
                "dependencies": ["staging_validate"]
            },
            {
                "name": "production_validate",
                "stage": "production_validate",
                "description": "Validate production deployment",
                "required": True,
                "timeout_minutes": 20,
                "dependencies": ["production_deploy"]
            }
        ],
        "monitoring": {
            "telemetry": {
                "storage_dir": "./deployment-telemetry"
            },
            "slo": {
                "storage_dir": "./deployment-slo"
            },
            "scorecard": {
                "storage_dir": "./deployment-scorecard",
                "update_interval_seconds": 30
            },
            "adversarial": {
                "storage_dir": "./deployment-adversarial"
            },
            "drift": {
                "storage_dir": "./deployment-drift"
            }
        },
        "security": {
            "storage_dir": "./deployment-security",
            "rate_limiter": {
                "per_user": {"requests": 1000, "window": 3600},
                "per_ip": {"requests": 5000, "window": 3600}
            }
        },
        "artifacts_dir": "./deployment-artifacts"
    }


async def main():
    """Example usage of deployment orchestrator."""
    # Create default configuration
    config = create_default_deployment_config()
    config_path = Path("deployment-config.json")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run deployment pipeline
    orchestrator = DeploymentOrchestrator(config_path)
    success = await orchestrator.run_deployment_pipeline()
    
    if success:
        logger.info("🎉 Deployment pipeline completed successfully!")
        logger.info("CI status: GREEN ✅")
    else:
        logger.error("💥 Deployment pipeline failed!")
        logger.error("CI status: RED ❌")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)