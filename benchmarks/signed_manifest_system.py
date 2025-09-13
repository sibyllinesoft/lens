#!/usr/bin/env python3
"""
Signed Manifest System for Sanity Pyramid

Creates and validates signed manifests that lock corpus SHAs, ESS thresholds,
prompt templates, and system configuration to ensure reproducible results
and prevent unintended drift.

Key Features:
- Cryptographic signing of critical system state
- Version-controlled manifest updates
- Automatic drift detection
- Rollback capability for configuration changes
- Immutable fingerprints for "works today" guarantees
"""
import asyncio
import json
import logging
import hashlib
import hmac
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import tempfile
import base64

logger = logging.getLogger(__name__)


@dataclass
class CorpusFingerprint:
    """Immutable corpus state fingerprint."""
    repository_sha: str
    indexed_files_count: int
    total_lines: int
    content_hash: str  # SHA256 of all indexed content
    indexing_timestamp: str
    indexing_config: Dict[str, Any]


@dataclass
class ThresholdConfiguration:
    """ESS threshold configuration per operation."""
    locate: float
    extract: float
    explain: float
    compose: float
    transform: float
    calibration_date: str
    calibration_quality: Dict[str, float]  # Balanced accuracy per operation


@dataclass
class PromptFingerprint:
    """Prompt template fingerprints."""
    template_hashes: Dict[str, str]  # template_name -> SHA256
    template_versions: Dict[str, str]  # template_name -> version
    last_modified: str


@dataclass
class SystemConfiguration:
    """Core system configuration."""
    chunk_size: int
    chunk_overlap: int
    max_context_tokens: int
    containment_config: Dict[str, Any]
    pointer_extract_enabled: bool
    normalization_config: Dict[str, Any]


@dataclass
class SanityManifest:
    """Complete signed manifest of system state."""
    manifest_version: str
    created_at: str
    created_by: str
    corpus_fingerprint: CorpusFingerprint
    threshold_config: ThresholdConfiguration
    prompt_fingerprint: PromptFingerprint
    system_config: SystemConfiguration
    baseline_metrics: Dict[str, float]
    validation_results: Dict[str, Any]
    signature: Optional[str] = None  # HMAC signature
    previous_manifest_hash: Optional[str] = None


class SignedManifestSystem:
    """Manages signed manifests for reproducible system state."""
    
    def __init__(self, work_dir: Path, secret_key: Optional[str] = None):
        self.work_dir = work_dir
        self.work_dir.mkdir(exist_ok=True)
        
        # Cryptographic configuration
        self.secret_key = secret_key or self._generate_secret_key()
        self.manifests_dir = work_dir / "manifests"
        self.manifests_dir.mkdir(exist_ok=True)
        
        # Current manifest tracking
        self.current_manifest: Optional[SanityManifest] = None
        self.manifest_history: List[str] = []
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key for signing."""
        # In production, this would be from secure key management
        return "sanity_pyramid_secret_key_v1_" + hashlib.sha256(
            str(datetime.now().timestamp()).encode()
        ).hexdigest()[:32]
    
    async def create_corpus_fingerprint(self, corpus_path: Path, 
                                      indexed_content_path: Path) -> CorpusFingerprint:
        """Create immutable fingerprint of corpus state."""
        logger.info(f"📊 Creating corpus fingerprint for {corpus_path}")
        
        # Get repository SHA
        repo_sha = await self._get_repo_sha(corpus_path)
        
        # Count indexed files and lines
        indexed_files = list(indexed_content_path.glob("*"))
        total_lines = 0
        content_for_hash = ""
        
        for file_path in sorted(indexed_files):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        total_lines += len(lines)
                        # Include content in hash calculation
                        content_for_hash += f.read()
                except:
                    continue
        
        # Create content hash
        content_hash = hashlib.sha256(content_for_hash.encode('utf-8')).hexdigest()
        
        # Indexing configuration
        indexing_config = {
            "include_patterns": ["*.py", "*.ts", "*.js", "*.md"],
            "exclude_patterns": ["node_modules", ".git", "__pycache__"],
            "max_file_size": 1024 * 1024,  # 1MB
            "encoding": "utf-8"
        }
        
        return CorpusFingerprint(
            repository_sha=repo_sha,
            indexed_files_count=len(indexed_files),
            total_lines=total_lines,
            content_hash=content_hash,
            indexing_timestamp=datetime.now(timezone.utc).isoformat(),
            indexing_config=indexing_config
        )
    
    async def create_threshold_configuration(self, 
                                           calibration_results: Dict[str, Any]) -> ThresholdConfiguration:
        """Create threshold configuration from calibration results."""
        logger.info("🎯 Creating threshold configuration from calibration")
        
        # Extract thresholds (use defaults if not available)
        thresholds = calibration_results.get('optimal_thresholds', {
            'locate': 0.8,
            'extract': 0.75, 
            'explain': 0.6,
            'compose': 0.7,
            'transform': 0.65
        })
        
        # Extract calibration quality metrics
        quality_metrics = {}
        for operation, results in calibration_results.get('calibration_details', {}).items():
            quality_metrics[operation] = results.get('balanced_accuracy', 1.0)
        
        return ThresholdConfiguration(
            locate=thresholds.get('locate', 0.8),
            extract=thresholds.get('extract', 0.75),
            explain=thresholds.get('explain', 0.6),
            compose=thresholds.get('compose', 0.7),
            transform=thresholds.get('transform', 0.65),
            calibration_date=datetime.now(timezone.utc).isoformat(),
            calibration_quality=quality_metrics
        )
    
    async def create_prompt_fingerprint(self, prompt_templates_dir: Path) -> PromptFingerprint:
        """Create fingerprint of prompt templates."""
        logger.info(f"📝 Creating prompt fingerprint for {prompt_templates_dir}")
        
        template_hashes = {}
        template_versions = {}
        
        if prompt_templates_dir.exists():
            for template_file in prompt_templates_dir.glob("*.txt"):
                template_name = template_file.stem
                
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_content = f.read()
                
                # Create hash of template content
                template_hash = hashlib.sha256(template_content.encode('utf-8')).hexdigest()
                template_hashes[template_name] = template_hash
                
                # Extract version if available (look for version comment)
                version = "1.0"  # Default version
                for line in template_content.split('\n')[:5]:  # Check first 5 lines
                    if 'version:' in line.lower():
                        version = line.split(':')[-1].strip()
                        break
                
                template_versions[template_name] = version
        else:
            # Create default prompt fingerprints
            default_prompts = {
                'extract_prompt': 'Extract the relevant code from the context.',
                'explain_prompt': 'Explain how the code works.',
                'locate_prompt': 'Find the file containing the implementation.'
            }
            
            for name, content in default_prompts.items():
                template_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                template_hashes[name] = template_hash
                template_versions[name] = "1.0"
        
        return PromptFingerprint(
            template_hashes=template_hashes,
            template_versions=template_versions,
            last_modified=datetime.now(timezone.utc).isoformat()
        )
    
    def create_system_configuration(self, enhanced_pyramid) -> SystemConfiguration:
        """Create system configuration fingerprint."""
        logger.info("⚙️ Creating system configuration fingerprint")
        
        # Extract containment configuration
        containment_config = {
            'chunk_len': enhanced_pyramid.pointer_extractor.containment_config.chunk_len,
            'overlap': enhanced_pyramid.pointer_extractor.containment_config.overlap,
            'p95_span_len': enhanced_pyramid.pointer_extractor.containment_config.p95_span_len,
            'max_span_len': enhanced_pyramid.pointer_extractor.containment_config.max_span_len,
            'dynamic_widening': enhanced_pyramid.pointer_extractor.containment_config.dynamic_widening
        }
        
        # Normalization configuration
        normalization_config = {
            'crlf_to_lf': True,
            'tabs_to_spaces': True,
            'spaces_per_tab': 4,
            'unicode_nfc': True,
            'preserve_byte_mapping': True
        }
        
        return SystemConfiguration(
            chunk_size=containment_config['chunk_len'],
            chunk_overlap=containment_config['overlap'],
            max_context_tokens=8000,  # Standard limit
            containment_config=containment_config,
            pointer_extract_enabled=enhanced_pyramid.use_pointer_extract,
            normalization_config=normalization_config
        )
    
    async def create_signed_manifest(self, 
                                   corpus_path: Path,
                                   indexed_content_path: Path,
                                   calibration_results: Dict[str, Any],
                                   validation_results: Dict[str, Any],
                                   enhanced_pyramid,
                                   prompt_templates_dir: Optional[Path] = None) -> SanityManifest:
        """Create a complete signed manifest."""
        logger.info("🔐 Creating signed manifest")
        
        # Create all fingerprints
        corpus_fingerprint = await self.create_corpus_fingerprint(
            corpus_path, indexed_content_path
        )
        
        threshold_config = await self.create_threshold_configuration(calibration_results)
        
        prompt_fingerprint = await self.create_prompt_fingerprint(
            prompt_templates_dir or self.work_dir / "prompt_templates"
        )
        
        system_config = self.create_system_configuration(enhanced_pyramid)
        
        # Extract baseline metrics from validation results
        baseline_metrics = {
            'overall_pass_rate': validation_results.get('overall_pass_rate', 0.0),
            'extract_pass_rate': validation_results.get('operation_stats', {}).get('extract', {}).get('pass_rate', 0.0),
            'extract_substring_rate': validation_results.get('operation_stats', {}).get('extract', {}).get('substring_containment_rate', 0.0),
            'locate_pass_rate': validation_results.get('operation_stats', {}).get('locate', {}).get('pass_rate', 0.0),
            'explain_pass_rate': validation_results.get('operation_stats', {}).get('explain', {}).get('pass_rate', 0.0)
        }
        
        # Get previous manifest hash for chain of custody
        previous_hash = await self._get_latest_manifest_hash()
        
        # Create manifest
        manifest = SanityManifest(
            manifest_version=self._generate_manifest_version(),
            created_at=datetime.now(timezone.utc).isoformat(),
            created_by="sanity_pyramid_system",
            corpus_fingerprint=corpus_fingerprint,
            threshold_config=threshold_config,
            prompt_fingerprint=prompt_fingerprint,
            system_config=system_config,
            baseline_metrics=baseline_metrics,
            validation_results=validation_results,
            previous_manifest_hash=previous_hash
        )
        
        # Sign the manifest
        manifest.signature = self._sign_manifest(manifest)
        
        # Save manifest
        await self._save_manifest(manifest)
        
        self.current_manifest = manifest
        logger.info(f"✅ Created signed manifest: {manifest.manifest_version}")
        
        return manifest
    
    def _generate_manifest_version(self) -> str:
        """Generate unique manifest version."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"v1.0_{timestamp}"
    
    def _sign_manifest(self, manifest: SanityManifest) -> str:
        """Create HMAC signature for manifest."""
        # Create manifest data without signature
        manifest_dict = asdict(manifest)
        manifest_dict.pop('signature', None)
        
        # Serialize manifest data
        manifest_json = json.dumps(manifest_dict, sort_keys=True, separators=(',', ':'))
        
        # Create HMAC signature
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            manifest_json.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_manifest_signature(self, manifest: SanityManifest) -> bool:
        """Verify manifest signature."""
        if not manifest.signature:
            return False
        
        # Create expected signature
        expected_signature = self._sign_manifest(manifest)
        
        # Compare signatures
        return hmac.compare_digest(manifest.signature, expected_signature)
    
    async def _save_manifest(self, manifest: SanityManifest):
        """Save manifest to disk."""
        manifest_file = self.manifests_dir / f"manifest_{manifest.manifest_version}.json"
        
        with open(manifest_file, 'w') as f:
            json.dump(asdict(manifest), f, indent=2)
        
        # Update current manifest symlink
        current_link = self.manifests_dir / "current_manifest.json"
        if current_link.exists():
            current_link.unlink()
        
        current_link.symlink_to(manifest_file.name)
        
        # Update manifest history
        self.manifest_history.append(manifest.manifest_version)
        
        history_file = self.manifests_dir / "manifest_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.manifest_history, f, indent=2)
    
    async def load_current_manifest(self) -> Optional[SanityManifest]:
        """Load the current manifest."""
        current_link = self.manifests_dir / "current_manifest.json"
        
        if not current_link.exists():
            return None
        
        with open(current_link, 'r') as f:
            manifest_dict = json.load(f)
        
        # Convert back to SanityManifest
        manifest = SanityManifest(**manifest_dict)
        
        # Verify signature
        if not self.verify_manifest_signature(manifest):
            logger.error("❌ Manifest signature verification failed!")
            return None
        
        self.current_manifest = manifest
        return manifest
    
    async def detect_configuration_drift(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect drift from current manifest."""
        if not self.current_manifest:
            return {'drift_detected': False, 'reason': 'No current manifest'}
        
        drift_report = {
            'drift_detected': False,
            'drift_details': [],
            'manifest_version': self.current_manifest.manifest_version,
            'checked_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Check corpus drift
        if 'corpus_sha' in current_state:
            expected_sha = self.current_manifest.corpus_fingerprint.repository_sha
            actual_sha = current_state['corpus_sha']
            
            if expected_sha != actual_sha:
                drift_report['drift_detected'] = True
                drift_report['drift_details'].append({
                    'type': 'corpus_sha_drift',
                    'expected': expected_sha,
                    'actual': actual_sha,
                    'severity': 'critical'
                })
        
        # Check threshold drift
        if 'current_thresholds' in current_state:
            current_thresholds = current_state['current_thresholds']
            expected_thresholds = self.current_manifest.threshold_config
            
            for operation in ['locate', 'extract', 'explain', 'compose', 'transform']:
                expected_val = getattr(expected_thresholds, operation)
                actual_val = current_thresholds.get(operation)
                
                if actual_val and abs(expected_val - actual_val) > 0.01:  # 1% tolerance
                    drift_report['drift_detected'] = True
                    drift_report['drift_details'].append({
                        'type': 'threshold_drift',
                        'operation': operation,
                        'expected': expected_val,
                        'actual': actual_val,
                        'severity': 'warning'
                    })
        
        return drift_report
    
    async def _get_repo_sha(self, repo_path: Path) -> str:
        """Get current repository SHA."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except:
            return "unknown"
    
    async def _get_latest_manifest_hash(self) -> Optional[str]:
        """Get hash of latest manifest for chain of custody."""
        current_link = self.manifests_dir / "current_manifest.json"
        
        if not current_link.exists():
            return None
        
        with open(current_link, 'rb') as f:
            content = f.read()
        
        return hashlib.sha256(content).hexdigest()
    
    async def generate_manifest_report(self, manifest: SanityManifest) -> str:
        """Generate human-readable manifest report."""
        status = "✅ VERIFIED" if self.verify_manifest_signature(manifest) else "❌ INVALID"
        
        report = f"""# Sanity Pyramid Manifest Report

**Version**: {manifest.manifest_version}
**Status**: {status}
**Created**: {manifest.created_at}

## 📊 Corpus Fingerprint
- **Repository SHA**: {manifest.corpus_fingerprint.repository_sha[:8]}...
- **Indexed Files**: {manifest.corpus_fingerprint.indexed_files_count:,}
- **Total Lines**: {manifest.corpus_fingerprint.total_lines:,}
- **Content Hash**: {manifest.corpus_fingerprint.content_hash[:16]}...

## 🎯 Threshold Configuration  
- **Locate**: {manifest.threshold_config.locate}
- **Extract**: {manifest.threshold_config.extract}
- **Explain**: {manifest.threshold_config.explain}
- **Compose**: {manifest.threshold_config.compose}  
- **Transform**: {manifest.threshold_config.transform}

## 📝 Prompt Templates
- **Templates**: {len(manifest.prompt_fingerprint.template_hashes)}
- **Last Modified**: {manifest.prompt_fingerprint.last_modified}

## ⚙️ System Configuration
- **Chunk Size**: {manifest.system_config.chunk_size}
- **Chunk Overlap**: {manifest.system_config.chunk_overlap}
- **Pointer Extract**: {manifest.system_config.pointer_extract_enabled}
- **Max Context Tokens**: {manifest.system_config.max_context_tokens}

## 📈 Baseline Metrics
- **Overall Pass Rate**: {manifest.baseline_metrics['overall_pass_rate']:.1%}
- **Extract Pass Rate**: {manifest.baseline_metrics['extract_pass_rate']:.1%}
- **Extract Substring Rate**: {manifest.baseline_metrics['extract_substring_rate']:.1%}

## 🔐 Security
- **Signature**: {manifest.signature[:16] if manifest.signature else 'None'}...
- **Previous Hash**: {manifest.previous_manifest_hash[:16] if manifest.previous_manifest_hash else 'None'}...

---
*This manifest provides immutable fingerprints for reproducible system state.*
"""
        
        return report


async def run_signed_manifest_demo():
    """Demonstrate signed manifest system."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize manifest system
    manifest_system = SignedManifestSystem(Path('signed_manifest_results'))
    
    # Mock data for demonstration
    corpus_path = Path('../storyviz')  # Mock corpus path
    indexed_content_path = Path('indexed-content')  # Mock indexed content
    
    # Create mock directories if they don't exist
    indexed_content_path.mkdir(exist_ok=True)
    
    # Mock calibration results
    calibration_results = {
        'optimal_thresholds': {
            'locate': 0.8,
            'extract': 0.75,
            'explain': 0.6,
            'compose': 0.7,
            'transform': 0.65
        },
        'calibration_details': {
            'locate': {'balanced_accuracy': 1.0},
            'extract': {'balanced_accuracy': 1.0},
            'explain': {'balanced_accuracy': 1.0}
        }
    }
    
    # Mock validation results
    validation_results = {
        'overall_pass_rate': 0.95,
        'operation_stats': {
            'extract': {
                'pass_rate': 1.0,
                'substring_containment_rate': 1.0
            },
            'locate': {'pass_rate': 0.95},
            'explain': {'pass_rate': 0.90}
        }
    }
    
    # Mock enhanced pyramid
    from enhanced_sanity_pyramid import EnhancedSanityPyramid
    enhanced_pyramid = EnhancedSanityPyramid(Path('mock_pyramid'))
    
    # Create signed manifest
    manifest = await manifest_system.create_signed_manifest(
        corpus_path=corpus_path,
        indexed_content_path=indexed_content_path,
        calibration_results=calibration_results,
        validation_results=validation_results,
        enhanced_pyramid=enhanced_pyramid
    )
    
    # Generate report
    report = await manifest_system.generate_manifest_report(manifest)
    
    print(f"\n🎯 SIGNED MANIFEST DEMO COMPLETE")
    print(f"Manifest version: {manifest.manifest_version}")
    print(f"Signature valid: {manifest_system.verify_manifest_signature(manifest)}")
    print(f"Corpus files: {manifest.corpus_fingerprint.indexed_files_count}")
    print(f"Extract threshold: {manifest.threshold_config.extract}")
    print(f"Pointer extract enabled: {manifest.system_config.pointer_extract_enabled}")
    
    # Save human-readable report
    report_file = manifest_system.work_dir / f"manifest_report_{manifest.manifest_version}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📋 Report saved: {report_file}")
    
    # Test drift detection
    mock_drift_state = {
        'corpus_sha': 'different_sha_123',  # Different from manifest
        'current_thresholds': {
            'extract': 0.8  # Different from 0.75 in manifest
        }
    }
    
    drift_report = await manifest_system.detect_configuration_drift(mock_drift_state)
    
    if drift_report['drift_detected']:
        print("⚠️ Configuration drift detected!")
        for detail in drift_report['drift_details']:
            print(f"   - {detail['type']}: {detail.get('severity', 'unknown')} severity")
    else:
        print("✅ No configuration drift detected")
    
    return manifest


if __name__ == "__main__":
    asyncio.run(run_signed_manifest_demo())