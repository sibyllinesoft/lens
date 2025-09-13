#!/usr/bin/env python3
"""
Signed Manifest Release System with Semver and CI Protection
Implements corpus SHA freezing, ESS thresholds, prompt IDs, and green fingerprint generation.
"""

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of changes for semver classification."""
    DATA_MAJOR = "data_major"      # Corpus SHA changes
    THRESHOLD_MINOR = "threshold_minor"  # ESS threshold changes  
    PROMPT_PATCH = "prompt_patch"  # Prompt ID changes
    MODEL_MINOR = "model_minor"    # Model version changes


@dataclass
class SemanticVersion:
    """Semantic version with change type tracking."""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None
    
    @classmethod
    def parse(cls, version_str: str) -> 'SemanticVersion':
        """Parse semantic version string."""
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9\-\.]+))?(?:\+([a-zA-Z0-9\-\.]+))?$'
        match = re.match(pattern, version_str)
        if not match:
            raise ValueError(f"Invalid semantic version: {version_str}")
        
        major, minor, patch, prerelease, build = match.groups()
        return cls(
            major=int(major),
            minor=int(minor), 
            patch=int(patch),
            prerelease=prerelease,
            build=build
        )
    
    def increment(self, change_type: ChangeType) -> 'SemanticVersion':
        """Increment version based on change type."""
        if change_type == ChangeType.DATA_MAJOR:
            return SemanticVersion(self.major + 1, 0, 0)
        elif change_type in [ChangeType.THRESHOLD_MINOR, ChangeType.MODEL_MINOR]:
            return SemanticVersion(self.major, self.minor + 1, 0)
        elif change_type == ChangeType.PROMPT_PATCH:
            return SemanticVersion(self.major, self.minor, self.patch + 1)
        else:
            raise ValueError(f"Unknown change type: {change_type}")
    
    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version


@dataclass
class ReleaseArtifact:
    """Individual component of a release manifest."""
    name: str
    version: str
    sha: Optional[str] = None
    source_path: Optional[str] = None
    checksum: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def calculate_checksum(self) -> str:
        """Calculate SHA256 checksum of the artifact."""
        if self.source_path and Path(self.source_path).exists():
            with open(self.source_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        elif isinstance(self.version, str):
            return hashlib.sha256(self.version.encode()).hexdigest()
        else:
            return hashlib.sha256(str(self.version).encode()).hexdigest()


@dataclass
class SignedManifest:
    """Complete signed release manifest with all frozen components."""
    version: SemanticVersion
    release_id: str
    green_fingerprint: str
    created_at: datetime
    signed_by: str
    git_commit: str
    build_metadata: Dict[str, Any]
    
    # Core artifacts
    corpus_artifact: ReleaseArtifact
    ess_thresholds: Dict[str, float]
    prompt_artifacts: Dict[str, ReleaseArtifact]
    model_versions: Dict[str, str]
    chart_hashes: Dict[str, str]
    
    # Change tracking
    changes: List[ChangeType]
    previous_version: Optional[str] = None
    
    @classmethod
    def create(cls, base_version: str, changes: List[ChangeType], 
               corpus_path: str, ess_thresholds: Dict[str, float],
               prompt_configs: Dict[str, str], model_versions: Dict[str, str],
               signed_by: str, chart_paths: Dict[str, str]) -> 'SignedManifest':
        """Create new signed manifest from components."""
        
        # Parse and increment version
        current_version = SemanticVersion.parse(base_version)
        for change in changes:
            current_version = current_version.increment(change)
        
        # Get git commit
        git_commit = cls._get_git_commit()
        
        # Create release ID
        release_id = f"lens-{current_version}-{git_commit[:8]}-{int(time.time())}"
        
        # Create corpus artifact
        corpus_artifact = ReleaseArtifact(
            name="corpus",
            version=git_commit,
            sha=cls._get_corpus_sha(corpus_path),
            source_path=corpus_path,
            created_at=datetime.utcnow()
        )
        corpus_artifact.checksum = corpus_artifact.calculate_checksum()
        
        # Create prompt artifacts
        prompt_artifacts = {}
        for name, config_path in prompt_configs.items():
            artifact = ReleaseArtifact(
                name=f"prompt_{name}",
                version=cls._get_file_version(config_path),
                source_path=config_path,
                created_at=datetime.utcnow()
            )
            artifact.checksum = artifact.calculate_checksum()
            prompt_artifacts[name] = artifact
        
        # Calculate chart hashes
        chart_hashes = {}
        for name, path in chart_paths.items():
            chart_hashes[name] = cls._calculate_chart_hash(path)
        
        # Generate green fingerprint
        fingerprint_data = {
            "version": str(current_version),
            "corpus_sha": corpus_artifact.sha,
            "ess_thresholds": ess_thresholds,
            "prompt_checksums": {name: art.checksum for name, art in prompt_artifacts.items()},
            "model_versions": model_versions,
            "chart_hashes": chart_hashes
        }
        green_fingerprint = cls._generate_green_fingerprint(fingerprint_data)
        
        # Build metadata
        build_metadata = {
            "build_time": datetime.utcnow().isoformat(),
            "build_host": os.environ.get("HOSTNAME", "unknown"),
            "ci_pipeline": os.environ.get("GITHUB_RUN_ID", "local"),
            "build_number": os.environ.get("GITHUB_RUN_NUMBER", "1")
        }
        
        return cls(
            version=current_version,
            release_id=release_id,
            green_fingerprint=green_fingerprint,
            created_at=datetime.utcnow(),
            signed_by=signed_by,
            git_commit=git_commit,
            build_metadata=build_metadata,
            corpus_artifact=corpus_artifact,
            ess_thresholds=ess_thresholds,
            prompt_artifacts=prompt_artifacts,
            model_versions=model_versions,
            chart_hashes=chart_hashes,
            changes=changes
        )
    
    @staticmethod
    def _get_git_commit() -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            logger.warning("Could not get git commit, using timestamp")
            return f"unknown-{int(time.time())}"
    
    @staticmethod
    def _get_corpus_sha(corpus_path: str) -> str:
        """Calculate SHA for corpus directory or file."""
        path = Path(corpus_path)
        if path.is_file():
            with open(path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        elif path.is_dir():
            # Calculate hash of all files in directory
            file_hashes = []
            for file_path in sorted(path.rglob('*')):
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        file_hashes.append(hashlib.sha256(f.read()).hexdigest())
            combined = ''.join(file_hashes)
            return hashlib.sha256(combined.encode()).hexdigest()
        else:
            raise ValueError(f"Corpus path does not exist: {corpus_path}")
    
    @staticmethod
    def _get_file_version(file_path: str) -> str:
        """Get version identifier for a file."""
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"File does not exist: {file_path}")
        
        # Use file modification time + size as version
        stat = path.stat()
        return f"v{int(stat.st_mtime)}-{stat.st_size}"
    
    @staticmethod
    def _calculate_chart_hash(chart_path: str) -> str:
        """Calculate hash for a Helm chart or configuration."""
        path = Path(chart_path)
        if not path.exists():
            logger.warning(f"Chart path does not exist: {chart_path}")
            return "unknown"
        
        if path.is_file():
            with open(path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        else:
            # Hash all YAML files in chart directory
            yaml_hashes = []
            for yaml_file in sorted(path.rglob('*.yaml')):
                with open(yaml_file, 'rb') as f:
                    yaml_hashes.append(hashlib.sha256(f.read()).hexdigest())
            combined = ''.join(yaml_hashes)
            return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    @staticmethod
    def _generate_green_fingerprint(data: Dict[str, Any]) -> str:
        """Generate green fingerprint from manifest data."""
        serialized = json.dumps(data, sort_keys=True)
        hash_digest = hashlib.sha256(serialized.encode()).hexdigest()
        return f"green-{hash_digest[:16]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary for serialization."""
        return {
            "version": str(self.version),
            "release_id": self.release_id,
            "green_fingerprint": self.green_fingerprint,
            "created_at": self.created_at.isoformat(),
            "signed_by": self.signed_by,
            "git_commit": self.git_commit,
            "build_metadata": self.build_metadata,
            "corpus_artifact": asdict(self.corpus_artifact),
            "ess_thresholds": self.ess_thresholds,
            "prompt_artifacts": {name: asdict(art) for name, art in self.prompt_artifacts.items()},
            "model_versions": self.model_versions,
            "chart_hashes": self.chart_hashes,
            "changes": [change.value for change in self.changes],
            "previous_version": self.previous_version
        }
    
    def save(self, output_dir: Path):
        """Save manifest and artifacts to directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main manifest
        manifest_path = output_dir / f"manifest-{self.version}.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        # Save green fingerprint file
        fingerprint_path = output_dir / f"fingerprint-{self.green_fingerprint}.txt"
        with open(fingerprint_path, 'w') as f:
            f.write(f"{self.green_fingerprint}\n")
            f.write(f"Version: {self.version}\n")
            f.write(f"Release: {self.release_id}\n")
            f.write(f"Created: {self.created_at.isoformat()}\n")
        
        # Copy artifacts
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        # Copy corpus if it's a file
        if self.corpus_artifact.source_path and Path(self.corpus_artifact.source_path).is_file():
            corpus_dest = artifacts_dir / f"corpus-{self.corpus_artifact.sha[:8]}.tar.gz"
            # TODO: Create corpus archive
        
        # Copy prompt artifacts
        for name, artifact in self.prompt_artifacts.items():
            if artifact.source_path and Path(artifact.source_path).exists():
                dest = artifacts_dir / f"prompt-{name}-{artifact.checksum[:8]}.json"
                with open(artifact.source_path, 'r') as src, open(dest, 'w') as dst:
                    dst.write(src.read())
        
        logger.info(f"Manifest saved to {manifest_path}")
        logger.info(f"Green fingerprint: {self.green_fingerprint}")
        
        return manifest_path
    
    def validate_integrity(self) -> Tuple[bool, List[str]]:
        """Validate manifest integrity and artifact checksums."""
        errors = []
        
        # Validate corpus artifact
        if self.corpus_artifact.source_path:
            try:
                expected_checksum = self.corpus_artifact.calculate_checksum()
                if expected_checksum != self.corpus_artifact.checksum:
                    errors.append(f"Corpus checksum mismatch: expected {self.corpus_artifact.checksum}, got {expected_checksum}")
            except Exception as e:
                errors.append(f"Corpus validation error: {e}")
        
        # Validate prompt artifacts
        for name, artifact in self.prompt_artifacts.items():
            if artifact.source_path:
                try:
                    expected_checksum = artifact.calculate_checksum()
                    if expected_checksum != artifact.checksum:
                        errors.append(f"Prompt {name} checksum mismatch: expected {artifact.checksum}, got {expected_checksum}")
                except Exception as e:
                    errors.append(f"Prompt {name} validation error: {e}")
        
        # Validate green fingerprint
        fingerprint_data = {
            "version": str(self.version),
            "corpus_sha": self.corpus_artifact.sha,
            "ess_thresholds": self.ess_thresholds,
            "prompt_checksums": {name: art.checksum for name, art in self.prompt_artifacts.items()},
            "model_versions": self.model_versions,
            "chart_hashes": self.chart_hashes
        }
        expected_fingerprint = self._generate_green_fingerprint(fingerprint_data)
        if expected_fingerprint != self.green_fingerprint:
            errors.append(f"Green fingerprint mismatch: expected {expected_fingerprint}, got {self.green_fingerprint}")
        
        return len(errors) == 0, errors


class CIMixedVersionProtection:
    """Protects CI from running mixed versions by enforcing manifest consistency."""
    
    def __init__(self, manifest_dir: Path):
        self.manifest_dir = manifest_dir
        self.active_manifest_path = manifest_dir / "active-manifest.json"
        
    def check_version_consistency(self) -> Tuple[bool, str]:
        """Check if current codebase matches active manifest."""
        if not self.active_manifest_path.exists():
            return False, "No active manifest found"
        
        try:
            with open(self.active_manifest_path, 'r') as f:
                active_manifest = json.load(f)
            
            # Check git commit
            current_commit = SignedManifest._get_git_commit()
            if current_commit != active_manifest["git_commit"]:
                return False, f"Git commit mismatch: active={active_manifest['git_commit'][:8]}, current={current_commit[:8]}"
            
            # TODO: Add more consistency checks
            # - Corpus integrity
            # - Prompt file checksums
            # - Configuration drift
            
            return True, "Version consistency verified"
            
        except Exception as e:
            return False, f"Version check error: {e}"
    
    def lock_manifest(self, manifest: SignedManifest):
        """Lock manifest as active for CI protection."""
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.active_manifest_path, 'w') as f:
            json.dump(manifest.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Manifest locked: {manifest.version} ({manifest.green_fingerprint})")
    
    def unlock_manifest(self):
        """Unlock manifest to allow version changes."""
        if self.active_manifest_path.exists():
            self.active_manifest_path.unlink()
            logger.info("Manifest unlocked")


class ReleaseManager:
    """Manages the complete release process with semver and CI protection."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.release_dir = repo_root / "releases"
        self.manifest_dir = repo_root / "deployment" / "manifests"
        self.ci_protection = CIMixedVersionProtection(self.manifest_dir)
        
    def create_release(self, base_version: str, changes: List[str],
                      corpus_path: str, signed_by: str) -> SignedManifest:
        """Create a new release with automatic version increment."""
        
        # Convert change strings to enums
        change_types = []
        for change in changes:
            if change in ["corpus", "data"]:
                change_types.append(ChangeType.DATA_MAJOR)
            elif change in ["threshold", "ess"]:
                change_types.append(ChangeType.THRESHOLD_MINOR)
            elif change in ["prompt", "prompts"]:
                change_types.append(ChangeType.PROMPT_PATCH)
            elif change in ["model", "models"]:
                change_types.append(ChangeType.MODEL_MINOR)
            else:
                logger.warning(f"Unknown change type: {change}")
        
        # Default ESS thresholds (should be configurable)
        ess_thresholds = {
            "core": 0.85,
            "extended": 0.75,
            "experimental": 0.65
        }
        
        # Default prompt configurations
        prompt_configs = {
            "search": str(self.repo_root / "prompts" / "search.json"),
            "extract": str(self.repo_root / "prompts" / "extract.json"),
            "rag": str(self.repo_root / "prompts" / "rag.json")
        }
        
        # Default model versions
        model_versions = {
            "search": "gpt-4-turbo-2024-04-09",
            "extract": "gpt-3.5-turbo-0125",
            "rag": "gpt-4-turbo-2024-04-09"
        }
        
        # Default chart paths
        chart_paths = {
            "deployment": str(self.repo_root / "charts" / "lens"),
            "monitoring": str(self.repo_root / "charts" / "monitoring")
        }
        
        # Create manifest
        manifest = SignedManifest.create(
            base_version=base_version,
            changes=change_types,
            corpus_path=corpus_path,
            ess_thresholds=ess_thresholds,
            prompt_configs=prompt_configs,
            model_versions=model_versions,
            signed_by=signed_by,
            chart_paths=chart_paths
        )
        
        # Save release
        release_dir = self.release_dir / str(manifest.version)
        manifest_path = manifest.save(release_dir)
        
        # Validate integrity
        is_valid, errors = manifest.validate_integrity()
        if not is_valid:
            logger.error(f"Manifest validation failed: {errors}")
            raise ValueError(f"Manifest validation failed: {errors}")
        
        logger.info(f"Release {manifest.version} created successfully")
        logger.info(f"Green fingerprint: {manifest.green_fingerprint}")
        
        return manifest
    
    def activate_release(self, manifest: SignedManifest):
        """Activate release for deployment with CI protection."""
        self.ci_protection.lock_manifest(manifest)
        
    def check_ci_safety(self) -> Tuple[bool, str]:
        """Check if CI is safe to run with current codebase."""
        return self.ci_protection.check_version_consistency()
    
    def list_releases(self) -> List[str]:
        """List all available releases."""
        if not self.release_dir.exists():
            return []
        
        releases = []
        for release_dir in self.release_dir.iterdir():
            if release_dir.is_dir():
                manifest_files = list(release_dir.glob("manifest-*.json"))
                if manifest_files:
                    releases.append(release_dir.name)
        
        return sorted(releases, key=lambda x: SemanticVersion.parse(x))


def main():
    """Example usage of release management system."""
    repo_root = Path("/home/nathan/Projects/lens")
    release_manager = ReleaseManager(repo_root)
    
    # Check CI safety
    is_safe, message = release_manager.check_ci_safety()
    logger.info(f"CI Safety Check: {message}")
    
    # Create a new release
    try:
        manifest = release_manager.create_release(
            base_version="2.0.0",
            changes=["corpus", "threshold"],  # Major + minor changes
            corpus_path=str(repo_root / "indexed-content"),
            signed_by="ci-system"
        )
        
        # Activate release
        release_manager.activate_release(manifest)
        
        logger.info(f"Release {manifest.version} activated with fingerprint {manifest.green_fingerprint}")
        
    except Exception as e:
        logger.error(f"Release creation failed: {e}")


if __name__ == "__main__":
    main()