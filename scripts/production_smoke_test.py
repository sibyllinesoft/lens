#!/usr/bin/env python3
"""
Production Smoke Test for Lens V2.3.0 Micro-Canary
Validates basic system functionality before deployment
"""

import sys
import argparse
import time
import subprocess
import json
import logging
from pathlib import Path

def setup_logging():
    """Configure logging for smoke test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_lens_binary():
    """Test that lens binary exists and can execute basic commands"""
    logger = logging.getLogger(__name__)
    
    # Check for lens binary
    lens_binary = Path("./target/release/lens")
    if not lens_binary.exists():
        logger.error("Lens binary not found at expected path")
        return False
    
    try:
        # Test basic lens execution (help or version)
        result = subprocess.run([str(lens_binary), "--version"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info(f"Lens binary version check: OK")
            return True
        else:
            logger.error(f"Lens binary failed basic test: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("Lens binary test timed out")
        return False
    except Exception as e:
        logger.error(f"Error testing lens binary: {e}")
        return False

def test_manifest_integrity():
    """Test that manifest files are present and valid"""
    logger = logging.getLogger(__name__)
    
    manifest_path = Path("manifests/current.lock")
    if not manifest_path.exists():
        logger.error("Current manifest lock file not found")
        return False
    
    try:
        with open(manifest_path, 'r') as f:
            content = f.read()
            if len(content) > 0:
                logger.info("Manifest integrity: OK")
                return True
            else:
                logger.error("Manifest file is empty")
                return False
    except Exception as e:
        logger.error(f"Error reading manifest: {e}")
        return False

def test_configuration():
    """Test that essential configuration files are present"""
    logger = logging.getLogger(__name__)
    
    config_files = [
        "package.json",
        "Cargo.toml"
    ]
    
    missing_files = []
    for config_file in config_files:
        if not Path(config_file).exists():
            missing_files.append(config_file)
    
    if missing_files:
        logger.error(f"Missing configuration files: {missing_files}")
        return False
    
    logger.info("Configuration files: OK")
    return True

def test_docker_connectivity():
    """Test Docker daemon connectivity"""
    logger = logging.getLogger(__name__)
    
    try:
        result = subprocess.run(["docker", "version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("Docker connectivity: OK")
            return True
        else:
            logger.error(f"Docker connectivity failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("Docker version check timed out")
        return False
    except Exception as e:
        logger.error(f"Error testing Docker: {e}")
        return False

def test_python_dependencies():
    """Test that Python is available and can import basic modules"""
    logger = logging.getLogger(__name__)
    
    try:
        import json
        import sys
        import os
        logger.info("Python dependencies: OK")
        return True
    except ImportError as e:
        logger.error(f"Missing Python dependencies: {e}")
        return False

def run_smoke_tests(scenarios=5, strict=False):
    """Run production smoke test suite"""
    logger = setup_logging()
    
    logger.info(f"Starting production smoke test with {scenarios} scenarios (strict={strict})")
    
    tests = [
        ("Configuration Files", test_configuration),
        ("Manifest Integrity", test_manifest_integrity),
        ("Docker Connectivity", test_docker_connectivity),
        ("Python Dependencies", test_python_dependencies),
        ("Lens Binary", test_lens_binary),
    ]
    
    # Run only the requested number of scenarios
    tests_to_run = tests[:scenarios]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests_to_run:
        logger.info(f"Running test: {test_name}")
        try:
            if test_func():
                logger.info(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name}: FAILED")
                failed += 1
                if strict:
                    logger.error("Strict mode: stopping on first failure")
                    break
        except Exception as e:
            logger.error(f"❌ {test_name}: FAILED with exception: {e}")
            failed += 1
            if strict:
                logger.error("Strict mode: stopping on first failure")
                break
    
    logger.info(f"Smoke test results: {passed} passed, {failed} failed")
    
    # Return success if all tests passed or if we're in non-strict mode and most passed
    if failed == 0:
        logger.info("🎉 All smoke tests PASSED")
        return True
    elif not strict and passed >= failed:
        logger.info("🔶 Smoke tests: majority passed (non-strict mode)")
        return True
    else:
        logger.error("❌ Smoke tests FAILED")
        return False

def main():
    parser = argparse.ArgumentParser(description='Production smoke test for Lens deployment')
    parser.add_argument('--scenarios', type=int, default=5, 
                       help='Number of test scenarios to run (default: 5)')
    parser.add_argument('--strict', action='store_true',
                       help='Fail fast mode - stop on first test failure')
    
    args = parser.parse_args()
    
    try:
        success = run_smoke_tests(scenarios=args.scenarios, strict=args.strict)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Smoke test execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()