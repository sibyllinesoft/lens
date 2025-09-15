#!/usr/bin/env python3
"""
Production Smoke Test for Lens V2.3.0 Micro-Canary
Validates core system functionality before and during micro-canary execution.
"""

import sys
import time
import hashlib
import subprocess
from datetime import datetime, timezone

def test_docker_images():
    """Test 1: Verify required Docker images are available."""
    print("🔍 Test 1/5: Docker Images Availability")
    
    required_images = [
        "lens-production:baseline-stable",
        "lens-production:green-aa77b469"
    ]
    
    try:
        result = subprocess.run(['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}'], 
                             capture_output=True, text=True, check=True)
        available_images = set(result.stdout.strip().split('\n'))
        
        missing = set(required_images) - available_images
        if missing:
            print(f"   ❌ FAIL: Missing images: {missing}")
            return False
        
        print(f"   ✅ PASS: All required Docker images available")
        return True
        
    except Exception as e:
        print(f"   ❌ FAIL: Docker check failed: {e}")
        return False

def test_manifest_integrity():
    """Test 2: Verify manifest file exists and is valid."""
    print("🔍 Test 2/5: Manifest Integrity")
    
    try:
        with open('manifests/current.lock') as f:
            manifest_content = f.read()
        
        if len(manifest_content) < 100:
            print(f"   ❌ FAIL: Manifest too short ({len(manifest_content)} chars)")
            return False
        
        # Calculate hash for integrity
        manifest_hash = hashlib.sha256(manifest_content.encode()).hexdigest()[:16]
        print(f"   ✅ PASS: Manifest valid, hash: {manifest_hash}")
        return True
        
    except Exception as e:
        print(f"   ❌ FAIL: Manifest check failed: {e}")
        return False

def test_baseline_data():
    """Test 3: Verify baseline data file exists and is readable."""
    print("🔍 Test 3/5: Baseline Data Access")
    
    try:
        import pandas as pd
        baseline_path = "reports/active/2025-09-13_152035_v2.2.2/operational/rollup.csv"
        df = pd.read_csv(baseline_path)
        
        if len(df) < 10:
            print(f"   ❌ FAIL: Baseline data insufficient ({len(df)} rows)")
            return False
        
        print(f"   ✅ PASS: Baseline data loaded ({len(df)} rows)")
        return True
        
    except Exception as e:
        print(f"   ❌ FAIL: Baseline data check failed: {e}")
        return False

def test_monitoring_directory():
    """Test 4: Verify monitoring directory structure can be created."""
    print("🔍 Test 4/5: Monitoring Directory Structure")
    
    try:
        import os
        from pathlib import Path
        
        test_dir = Path(f"reports/{datetime.now().strftime('%Y%m%d')}/v2.3.0_microcanary")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create required subdirectories
        for subdir in ['operational', 'packets', 'technical', 'marketing', 'executive']:
            (test_dir / subdir).mkdir(exist_ok=True)
        
        # Test write access
        test_file = test_dir / 'operational' / 'health_check.json'
        with open(test_file, 'w') as f:
            f.write('{"test": "success"}')
        
        # Clean up
        test_file.unlink()
        
        print(f"   ✅ PASS: Monitoring directories created at {test_dir}")
        return True
        
    except Exception as e:
        print(f"   ❌ FAIL: Directory structure test failed: {e}")
        return False

def test_network_connectivity():
    """Test 5: Verify basic network connectivity for monitoring."""
    print("🔍 Test 5/5: Network Connectivity")
    
    try:
        # Test localhost connectivity (for monitoring endpoints)
        result = subprocess.run(['ping', '-c', '1', '-W', '2', '127.0.0.1'], 
                             capture_output=True, check=True)
        
        print(f"   ✅ PASS: Network connectivity verified")
        return True
        
    except Exception as e:
        print(f"   ❌ FAIL: Network connectivity test failed: {e}")
        return False

def main():
    """Run all smoke tests and report results."""
    print("🚀 LENS V2.3.0 PRODUCTION SMOKE TEST")
    print("=" * 50)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    
    tests = [
        test_docker_images,
        test_manifest_integrity,
        test_baseline_data,
        test_monitoring_directory,
        test_network_connectivity
    ]
    
    passed = 0
    failed = 0
    
    start_time = time.time()
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ EXCEPTION: {e}")
            failed += 1
        print()
    
    duration = time.time() - start_time
    
    print("=" * 50)
    print(f"📊 SMOKE TEST RESULTS: {passed}/5 PASSED")
    print(f"⏱️  Duration: {duration:.1f}s")
    
    if failed > 0:
        print(f"❌ FAILED: {failed} test(s) failed")
        sys.exit(1)
    else:
        print("✅ SUCCESS: All smoke tests passed")
        sys.exit(0)

if __name__ == "__main__":
    main()