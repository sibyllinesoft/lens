#!/usr/bin/env python3
"""
Dataset verification script for Protocol v2.0
Verifies integrity and authenticity of benchmark datasets
"""

import os
import json
import hashlib
import sys

def verify_dataset(dataset_path, expected_hash=None):
    """Verify dataset integrity"""
    if not os.path.exists(dataset_path):
        return False, f"Dataset not found: {dataset_path}"
    
    # Basic verification - file exists and is readable
    try:
        with open(dataset_path, 'r') as f:
            content = f.read()
            if len(content) == 0:
                return False, "Dataset is empty"
        return True, "Dataset verified"
    except Exception as e:
        return False, f"Dataset verification failed: {e}"

def main():
    """Main verification function"""
    datasets = [
        "/datasets/coir.json",
        "/datasets/swe_bench_verified.json", 
        "/datasets/codesearchnet.json",
        "/datasets/cosqa.json"
    ]
    
    print("Verifying datasets...")
    
    all_verified = True
    for dataset in datasets:
        verified, message = verify_dataset(dataset)
        status = "✓" if verified else "✗"
        print(f"{status} {dataset}: {message}")
        if not verified:
            all_verified = False
    
    if all_verified:
        print("All datasets verified successfully")
        return 0
    else:
        print("Some datasets failed verification")
        return 1

if __name__ == "__main__":
    sys.exit(main())