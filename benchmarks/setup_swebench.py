#!/usr/bin/env python3
"""
SWE-bench Verified Dataset Setup
Industry standard for code task evaluation
"""

import json
import requests
from pathlib import Path
from typing import Dict, List, Any

class SWEBenchSetup:
    def __init__(self):
        self.base_url = "https://raw.githubusercontent.com/princeton-nlp/SWE-bench/main"
        self.dataset_dir = Path("./datasets/swe-bench-verified")
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
    
    def download_verified_dataset(self):
        """Download SWE-bench Verified (500 expert-screened items)"""
        print("📥 Downloading SWE-bench Verified dataset...")
        
        verified_url = f"{self.base_url}/swebench/verified/test.jsonl"
        response = requests.get(verified_url)
        
        if response.status_code == 200:
            with open(self.dataset_dir / "verified_test.jsonl", "w") as f:
                f.write(response.text)
            print("✅ SWE-bench Verified downloaded")
        else:
            print("❌ Failed to download SWE-bench Verified")
    
    def create_witness_spans(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert PR diffs to witness spans for evaluation"""
        witness_spans = []
        
        for item in items:
            # Extract spans from patch/diff
            patch = item.get('patch', '')
            test_patch = item.get('test_patch', '')
            
            # Parse diff to get file:line ranges
            spans = self.parse_diff_spans(patch)
            
            witness_spans.append({
                'instance_id': item['instance_id'],
                'problem_statement': item['problem_statement'],
                'witness_spans': spans,
                'success_criteria': 'FAIL→PASS test transition',
                'repository': item.get('repo', ''),
                'base_commit': item.get('base_commit', ''),
                'test_files': self.parse_test_files(test_patch)
            })
        
        return witness_spans
    
    def parse_diff_spans(self, patch: str) -> List[Dict[str, Any]]:
        """Parse diff to extract file:line spans"""
        spans = []
        current_file = None
        
        for line in patch.split('\n'):
            if line.startswith('diff --git'):
                # Extract file path
                parts = line.split()
                if len(parts) >= 4:
                    current_file = parts[3][2:]  # Remove 'b/' prefix
            elif line.startswith('@@') and current_file:
                # Extract line range
                import re
                match = re.search(r'@@.*\+(\d+),?(\d+)?', line)
                if match:
                    start_line = int(match.group(1))
                    line_count = int(match.group(2)) if match.group(2) else 1
                    
                    spans.append({
                        'file': current_file,
                        'start_line': start_line,
                        'end_line': start_line + line_count - 1,
                        'type': 'witness_span'
                    })
        
        return spans
    
    def parse_test_files(self, test_patch: str) -> List[str]:
        """Extract test file paths from test patch"""
        test_files = []
        for line in test_patch.split('\n'):
            if line.startswith('diff --git'):
                parts = line.split()
                if len(parts) >= 4:
                    test_files.append(parts[3][2:])
        return test_files

if __name__ == "__main__":
    setup = SWEBenchSetup()
    setup.download_verified_dataset()
    print("🎯 SWE-bench Verified setup complete")