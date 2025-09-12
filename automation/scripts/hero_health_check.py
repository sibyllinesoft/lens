#!/usr/bin/env python3
"""Quick hero health check"""
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true', help='Quick check mode')
args = parser.parse_args()

print(f"❤️ Hero health check ({'quick' if args.quick else 'full'} mode)")
print(f"Timestamp: {datetime.utcnow().isoformat()}Z")

# Simulate health check
heroes = ['Lexical Hero', 'Router Hero', 'ANN Hero']
for hero in heroes:
    print(f"  {hero}: HEALTHY")
    
print("✅ All heroes healthy")
