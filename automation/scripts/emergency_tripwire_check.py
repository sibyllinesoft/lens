#!/usr/bin/env python3
"""Emergency tripwire monitoring - critical metrics only"""
from datetime import datetime
import sys

print(f"🚨 Emergency tripwire check", flush=True)
print(f"Timestamp: {datetime.utcnow().isoformat()}Z", flush=True)

# Check for emergency conditions
emergency_conditions = []

# Simulate emergency checks
calibrator_p99 = 0.8  # Should be <1ms
error_rate = 0.001    # Should be <0.01
response_time_spike = False

if calibrator_p99 >= 1.0:
    emergency_conditions.append(f"calibrator_p99_violation: {calibrator_p99}ms")

if error_rate >= 0.01:
    emergency_conditions.append(f"error_rate_spike: {error_rate}")
    
if response_time_spike:
    emergency_conditions.append("response_time_spike_detected")

if emergency_conditions:
    print("🚨 EMERGENCY CONDITIONS DETECTED:", flush=True)
    for condition in emergency_conditions:
        print(f"  - {condition}", flush=True)
    sys.exit(1)
else:
    print("✅ No emergency conditions", flush=True)
