#!/bin/bash

# Lens v2.2 Weekly Validation Cron Job
# Runs every Sunday at 02:00 local time
# Validates standing tripwires and triggers auto-revert if needed

set -euo pipefail

# Configuration
FINGERPRINT="v22_1f3db391_1757345166574"
BASELINE_DIR="./cron-tripwires/baselines"
LOG_DIR="./cron-tripwires/logs"
ALERT_DIR="./cron-tripwires/alerts"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOG_DIR/cron-validation-$TIMESTAMP.log"

# Ensure directories exist
mkdir -p "$BASELINE_DIR" "$LOG_DIR" "$ALERT_DIR"

echo "🔍 Starting Weekly Tripwire Validation - $TIMESTAMP" | tee "$LOG_FILE"
echo "📄 Baseline Fingerprint: $FINGERPRINT" | tee -a "$LOG_FILE"
echo "📅 Schedule: Weekly Sunday 02:00" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to log with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to trigger P0 alert
trigger_p0_alert() {
    local message="$1"
    local tripwire="$2"
    
    log "🚨 P0 ALERT: $tripwire - $message"
    
    # Create alert file
    cat > "$ALERT_DIR/p0-alert-$TIMESTAMP.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "severity": "P0",
    "tripwire": "$tripwire",
    "message": "$message",
    "fingerprint": "$FINGERPRINT",
    "log_file": "$LOG_FILE",
    "auto_revert_triggered": true
}
EOF
    
    # Send alert to monitoring system
    if command -v curl >/dev/null 2>&1; then
        curl -X POST "https://alerts.lens.dev/webhook" \
            -H "Content-Type: application/json" \
            -d @"$ALERT_DIR/p0-alert-$TIMESTAMP.json" \
            --max-time 10 || log "⚠️  Failed to send webhook alert"
    fi
    
    # Email alert (if configured)
    if command -v mail >/dev/null 2>&1; then
        echo "$message" | mail -s "🚨 Lens P0: $tripwire Tripwire Failed" ops-team@lens.dev || true
    fi
    
    return 0
}

# Function to trigger auto-revert
auto_revert_to_baseline() {
    local reason="$1"
    
    log "🔄 Initiating auto-revert to baseline fingerprint: $FINGERPRINT"
    log "📝 Reason: $reason"
    
    # Revert configuration to last known good state
    if [ -f "$BASELINE_DIR/config-$FINGERPRINT.json" ]; then
        cp "$BASELINE_DIR/config-$FINGERPRINT.json" "./config.json"
        log "✅ Configuration reverted to baseline"
    else
        log "❌ Baseline configuration not found"
        return 1
    fi
    
    # Restart services with baseline configuration
    if command -v systemctl >/dev/null 2>&1; then
        systemctl restart lens-search || log "⚠️  Failed to restart lens-search service"
        systemctl restart lens-api || log "⚠️  Failed to restart lens-api service"
        log "🔄 Services restarted with baseline configuration"
    fi
    
    # Verify revert was successful
    sleep 30
    if curl -f http://localhost:3000/health >/dev/null 2>&1; then
        log "✅ Auto-revert successful - services healthy"
        return 0
    else
        log "❌ Auto-revert failed - services unhealthy"
        return 1
    fi
}

# Main validation execution
main() {
    log "🚀 Starting v2.2 validation run with current HEAD"
    
    # Run the benchmark suite without changing configs
    if ! node validate-weekly-tripwires.js --baseline "$FINGERPRINT" >> "$LOG_FILE" 2>&1; then
        log "❌ Tripwire validation script failed"
        trigger_p0_alert "Validation script execution failed" "EXECUTION_FAILURE"
        exit 1
    fi
    
    # Parse validation results
    if [ -f "./validation-results-$TIMESTAMP.json" ]; then
        log "📊 Validation results generated successfully"
        
        # Check each tripwire
        node check-tripwire-results.js "./validation-results-$TIMESTAMP.json" >> "$LOG_FILE" 2>&1
        TRIPWIRE_EXIT_CODE=$?
        
        if [ $TRIPWIRE_EXIT_CODE -eq 0 ]; then
            log "✅ All tripwires PASSED - system healthy"
            
            # Update last successful validation timestamp
            echo "$(date -Iseconds)" > "$BASELINE_DIR/last-success.timestamp"
            
        elif [ $TRIPWIRE_EXIT_CODE -eq 1 ]; then
            log "🚨 One or more tripwires FAILED"
            
            # Trigger alerts and auto-revert
            trigger_p0_alert "Standing tripwires detected drift from baseline" "TRIPWIRE_FAILURE"
            
            if auto_revert_to_baseline "Tripwire validation failed"; then
                log "✅ Auto-revert completed successfully"
            else
                log "❌ Auto-revert FAILED - manual intervention required"
                trigger_p0_alert "Auto-revert failed - manual intervention required" "REVERT_FAILURE"
            fi
            
        else
            log "❌ Tripwire validation script error (exit code: $TRIPWIRE_EXIT_CODE)"
            trigger_p0_alert "Tripwire validation script error" "VALIDATION_ERROR"
        fi
        
    else
        log "❌ Validation results file not found"
        trigger_p0_alert "Validation results not generated" "MISSING_RESULTS"
    fi
    
    log "✅ Weekly validation cron job completed"
}

# Cleanup function
cleanup() {
    log "🧹 Cleaning up temporary files..."
    
    # Keep logs for 30 days
    find "$LOG_DIR" -name "cron-validation-*.log" -mtime +30 -delete 2>/dev/null || true
    
    # Keep alerts for 90 days  
    find "$ALERT_DIR" -name "p0-alert-*.json" -mtime +90 -delete 2>/dev/null || true
    
    # Keep validation results for 7 days
    find . -name "validation-results-*.json" -mtime +7 -delete 2>/dev/null || true
    
    log "✅ Cleanup completed"
}

# Trap cleanup on exit
trap cleanup EXIT

# Execute main function
main

exit 0
