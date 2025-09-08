#!/bin/bash

# Lens v2.2 Weekly Cron Installation Script
# Sets up cron job for Sunday 02:00 execution

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CRON_SCRIPT="$SCRIPT_DIR/weekly-validation.sh"
CRON_ENTRY="0 2 * * 0 $CRON_SCRIPT"

echo "⏰ Installing Lens v2.2 Weekly Validation Cron Job"
echo "📄 Script: $CRON_SCRIPT"
echo "📅 Schedule: Sundays at 02:00 (local time)"
echo ""

# Make script executable
chmod +x "$CRON_SCRIPT"
echo "✅ Made script executable"

# Backup existing crontab
crontab -l > /tmp/crontab-backup-$(date +%Y%m%d-%H%M%S) 2>/dev/null || true
echo "✅ Backed up existing crontab"

# Check if cron entry already exists
if crontab -l 2>/dev/null | grep -q "$CRON_SCRIPT"; then
    echo "⚠️  Cron entry already exists - removing old entry"
    crontab -l 2>/dev/null | grep -v "$CRON_SCRIPT" | crontab -
fi

# Add new cron entry
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
echo "✅ Added new cron entry"

# Verify installation
echo ""
echo "📋 Current crontab entries:"
crontab -l | grep -E "(weekly-validation|lens)" || echo "No Lens cron entries found"

echo ""
echo "🎉 Weekly cron installation complete!"
echo "⏰ Next run: $(date -d 'next sunday 02:00')"
echo "📝 Logs will be written to: ./cron-tripwires/logs/"
echo "🚨 P0 alerts will be written to: ./cron-tripwires/alerts/"

# Test cron script syntax
echo ""
echo "🧪 Testing cron script syntax..."
if bash -n "$CRON_SCRIPT"; then
    echo "✅ Cron script syntax is valid"
else
    echo "❌ Cron script has syntax errors"
    exit 1
fi

echo ""
echo "💡 To test the cron job manually:"
echo "   $CRON_SCRIPT"
echo ""
echo "💡 To remove the cron job:"
echo "   crontab -e  # Remove the line containing weekly-validation.sh"
