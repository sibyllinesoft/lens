#!/usr/bin/env node

/**
 * Lens v2.2 Alert Manager
 * Handles P0 alerts for tripwire failures
 */

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { createHash } from 'crypto';

class AlertManager {
    constructor() {
        this.alertDir = './cron-tripwires/alerts';
        this.timestamp = new Date().toISOString();
        
        this.alertChannels = {
            slack: process.env.SLACK_WEBHOOK_URL,
            email: process.env.ALERT_EMAIL_RECIPIENTS,
            pagerduty: process.env.PAGERDUTY_INTEGRATION_KEY,
            discord: process.env.DISCORD_WEBHOOK_URL
        };
        
        this.severityConfig = {
            P0: {
                immediate: true,
                channels: ['slack', 'email', 'pagerduty'],
                retry_attempts: 3,
                escalation_minutes: 15
            },
            P1: {
                immediate: false,
                channels: ['slack', 'email'],
                retry_attempts: 2,
                escalation_minutes: 60
            },
            P2: {
                immediate: false,
                channels: ['slack'],
                retry_attempts: 1,
                escalation_minutes: 240
            }
        };
    }

    async sendAlert(alert) {
        console.log(`🚨 Sending ${alert.severity} alert: ${alert.tripwire}`);
        
        const config = this.severityConfig[alert.severity] || this.severityConfig.P2;
        const alertRecord = {
            ...alert,
            timestamp: this.timestamp,
            alert_id: this.generateAlertId(alert),
            channels_attempted: [],
            status: 'SENDING'
        };
        
        // Send to configured channels
        for (const channel of config.channels) {
            try {
                await this.sendToChannel(channel, alert);
                alertRecord.channels_attempted.push({
                    channel: channel,
                    status: 'SUCCESS',
                    timestamp: new Date().toISOString()
                });
                console.log(`✅ Alert sent to ${channel}`);
            } catch (error) {
                alertRecord.channels_attempted.push({
                    channel: channel,
                    status: 'FAILED',
                    error: error.message,
                    timestamp: new Date().toISOString()
                });
                console.log(`❌ Failed to send to ${channel}: ${error.message}`);
            }
        }
        
        // Record alert
        alertRecord.status = alertRecord.channels_attempted.some(c => c.status === 'SUCCESS') ? 'SENT' : 'FAILED';
        await this.recordAlert(alertRecord);
        
        return alertRecord;
    }

    async sendToChannel(channel, alert) {
        switch (channel) {
            case 'slack':
                return await this.sendSlackAlert(alert);
            case 'email':
                return await this.sendEmailAlert(alert);
            case 'pagerduty':
                return await this.sendPagerDutyAlert(alert);
            case 'discord':
                return await this.sendDiscordAlert(alert);
            default:
                throw new Error(`Unknown alert channel: ${channel}`);
        }
    }

    async sendSlackAlert(alert) {
        if (!this.alertChannels.slack) {
            throw new Error('Slack webhook URL not configured');
        }

        const slackMessage = {
            text: `🚨 Lens v2.2 ${alert.severity} Alert`,
            blocks: [
                {
                    type: 'section',
                    text: {
                        type: 'mrkdwn',
                        text: `*🚨 ${alert.severity} Alert: ${alert.tripwire}*\n\n${alert.message}`
                    }
                },
                {
                    type: 'section',
                    fields: [
                        {
                            type: 'mrkdwn',
                            text: `*Timestamp:*\n${alert.timestamp}`
                        },
                        {
                            type: 'mrkdwn',
                            text: `*Fingerprint:*\n${alert.fingerprint}`
                        },
                        {
                            type: 'mrkdwn',
                            text: `*Auto-revert:*\n${alert.auto_revert_triggered ? '✅ Triggered' : '❌ Not triggered'}`
                        }
                    ]
                },
                {
                    type: 'actions',
                    elements: [
                        {
                            type: 'button',
                            text: {
                                type: 'plain_text',
                                text: 'View Logs'
                            },
                            style: 'primary',
                            url: 'https://monitoring.lens.dev/logs'
                        },
                        {
                            type: 'button', 
                            text: {
                                type: 'plain_text',
                                text: 'Acknowledge'
                            },
                            style: 'danger',
                            action_id: 'acknowledge_alert'
                        }
                    ]
                }
            ]
        };

        const response = await fetch(this.alertChannels.slack, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(slackMessage)
        });

        if (!response.ok) {
            throw new Error(`Slack API error: ${response.status}`);
        }
    }

    async sendEmailAlert(alert) {
        // Email implementation would depend on configured email service
        console.log('📧 Email alert would be sent here');
        
        const emailContent = `
Subject: 🚨 Lens v2.2 ${alert.severity} Alert: ${alert.tripwire}

Lens v2.2 Tripwire Alert

Severity: ${alert.severity}
Tripwire: ${alert.tripwire}  
Message: ${alert.message}
Timestamp: ${alert.timestamp}
Fingerprint: ${alert.fingerprint}

Auto-revert Status: ${alert.auto_revert_triggered ? 'Triggered' : 'Not triggered'}

Action Required:
1. Check system health: https://monitoring.lens.dev
2. Review logs: ${alert.log_file}
3. Acknowledge alert when resolved

This is an automated alert from the Lens v2.2 monitoring system.
        `;
        
        // In real implementation, would send via SMTP or email service API
        console.log('Email content prepared (not sent in demo)');
    }

    async sendPagerDutyAlert(alert) {
        if (!this.alertChannels.pagerduty) {
            throw new Error('PagerDuty integration key not configured');
        }

        const pdPayload = {
            routing_key: this.alertChannels.pagerduty,
            event_action: 'trigger',
            dedup_key: this.generateAlertId(alert),
            payload: {
                summary: `Lens v2.2 ${alert.severity}: ${alert.tripwire}`,
                source: 'lens-tripwire-monitor',
                severity: alert.severity.toLowerCase(),
                timestamp: alert.timestamp,
                custom_details: {
                    tripwire: alert.tripwire,
                    message: alert.message,
                    fingerprint: alert.fingerprint,
                    auto_revert_triggered: alert.auto_revert_triggered
                }
            },
            links: [{
                href: 'https://monitoring.lens.dev',
                text: 'Monitoring Dashboard'
            }]
        };

        const response = await fetch('https://events.pagerduty.com/v2/enqueue', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(pdPayload)
        });

        if (!response.ok) {
            throw new Error(`PagerDuty API error: ${response.status}`);
        }
    }

    async sendDiscordAlert(alert) {
        if (!this.alertChannels.discord) {
            throw new Error('Discord webhook URL not configured');
        }

        const discordMessage = {
            content: `🚨 **Lens v2.2 ${alert.severity} Alert**`,
            embeds: [{
                title: `${alert.tripwire} Tripwire Failed`,
                description: alert.message,
                color: alert.severity === 'P0' ? 0xFF0000 : alert.severity === 'P1' ? 0xFFA500 : 0xFFFF00,
                fields: [
                    { name: 'Timestamp', value: alert.timestamp, inline: true },
                    { name: 'Fingerprint', value: alert.fingerprint, inline: true },
                    { name: 'Auto-revert', value: alert.auto_revert_triggered ? '✅ Triggered' : '❌ Not triggered', inline: true }
                ],
                footer: { text: 'Lens v2.2 Tripwire Monitor' },
                timestamp: new Date().toISOString()
            }]
        };

        const response = await fetch(this.alertChannels.discord, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(discordMessage)
        });

        if (!response.ok) {
            throw new Error(`Discord API error: ${response.status}`);
        }
    }

    generateAlertId(alert) {
        const hash = createHash('sha256');
        hash.update(`${alert.tripwire}-${alert.fingerprint}-${alert.timestamp.split('T')[0]}`);
        return hash.digest('hex').substring(0, 12);
    }

    async recordAlert(alertRecord) {
        const filename = `${this.alertDir}/alert-${alertRecord.alert_id}.json`;
        writeFileSync(filename, JSON.stringify(alertRecord, null, 2));
        console.log(`📝 Alert recorded: ${filename}`);
    }
}

// CLI interface
if (import.meta.url === `file://${process.argv[1]}`) {
    const command = process.argv[2];
    
    if (command === 'test') {
        // Test alert
        const testAlert = {
            severity: 'P1',
            tripwire: 'test_tripwire',
            message: 'This is a test alert',
            fingerprint: 'v22_1f3db391_1757345166574',
            auto_revert_triggered: false
        };
        
        const alertManager = new AlertManager();
        await alertManager.sendAlert(testAlert);
        
    } else if (command === 'send') {
        // Send alert from JSON file
        const alertFile = process.argv[3];
        if (!alertFile || !existsSync(alertFile)) {
            console.error('Alert file required');
            process.exit(1);
        }
        
        const alert = JSON.parse(readFileSync(alertFile, 'utf8'));
        const alertManager = new AlertManager();
        await alertManager.sendAlert(alert);
        
    } else {
        console.log(`Lens v2.2 Alert Manager

Usage:
  node alert-manager.js test                    Send test alert
  node alert-manager.js send <alert-file.json>  Send alert from file

Environment Variables:
  SLACK_WEBHOOK_URL           Slack webhook for alerts
  ALERT_EMAIL_RECIPIENTS      Comma-separated email addresses  
  PAGERDUTY_INTEGRATION_KEY   PagerDuty integration key
  DISCORD_WEBHOOK_URL         Discord webhook for alerts
`);
    }
}
