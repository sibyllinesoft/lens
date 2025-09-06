/**
 * API Endpoints for RAPTOR Configuration & Rollout Management
 * 
 * Provides HTTP endpoints for managing RAPTOR configuration,
 * rollout controls, policy management, and kill switch operations.
 */

import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { ConfigRolloutManager, RaptorConfig, PolicyRule } from './config-rollout.js';

export interface ConfigEndpointOptions {
  prefix?: string;
  authentication?: boolean;
  rateLimit?: {
    max: number;
    timeWindow: string;
  };
}

/**
 * Register RAPTOR configuration endpoints
 */
export async function registerConfigEndpoints(
  fastify: FastifyInstance,
  configManager: ConfigRolloutManager,
  options: ConfigEndpointOptions = {}
): Promise<void> {
  const prefix = options.prefix || '/raptor/config';

  // Get current configuration
  fastify.get(`${prefix}/current`, async (request: FastifyRequest, reply: FastifyReply) => {
    try {
      const config = configManager.getConfig();
      const rolloutState = configManager.getRolloutState();
      
      return {
        config,
        rollout_state: rolloutState,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      reply.code(500).send({
        error: 'Failed to get configuration',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  // Update configuration
  fastify.put(`${prefix}/update`, {
    schema: {
      body: {
        type: 'object',
        properties: {
          config: { type: 'object' },
          reason: { type: 'string' }
        },
        required: ['config']
      }
    }
  }, async (request: FastifyRequest<{
    Body: { config: Partial<RaptorConfig>; reason?: string }
  }>, reply: FastifyReply) => {
    try {
      const { config: updates, reason } = request.body;
      const validation = await configManager.updateConfig(updates);
      
      if (!validation.valid) {
        reply.code(400).send({
          error: 'Configuration validation failed',
          validation
        });
        return;
      }

      return {
        success: true,
        validation,
        updated_config: configManager.getConfig(),
        reason: reason || 'API update',
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      reply.code(500).send({
        error: 'Failed to update configuration',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  // Validate configuration (dry-run)
  fastify.post(`${prefix}/validate`, {
    schema: {
      body: {
        type: 'object',
        properties: {
          config: { type: 'object' }
        },
        required: ['config']
      }
    }
  }, async (request: FastifyRequest<{
    Body: { config: RaptorConfig }
  }>, reply: FastifyReply) => {
    try {
      const validation = configManager.validateConfig(request.body.config);
      
      return {
        validation,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      reply.code(500).send({
        error: 'Failed to validate configuration',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  // Rollout management endpoints
  fastify.get(`${prefix}/rollout/status`, async (request: FastifyRequest, reply: FastifyReply) => {
    try {
      const rolloutState = configManager.getRolloutState();
      
      return {
        rollout_state: rolloutState,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      reply.code(500).send({
        error: 'Failed to get rollout status',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  // Update rollout target
  fastify.put(`${prefix}/rollout/target`, {
    schema: {
      body: {
        type: 'object',
        properties: {
          target_percentage: { type: 'number', minimum: 0, maximum: 100 },
          reason: { type: 'string' }
        },
        required: ['target_percentage']
      }
    }
  }, async (request: FastifyRequest<{
    Body: { target_percentage: number; reason?: string }
  }>, reply: FastifyReply) => {
    try {
      const { target_percentage, reason } = request.body;
      
      await configManager.updateRolloutTarget(target_percentage);
      
      return {
        success: true,
        new_target: target_percentage,
        current_state: configManager.getRolloutState(),
        reason: reason || 'Manual API update',
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      reply.code(500).send({
        error: 'Failed to update rollout target',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  // Test rollout decision for user
  fastify.post(`${prefix}/rollout/test`, {
    schema: {
      body: {
        type: 'object',
        properties: {
          user_id: { type: 'string' },
          query_intent: { type: 'string' }
        }
      }
    }
  }, async (request: FastifyRequest<{
    Body: { user_id?: string; query_intent?: string }
  }>, reply: FastifyReply) => {
    try {
      const { user_id, query_intent } = request.body;
      const decision = configManager.makeRolloutDecision(user_id, query_intent);
      
      return {
        decision,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      reply.code(500).send({
        error: 'Failed to test rollout decision',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  // Kill switch endpoints
  fastify.post(`${prefix}/killswitch/activate`, {
    schema: {
      body: {
        type: 'object',
        properties: {
          reason: { type: 'string' }
        },
        required: ['reason']
      }
    }
  }, async (request: FastifyRequest<{
    Body: { reason: string }
  }>, reply: FastifyReply) => {
    try {
      const { reason } = request.body;
      
      await configManager.activateKillSwitch(reason);
      
      return {
        success: true,
        kill_switch_active: true,
        reason,
        activated_at: new Date().toISOString()
      };
    } catch (error) {
      reply.code(500).send({
        error: 'Failed to activate kill switch',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  fastify.post(`${prefix}/killswitch/deactivate`, {
    schema: {
      body: {
        type: 'object',
        properties: {
          reason: { type: 'string' }
        },
        required: ['reason']
      }
    }
  }, async (request: FastifyRequest<{
    Body: { reason: string }
  }>, reply: FastifyReply) => {
    try {
      const { reason } = request.body;
      
      await configManager.deactivateKillSwitch(reason);
      
      return {
        success: true,
        kill_switch_active: false,
        reason,
        deactivated_at: new Date().toISOString()
      };
    } catch (error) {
      reply.code(500).send({
        error: 'Failed to deactivate kill switch',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  // Policy management endpoints
  fastify.get(`${prefix}/policies`, async (request: FastifyRequest, reply: FastifyReply) => {
    try {
      const policies = configManager.getPolicies();
      
      return {
        policies,
        count: policies.length,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      reply.code(500).send({
        error: 'Failed to get policies',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  fastify.post(`${prefix}/policies`, {
    schema: {
      body: {
        type: 'object',
        properties: {
          policy: {
            type: 'object',
            properties: {
              id: { type: 'string' },
              name: { type: 'string' },
              condition: { type: 'string' },
              action: { type: 'string', enum: ['allow', 'deny', 'modify'] },
              parameters: { type: 'object' },
              priority: { type: 'number' },
              enabled: { type: 'boolean' }
            },
            required: ['id', 'name', 'condition', 'action', 'priority']
          }
        },
        required: ['policy']
      }
    }
  }, async (request: FastifyRequest<{
    Body: { policy: PolicyRule }
  }>, reply: FastifyReply) => {
    try {
      const { policy } = request.body;
      
      configManager.addPolicy(policy);
      
      return {
        success: true,
        policy_added: policy,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      reply.code(500).send({
        error: 'Failed to add policy',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  fastify.delete(`${prefix}/policies/:policyId`, async (request: FastifyRequest<{
    Params: { policyId: string }
  }>, reply: FastifyReply) => {
    try {
      const { policyId } = request.params;
      const removed = configManager.removePolicy(policyId);
      
      if (!removed) {
        reply.code(404).send({
          error: 'Policy not found',
          policy_id: policyId
        });
        return;
      }

      return {
        success: true,
        removed_policy_id: policyId,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      reply.code(500).send({
        error: 'Failed to remove policy',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  // Feature flag control endpoints
  fastify.get(`${prefix}/features`, async (request: FastifyRequest, reply: FastifyReply) => {
    try {
      const config = configManager.getConfig();
      
      return {
        features: config.features,
        enabled_count: Object.values(config.features).filter(Boolean).length,
        total_count: Object.keys(config.features).length,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      reply.code(500).send({
        error: 'Failed to get feature flags',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  fastify.put(`${prefix}/features/:featureName`, {
    schema: {
      body: {
        type: 'object',
        properties: {
          enabled: { type: 'boolean' },
          reason: { type: 'string' }
        },
        required: ['enabled']
      }
    }
  }, async (request: FastifyRequest<{
    Params: { featureName: string };
    Body: { enabled: boolean; reason?: string }
  }>, reply: FastifyReply) => {
    try {
      const { featureName } = request.params;
      const { enabled, reason } = request.body;
      
      const currentConfig = configManager.getConfig();
      
      if (!(featureName in currentConfig.features)) {
        reply.code(404).send({
          error: 'Feature not found',
          feature_name: featureName,
          available_features: Object.keys(currentConfig.features)
        });
        return;
      }

      const updates = {
        features: {
          ...currentConfig.features,
          [featureName]: enabled
        }
      };
      
      const validation = await configManager.updateConfig(updates);
      
      if (!validation.valid) {
        reply.code(400).send({
          error: 'Feature update validation failed',
          validation
        });
        return;
      }

      return {
        success: true,
        feature_name: featureName,
        enabled,
        reason: reason || 'Manual API update',
        validation,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      reply.code(500).send({
        error: 'Failed to update feature flag',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  // Batch feature update
  fastify.put(`${prefix}/features`, {
    schema: {
      body: {
        type: 'object',
        properties: {
          features: { type: 'object' },
          reason: { type: 'string' }
        },
        required: ['features']
      }
    }
  }, async (request: FastifyRequest<{
    Body: { features: Record<string, boolean>; reason?: string }
  }>, reply: FastifyReply) => {
    try {
      const { features, reason } = request.body;
      
      const currentConfig = configManager.getConfig();
      const updates = {
        features: {
          ...currentConfig.features,
          ...features
        }
      };
      
      const validation = await configManager.updateConfig(updates);
      
      if (!validation.valid) {
        reply.code(400).send({
          error: 'Batch feature update validation failed',
          validation
        });
        return;
      }

      return {
        success: true,
        updated_features: features,
        reason: reason || 'Batch API update',
        validation,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      reply.code(500).send({
        error: 'Failed to batch update feature flags',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  // Health and monitoring endpoints
  fastify.get(`${prefix}/health`, async (request: FastifyRequest, reply: FastifyReply) => {
    try {
      const rolloutState = configManager.getRolloutState();
      const config = configManager.getConfig();
      
      return {
        system_health: {
          enabled: config.enabled,
          kill_switch_active: config.kill_switch_active,
          rollout_percentage: rolloutState.current_percentage,
          status: rolloutState.status,
          health_metrics: rolloutState.health_metrics
        },
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      reply.code(500).send({
        error: 'Failed to get health status',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  // Export configuration
  fastify.get(`${prefix}/export`, async (request: FastifyRequest, reply: FastifyReply) => {
    try {
      const config = configManager.getConfig();
      const policies = configManager.getPolicies();
      const rolloutState = configManager.getRolloutState();
      
      const exportData = {
        config,
        policies,
        rollout_state: rolloutState,
        exported_at: new Date().toISOString(),
        export_version: '1.0'
      };
      
      reply.header('Content-Type', 'application/json');
      reply.header('Content-Disposition', `attachment; filename="raptor-config-export-${new Date().toISOString().split('T')[0]}.json"`);
      
      return exportData;
    } catch (error) {
      reply.code(500).send({
        error: 'Failed to export configuration',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  // Import configuration
  fastify.post(`${prefix}/import`, {
    schema: {
      body: {
        type: 'object',
        properties: {
          config: { type: 'object' },
          policies: { type: 'array' },
          overwrite: { type: 'boolean' },
          reason: { type: 'string' }
        },
        required: ['config']
      }
    }
  }, async (request: FastifyRequest<{
    Body: {
      config: RaptorConfig;
      policies?: PolicyRule[];
      overwrite?: boolean;
      reason?: string;
    }
  }>, reply: FastifyReply) => {
    try {
      const { config, policies, overwrite = false, reason } = request.body;
      
      // Validate imported config
      const validation = configManager.validateConfig(config);
      if (!validation.valid) {
        reply.code(400).send({
          error: 'Imported configuration validation failed',
          validation
        });
        return;
      }

      // Update config
      await configManager.updateConfig(config);
      
      // Update policies if provided
      let importedPolicies = 0;
      if (policies && policies.length > 0) {
        if (overwrite) {
          // Clear existing policies
          const existingPolicies = configManager.getPolicies();
          for (const policy of existingPolicies) {
            configManager.removePolicy(policy.id);
          }
        }
        
        // Add new policies
        for (const policy of policies) {
          configManager.addPolicy(policy);
          importedPolicies++;
        }
      }

      return {
        success: true,
        imported_config: true,
        imported_policies: importedPolicies,
        overwrite_mode: overwrite,
        validation,
        reason: reason || 'Configuration import',
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      reply.code(500).send({
        error: 'Failed to import configuration',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  console.log(`✅ RAPTOR config endpoints registered with prefix: ${prefix}`);
}

/**
 * Configuration endpoint schemas for OpenAPI documentation
 */
export const configEndpointSchemas = {
  getCurrentConfig: {
    description: 'Get current RAPTOR configuration and rollout state',
    tags: ['RAPTOR Config'],
    response: {
      200: {
        type: 'object',
        properties: {
          config: { type: 'object' },
          rollout_state: { type: 'object' },
          timestamp: { type: 'string' }
        }
      }
    }
  },
  
  updateConfig: {
    description: 'Update RAPTOR configuration with validation',
    tags: ['RAPTOR Config'],
    body: {
      type: 'object',
      properties: {
        config: { type: 'object' },
        reason: { type: 'string' }
      },
      required: ['config']
    }
  },
  
  activateKillSwitch: {
    description: 'Activate emergency kill switch to disable RAPTOR',
    tags: ['RAPTOR Config'],
    body: {
      type: 'object',
      properties: {
        reason: { type: 'string' }
      },
      required: ['reason']
    }
  }
};