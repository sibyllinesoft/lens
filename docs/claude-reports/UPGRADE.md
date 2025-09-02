# Lens Upgrade Guide

This document provides step-by-step instructions for upgrading lens between versions.

## Current Version: v1.0.0

### From v0.x to v1.0.0

**⚠️ Breaking Changes:**
- API responses now include mandatory `api_version`, `index_version`, and `policy_version` fields
- `/compat/check` endpoint added for version compatibility validation  
- `/compat/bundles` endpoint added for nightly bundle compatibility checks
- CLI now supports migration commands

**Migration Steps:**

1. **Backup your data** (if applicable):
   ```bash
   # Backup any existing index data
   cp -r ./data ./data-backup-$(date +%Y%m%d)
   ```

2. **Stop the current lens service**:
   ```bash
   # Stop lens if running as a service
   pkill -f "lens" || true
   ```

3. **Install the new version**:
   ```bash
   # Using npm
   npm install -g lens@1.0.0

   # Or using the tarball
   tar -xzf lens-1.0.0.tar.gz
   cd lens-1.0.0
   npm install --production
   ```

4. **Run compatibility check** (new feature):
   ```bash
   # Check if your current setup is compatible
   lens compat-check --api-version v1 --index-version v1 --policy-version v1
   
   # Check against nightly bundles
   curl "http://localhost:3000/compat/bundles"
   ```

5. **Migrate existing indexes** (if any):
   ```bash
   # Migrate from v0 to v1 (currently a no-op)
   lens migrate-index --from v0 --to v1 --verbose
   ```

6. **Update configuration** (if needed):
   - No configuration changes required for v1.0.0
   - New optional environment variables available for security features

7. **Restart the service**:
   ```bash
   # Start lens server
   npm start
   
   # Or if installed globally
   lens server
   ```

8. **Verify the upgrade**:
   ```bash
   # Check version
   curl http://localhost:3000/health
   
   # Or using the CLI
   lens version
   ```

### API Changes in v1.0.0

**New Response Fields:**
All search responses now include:
```json
{
  "hits": [...],
  "api_version": "v1",
  "index_version": "v1",
  "policy_version": "v1",
  "..."
}
```

**New Endpoints:**
- `GET /compat/check?api_version=v1&index_version=v1&policy_version=v1` - Check version compatibility
- `GET /compat/bundles?allow_compat=false` - Check compatibility against nightly bundles

**Client Updates Required:**
If you have custom clients, update them to handle the new response fields:

```typescript
// Before
interface SearchResponse {
  hits: SearchHit[];
  total: number;
  latency_ms: LatencyBreakdown;
  trace_id: string;
}

// After (v1.0.0)
interface SearchResponse {
  hits: SearchHit[];
  total: number;
  latency_ms: LatencyBreakdown;
  trace_id: string;
  api_version: 'v1';      // New required field
  index_version: 'v1';    // New required field
  policy_version: 'v1';   // New required field
}
```

### Security Enhancements

**New Build Features:**
- SBOM (Software Bill of Materials) generation
- SAST (Static Application Security Testing) integration
- Enhanced dependency auditing

**Build with security features:**
```bash
# Generate SBOM and run security scans
lens build --sbom --sast --lock

# Build container with security artifacts
lens build --container --sbom --sast
```

### CLI Enhancements

**New Commands:**
```bash
# List available migrations
lens list-migrations

# Perform index migration
lens migrate-index --from v0 --to v1

# Check version compatibility
lens compat-check --api-version v1 --index-version v1 --policy-version v1

# Show detailed version info
lens version
```

### Rollback Instructions

If you need to rollback to a previous version:

1. **Stop the current service**:
   ```bash
   pkill -f "lens" || true
   ```

2. **Restore from backup** (if data was affected):
   ```bash
   # Restore data backup
   rm -rf ./data
   cp -r ./data-backup-YYYYMMDD ./data
   ```

3. **Install previous version**:
   ```bash
   # Install specific version
   npm install -g lens@0.9.0
   ```

4. **Restart service**:
   ```bash
   npm start
   ```

### Troubleshooting

**Common Issues:**

1. **Version Compatibility Errors**:
   ```bash
   # Check compatibility
   lens compat-check --api-version v1 --index-version v1 --policy-version v1
   
   # Check nightly bundle compatibility
   curl "http://localhost:3000/compat/bundles"
   
   # Force compatibility (not recommended)
   curl "http://localhost:3000/compat/check?api_version=v1&index_version=v1&policy_version=v1&allow_compat=true"
   ```

2. **Migration Failures**:
   ```bash
   # Run migration in dry-run mode first
   lens migrate-index --from v0 --to v1 --dry-run --verbose
   
   # Check migration logs
   lens list-migrations
   ```

3. **Build Issues**:
   ```bash
   # Clean build
   rm -rf node_modules dist
   npm install
   npm run build
   
   # Or use the secure build
   lens build --lock
   ```

4. **Permission Issues**:
   ```bash
   # Fix file permissions
   chmod +x scripts/build-secure.sh
   
   # Check user permissions for data directories
   ls -la ./data ./segments ./logs
   ```

### Performance Notes

- No performance regressions expected in v1.0.0
- New version compatibility checks add <10ms to response times
- Security scanning during build may increase build time by 30-60 seconds

### Support

For upgrade issues:
1. Check the troubleshooting section above
2. Review server logs for detailed error messages
3. Use `lens version` to verify installation
4. Run compatibility checks to identify version mismatches

### Next Steps

After successful upgrade:
1. Run the full test suite if you have custom integrations
2. Monitor performance for the first few days
3. Consider enabling security features in your CI/CD pipeline
4. Update any documentation or deployment scripts

---

**Version History:**
- v1.0.0: Initial stable release with version management and security features
- v0.9.x: Pre-release versions (deprecated)