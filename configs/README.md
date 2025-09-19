# Configuration Packs

This directory collects example configuration bundles used during benchmarking
and deployment dry-runs. The files are not consumed by the Lens binary at
runtime, but act as documented starting points for operators:

- `*_pack_*.yaml` files outline candidate search/index tunings for different
  environments (ANN heavy, lexical focused, router experiments).
- `baseline.json`, `production_monitoring_config.json`, and similar JSON files
  capture historical Prometheus/Grafana wiring and NATS routing strategies.
- The `settings/` and `benchmarks/` sub-directories store supporting inputs for
  the above packs.

Feel free to prune or replace these artefacts when adopting Lens in your own
infrastructure. If you create a new profile, document the expected usage and
inputs here so future maintainers understand how to reproduce the setup.
