**TL;DR:** Wire the framework to real endpoints, run an A/A→A/B canary under the v2.2 gates, and replace every simulator with production signals; ship Sprint-1 tail-taming behind a flag, then lock replication on real pools. Below is a crisp, integration-first plan with acceptance criteria.

### 0) Assumptions

* Lens `/search` is NDJSON and returns span-accurate hits with `why:[lex|struct|sem]`, latency, and shard ids.
* Metrics engine is `@lens/metrics` and the scoreboard uses pooled-qrels from v2.2.
* Simulators currently feed the monitoring, transparency pages, and weekly cron.

### 1) Convert “simulated” → “real” (48 hours, merge in stages)

**Idea:** Replace data sources at the boundary, not in consumers. Keep simulator as a fallback via env flag.

**Mechanism:**

* **Config:** `DATA_SOURCE=sim|prod`; `LENS_ENDPOINTS=[…]`; `AUTH_TOKEN=…`; `SLA_MS=150`.
* **Client:** idempotent, retry-aware fetcher with `deadline = SLA_MS` and `cancel_on_first`.
* **Emit:** one canonical record per query → `agg.parquet` row schema, plus `hits.parquet`.
* **Map fields:** `req_ts, shard, lat_ms, within_sla=(lat_ms<=150), why_mix_* from hit.why counts`.

**Trade-offs:** Minimal churn, but you must hard-fail on schema drift to preserve artifact binding.

**Next steps:**

1. Implement `ProdIngestor` beside `SimIngestor`; add parity tests (same query ids → same row counts).
2. Add **schema guard**: refuse writes if any required column missing; stamp `attestation_sha256`.
3. Flip staging to `DATA_SOURCE=prod`, run 10k queries (mixed suites) to populate *new* `runs/staging/`.
4. Run `bench score` span-only with pooled-qrels; publish staging dashboard privately.

**Definition of Done (DoD):** No simulator process is required to render the live dashboards; staging run passes **all gates** (ECE, power, pool, p99/p95).

---

### 2) Sprint-1 (Tail-taming) integration (one week, gated)

**Idea:** Land tail-taming behind a router flag; measure SLA-bounded recall not just latency.

**Mechanism:**

* **Hedged probes:** secondary probe at `t = min(6ms, 0.1·p50_shard)`; cancel on first success.
* **Cooperative cancel:** shard workers observe `ctx` cancellation; maintain visited-set reuse.
* **Cross-shard TA/NRA:** global threshold maintains min required score; stop shards early.
* **Learning-to-stop:** feature vector ⟨top-k gap, shard residuals, elapsed, efSearch⟩ → logistic stop; clamp with monotone floor.

**Gates (all must hold vs v2.2):**

* `p99_latency`: −10–15% *and* `p99/p95 ≤ 2.0`
* `SLA-Recall@50`: ≥ baseline (Δ ≥ 0.0 pp)
* `QPS@150ms`: +10–15%
* Cost: ≤ +5%

**Rollout:** 5%→25%→50%→100% traffic by repo bucket; tripwire auto-revert on any gate breach for two consecutive 15-min windows.

**Next steps:**

1. Add router flags: `TAIL_HEDGE=true`, `HEDGE_DELAY_MS`, `TA_STOP=true`, `LTS_STOP=true`.
2. Instrument per-shard spans: `probe_id, issued_ts, first_byte_ts, cancel_ts`.
3. Canary A/A (flags off vs off) to establish *noise floor*; then A/B: control(flags off) vs test(flags on).
4. Score **SLA-Recall\@50** hourly; block promotion unless CI width ≤ 0.03 (paired bootstrap B≥2000).

**DoD:** Canary at 100% with green gates for 24h; commit flags default-on; artifacts stamped with new cfg hash.

---

### 3) Replication kit: move to *real pools* (2–3 days)

**Idea:** Your kit is solid but simulated; swap in production pool manifests and enforce parity embeddings + SLA mask.

**Mechanism:**

* Bundle `pool/` built from union of **in-SLA** top-k across systems; include `pool_counts_by_system.csv`.
* Freeze `Gemma-256` parity weights (digest in attestation).
* Tighten `make repro`: assert **ECE ≤ 0.02** per intent×language; clamp isotonic slope to \[0.9, 1.1].

**Next steps:**

1. Publish `pool/` and `hero_span_v22.csv` from *production runs* (not simulator).
2. Update kit README with **SLA note** and **fingerprint** `v22_1f3db391_1757345166574`.
3. Run partner dry-run on 1k queries; verify ±0.1 pp tolerance, then full 48,768.

**DoD:** External lab returns attested `hero_span_v22.csv` within tolerance; we host “replicated” badge.

---

### 4) Transparency & weekly cron: bind to prod (1 day)

**Idea:** Keep the simulator for local dev, but public pages must source production fingerprints.

**Mechanism:**

* Cron at Sun 02:00 uses `DATA_SOURCE=prod`; on green gates → publish new fingerprint; else auto-revert and open P0.
* Leaderboard renders CI whiskers and `p99/p95` per system; link pool audit & ECE reliability diagrams.

**Next steps:** Wire the cron job creds to call Lens endpoints and write to the immutable bucket; add “pool membership” counts widget.

**DoD:** First cron run produces a *public* green fingerprint without manual edits.

---

### 5) Sprint-2 prep (in parallel, but don’t ship yet)

**Idea:** Build the harness so shipping is a config change post Sprint-1.

**Mechanism:**

* Lexical phrase/prox scorer with impact-ordered postings; backoff “panic exactifier” under high entropy.
* **Gate:** +1–2 pp on lexical slices, ≤ +0.5 ms p95.
* Precompute phrase windows for hot n-grams to keep SLA flat.

**DoD:** Benchmark report with Pareto curves (quality vs ms) and reproducible cfg hashes.

---

### 6) Calibration sanity (continuous)

**Idea:** Guard against confidence drift as tails change.

**Mechanism:**

* Refit isotonic per intent×language each weekly cron; **slope clamp \[0.9,1.1]**; assert `ECE ≤ 0.02`.
* Tripwire: if clamp activates >10% of bins, open P1 for calibration drift.

---

### 7) Minimal pseudocode (for avoidance of doubt)

```pseudo
for q in suite:
  t0 = now()
  ctl = search(q, flags=OFF, deadline=150)
  tst = search(q, flags=TAIL_TAMING, deadline=150)
  log_rows = toAggRows([ctl, tst], qid=q.id, cfg_hash, sla=150)
  writeParquet(agg, log_rows); writeParquet(hits, toHits(ctl,tst))
score = benchScore(agg, pool, credit=span_only, bootstrap=2000, permute=holm)
assert gates(score)  // ECE, power, CI width, p99/p95, SLA-Recall@50
```

---

### 8) Owner map & timeline

* **Prod wiring (Section 1):** 2 days — *Infra owner*.
* **Sprint-1 canary (Section 2):** 5–7 days — *Search core*.
* **Replication swap (Section 3):** 2–3 days — *Benchmarks owner*.
* **Cron & transparency (Section 4):** 1 day — *DevEx*.
* **Calibration (Section 6):** ongoing — *ML owner*.

---

### 9) Risks → Mitigations

* **Pool drift:** lock pool build from in-SLA tops; publish membership deltas.
* **Hidden cost creep:** budget tracker per query class; fail build if >5% delta.
* **Router mis-spend:** entropy-conditioned spend cap; conformal guard for upshifts.

---

When these are green, Sprint-1 ships by flipping defaults; you’ll have removed the last “simulated” crutch, proven tail wins under SLA-Recall, and set the table for small, defensible lexical and ANN gains next.
