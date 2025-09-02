**TL;DR:** You’re ready to ship; now lock it in with a short **pre-GA verification**, a **canary plan** with crisp abort rules, an **online calibration check** (reliability → τ), and a **post-deploy drift watch**. Tag versions, freeze artifacts, and roll.

**Idea → mechanism.** Treat your current setup as the new baseline. Before user traffic, run a final **AnchorSmoke → LadderFull** on the exact corpus snapshot, logging: P\@1, nDCG\@10, Recall\@50, span, “why” histogram, results/query, and LTR model hash. Export a **release fingerprint** (policy JSON, τ from reliability curve, LTR weights, dedup params, feature schema). Verify “true recall” (candidates) is unchanged by dynamic\_topn/dedup; user-visible recall is then tracked at fixed k (e.g., 50) to avoid false regressions.

**Trade-offs → rollout.** Go canary 5%→25%→100% with **one knob per wave** (A: early-exit, B: dynamic\_topn(τ), C: dedup). Success at each step requires: Anchor deltas within gates (ΔnDCG\@10 ≥ 0, Recall\@50 Δ ≥ 0), p95 ≤ +10% vs pre-canary, p99 ≤ 2×p95, span = 100%, and **CUSUM alarms quiet** over a 24-hour window. Abort if hard-negative leakage to top-5 rises >1.0% abs, or results/query drift >±1 from the target envelope.

**Mechanism → guardrails online.** Keep **isotonic** as the final layer and recompute the **reliability diagram** on canary clicks/impressions daily; re-solve τ to maintain your 5 ± 2 results/query target, but only update after a 2-day holdout (prevent feedback loops). LTR: monitor feature sparsity/skew; if feature drift >3σ vs training, freeze LTR and fall back to calibrated Stage-C score. Keep **sentinel zero-result** probes (e.g., “class”, “def”) firing hourly; any miss trips a kill switch.

**Next steps (measurable, minimal ceremony).**

1. **Tag + freeze:** `policy_version++`, record `{api,index,policy}`, τ, model hashes in `config_fingerprint.json`.
2. **Final bench (pinned):** run AnchorSmoke + LadderFull; attach artifacts and sign-off summary.
3. **Start canary:** enable **Block A** only; watch the dashboard KPIs and alarms for 24h; promote or auto-rollback by flag.
4. **Then Block B → Block C**, one per day, same gates.
   When all three are green at 100%, declare GA and set the drift alarms (Anchor P\@1, Recall\@50; Ladder positives-in-candidates; LSIF/tree-sitter coverage) to page on sustained deviation.
