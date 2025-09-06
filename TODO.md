**TL;DR:** Leave search alone. Add a small LSP-backed SPI to Lens—diagnostics, format, selection/folding ranges, rename (prepare+workspace edit), code actions, and call/type hierarchy—plus deterministic ordering, caching by `source_hash`, and metrics. These unlock safer `spatch/srefactor`, tighter `scontext`, and smarter `simpact` without changing your current search path.

### Assumptions (explicit)

* You already expose `/v1/spi/{search,resolve,context,xref}` and bake LSP signals into **search**.
* Lens can talk to per-lang LSP servers you run sidecar or in-proc.
* `ref = lens://{repo_sha}/{path}@{source_hash}#B{start}:{end}` is the stable address.

---

## 1) New SPI endpoints (LSP-backed, read-only + edit synthesis)

All endpoints must: accept `budget_ms`, return `duration_ms`, sort outputs deterministically, and be cacheable by `(repo_sha, path, source_hash)`. Return `429/504` on budget exceed, not silent truncation.

**Capabilities**

* `GET /v1/spi/lsp/capabilities?repo_sha=…`

  * → `{ languages:[{lang, features:[diagnostics,format,selectionRanges,foldingRanges,prepareRename,rename,codeActions,callHierarchy,typeHierarchy]}] }`

**Diagnostics (fast verify gate)**

* `POST /v1/spi/lsp/diagnostics`

  * Req: `{ files:[{path, source_hash}], budget_ms }`
  * Res: `{ diags:[{path, source_hash, items:[{range:{b0,b1}, severity∈{hint,info,warning,error}, code?, message}]}], duration_ms }`

**Format (idempotence & normalization)**

* `POST /v1/spi/lsp/format`

  * Req: `{ ref? , path?, range? , options? , budget_ms }`
  * Res: `{ edits:[{path, range:{b0,b1}, new_text}], idempotent:boolean, duration_ms }`
  * Contract: applying `edits` twice yields no diff when `idempotent=true`.

**Selection & Folding (snippet fences)**

* `POST /v1/spi/lsp/selectionRanges`

  * Req: `{ refs:[ref], budget_ms }`
  * Res: `{ chains:[[{range, parent_ix?}]], duration_ms }`  // outer→inner chain per ref
* `POST /v1/spi/lsp/foldingRanges`

  * Req: `{ files:[{path, source_hash}], budget_ms }`
  * Res: `{ folds:[{path, ranges:[{b0,b1,kind}] }], duration_ms }`

**Rename / Workspace Edit (safe multi-file changes)**

* `POST /v1/spi/lsp/prepareRename`

  * Req: `{ ref, budget_ms }`
  * Res: `{ allowed:boolean, placeholder?:string, range?:{b0,b1}, reason? }`
* `POST /v1/spi/lsp/rename`

  * Req: `{ ref, new_name, budget_ms }`
  * Res: `{ workspaceEdit:{ changes:[{path, source_hash, edits:[{b0,b1,new_text}]}] }, duration_ms }`
  * Determinism: sort `changes` by `path`, `edits` by `b0`.

**Code Actions (quick-fixes & refactors as proposals)**

* `POST /v1/spi/lsp/codeActions`

  * Req: `{ ref, kinds?:[\"quickfix\",\"refactor\",\"source.organizeImports\",…], diagnostics?[], budget_ms }`
  * Res: `{ actions:[{title, kind, workspaceEdit?, data?}], duration_ms }`
  * Do **not** execute arbitrary commands; only return pure text edits.

**Hierarchy (impact ranking substrate)**

* `POST /v1/spi/lsp/hierarchy`

  * Req: `{ ref, kind∈{\"call\",\"type\"}, dir∈{\"incoming\",\"outgoing\"}, depth?:int, fanout_cap?:int, budget_ms }`
  * Res: `{ nodes:[{symbol_id,name,kind,def_ref?}], edges:[{src:symbol_id,dst:symbol_id,role}], truncated:boolean, duration_ms }`

---

## 2) Engine/runtime changes (minimal but important)

* **Caching & invalidation**

  * Cache responses by `(repo_sha, path, source_hash)`; TTL = 10–60s.
  * Invalidate on index change or when `source_hash` differs from working tree.
  * Batch LSP calls per language server to reduce chattiness; enforce `budget_ms`.

* **Determinism**

  * Sort diagnostics by `(severity desc, b0 asc, code asc?)`.
  * Sort actions by `(kind, title, first_edit.b0)`.
  * Canonicalize newline/encoding in `format` and `workspaceEdit` outputs.
  * Include `seed` echo and `trace:{stages[],timings}` in every response.

* **Safety & sandbox**

  * Formatting may shell out (prettier, rustfmt, gofmt): execute in a sandbox with CPU/mem caps and a 2× budget sub-cap.
  * Reject code actions that contain ExecuteCommand-type side effects; allow only text edits.

* **Observability**

  * New metrics: `lsp_diag_latency_ms{lang}`, `lsp_format_idempotent_rate`, `lsp_rename_size_edits`, `lsp_cache_hit_ratio`, `lsp_timeout_rate`.
  * Structured logs include repo, path, source\_hash, counts, truncated flag.

---

## 3) Data model tweaks (no search changes)

* Introduce a **LensSymbol** view (in-memory or persisted) to attach stable `symbol_id` and `moniker` to `ref`s returned by hierarchy/rename:

  * `{ symbol_id, lang, name, kind, def_ref, container[], moniker? }`
  * Build on demand from LSP or from your existing symbol index; **not** required to modify `/search` responses.

---

## 4) Backward compatibility & rollout

* All endpoints are **additive** under `/v1/spi/lsp/*`. No changes to `/search`.
* Feature flag per language (`lsp.enabled.go=true`, etc.).
* Canary policy: error budget <1% for `lsp_*` endpoints in a week before GA; auto-disable specific langs on crash/timeout spikes.

---

## 5) Contract tests (must pass before GA)

* **Format idempotence:** apply returned edits twice ⇒ identical tree/hash.
* **Diagnostics stability:** re-ordering of unrelated edits doesn’t reshuffle diagnostics ordering.
* **Rename determinism:** same input ⇒ identical `workspaceEdit` byte ranges across runs.
* **Hierarchy truncation:** `truncated=true` when `(fanout,depth)` limits hit; never partial without the flag.
* **Budget behavior:** when `budget_ms` is too small, return `504` with `timed_out:true`, not empty 200s.

---

## 6) Minimal JSON schemas (for Lens only)

```json
// Diagnostics
{ "diags": [
  { "path":"src/a.go","source_hash":"…",
    "items":[{"range":{"b0":123,"b1":135},"severity":"error","code":"E1001","message":"undefined x"}]
  }
], "duration_ms": 32, "timed_out": false, "trace": {...} }

// Rename (workspace edits)
{ "workspaceEdit": {
    "changes": [
      {"path":"pkg/f.go","source_hash":"…",
       "edits":[{"b0":250,"b1":253,"new_text":"Foo"}]}
    ]},
  "duration_ms": 71
}
```

---

## 7) Sequenced implementation (2 sprints)

* **Sprint A:** capabilities, diagnostics, format, selection/folding; caching, metrics, tests.
* **Sprint B:** prepareRename/rename, codeActions (text edits only), hierarchy; determinism polish; rollout per language.

