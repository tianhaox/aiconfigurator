# Design Doc: WebApp Redesign on Task v2

**Author:** Tianhao Xu (tianhaox@nvidia.com)
**First Proposed:** 2026-06-14
**Last Updated:** 2026-06-14
**Status:** Draft / In exploration

| Stakeholder | Approval date | Comments |
|-------------|---------------|----------|
|             |               |          |
|             |               |          |

---

## 1 Objective

Replace the legacy Gradio webapp with a focused, modern frontend whose sole reason
to exist is **the things the CLI cannot do well**: interactive exploration, side-by-side
comparison of multiple experiments, visual performance breakdown, user-driven YAML
overrides for custom experiments, and a natural-language (LLM) entry point — all built
as a thin layer on top of the now-clean **Task v2** API.

### Goals

- **G1 — Visualize what the CLI renders as ASCII.** Turn the static-estimation perf
  breakdown (`estimate_detail_report.py` ASCII bars) into interactive charts: per-op
  latency, memory decomposition, energy, and data-source attribution.
- **G2 — Multi-experiment comparison.** Let a user define several experiment groups
  (each a Task config), run them, and compare results side-by-side (Pareto fronts,
  metric tables, breakdowns). Replaces the old in-memory `gr.State` comparison tab.
- **G3 — Interactive exploration.** Sweep results (the `run()` DataFrame, 45/60 columns)
  rendered as interactive, filterable, sortable tables + scatter/Pareto plots.
- **G4 — YAML override for custom experiments.** Expose the flat Task v2 field set
  (defaults + constraints + per-backend support) so users can override fields and run
  bespoke experiments, with strict validation and friendly errors surfaced in the UI.
- **G5 — LLM natural-language experiments.** Let users describe intent in natural
  language; an LLM produces/edits a Task YAML, validation errors feed back for
  self-correction, then the experiment runs and results render.
- **G6 — Thin and standalone.** No business logic in the frontend. The Task v2 SDK
  remains the single source of truth for config construction, search, and results.

### Non-goals

- Re-implementing the optimization engine — Task v2 / `sweep.py` stay as-is.
- Keeping Gradio or any of its custom JS / monkey-patches.
- A persistent multi-user backend / accounts / RBAC (single-user / team-internal first).
- The legacy GPU profiling subsystem — **dropped entirely** (1.5k-line tab + 7 JS
  modules), not ported. Only the new requirements (§1 goals) are in scope.

---

## 2 Background

### 2.1 Current state (legacy Gradio webapp)

`src/aiconfigurator/webapp/` ≈ **5.6k lines Python + 1.4k lines JS/CSS**, 29 Python files.
The complexity is architectural, not functional:

- `events/event_fn.py` (**1391 lines**) mixes *form collection → config construction →
  sweep → Pareto → hand-built Plotly HTML* in one file.
- `events/event_handler.py` (**507 lines**) is hand-wired Gradio event plumbing
  (50+ input→output combinations, no reactive framework).
- `components/profiling/` (**1562 lines** Python + 7 mutually-coupled raw JS modules)
  binds to Gradio `elem_id`s and loads Chart.js/DataTables from CDN at runtime.
- 39+ form widgets with cascading dropdowns (system→backend→version→database).
- All state is in-memory `gr.State`; results compared by manual DataFrame concat.

**Key observation:** most of this code does work that Task v2 now owns. The webapp
was carrying an orchestration layer that has since moved into the SDK.

### 2.2 The Task v2 opportunity

`src/aiconfigurator/sdk/task_v2.py` exposes a clean, **flat, serializable** dataclass:

- `Task.from_yaml(dict, **overrides)` / `to_yaml()` / `to_dict()` — perfect round-trip,
  ~100 fields mapped 1:1, **strict validation with friendly errors** (misspelled keys
  rejected, not silently ignored).
- `task.run() -> pd.DataFrame` — single entry point; agg = `common.ColumnsAgg` (45 cols),
  disagg = `common.ColumnsDisagg` (60 cols). Feasible (SLA-filtered) candidates;
  `picking.get_pareto_front()` computes the frontier.
- `task.run_single_agg/disagg(...) -> dict` — fast single-point what-if eval.
- `module_bridge.task_config_to_generator_config(task, result_row)` — winning row →
  deployment config.

This means a new frontend only needs to: **collect a Task config → call `run()` /
`cli_estimate()` → visualize the result.** The engine is untouched.

### 2.3 Perf-breakdown data path (load-bearing for G1)

`cli_estimate(mode="static"|"static_ctx"|"static_gen", ...)` → `EstimateResult`
(`src/aiconfigurator/cli/api.py`):

```python
@dataclass
class EstimateResult:
    ttft: float; tpot: float; power_w: float
    raw: dict                 # all scalar metrics (ColumnsStatic)
    mode: str
    summary: object | None    # InferenceSummary (static modes only)
    per_ops_data: dict | None # per-op latency/energy dicts
    per_ops_source: dict | None
```

`InferenceSummary` exposes plain, **JSON-serializable** dicts:

- per-op latency — `context_latency_dict`, `generation_latency_dict`, `encoder_*`
  (keys: `qkv_gemm`, `context_attention`, `context_full_moe`, …; values: ms)
- per-op energy — `*_energy_wms_dict` (W·ms)
- memory — `{total, weights, kvcache, activations, nccl, others}` (GiB)
- data source — `*_source_dict` (`silicon` / `empirical` / `sol` / `mixed`) for
  traceability of every op's prediction provenance

Today this is rendered as ASCII bars in `estimate_detail_report.py` — exactly the
"CLI is not intuitive" case G1 targets.

**Update (verified in prototype):** all three modes expose a per-op breakdown.
Static reads `InferenceSummary.get_context/generation_latency_dict()`; agg and disagg
read `EstimateResult.per_ops_data` (agg: `mix_step` / `genonly_step`; disagg:
`prefill` / `decode`), each with a parallel `per_ops_source`. The only disagg gap is
the **memory breakdown** (no single `summary`; only `(p)memory` / `(d)memory` scalar
totals) and per-token KV (no summary). The earlier "disagg has no per-op breakdown"
note was wrong.

---

## 3 The Five Pillars (requirements → data source)

| Pillar | Backing API (exists today) | New work |
|---|---|---|
| Multi-experiment comparison (G2) | `run()` DataFrame + serializable `Task` as the group key | session/store layer to hold N experiments + comparison views |
| Interactive visualization (G3) | 45/60-col DataFrame; `get_pareto_front()` | chart layer (replaces hand-built Plotly HTML) |
| YAML override custom experiments (G4) | `from_yaml(dict, **overrides)`, strict validation | expose field schema (defaults/constraints/backend support) to UI |
| LLM natural-language (G5) | flat strict YAML schema; validation errors as feedback | LLM orchestration endpoint + schema description prompt |
| Perf breakdown (G1) | `cli_estimate(static)` → `EstimateResult` (JSON-serializable) | interactive breakdown charts (replace ASCII) |

---

## 4 Architecture

**Frontend stack (decided): Vite + React + TypeScript.** Rationale: the requirements
(interactive charts, multi-experiment comparison, live chat) are genuinely high-
interaction — the SPA sweet spot, where server-rendered / HTMX approaches force you to
hand-write JS anyway. React has the largest ecosystem (Plotly/ECharts charts,
assistant-ui / Vercel AI SDK chat, shadcn-ui / Mantine components → little hand CSS)
**and the most AI-coding support** — decisive since the frontend will be AI-authored
and -maintained rather than hand-written by the (Python-centric) team. Plain Vite SPA,
not Next.js: this is an internal tool behind FastAPI, no SSR/SEO need.

```
┌─────────────────────────────────────────────┐
│  Frontend — Vite + React + TS                │
│  - experiment builder (form + raw YAML)      │
│  - results: interactive tables + charts      │
│  - comparison view (N experiments)           │
│  - breakdown view (per-op / memory / energy) │
│  - chat panel (NL → Task config)             │
└───────────────┬─────────────────────────────┘
                │ REST / JSON  (+ WS or polling for progress)
┌───────────────▼─────────────────────────────┐
│  Thin API server (FastAPI)                   │
│  - /schema      Task field schema for UI     │
│  - /experiments run() as background job      │
│  - /estimate    cli_estimate(static)         │
│  - /databases   supported system/backend/ver │
│  - /chat        LLM → Task YAML (+ validate)  │
│  NO business logic — pure adapter            │
└───────────────┬─────────────────────────────┘
                │ in-process Python calls
┌───────────────▼─────────────────────────────┐
│  Task v2 SDK (unchanged)                     │
│  task.run() / run_single() / cli_estimate()  │
│  from_yaml / to_yaml / validate              │
│  perf_database / sweep / picking / bridge    │
└─────────────────────────────────────────────┘
```

The frontend never touches SDK internals; the API server is a stateless adapter plus
a job/session store. The SDK is the single source of truth.

### 4.1 Async / progress

`task.run()` is **synchronous, blocking, can take minutes, and has no progress
callback**. The API must therefore:

- run experiments as **background jobs** (job id returned immediately);
- expose status via **polling or WebSocket** (no streaming exists in the SDK today —
  v1 reports coarse states: queued / running / done / failed; finer progress would
  require an SDK-side callback hook, out of scope for v1).

### 4.2 Database coupling

`run()` loads the perf database internally (system/backend/version-specific). The API
should pre-list available DBs via `perf_database.get_supported_databases()` so the UI
can constrain choices, and surface `database_mode` (silicon/hybrid/empirical/sol).

---

## 5 API contract (v1 sketch)

| Endpoint | Method | Purpose | SDK call |
|---|---|---|---|
| `/api/schema` | GET | Task field schema: defaults, constraints, per-backend support, enums | derived from `task_v2` + `common` enums |
| `/api/databases` | GET | supported system/backend/version + modes | `perf_database.get_supported_databases()` |
| `/api/validate` | POST | validate a Task YAML/dict, return errors | `Task.from_yaml(...).validate()` |
| `/api/experiments` | POST | submit experiment(s) → job id | `Task.from_yaml().run()` in background |
| `/api/experiments/{id}` | GET | status + result (DataFrame as JSON + Pareto front) | job store |
| `/api/estimate` | POST | static perf breakdown | `cli_estimate(mode="static")` |
| `/api/chat` | POST | NL → Task YAML (validated), or NL edits to existing config | LLM + `/validate` loop |
| `/api/deploy-config` | POST | result row → deployment config | `module_bridge.task_config_to_generator_config()` |

Result payloads: DataFrames → `df.to_dict(orient="records")`; breakdown dicts pass
through directly (already JSON-safe).

---

## 6 LLM integration (G5)

The flat, strict Task v2 YAML schema is an ideal LLM target:

1. System prompt carries the field schema from `/api/schema` (names, types, enums,
   defaults, per-backend support) + a few canonical `example.yaml` shots.
2. User describes intent ("compare DeepSeek-V3 on H200 trtllm fp8_block, agg vs disagg,
   ISL 4k OSL 1k, TTFT 1s") → LLM emits a Task YAML (or a patch over the current one).
3. Server calls `/api/validate`; on failure the **structured validation error feeds
   back** to the LLM for self-correction (the strict-rejection behavior is a feature
   here, not a nuisance).
4. On success → submit as an experiment; results render in the same views as manual runs.

Per repo guidance, default to the latest Claude models for this integration. Provider
and model selection is an open question (§9).

---

## 7 Data flow (worked example: G1 breakdown)

```
UI: pick model/system/backend + batch size, "show breakdown"
  → POST /api/estimate {mode: "static", ...overrides}
  → server: cli_estimate(mode="static", ...) → EstimateResult
  → serialize summary: {context_latency, generation_latency, memory,
                        context_energy, generation_energy, *_source}
  → JSON to frontend
  → render: stacked/sorted bar of per-op latency (color by source tag),
            memory donut (weights/kv/act/nccl/others),
            energy bars, source-attribution legend
```

This replaces `_format_summary_time_section` / `_format_memory_section` ASCII output
with interactive charts — same data, better medium.

---

## 8 Migration & scope of the cut

Legacy tabs and their disposition:

| Legacy tab | v1 disposition |
|---|---|
| Static Estimation | **Keep, upgrade** → G1 interactive breakdown |
| Agg / Disagg Pareto | **Keep, upgrade** → G3 interactive sweep results |
| Pareto Comparison | **Keep, upgrade** → G2 multi-experiment comparison |
| Disagg PD Ratio | **Keep** (thin view over disagg results) |
| Support Matrix | **Keep** (624 lines → driven by `/api/databases`) |
| README | **Keep** (static content) |
| **Profiling** (1.5k lines + 7 JS modules) | **Dropped entirely** — not ported |

Cut entirely: Gradio, `gradio_patches.py`, hand-wired `event_handler.py`, hand-built
Plotly HTML, the 7 coupled JS modules.

---

## 9 Open questions

1. ~~Frontend tech stack~~ — **Decided: Vite + React + TypeScript** (see §4).
2. **Disagg per-op breakdown gap** — G1 only has data for agg/static. Do we (a) ship
   breakdown for agg only in v1, or (b) extend `DisaggInferenceSession` to expose a
   per-op summary at the API boundary? *Recommend (a) for v1.*
3. **Progress reporting** — coarse job states for v1; an SDK-side progress callback in
   `sweep.py` would enable real progress bars later. *Defer the SDK hook.*
4. **LLM provider/model** — which model, where hosted, auth. *Pending.*
5. **Persistence** — in-memory job store for v1, or a real store so experiments survive
   restart / are shareable? *Recommend in-memory v1, pluggable later.*
6. **Profiling tab** — re-implement, drop, or keep as a separate legacy launch? *Defer.*

---

## 10 Appendix: key files

- Task v2: `src/aiconfigurator/sdk/task_v2.py` (dataclass, from_yaml/to_yaml, run, run_single)
- Result schema: `src/aiconfigurator/sdk/common.py` (`ColumnsAgg`, `ColumnsDisagg`, `ColumnsStatic`)
- Sweep: `src/aiconfigurator/sdk/sweep.py`
- Breakdown: `src/aiconfigurator/sdk/inference_session.py` (`run_static`),
  `src/aiconfigurator/sdk/inference_summary.py` (`InferenceSummary`),
  `src/aiconfigurator/cli/api.py` (`cli_estimate`, `EstimateResult`),
  `src/aiconfigurator/cli/estimate_detail_report.py` (current ASCII rendering)
- Generator bridge: `src/aiconfigurator/generator/module_bridge.py`
- Example config: `src/aiconfigurator/cli/example.yaml`
- Legacy webapp (to retire): `src/aiconfigurator/webapp/`
