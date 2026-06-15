# webapp2 (prototype)

A thin **FastAPI + React** rebuild of the webapp on top of Task v2 / `cli_estimate`.
This first cut implements one view end-to-end: the **interactive static perf
breakdown** (the modern replacement for the CLI's ASCII bars).

See `docs/design/webapp_redesign_design.md` for the full design.

## Run

Two processes. From the repo root:

**Backend** (FastAPI on :8000)
```bash
uv pip install fastapi "uvicorn[standard]"   # once
uv run uvicorn webapp2.backend.server:app --reload --port 8000
```

**Frontend** (Vite dev server on :5173, proxies /api → :8000)
```bash
cd webapp2/frontend
npm install      # once
npm run dev
```

Open http://localhost:5173.

## What works

- Cascading system → backend → version selectors, driven by real
  `perf_database.get_supported_databases()`.
- `Estimate` runs a real static-mode `cli_estimate` and renders:
  - scalar cards (TTFT / TPOT / memory / power),
  - prefill & decode per-op latency bars, **colored by data source**
    (silicon / empirical / sol / mixed),
  - memory breakdown donut.
- SDK/validation errors surface in the UI.

## Not yet (next cuts)

- Multi-experiment comparison, sweep results (`task.run()` DataFrame), background jobs.
- LLM natural-language → Task YAML.
- YAML override editor.
