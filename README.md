# Backend — Autoeval API + CLI

FastAPI service that ingests `task_name.zip` bundles of agent trajectories,
classifies each one with OpenAI, and persists results to either local disk or a
Google Cloud Storage bucket. Also ships the original CLI scripts that run
directly against a Harbor-style `jobs/<run>/` directory you supply.

This folder is its own git working tree (has its own `.git/` and `.gitignore`).
You can develop and version it independently from the rest of the repo.

---

## Layout

```
backend/
├── .env                       # secrets + config (gitignored)
├── .env.example               # template
├── .dockerignore
├── .gitignore
├── Dockerfile                 # backend image (Python 3.13 slim)
├── requirements.txt
├── README.md
│
├── golden_trajectory.py       # CLI: GT-rubric evaluator (GT1..GT5, SC1..SC4)
├── failure_trajectory.py      # CLI: 9-category failure classifier
├── run_all.py                 # CLI: runs both over a Harbor jobs dir
├── make_sample_zip.py         # utility: builds a test zip from a Harbor jobs dir
├── system_prompt.md           # GT-rubric system prompt
├── knowledge_base.md          # GT-rubric definitions
│
├── platform_app/              # the FastAPI app
│   ├── __init__.py
│   ├── main.py                # ASGI entrypoint
│   ├── config.py              # pydantic-settings + load_dotenv()
│   ├── api.py                 # REST routes
│   ├── pipeline.py            # zip ingest, episode discovery, background worker
│   ├── storage.py             # LocalStorage / GCSStorage abstraction
│   ├── manifest.py            # index.json / manifest.json read+write
│   └── evaluators.py          # thin wrappers over golden_/failure_ modules
│
└── .venv/                     # local virtualenv (uv-managed)
```

> Heads-up: there's a `package-lock.json` in this folder from when `npm install`
> was run here by mistake. It does nothing for the backend — safe to delete
> (`rm package-lock.json`).

---

## Dependencies

`requirements.txt`:

```
openai                # the LLM client
pandas + openpyxl     # XLSX summary writer
python-dotenv         # .env loader
fastapi + uvicorn     # API + ASGI server
python-multipart      # multipart upload parsing
pydantic-settings     # env-driven config
google-cloud-storage  # GCS backend (>=3.10 for v2/gRPC support)
google-crc32c         # checksum library for streaming uploads
grpc-google-iam-v1    # required by google-cloud-storage v2 (appendable / zonal)
```

Install:

```bash
pip install -r requirements.txt
# or with uv (what .venv was built with):
uv pip install --python .venv/bin/python3 -r requirements.txt
```

---

## Environment variables

See [`.env.example`](.env.example). Summary:

| Var | Required | Default | Notes |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | — | OpenAI key |
| `OPENAI_MODEL` | no | `gpt-5.1` | Override per deployment |
| `OPENAI_MAX_TOKENS` | no | `16000` | Cap on completion tokens |
| `MAX_PARALLEL_EPISODES` | no | `10` | How many episodes evaluate concurrently inside a run. Lower if you hit OpenAI TPM caps. |
| `INGEST_PARALLELISM` | no | `16` | How many zip entries the ingest writes to the bucket in parallel. Important on slow buckets (Rapid / HNS). |
| `STORAGE_BACKEND` | yes | `local` | `local` for dev, `gcs` for production |
| `STORAGE_LOCAL_DIR` | when local | `./storage` | Where to put zips and results |
| `GCS_BUCKET` | when gcs | — | Bucket name only, no `gs://`. Hyphens vs underscores matter (`autoeval-pipeline` ≠ `autoeval_pipeline`). |
| `GCS_FORCE_APPENDABLE` | no | `false` | Force appendable / gRPC writes for Rapid / Zonal buckets. Leave `false` and the backend auto-detects from a GCS 400. |
| `GOOGLE_APPLICATION_CREDENTIALS` | when gcs (dev only) | — | Absolute path to SA JSON key. On Cloud Run, the attached SA is used automatically. |
| `API_TOKEN` | no | empty | If set, all `/api/*` requests must include `Authorization: Bearer <token>` |
| `CORS_ALLOW_ORIGINS` | no | `http://localhost:5173` | Comma-separated origins. Set to the deployed frontend URL in prod. |

> **`.env` pitfall**: don't put inline `# comments` after a value — `python-dotenv` keeps the comment as part of the value. Comments go on their own line above, or quote: `KEY=""`.

---

## Local dev

```bash
cd backend
cp .env.example .env             # fill OPENAI_API_KEY
.venv/bin/python3 -m uvicorn platform_app.main:app --reload --port 8000
```

Smoke test:

```bash
curl -s http://localhost:8000/api/health     # -> {"status":"ok"}
curl -s http://localhost:8000/api/runs       # -> []
```

`pydantic-settings` is `lru_cache`-d in [`config.py`](platform_app/config.py); changes to `.env` need a full **process restart** (Ctrl+C and start again) — `--reload` alone won't clear the cache.

---

## REST API

Base: `http://localhost:8000` (dev) or `https://autoeval-backend-<hash>-<region>.a.run.app` (prod).

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness probe |
| POST | `/api/runs` | multipart `file=task.zip` → `{"run_id":"…", "status":"queued", "total_episodes":N}` |
| GET | `/api/runs` | All runs, newest first |
| GET | `/api/runs/{id}` | Full manifest for one run (episodes, statuses, summaries) |
| GET | `/api/runs/{id}/episodes/{episode_id}` | Per-episode eval JSON |
| GET | `/api/runs/{id}/golden_summary.csv` | CSV download |
| GET | `/api/runs/{id}/failure_summary.xlsx` | XLSX download |
| POST | `/api/runs/{id}/pause` | Stop scheduling new episodes; finish in-flight ones; land in `paused` |
| POST | `/api/runs/{id}/resume` | Clear pause flag and re-schedule the worker for remaining episodes |
| DELETE | `/api/runs/{id}` | Remove the run and all its storage |

When `API_TOKEN` is set, every `/api/*` request must carry `Authorization: Bearer <token>`.

```bash
curl -F "file=@/path/to/task.zip" \
     -H "Authorization: Bearer $API_TOKEN" \
     http://localhost:8000/api/runs
```

---

## How a run is processed

```
POST /api/runs (file=task.zip)
   │  (runs in a worker thread; the event loop stays responsive)
   ├──▶ ingest_zip(): extract into <storage>/runs/<run_id>/extracted/...
   │     • INGEST_PARALLELISM concurrent bucket writes
   ├──▶ build manifest with episodes discovered under Golden_/Failure_
   └──▶ schedule BackgroundTask: pipeline.process_run(run_id)

process_run:
   │  (per-run lock prevents concurrent workers for the same run_id)
   ├── reset any "running" episodes to "pending" (crash recovery)
   ├── reconcile in-memory pause flag with manifest.pause_requested
   │
   └── ThreadPoolExecutor(max_workers=MAX_PARALLEL_EPISODES), per episode:
         │
         ├── if pause_event is set → skip; leave as "pending"
         ├── if golden:
         │     1× OpenAI call → GT rubric verdict
         │     write eval_golden.json + summary row
         │
         └── if failure:
               1× OpenAI call → 9-category failure classification
               1× OpenAI call → GT class label (+ justification only)
               write eval_failure.json + summary row (incl. GT Class(AI))

After episodes finish:
   ├── if pause requested AND some episodes still "pending"  → status="paused"
   ├── else: write golden_summary.csv + failure_summary.xlsx → status="done"
   └── if no episodes were discovered                          → status="failed"
```

The browser polls `/api/runs/{id}` every 3 s while the run is in motion. Each manifest write is atomic on local storage (`tmp → rename`) and idempotent on GCS (replace-on-write).

### Run statuses

```
queued ─┬─► running ──┬─► done                (normal completion)
        │             ├─► failed              (catastrophic error or empty zip)
        │             └─► pausing ─► paused ─┐
        │                                    │
        └────────────────── resume ◄─────────┘
```

- `queued` — manifest exists; worker not started yet.
- `running` — at least one episode currently in flight.
- `pausing` — pause requested while episodes were running; waiting for in-flight calls to finish.
- `paused` — pause complete; no episodes running; pending episodes remain. Stays here until `/resume`.
- `done` / `failed` — terminal.

### Parallelism knobs

| Knob | Default | Where it kicks in |
|---|---|---|
| `INGEST_PARALLELISM` | 16 | `ThreadPoolExecutor` around the zip-extracted bucket writes |
| `MAX_PARALLEL_EPISODES` | 10 | `ThreadPoolExecutor` around per-episode evaluation |

Both threads pools share a single `OpenAI` client (thread-safe in the SDK) and a single in-memory manifest dict protected by a `Lock`. There is no contention between different bucket keys, so the bucket-side throughput is the real cap.

### Pause / resume / crash-recovery

- **Pause**: `POST /api/runs/{id}/pause` sets an in-process `threading.Event` *and* writes `pause_requested=true` to the manifest. Worker threads check the event before each episode starts and bail without touching the episode (leaves it `pending`). Status flips `running → pausing → paused` once in-flight episodes wrap.
- **Resume**: `POST /api/runs/{id}/resume` clears the flag and schedules a fresh `process_run` via `BackgroundTasks`. The per-run lock prevents racing workers if a stale one is still around.
- **Crash recovery**: when `process_run` starts, any episode left in `running` (orphaned by a prior worker that died) is reset to `pending` and re-evaluated. The persistent `pause_requested` flag survives uvicorn restarts, so a paused run stays paused.

### Idempotency guarantees

- Episode-level: each evaluator call is independent. Re-running an episode produces a new `eval_*.json` and a fresh row in the next summary spreadsheet.
- Run-level: re-invoking `process_run(run_id)` is safe — the per-run lock prevents two workers from racing, and any non-`done` episode is up for grabs again.

### Required zip layout

```
task_name.zip
├── Golden_trajectories/
│   ├── episode1/
│   │   └── agent/trajectory.json   (or just trajectory.json)
│   └── episodeN/
└── Failure_trajectories/
    ├── episode1/
    └── episodeN/
```

Episodes can use either full Harbor layout (`agent/trajectory.json` + sibling `result.json` / `config.json`) or a bare `trajectory.json`.

### Output layout in storage

```
<storage>/
├── index.json                          # array of run summaries
└── runs/<run_id>/
    ├── manifest.json
    ├── original.zip
    ├── extracted/
    │   ├── Golden_trajectories/episodeN/agent/trajectory.json
    │   └── Failure_trajectories/episodeN/trajectory.json
    ├── results/
    │   ├── Golden_trajectories/episodeN/eval_golden.json
    │   └── Failure_trajectories/episodeN/eval_failure.json
    ├── golden_summary.csv
    └── failure_summary.xlsx
```

Local backend writes under `STORAGE_LOCAL_DIR`; GCS backend writes into `gs://$GCS_BUCKET/`.

---

## Evaluator outputs

### Golden — GT class + success criteria

| Label | Meaning |
|---|---|
| **GT1** | High-Quality Tool-Use CoT |
| **GT2** | Long-Horizon Planning |
| **GT3** | Error Identification & Recovery |
| **GT4** | Multi-Tool Orchestration |
| **GT5** | Code Understanding |

Plus per-trajectory `SC1..SC4 ∈ {PASS, FAIL, WEAK PASS, NA}`, three boolean hard-requirement flags, and a final verdict `ACCEPT as Golden Trajectory | BORDERLINE | REJECT`.

### Failure — 9-category classification + GT shape

9-category labels: `Disobey Specification, Step Repetition, Unaware of termination conditions, Reasoning-Action Mismatch, Context Loss, Task Derailment, Premature termination, No or incorrect Verification, Weak Verification`.

Each failure episode also receives a GT-class label + 1-sentence justification (only those two — SC1..SC4, verdict, and hard requirements are intentionally dropped for failures).

### Summary spreadsheet columns

`golden_summary.csv`:
```
Task, Agent, Model, GT Class(AI), GT Justification(AI),
Success Criteria (AI), GT Class(Human), Success Criteria (Human),
HITL Remarks, Task name, Trajectory name
```

`failure_summary.xlsx`:
```
agent, benchmark, trial_id, status,
GT Class(AI), GT Justification(AI),
failure_type, reason, root_cause, fix
```

---

## CLI (file-based, no HTTP)

These work directly on Harbor trial directories you point them at. The data lives wherever you keep it — there's no `jobs_new/` shipped inside `backend/`.

```bash
# Evaluate every passed trial in a run (verifier/reward.txt == "1")
python golden_trajectory.py /path/to/jobs/2026-05-08__16-23-19/

# Classify every failed trial in a run
python failure_trajectory.py /path/to/jobs/2026-05-08__16-23-19/

# Both, end-to-end
python run_all.py /path/to/jobs/2026-05-08__16-23-19/

# Build a small test zip from one passed + one failed trial
python make_sample_zip.py /path/to/jobs/2026-05-08__16-23-19/
#   → sample.zip in the cwd
```

CLI outputs land next to each trial (`eval_golden.json` / `eval_failure.json`) plus `golden_summary.csv` / `failure_summary.xlsx` at the run root.

---

## GCS setup (one-time)

For local dev pointed at GCS:

1. **Create a Standard regional bucket.** Do **not** enable "Appendable objects", "Hierarchical namespace", or "Rapid storage" unless you understand the trade-off — those modes reject normal multipart uploads. If you do need a Rapid/Zonal bucket, set `GCS_FORCE_APPENDABLE=true`.
2. **Grant the service account `roles/storage.objectAdmin` on the bucket.** Project-wide bucket-list permission is not needed.
3. **Wire it up in `.env`**:
   ```
   STORAGE_BACKEND=gcs
   GCS_BUCKET=<bucket-name>
   GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/sa-key.json
   ```
4. **Restart uvicorn** (full restart, not just `--reload`).

Verify:
```bash
.venv/bin/python3 -c "
from dotenv import load_dotenv; load_dotenv()
from platform_app.storage import get_storage
s = get_storage()
s.put_text('smoketest.txt', 'hello')
print('ok:', s.exists('smoketest.txt'))
"
```

---

## Deploy to Cloud Run

One image, one service. Background work survives between requests when the instance stays warm.

```bash
export PROJECT_ID=...
export REGION=us-central1

# Build (context = backend/)
gcloud builds submit . \
  --tag $REGION-docker.pkg.dev/$PROJECT_ID/autoeval/backend:latest

# Deploy
gcloud run deploy autoeval-backend \
  --image=$REGION-docker.pkg.dev/$PROJECT_ID/autoeval/backend:latest \
  --region=$REGION \
  --allow-unauthenticated \
  --service-account=autoeval-run@$PROJECT_ID.iam.gserviceaccount.com \
  --set-env-vars="STORAGE_BACKEND=gcs,GCS_BUCKET=$BUCKET,OPENAI_MODEL=gpt-5.1" \
  --set-secrets="OPENAI_API_KEY=openai-api-key:latest,API_TOKEN=api-token:latest" \
  --min-instances=1 --max-instances=3 \
  --no-cpu-throttling --cpu=1 --memory=1Gi \
  --timeout=3600
```

After the frontend service is also up, set CORS:

```bash
gcloud run services update autoeval-backend \
  --region=$REGION \
  --update-env-vars="CORS_ALLOW_ORIGINS=https://autoeval-frontend-<hash>-<region>.a.run.app"
```

`--min-instances=1 --no-cpu-throttling` is what makes the in-process `BackgroundTasks` worker reliable on Cloud Run. Without it, the instance can scale to zero mid-run.

### Service account permissions (production)

The Cloud Run SA needs:

- `roles/storage.objectAdmin` on the GCS bucket
- `roles/secretmanager.secretAccessor` on each secret referenced via `--set-secrets`

Project-wide `storage.buckets.list` is **not** required — bucket-scoped access is enough.

---

## Common gotchas

- **`401 invalid or missing bearer token`** — `API_TOKEN` is set in `.env`. Either clear it (`API_TOKEN=` with absolutely nothing after) or send `Authorization: Bearer <same value>` from the client. Check for stray spaces or inline comments.
- **`DefaultCredentialsError`** — `GOOGLE_APPLICATION_CREDENTIALS` isn't reaching the Google SDK. `load_dotenv()` in [`config.py`](platform_app/config.py) pushes it into `os.environ` on import; if you bypass `config.py`, set it manually before importing `google.cloud`.
- **`This bucket requires appendable objects`** — bucket created in Rapid / Zonal / HNS mode. Either recreate as Standard regional, or set `GCS_FORCE_APPENDABLE=true` (requires `grpc-google-iam-v1` from `requirements.txt`). Note: appendable writes are ~40× slower per object than multipart — see *Run takes much longer than expected* below.
- **Run takes much longer than expected on a Rapid/HNS bucket** — each bucket write costs ~1.3 s instead of ~30 ms. Mitigate with high `INGEST_PARALLELISM` and `MAX_PARALLEL_EPISODES`. For sub-15-minute runs of ~100 episodes, switch to a Standard regional bucket.
- **OpenAI 429 / TPM rate limits** — lower `MAX_PARALLEL_EPISODES` (try 5). The retry/backoff in `call_model` will absorb short bursts but sustained 429s mean the concurrency is too high for your account tier.
- **Run stuck in `pausing`** — an in-flight OpenAI call has not finished yet (no mid-call cancel). Wait for it; once the last in-flight episode lands the status flips to `paused`.
- **Run shows `running` but no progress** — likely a crashed worker. Click **Resume** in the UI; `process_run` resets stale `running` episodes to `pending` and re-evaluates them.
- **`gpt-5.1` rejecting `temperature` or `max_tokens`** — the evaluators pass `max_completion_tokens` and omit `temperature`. Don't add them back without testing.

---

## Quick smoke test (no real OpenAI call)

```bash
.venv/bin/python3 -c "
import sys; sys.path.insert(0,'.')
from fastapi.testclient import TestClient
from platform_app.main import app
c = TestClient(app)
print('health:', c.get('/api/health').json())
print('runs:', c.get('/api/runs').json())
"
```

Both should return without errors. If `/api/runs` 500s, see the GCS / credentials gotchas above.
