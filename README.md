# Backend — Autoeval API + CLI

FastAPI service that ingests `task_name.zip` bundles of agent trajectories,
classifies each one with OpenAI, and persists results to either local disk or a
Google Cloud Storage bucket. Also ships the original CLI scripts that run
directly against a Harbor-style `jobs_new/<run>/` directory.

---

## Layout

```
backend/
├── .env                       # secrets + config (gitignored)
├── .env.example               # template
├── Dockerfile                 # backend image (Python 3.13 slim)
├── requirements.txt
│
├── golden_trajectory.py       # CLI: GT-rubric evaluator (GT1..GT5, SC1..SC4)
├── failure_trajectory.py      # CLI: 9-category failure classifier
├── run_all.py                 # CLI: runs both over jobs_new/<run>/
├── make_sample_zip.py         # utility: builds a test zip from jobs_new/
├── system_prompt.md           # GT-rubric system prompt
├── knowledge_base.md          # GT-rubric definitions
│
├── benchmark/                 # task definitions (CLI input)
├── jobs_new/                  # Harbor trial outputs (CLI input)
│
└── platform_app/              # the FastAPI app
    ├── main.py                # ASGI entrypoint
    ├── config.py              # pydantic-settings env config + load_dotenv()
    ├── api.py                 # REST routes
    ├── pipeline.py            # zip ingest, episode discovery, background worker
    ├── storage.py             # LocalStorage / GCSStorage abstraction
    ├── manifest.py            # index.json / manifest.json read+write
    └── evaluators.py          # thin wrappers over golden_/failure_ modules
```

---

## Environment variables

See [`.env.example`](.env.example). Summary:

| Var | Required | Default | Notes |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | — | OpenAI key for both evaluators |
| `OPENAI_MODEL` | no | `gpt-5.1` | Override per deployment |
| `OPENAI_MAX_TOKENS` | no | `16000` | Cap on completion tokens |
| `STORAGE_BACKEND` | yes | `local` | `local` for dev, `gcs` for production |
| `STORAGE_LOCAL_DIR` | when local | `./storage` | Where to put zips and results |
| `GCS_BUCKET` | when gcs | — | Bucket name, no `gs://` prefix |
| `GOOGLE_APPLICATION_CREDENTIALS` | when gcs (dev only) | — | Absolute path to SA JSON key. On Cloud Run, the attached SA is used automatically. |
| `API_TOKEN` | no | empty | If set, all `/api/*` requests must include `Authorization: Bearer <token>` |
| `CORS_ALLOW_ORIGINS` | no | `http://localhost:5173` | Comma-separated origins. Set to the deployed frontend URL in prod. |

> **`.env` pitfall**: do **not** put inline `# comments` after a value — `python-dotenv` keeps the comment text as part of the value. Put comments on their own line above, or use `KEY=""`.

---

## Local dev

```bash
cd backend
cp .env.example .env             # fill OPENAI_API_KEY
pip install -r requirements.txt  # or use the existing .venv
uvicorn platform_app.main:app --reload --port 8000
```

Smoke test:

```bash
curl -s http://localhost:8000/api/health     # -> {"status":"ok"}
curl -s http://localhost:8000/api/runs       # -> []
```

`pydantic-settings` is `lru_cache`-d in [`config.py`](platform_app/config.py); changes to `.env` need a **process restart**, not just `--reload`.

---

## REST API

`base = http://localhost:8000` in dev, `https://autoeval-backend-<hash>-<region>.a.run.app` in prod.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness probe |
| POST | `/api/runs` | multipart `file=task.zip` → `{"run_id":"...", "status":"queued", "total_episodes":N}` |
| GET | `/api/runs` | All runs, newest first |
| GET | `/api/runs/{id}` | Full manifest for one run (episodes, statuses, summaries) |
| GET | `/api/runs/{id}/episodes/{episode_id}` | Per-episode eval JSON |
| GET | `/api/runs/{id}/golden_summary.csv` | CSV download |
| GET | `/api/runs/{id}/failure_summary.xlsx` | XLSX download |
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
   │
   ├──▶ ingest_zip(): extract into <storage>/runs/<run_id>/extracted/...
   ├──▶ build manifest with episodes discovered under Golden_/Failure_
   └──▶ schedule BackgroundTask: pipeline.process_run(run_id)

process_run loop, per episode:
   │
   ├── load trajectory.json from storage
   ├── if golden:
   │     1× OpenAI call → GT rubric verdict
   │     write eval_golden.json + summary row
   │
   └── if failure:
         1× OpenAI call → 9-category failure classification
         1× OpenAI call → GT class label (+ justification only)
         write eval_failure.json + summary row (incl. GT Class(AI))

After all episodes:
   ├── write golden_summary.csv
   └── write failure_summary.xlsx
```

Each manifest write is atomic: temp object → rename. The browser polls `/api/runs/{id}` and sees status flip from `queued` → `running` → `done` (or `failed`).

Background-task survival on Cloud Run depends on `--min-instances=1 --no-cpu-throttling`; see "Deploy" below.

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

The 9-category labels: `Disobey Specification, Step Repetition, Unaware of termination conditions, Reasoning-Action Mismatch, Context Loss, Task Derailment, Premature termination, No or incorrect Verification, Weak Verification`.

Each failure episode also receives a GT-class label + 1-sentence justification (only the class + justification — SC1..SC4, verdict, and hard requirements from the GT rubric are deliberately discarded for failures).

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

Run from inside `backend/`. Operates directly on Harbor trial directories.

```bash
# Evaluate every passed trial in a run (verifier/reward.txt == "1")
python golden_trajectory.py jobs_new/2026-05-08__16-23-19/

# Classify every failed trial in a run
python failure_trajectory.py jobs_new/2026-05-08__16-23-19/

# Both, end-to-end
python run_all.py jobs_new/2026-05-08__16-23-19/
```

Outputs land next to each trial: `eval_golden.json` / `eval_failure.json`, plus `golden_summary.csv` / `failure_summary.xlsx` at the run root.

`make_sample_zip.py` packs a 1-golden / 1-failure subset from `jobs_new/` into a `sample.zip` you can drop into the platform's upload card for end-to-end testing.

---

## GCS setup (one-time)

For local dev pointed at GCS:

1. Create a Standard regional bucket (do **not** enable "Appendable objects", "Hierarchical namespace", or "Rapid storage" — they reject normal multipart uploads).
2. Grant the service account `roles/storage.objectAdmin` on the bucket.
3. Put the JSON key path in `.env`:
   ```
   STORAGE_BACKEND=gcs
   GCS_BUCKET=<bucket-name>
   GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/sa-key.json
   ```
4. Restart uvicorn (full restart, not just `--reload`).

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

One image, one service. Background work survives between requests because the instance stays warm.

```bash
export PROJECT_ID=...
export REGION=us-central1

# Build (context = backend/, run from repo root)
gcloud builds submit backend \
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

After the backend's URL is known, set CORS to permit the frontend origin:

```bash
gcloud run services update autoeval-backend \
  --region=$REGION \
  --update-env-vars="CORS_ALLOW_ORIGINS=https://autoeval-frontend-<hash>-<region>.a.run.app"
```

The `--min-instances=1 --no-cpu-throttling` combination is what makes the in-process `BackgroundTasks` worker reliable on Cloud Run. Without it, the instance can scale to zero mid-run.

### Service account permissions (production)

The Cloud Run SA needs:

- `roles/storage.objectAdmin` on the GCS bucket
- `roles/secretmanager.secretAccessor` on each secret you reference via `--set-secrets`

It does **not** need project-wide `storage.buckets.list` — bucket-level access is enough.

---

## Common gotchas

- **`401 invalid or missing bearer token`** — `API_TOKEN` is set in `.env`. Either clear it (`API_TOKEN=` with nothing after) or send `Authorization: Bearer <same value>` from the client. Check for hidden inline comments.
- **`DefaultCredentialsError`** — `GOOGLE_APPLICATION_CREDENTIALS` isn't reaching the Google SDK. `load_dotenv()` in [`config.py`](platform_app/config.py) pushes it into `os.environ` on import; if you bypass `config.py`, set it manually.
- **`This bucket requires appendable objects`** — the bucket was created in a Rapid/Zonal/HNS mode. Recreate as Standard regional.
- **`gpt-5.1` rejecting `temperature` or `max_tokens`** — the evaluators pass `max_completion_tokens` and omit `temperature`. Don't add them back without testing.

---

## Tests

There aren't any yet. Smoke checks:

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
