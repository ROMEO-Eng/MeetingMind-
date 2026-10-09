# MeetingMind backend

FastAPI owns transcript loading and normalization, chunking, local CPU embeddings, FAISS retrieval, structured parsing, meeting sessions, and API responses. Text generation is delegated through an authenticated remote client to the temporary Google Colab inference service. The backend does not download or load LLM weights.

## Run locally

Create a Python 3.11 environment and install the CPU-capable dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Start `notebooks/MeetingMind_Colab_Inference.ipynb` with a Colab GPU runtime. Copy its active temporary endpoint and bearer key into `.env` as `LLM_BASE_URL` and `LLM_API_KEY`, then run:

```powershell
$env:CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
python -m uvicorn backend.app.api.main:app --reload --port 8000
```

`GET /api/health` checks the configured Colab service with a bounded timeout. It does not load the remote or local model. The health contract reports model readiness only when Colab returns `model_ready: true` alongside `model_status: "ready"`. GPU status describes the Colab runtime, not the local laptop.

## API

- `GET /api/health` — local API reachability plus remote model and GPU status.
- `POST /api/analyse` — multipart fields `source_type` (`youtube`, `text`, or `upload`), with `youtube_url`, `transcript`, or `file`.
- `POST /api/qa` — JSON `{ "meeting_id": "...", "question": "..." }`; locally retrieves transcript evidence and sends only retrieved chunks to the remote QA endpoint.
- `POST /api/sample` — deterministic sample report that does not load model weights or require Colab.
- `GET /api/meetings/{meeting_id}/downloads?format=csv|markdown` — export action items or notes.

Meeting sessions and FAISS indexes are held in a bounded in-memory registry and are lost when the backend restarts. The Colab endpoint is temporary development infrastructure; do not send confidential transcripts through a public tunnel.
