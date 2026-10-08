# MeetingMind backend

FastAPI exposes the notebook's local Transformers/LangChain/FAISS workflow as a typed integration layer. Model and embedding weights load lazily on the first analysis or Q&A request and require a CUDA GPU.

## Run locally

From the repository root, create and activate a Python 3.11 environment, install the appropriate CUDA-enabled PyTorch build for your platform, then install the remaining requirements:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
$env:CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
python -m uvicorn backend.app.api.main:app --reload --port 8000
```

The PyTorch command above is for CUDA 12.4. Follow the official PyTorch selector for a different CUDA/runtime combination. The notebook can still run independently from `notebooks/MeetingMind_Final_Project.ipynb`.

## API

- `GET /api/health` — backend, GPU, and model readiness.
- `POST /api/analyse` — multipart fields `source_type` (`youtube`, `text`, or `upload`), with `youtube_url`, `transcript`, or `file`.
- `POST /api/qa` — JSON `{ "meeting_id": "...", "question": "..." }`; returns a grounded answer and source chunk references.
- `POST /api/sample` — deterministic sample report that does not load model weights.
- `GET /api/meetings/{meeting_id}/downloads?format=csv|markdown` — export action items or notes.

Meeting sessions are held in a bounded in-memory registry and are lost when the backend restarts. The API is intended for a trusted local/demo environment; add authentication, request throttling, and durable storage before exposing it publicly.
