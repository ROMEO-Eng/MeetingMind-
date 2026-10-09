# Demo walkthrough

## Requirements

- Windows with Docker Desktop running Linux containers.
- Git and a browser.
- An active Google Colab GPU runtime for live analysis and Q&A. The deterministic sample works without Colab.

## Start the local Docker demo

From the repository root in PowerShell:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set `LLM_BASE_URL` to the currently active Colab HTTPS tunnel origin only and `LLM_API_KEY` to its private bearer key. Keep `.env` private and untracked. `NEXT_PUBLIC_API_BASE_URL` should remain a browser-reachable address, normally `http://localhost:8000`; do not set it to the Compose service hostname `api`.

The default published ports are API `8000` and frontend `3000`. If those host ports are occupied, set `API_HOST_PORT` and/or `FRONTEND_HOST_PORT` in `.env`; when changing the API port, also set `NEXT_PUBLIC_API_BASE_URL` to `http://localhost:<API_HOST_PORT>`. Add the frontend origin to `CORS_ORIGINS` if using a non-default frontend port.

Start and check the two local services:

```powershell
docker compose up --build -d
docker compose ps
$health = Invoke-RestMethod http://localhost:8000/api/health
$health | Select-Object status, api_status, model_status, model_ready, gpu_available, model_name
Invoke-WebRequest http://localhost:3000 -UseBasicParsing | Select-Object StatusCode
```

Open `http://localhost:3000` in the browser. The frontend makes browser-side requests directly to the local FastAPI API at `http://localhost:8000`; the backend alone calls Colab. A degraded model status means the local API is responding but the remote model is not ready. Compose can still start the frontend in that state, and the deterministic sample does not require Colab.

View logs:

```powershell
docker compose logs --tail 100 api frontend
docker compose logs -f api
```

After the Colab runtime starts with a new temporary tunnel URL or key, update `.env` and recreate only the API container so it reads the new values:

```powershell
docker compose up -d --force-recreate api
```

Stop the demo and retain the cache volume:

```powershell
docker compose down
```

Do not use `docker compose down -v`. MeetingMind caches local embedding files in its separate `meetingmind_model-cache` volume. The previously configured `finalproject_model-cache` belongs to a different Compose project and is not mounted or modified. No LLM weights are downloaded by the Docker image. Meeting sessions and FAISS indexes are process-memory data and are lost when the API container restarts.

## Colab inference service

1. Open `notebooks/MeetingMind_Colab_Inference.ipynb` in Google Colab and select a GPU runtime.
2. Run the setup and server cells. They report the assigned GPU/VRAM, choose a suitable Qwen2.5 instruction model, and load it in 4-bit NF4.
3. Copy the printed temporary endpoint and bearer key into local `.env` as `LLM_BASE_URL` and `LLM_API_KEY`.
4. Wait until the notebook's `/health` response reports `model_ready: true`; the local MeetingMind API health endpoint independently reports remote readiness.
5. Keep the runtime active. The endpoint is temporary and stops when the Colab session stops.

## Web product demo

1. Start the backend and frontend using the Docker commands above, or use the local development commands in the main README.
2. Visit `http://localhost:3000`, then choose **Try sample meeting**. The deterministic sample populates the dashboard without loading the model.
3. Verify that the legal-approval task has no owner or deadline rather than an invented person/date.
4. Choose **Analyse a meeting** to open `/workspace`; load the sample or submit a transcript, YouTube URL, or file.
5. Review decision/task source IDs and excerpts, export notes, then ask a question and expand its retrieved sources.

The deterministic UI sample works without Colab. Real analysis and Q&A need the active Colab runtime. MeetingMind targets English meeting transcripts and asks the model to produce English output.

## Five-line presentation script

1. MeetingMind transforms meeting transcripts into structured, source-addressable intelligence.
2. A Qwen2.5 instruction model on Colab extracts summaries, decisions, tasks, key points, and open questions.
3. Missing task owners and deadlines stay empty instead of being guessed.
4. Local FAISS retrieves transcript evidence before the remote model answers a question.
5. The Next.js dashboard stays connected only to FastAPI; Colab is an inference service, not a second application.
