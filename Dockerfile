FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/meetingmind/.cache/huggingface

WORKDIR /app

COPY requirements.txt .
COPY requirements-runtime.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 \
    && python -m pip install --no-cache-dir -r requirements-runtime.txt \
    && python -c "import torch; assert torch.version.cuda is None, 'Expected CPU-only PyTorch'; print('CPU-only PyTorch:', torch.__version__)" \
    && python -m pip check

COPY backend ./backend

RUN useradd --create-home --shell /usr/sbin/nologin meetingmind \
    && mkdir -p /home/meetingmind/.cache/huggingface \
    && chown -R meetingmind:meetingmind /app /home/meetingmind

USER meetingmind

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=8)"

CMD ["uvicorn", "backend.app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
