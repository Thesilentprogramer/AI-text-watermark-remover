# HF Docker Space: https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.11-slim

RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=user requirements.txt requirements.txt

USER user
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && grep -vE '^(torch|pytest)' requirements.txt > /tmp/requirements-prod.txt \
    && pip install --no-cache-dir -r /tmp/requirements-prod.txt

COPY --chown=user . /app

ENV ENABLE_LOCAL_GEMMA=false
ENV ENABLE_PERPLEXITY=true

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
