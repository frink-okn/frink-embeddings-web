FROM ghcr.io/astral-sh/uv:debian

WORKDIR /app

# Send logs to Docker logs immediately
ENV PYTHONUNBUFFERED=1

# Add application to PYTHONPATH
ENV PYTHONPATH=/app/src

# Default number of gunicorn workers
ENV NUM_WORKERS=4

COPY pyproject.toml uv.lock README.md ./


RUN uv sync --frozen -p 3.12

COPY . .

EXPOSE 8000

# gunicorn configuration picked up from gunicorn.conf.py
CMD [".venv/bin/gunicorn",  "frink_embeddings_web.app:app"]