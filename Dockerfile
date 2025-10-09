# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:debian





# Set working directory
WORKDIR /app

# Copy your pyproject.toml (and optional poetry.lock)
COPY pyproject.toml ./
# COPY poetry.lock ./

# Copy the rest of your project
COPY ./* ./


# Install dependencies
RUN uv sync -p 3.12


WORKDIR /app/ru

# Default command (adjust as needed)
CMD ["uv", "run", "python",  "-m", "frink_embeddings_web.app"]