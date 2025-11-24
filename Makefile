DOCKER_NAME ?= frink-web

.PHONY: dev
dev:
	uv run flask --app frink_embeddings_web.app run --debug

.PHONY: dev-gunicorn
dev-gunicorn:
	uv run gunicorn frink_embeddings_web.app:app

.PHONY: docker-build
docker-build:
	docker build -t $(DOCKER_NAME) .

.PHONY: docker-run
docker-run:
	docker run --rm -p 8000:8000 $(DOCKER_NAME)

.PHONY: lint
	uv run ruff check src

.PHONY: lint
lint:
	uv run ruff check
	uv run ruff format --check

.PHONY: format
format:
	-uv run ruff check --fix
	-uv run ruff format
