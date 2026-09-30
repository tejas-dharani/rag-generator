.PHONY: install run dev test eval docker clean

install:
	uv sync

run:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

dev:
	uv run uvicorn app.main:app --reload --port 8000

test:
	uv run pytest

eval:
	uv run python -m eval.run_eval $(ARGS)

docker:
	docker compose up --build

clean:
	rm -rf data/chroma data/uploads data/rag.db
