up:
	docker-compose up -d
	ping 127.0.0.1 -c 2 > /dev/null # hack to wait 2 sec for ngrok to start
	docker logs ngrok

down:
	docker-compose down -v
	rm -rf notebook/.ipynb_checkpoints
	rm -rf notebook/.ipython
	rm -rf notebook/.jupyter
	rm -rf notebook/.local

format:
	uv run ruff format src

lint:
	uv run ruff check src

fix:
	uv run ruff check --fix src

type:
	uv run mypy --install-types --non-interactive src

test:
	uv run pytest -v -s --cov=src tests
