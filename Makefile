# Makefile

.PHONY: server

server:
	PYTHONPATH=. uv run python scripts/run_server.py