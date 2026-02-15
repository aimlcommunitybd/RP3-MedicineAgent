# Makefile

.PHONY: server

server:
	PYTHONPATH=. uv run python scripts/run_server.py # runs the medicalagent

run:
	# run the madicalcrew
	PYTHONPATH=. uv run python app/run.py 