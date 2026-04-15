VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PYTEST = $(VENV)/bin/pytest

.PHONY: setup transcribe sync test clean

setup:
	./setup.sh

transcribe:
ifndef FILE
	$(error FILE is required. Usage: make transcribe FILE=path/to/audio.mp3)
endif
	$(VENV)/bin/transcribe_podcast $(FILE) $(ARGS)

sync:
	$(PYTHON) podcast_sync.py $(ARGS)

test:
	$(PYTEST) tests/ -v

clean:
	rm -rf $(VENV) .models __pycache__ src/__pycache__ src/backend/__pycache__ tests/__pycache__
