PY ?= python

.PHONY: install test run pipeline-all clean

install:
	$(PY) -m pip install -r requirements-dev.txt

test:
	$(PY) -m pytest -q

pipeline-all:
	for s in FD001 FD002 FD003 FD004; do $(PY) pipeline.py --subset $$s; done

run:
	$(PY) server.py

clean:
	rm -rf runs/*
