.PHONY: setup run run-pro test verify clean sample-data perf-smoke

PYTHON := python

setup:
	$(PYTHON) -m pip install -r requirements.txt
	@echo "Dependencies installed."

run:
	$(PYTHON) -m streamlit run streamlit_app.py

run-pro:
	$(PYTHON) -m streamlit run streamlit_app.py -- --mode="Pro"

test:
	$(PYTHON) -m pytest

verify:
	$(PYTHON) -m compileall .
	$(PYTHON) -m pytest -q
	$(PYTHON) -m ruff check .

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf data/cache/manifests/*
	@echo "Cache usage cleaned."

sample-data:
	@echo "Sample data located in data/sample/"

perf-smoke:
	$(PYTHON) scripts/perf_smoke.py --base-url http://localhost:8000 --path /health --requests 25
