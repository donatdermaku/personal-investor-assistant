.PHONY: verify

verify:
	python -m compileall .
	python -m pytest -q
	python -m ruff check .
