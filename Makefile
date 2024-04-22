before_commit: check lint requirements

check:
	poetry check

install:
	poetry install --sync

lint:
	ruff format
	ruff check --fix

requirements: install
	poetry export --without-hashes -f requirements.txt --output requirements.txt
