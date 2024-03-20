before_commit: check requirements lint

check:
	poetry check

install:
	poetry install --sync

requirements: install
	poetry export --without-hashes -f requirements.txt --output requirements.txt

lint:
	ruff format
	ruff check --fix
