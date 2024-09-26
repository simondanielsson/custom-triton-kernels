.PHONY: install_dev test

install_dev:
	pip install -r requirements.txt
	pre-commit install

test:
	pytest tests
