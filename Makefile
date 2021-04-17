help:
	@echo "available commands"
	@echo " - dev    : install dev environment"
	@echo " - clean  : clean temporary folders and files"
	@echo " - test   : runs all unit tests"
	@echo " - lint   : checks code style"
	@echo " - docs   : creates documentation in html"

dev:
	pip install -r requirements-dev.txt
	pre-commit install

clean:
	rm -rf `find . -type d -name .pytest_cache`
	rm -rf `find . -type d -name __pycache__`
	rm -rf `find . -type d -name .ipynb_checkpoints`
	rm -rf docs/_build
	rm -f .coverage

test: clean
	coverage run -m pytest
	coverage report

lint: clean
	flake8

docs: clean
	cp docs/examples/notebooks.rst docs
	rm -rf docs/api docs/examples
	sphinx-apidoc -f -o docs/api pymove pymove/tests/
	jupyter nbconvert --to rst --output-dir docs/examples notebooks/[0-9]*.ipynb
	mv docs/notebooks.rst docs/examples
	make -C docs html
