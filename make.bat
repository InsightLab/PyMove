@echo off


IF /I "%1"=="help" GOTO help
IF /I "%1"=="dev" GOTO dev
IF /I "%1"=="clean" GOTO clean
IF /I "%1"=="test" GOTO test
IF /I "%1"=="lint" GOTO lint
IF /I "%1"=="docs" GOTO docs
GOTO error

:help
	@echo "available commands"
	@echo " - dev    : install dev environment"
	@echo " - clean  : clean temporary folders and files"
	@echo " - test   : runs all unit tests"
	@echo " - lint   : checks code style"
	@echo " - docs   : creates documentation in html"
	GOTO :EOF

:dev
	pip install -r requirements-dev.txt
	pre-commit install
	GOTO :EOF

:clean
	DEL /Q `find . d .pytest_cache` -rf -type -name
	DEL /Q `find . d __pycache__` -rf -type -name
	DEL /Q `find . d .ipynb_checkpoints` -rf -type -name
	DEL /Q docs/_build -rf
	DEL /Q .coverage /F
	GOTO :EOF

:test
	CALL make.bat clean
	coverage run -m pytest
	coverage report
	GOTO :EOF

:lint
	CALL make.bat clean
	flake8
	GOTO :EOF

:docs
	CALL make.bat clean
	DEL /Q docs/api docs/examples -rf
	sphinx-apidoc -f -o docs/api pymove pymove/tests/
	jupyter nbconvert --to rst --output-dir docs/examples examples/[0-9]*.ipynb
	make -C docs html
	GOTO :EOF

:error
    IF "%1"=="" (
        ECHO make: *** No targets specified and no makefile found.  Stop.
    ) ELSE (
        ECHO make: *** No rule to make target '%1%'. Stop.
    )
    GOTO :EOF
