# References: https://isshub.readthedocs.io/en/develop/git/content/Makefile.html
include colours.mk

CWD := $(shell PWD)
BASENAME := $(shell basename "$(PWD)")
SRC := $(subst -,_,$(BASENAME))
SRC_DIR := $(addsuffix /,$(SRC))
TESTS_DIR := $(addsuffix /tests,$(CWD))

PYTHON_VERSION_FULL := $(wordlist 2,4,$(subst ., ,$(shell python --version 2>&1)))
PYTHON_VERSION := $(addsuffix $(word 2,${PYTHON_VERSION_FULL}),$(word 1,${PYTHON_VERSION_FULL}))

.DEFAULT: help

.PHONY: help
help:
	@echo "Please use 'make <target>' where <target> is one of the following:"
	@echo "    black"
	@echo "        Format code using black"
	@echo "    isort"
	@echo "        Sort the imported libraries"
	@echo "    pretty"
	@echo "        Run all code formatters (isort, black)"
	@echo "    check-black"
	@echo "        Check if source code complies with black"
	@echo "    check-isort"
	@echo "        Check if imported libraries comply with isort"
	@echo "    flake8"
	@echo "        Check source code with flake8"
	@echo "    mypy"
	@echo "        Check static typing with mypy"
	@echo "    lint"
	@echo "        Run all code linters (check-black, check-isort, flake8, mypy)"
	@echo "    full-clean"
	@echo "        Remove unnecessary generated files and directories such as '__pycache__', '.coverage, etc."
	@echo "    precommit-install"
	@echo "        Install pre-commit hooks"
	@echo "    precommit-update"
	@echo "        Autoupdate pre-commit hooks"
	@echo "    precommit"
	@echo "        Check precommit against all files"
	@echo "    generate-docs"
	@echo "        generate documentation"

.PHONY: black
black: ## Run the black tool and update files that need to
	@echo "$(BGreen)Running black$(Color_Off)"
	black $(SRC_DIR)

.PHONY: isort
isort:
	@echo "$(BGreen)Running isort$(Color_Off)"
	isort --profile black $(SRC_DIR)

.PHONY: pretty
pretty:
pretty: isort black

.PHONY: check-black
check-black:
	@echo "$(BGreen)Checking black$(Color_Off)"
	black --check $(SRC_DIR)

.PHONY: check-isort
check-isort:
	@echo "$(BGreen)Checking isort$(Color_Off)"
	isort $(SRC_DIR) --check-only

.PHONY: mypy
mypy:  ## Run the mypy tool
	@echo "$(BGreen)Running mypy$(Color_Off)"
	mypy --config-file ./mypy.ini $(SRC_DIR)
	#rm -rf ./mypy_cache

.PHONY: flake8
flake8: ## Run the flake8 tool
	@echo "$(BGreen)Running flake8$(Color_Off)"
	@echo $(CWD)
	flake8 --max-line-length 99 --ignore W503,E203,E402 $(SRC_DIR)

.PHONY: lint
lint:
lint: check-black flake8 mypy

.PHONY: full-clean
full-clean:
	@echo "$(BGreen)Full Cleaning$(Color_Off)"
	@echo "$(BGreen)Remove necessary directories$(Color_Off)"
	find $(CWD) -type d  \( -name '__pycache__' -or -name '.pytest_cache' -or -name '.mypy_cache'  \) | xargs rm -fr
	find $(CWD) -type f  \( -name '.coverage' -or -name '.coverage.*'  \) | xargs rm -fr

.PHONY: precommit-install
install-precommit:
	pre-commit install

.PHONY: precommit-update
update-precommit:
	pre-commit autoupdate

.PHONY: precommit
precommit:
	pre-commit run --all-files

.PHONY: generate-docs
generate-docs:
	# remove all .rst files except from the index
	find docs/source/ -name "*.rst" | grep -v index.rst | xargs rm
	# generate the api docs
	sphinx-apidoc -f -o docs/source/ $(SRC_DIR)
	cd docs/ && make clean && make html
