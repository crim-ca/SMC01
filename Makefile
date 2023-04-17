ALL_NOTEBOOKS=$(shell find notebooks/ -type f -not -path *.ipynb_checkpoints* -name *.ipynb )
DIFF_NOTEBOOKS=$(shell git diff master HEAD --relative --name-only | grep --color=never -E .ipynb$$ )
STAGED_NOTEBOOKS=$(shell git diff --cached --name-only --relative | grep --color=never -E .ipynb$$ )
SELECTED_NOTEBOOKS=$(shell echo $(DIFF_NOTEBOOKS) $(STAGED_NOTEBOOKS) | tr ' ' '\n' | sort | uniq)
REPORTS_DIR=./docs/exports/
SRC_DIR=./smc01/


ifdef $(SLURM_CPUS_PER_TASK)
N_CPUS=$(SLURM_CPUS_PER_TASK)
else
N_CPUS=4
endif

# Automatic formatting.
format:
	isort $(SRC_DIR)
	black $(SRC_DIR)

# Check if configured style guidelines are respected.
lint:
	isort --check $(SRC_DIR)
	black --check $(SRC_DIR)
	flake8 $(SRC_DIR)

# Install the package for development through pip.
setup:
	pip install --editable ./

# Install a kernel for use with Jupyter Notebooks.
install-kernel:
	python -m ipykernel install --user --name 'smc01' --display-name "SMC01"


notebooks-export:
	echo $(SELECTED_NOTEBOOKS) | tr ' ' '\n' | parallel --no-run-if-empty -j $(N_CPUS) jupyter nbconvert --to html --output-dir $(REPORTS_DIR) {}

notebooks-run:
	echo $(SELECTED_NOTEBOOKS) | tr ' ' '\n' | parallel --no-run-if-empty -j $(N_CPUS) jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=600 {}

notebooks-run-export:
	echo $(SELECTED_NOTEBOOKS) | tr ' ' '\n' | parallel --no-run-if-empty -j $(N_CPUS) jupyter nbconvert --to html --output-dir $(REPORTS_DIR) --execute --ExecutePreprocessor.timeout=600 {}

notebooks-strip:
	echo $(SELECTED_NOTEBOOKS) | tr ' ' '\n' | parallel --no-run-if-empty -j $(N_CPUS) jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {}

notebooks-are-stripped:
	crim_notebooks_are_stripped -dir ./notebooks/

changed-notebooks:
	@echo $(SELECTED_NOTEBOOKS)

