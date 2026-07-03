# CollisionVision — developer shortcuts
# Usage: make <target> [VAR=value]

PYTHON ?= python
IMAGE  ?= collision-vision

# preprocess / infer overridable arguments
FORMAT      ?= via
ANNOTATIONS ?= data/raw/annotations.json
IMAGES      ?= data/raw/images
OUTPUT      ?= data/processed
CONFIG      ?= config.yaml
WEIGHTS     ?= models/best.pt
SOURCE      ?= data/raw/images

.PHONY: help install install-dev preprocess train infer app test docker-build docker-run clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:  ## Install the package and runtime dependencies
	$(PYTHON) -m pip install -e .

install-dev:  ## Install dev/test dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt && $(PYTHON) -m pip install -e .

preprocess:  ## Convert annotations to a YOLOv8-Seg dataset
	$(PYTHON) -m collision_vision.cli preprocess \
		--format $(FORMAT) --annotations $(ANNOTATIONS) --images $(IMAGES) --output $(OUTPUT)

train:  ## Fine-tune the model from config.yaml
	$(PYTHON) -m collision_vision.cli train --config $(CONFIG)

infer:  ## Run inference and save overlays
	$(PYTHON) -m collision_vision.cli infer --weights $(WEIGHTS) --source $(SOURCE)

app:  ## Launch the Streamlit app
	streamlit run app.py

test:  ## Run the test suite
	$(PYTHON) -m pytest

docker-build:  ## Build the Streamlit app image
	docker build -t $(IMAGE) .

docker-run:  ## Run the Streamlit app container on port 8501
	docker run --rm -p 8501:8501 -v "$(PWD)/models:/app/models" $(IMAGE)

clean:  ## Remove caches and build artifacts
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
