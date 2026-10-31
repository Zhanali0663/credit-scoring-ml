.PHONY: setup data test lint train demo docker-build docker-up clean

setup:                     ## Install all dependencies
	pip install -r requirements-dev.txt

data:                      ## Download the dataset and run cleaning
	python scripts/download_data.py

train:                     ## Run the full training pipeline
	python -m src.utils.pipeline

test:                      ## Run unit and integration tests
	pytest

lint:                      ## Static analysis
	ruff check src tests app

demo:                      ## Launch the Streamlit demo app
	streamlit run app/streamlit_app.py

docker-build:              ## Build the Docker image
	docker build -t credit-scoring-ml .

docker-up:                 ## Start the Streamlit app inside Docker
	docker compose up app

docker-train:              ## Run training inside Docker
	docker compose up training

clean:                     ## Remove cached artifacts
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
