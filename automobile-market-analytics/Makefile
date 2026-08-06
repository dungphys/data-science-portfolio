.PHONY: build test eda train business pipeline clean docker-build docker-test

build:
	pip install -r requirements.txt

test:
	pytest

eda:
	python scripts/run_eda.py

train:
	python scripts/run_training.py

business:
	python scripts/run_business_analysis.py

pipeline: eda train business

docker-build:
	docker compose build

docker-test:
	docker compose run --rm test

docker-eda:
	docker compose run --rm eda

docker-train:
	docker compose run --rm train

docker-business:
	docker compose run --rm business

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache results/coverage htmlcov .coverage