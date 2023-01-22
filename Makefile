install:
	pip install -r requirements_prod.txt

mypy:
	python -m mypy -p predictor
	rm -r -rf .mypy_cache

pylint:
	python -m pylint --fail-under 10 predictor

test:
	pytest

clean:
	rm -r -rf .mypy_cache
	find . | grep -E "(/__pycache__)" | xargs rm -rf
	rm -r -rf .pytest_cache .coverage .coverage.*
	python -m black predictor 
	isort predictor train.py

isort:
	isort predictor train.py
