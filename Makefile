.PHONY:	test lint virtualenv dist

lint:
	ruff format snowball tests
	ruff check --fix snowball tests


typing:
	mypy -p snowball


test:
	PYTHONPATH=. coverage run --source=./snowball -m pytest
	PYTHONPATH=. coverage report


clean:
	rm -rf build dist *.egg-info .coverage .pytest_cache .mypy_cache .pytest_cache src/*.egg-info


publish:
	make all
	python -m pip install --upgrade build
	python -m build
	python -m pip install --upgrade twine
	python -m twine upload --repository testpypi dist/*


all:
	make clean
	make lint
	make typing
	make test
