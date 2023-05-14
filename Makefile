.PHONY:	test lint virtualenv dist

lint:
	black -t py39 -l 120 snowball tests
	pycln -a snowball tests
	isort --profile black snowball tests
	PYTHONPATH=. pylint --rcfile=pylint.cfg snowball
	PYTHONPATH=. flake8 --config=setup.cfg snowball


typing:
	mypy --config mypy.ini -p snowball


test:
	PYTHONPATH=. coverage run --rcfile=setup.cfg --source=./snowball -m pytest
	PYTHONPATH=. coverage report --rcfile=setup.cfg


clean:
	rm -rf build dist *.egg-info .coverage .pytest_cache .mypy_cache .pytest_cache src/*.egg-info


publish:
	make clean
	python -m pip install --upgrade build
	python -m build
	python -m pip install --upgrade twine
	python -m twine upload --repository testpypi dist/*


all:
	make clean
	make lint
	make typing
	make test