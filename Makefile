.PHONY:	test lint virtualenv dist

lint:
	black -t py39 -l 120 snowball tests
	pycln -a breds tests
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


all:
	make clean
	make lint
	make typing
	make test