Snowball: Extracting Relations from Large Plain-Text Collections
================================================================

This is my own implementation of the the Snowball system to bootstrap relationship instances. You can find more details about the original system here: 

- Eugene Agichtein and Luis Gravano, [Snowball: Extracting Relations from Large Plain-Text Collections](http://www.mathcs.emory.edu/~eugene/papers/dl00.pdf). In Proceedings of the fifth ACM conference on Digital libraries. ACM, 200.

- H Yu, E Agichtein, [Extracting synonymous gene and protein terms from biological literature](http://bioinformatics.oxfordjournals.org/content/19/suppl_1/i340.full.pdf). In Bioinformatics, 19(suppl 1), 2003 - Oxford University Press


For more details about this particular implementation please refer to:

- David S Batista, Bruno Martins, and Mário J Silva. , [Semi-Supervised Bootstrapping of Relationship Extractors with Distributional Semantics](http://davidsbatista.net/assets/documents/publications/breds-emnlp_15.pdf). In Empirical Methods in Natural Language Processing. ACL, 2015. (Honorable Mention for Best Short Paper)

- David S Batista, Ph.D. Thesis, [Large-Scale Semantic Relationship Extraction for Information Discovery (Chapter 5)](http://davidsbatista.net/assets/documents/publications/dsbatista-phd-thesis-2016.pdf), Instituto Superior Técnico, University of Lisbon, 2016



A sample file containing sentences where the named-entities are already tagged can be [downloaded](http://data.davidsbatista.net/sentences.txt.bz2), which has 1 million sentences taken from the New York Times articles part of the English Gigaword Collection.

**NOTE**: look at the desription of [BREDS](https://github.com/davidsbatista/BREDS) to understand how to give a tagged document collection and seeds to setup the bootstrapping of relationship instances with Snowball, both systems have a similar setup.


# Contributing to BREDS

Improvements, adding new features and bug fixes are welcome. If you wish to participate in the development of BREDS, 
please read the following guidelines.

## The contribution process at a glance

1. Preparing the development environment
2. Code away!
3. Continuous Integration
4. Submit your changes by opening a pull request

Small fixes and additions can be submitted directly as pull requests, but larger changes should be discussed in 
an issue first. You can expect a reply within a few days, but please be patient if it takes a bit longer. 


## Preparing the development environment

Make sure you have Python3.9 installed on your system

macOs
```
brew install python@3.9
python3.9 -m pip install --user --upgrade pip
python3.9 -m pip install virtualenv
```

Clone the repository and prepare the development environment:

```sh
git clone git@github.com:davidsbatista/Snowball.git
cd Snowball            
python3.9 -m virtualenv venv         # create a new virtual environment for development using python3.9 
source venv/bin/activate             # activate the virtual environment
pip install -r requirements_dev.txt  # install the development requirements
pip install -e .                     # install Snowball in edit mode
```


## Continuous Integration

Snowball runs a continuous integration (CI) on all pull requests. This means that if you open a pull request (PR), a 
full  test suite is run on your PR: 

- The code is formatted using `black` and `isort` 
- Unused imports are auto-removed using `pycln`
- Linting is done using `pyling` and `flake8`
- Type checking is done using `mypy`
- Tests are run using `pytest`

Nevertheless, if you prefer to run the tests & formatting locally, it's possible too. 

```sh
make all
```

## Opening a Pull Request

Every PR should be accompanied by short description of the changes, including:
- Impact and  motivation for the changes
- Any open issues that are closed by this PR

---

Give a ⭐️ if this project helped you!