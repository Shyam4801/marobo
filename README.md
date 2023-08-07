# Bayesian Optimization


## Installation

We use the Poetry tool which is a dependency management and packaging tool in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you. Please follow the installation of poetry at https://python-poetry.org/docs/#installation

After you've installed poetry, you can install all the dependencies by running the following command in the root of the project:

```
poetry install
```

## Running Demos

```
poetry run python demos/test_1d.py
```

Look at the tests for more details.

### FYI

Output receives a dictionary containing ```history``` and ```optimization_time```.