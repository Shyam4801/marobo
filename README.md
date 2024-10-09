# Multi Agent Rollout for Bayesian Optimization 
Solving black-box global optimization problems efficiently across domains remains challenging especially for large scale optimization problems. Bayesian optimization has obtained important success as a black box optimization technique based on surrogates, but it still suffers when applied to large scale heterogeneous landscapes. Recent approaches have proposed non-myopic approximations and partitioning of the input domain into subregions to prioritize regions that capture important areas of the solution space. A Multi Agent Rollout formulation of Bayesian optimization (MAroBO) that partitions the input domain among finite set of agents for distributed sampling is proposed. This is the official implementation of MAroBO.

## Link to Paper - https://drive.google.com/file/d/1xbVwCjLj0LSU8X6q7Ztp2xAVJSuWFf_d/view?usp=sharing

## Algorithm Overview

MAroBO constructs an m-ary tree structure where:
- Active regions are occupied by agents
- Abandoned regions become inactive and taken over by a dedicated agent
- Intuitively speaking, the existense of a 'hyper-edge' among agents facilitates decision-making.

<img src="resources/tree.png" width="500" alt="MAroBO Tree Structure"/>


## Illustration of MAroBO with 4 Agents on Modified Branin and Himmelblau Function

<p float="left">
  <img src="resources/S2.gif" width="500" alt="MAroBO with 4 agents on Modified Branin function"/>
  <img src="resources/SS.gif" width="500" alt="MAroBO with 4 agents on Himmelblau function"/>
</p>

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

## Testing Synthetic functions
```
poetry run python tests/log_test.py Test_internalBO.rastrigin
```

Look at the tests for more details.

### FYI

Output receives a dictionary containing ```history``` and ```optimization_time```.