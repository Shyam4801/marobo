
from typing import Callable, Tuple
from copy import deepcopy
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm
from ..utils.visualize import contour
import plotly.graph_objects as go
import yaml

from .bointerface import BO_Interface
from .rolloutEI import RolloutEI
from ..gprInterface import GPR
from ..sampling import uniform_sampling, lhs_sampling
from ..utils import compute_robustness
from ..behavior import Behavior

from ..utils.volume import compute_volume
from ..agent.treeOperations import * 
from ..agent.partition import Node
# from ..agent.agent import Agent
from ..agent.constants import *
from ..agent.localgp import Prior
from ..agent.observation import Observations
from ..utils.logger import logtime, LOGPATH
from ..utils.savestuff import *
import random
import os
from itertools import permutations
import multiprocessing as mp
from multiprocessing import Pool
from joblib import Parallel, delayed
# from ..utils.plotlyExport import exportTreeUsingPlotly
from memory_profiler import profile

from dask.distributed import Client, LocalCluster

from joblib import parallel_backend, parallel_config
import joblib
import dask, time, json


class Routine:
    def __init__(self, routine):
        self.routine = routine

    def run(self, root_node):
        self.routine.run(root_node)




# if __name__ == "__main__":
#     # Create a root node
#     root = TreeNode("Root")

#     # Create strategies
#     main_strategy = MainStrategy()
#     rollout_strategy = RolloutStrategy()

#     # Use Main Strategy to build the tree
#     tree_builder = TreeBuilder(main_strategy)
#     tree_builder.build_tree(root)

#     # Use Rollout Strategy to build the tree
#     tree_builder.strategy = rollout_strategy
#     tree_builder.build_tree(root)
