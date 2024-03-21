from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm
import copy
from copy import deepcopy
import time 

from .rolloutEI import RolloutEI, simulate
from ..agent.partition import Node
from ..agent.treeOperations import * 

# from ..agent.agent import Agent
# from ..agent.constants import *
from ..utils.volume import compute_volume
from ..utils.logger import logtime, LOGPATH
import yaml
from joblib import Parallel, delayed
# from ..agent.constants import ROLLOUT, MAIN
from ..utils.treeExport import export_tree_image
from ..utils.visualize import plot_convergence
# from ..utils.plotlyExport import exportTreeUsingPlotly
# from tests.log_test import logrolldf
from ..utils.savestuff import *
from ..agent.localgp import Prior
# from multiprocessing import Pool
import multiprocessing as mp
from functools import partial

with open('config.yml', 'r') as file:
    configs = yaml.safe_load(file)


class RolloutRoutine:
    def __init__(self) -> None:
        pass


    def run(self, xroots, globalGP, num_agents, region_support, rng):
        # Xs_roots = self.evalConfigsinParallel(roots) #, sample, Xs_root, agents, num_agents, globalXtrain, globalYtrain, region_support, model, rng)
        print('xroots and agents b4 joblib : ', len(xroots))
        # Xs_roots = self.evalConfigs(xroots, num_agents, region_support, rng)

        mc = configs['sampling']['mc_iters']
        if configs['parallel']:
            # mp.set_start_method("spawn", force=True)
            # with mp.Pool(processes=1) as pool:
            #     print('pooling ')
            #     results = pool.map(partial(simulate, globalGP=globalGP, mc_iters=mc, num_agents=num_agents, horizon=4, rng=rng), xroots)

            results = Parallel(n_jobs=-1)(delayed(simulate)(Xs_root_item, globalGP=globalGP, mc_iters=mct, num_agents=num_agents, horizon=4, rng=rng) for mct in range(1,mc) for (Xs_root_item) in tqdm(xroots) )
        
        else:
            results=[]
            for xr in xroots:
                res = simulate(xr, globalGP=globalGP, mc_iters=2, num_agents=num_agents, horizon=4, rng=rng)
                results.append(res)
        

        Xs_roots = results
        print("Xs_root from joblib ",len(Xs_roots))

        print('Xs_roots: ', Xs_roots)

        return Xs_roots
    
    def evalConfigs(self, Xs_root, num_agents, region_support, rng):
        self.ei_roll = RolloutEI()
        # print('inside evalConfigs')
        # Define a helper function to be executed in parallel
        def evaluate_in_parallel(Xs_root_item):
            # print('Xs_root_item in eval config : ',Xs_root_item)
            agents = []
            return self.ei_roll.sample(Xs_root_item, num_agents, region_support, rng)

        # Execute the evaluation function in parallel for each Xs_root item
        results = Parallel(n_jobs=-1)(delayed(evaluate_in_parallel)(Xs_root_item) for (Xs_root_item) in tqdm(Xs_root))

        return results