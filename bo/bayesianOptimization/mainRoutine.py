from typing import Callable
import numpy as np
from .rolloutEI import RolloutEI
from ..utils import compute_robustness
from ..behavior import Behavior

from ..agent.treeOperations import * 
from ..agent.constants import *
from ..agent.localgp import Prior
from ..utils.savestuff import *

from .rolloutRoutine import RolloutRoutine

# Responsible for sampling locations one agent at a time 
class MainRoutine:
    def __init__(self) -> None:
         pass
    
    def sample(self, 
               sampleIdx, 
               xroots, 
               globalGP, 
               num_agents: int, 
               tf_dim: int, 
               test_function: Callable, 
               behavior:Behavior, 
               rng):
        
        """Actual agent outcomes (Step 1: Configuration generation followed by Step 2: Configuration rollout)

        Args: 
            xroots: Different configurations to rollout
            num_agents: Number of agents
            test_function_dimension: The dimensionality of the region. (Dimensionality of the test function)

        Return:
            minXroot: New Agent Configuration and their corresponding sampled locations after 'm' minimizations
                    
        """
        ei_roll = RolloutEI()
        m = 0
        # X_root = self.run(m, xroots, globalGP, num_agents, tf_dim, test_function, behavior, rng)
        for m in range(num_agents):
            
            # Rollout the configurations and get the config with min f_ro
            X_root = self.run(m, xroots, globalGP, num_agents, tf_dim, test_function, behavior, rng)
            # print(f'end of MAIN minz {m}')
            # print(f'- after end main {m} min'*20)
            # print_tree(X_root)
            # print('-'*20)

            # Get the next location from agent m and its true function value for different possible cuts
            roots = getRootConfigs(m, X_root, globalGP, 1, num_agents, tf_dim, test_function, behavior, rng, routine='MAIN')

            for i in roots:
                for id, l in enumerate(X_root.find_leaves()):
                    
                    if l.getStatus() == RegionState.ACTIVE.value and l.agentId == m:
                        xtr = l.samples.x_train[l.smpIndices]
                        ytr = globalGP.dataset.y_train[l.obsIndices]    # region f*
                        smpEIs = (-1 * ei_roll.cost(xtr, ytr, l.model, "multiple"))
                        maxEI = np.array([xtr[np.argmin(smpEIs), :]])

                        yofEI, _ = compute_robustness(maxEI, test_function, behavior, agent_sample=True)

                        print('agent actual sample decision : ', l.agentId, maxEI, yofEI)

                        globalGP.dataset = globalGP.dataset.appendSamples(maxEI, yofEI)
                        
                        localGP = Prior(globalGP.dataset, l.input_space)
                        model , indices = localGP.buildModel()
                        l.updateModel(indices, model)

            

            
            # Fix the configuration for agents 1 to (i-1) using rollout policy and get configs for agents i to m
            xroots, agentModels, globalGP = genSamplesForConfigsinParallel(m, globalGP, configs['configs']['smp'], num_agents, roots, "lhs_sampling", tf_dim, test_function, behavior, rng)
            xroots  = np.hstack((xroots))
            agentModels  = np.hstack((agentModels))

            for i in xroots:
                numag = 0
                for id, l in enumerate(i.find_leaves()): 
                    # print('l.avgRewardDist: ', l.avgRewardDist, l.input_space)
                    localGP = Prior(globalGP.dataset, l.input_space)
                    # print(l.__dict__)
                    try:
                        assert localGP.checkPoints(globalGP.dataset.x_train[l.obsIndices]) == True
                    except AssertionError:
                        print(l.__dict__)
                        exit(1)
                    if l.getStatus() == RegionState.ACTIVE.value:
                        numag += 1
                assert numag == num_agents

            

            print(f'- after main {m} min'*20)
            print_tree(X_root)
            print('-'*20)


        fincfgIdx = np.random.randint(len(roots))
        # print('m b4 reass part minXroot : ', m)
        # print_tree(X_root, 'MAIN')
        # print()
        # print_tree(roots[fincfgIdx])
        # exit(1)
        return roots[fincfgIdx] , globalGP

    # Responsible for rolling out the configurations and get the config with min f_ro
    def run(self, 
            m: int, 
            Xs_roots: list, 
            globalGP, 
            num_agents: int, 
            tf_dim: int, 
            test_function: Callable, 
            behavior: Behavior, 
            rng):
        
        """ Step 2: Configuration rollout

        Args: 
            m: ith agent to rollout fixing (1 to i-1) agent configurations
            Xs_roots: Configurations to rollout
            num_agents: Number of agents
            test_function_dimension: The dimensionality of the region. (Dimensionality of the test function)

        Return:
            minXroot: New Agent Configuration and their corresponding sampled locations

        
        """
        
        rollout  = RolloutRoutine()
        Xs_roots, F_nc = rollout.run(m, Xs_roots, globalGP, num_agents, test_function, tf_dim, behavior, rng)
        F_nc = np.hstack((F_nc))
        print('Fnc : ',F_nc)
        minXroot = Xs_roots[np.argmin(F_nc)]
        
        return minXroot 
    