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
               xroots: list, 
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
        for m in range(num_agents):
            

            # Rollout the configurations and get the config with min f_ro
            minXroot = self.run(m, xroots, globalGP, num_agents, tf_dim, test_function, behavior, rng)
            print(f'end of MAIN minz {m}')
            
            # Get the next location from agent m and its true function value for different possible cuts
            roots = getRootConfigs(m, minXroot, globalGP, 1, num_agents, tf_dim, test_function, behavior, rng, routine='MAIN')

            
            for i in roots:
                # print('in main routine')
                # print_tree(i, 'MAIN')
                for id, l in enumerate(i.find_leaves()):
                    ytr = globalGP.dataset.y_train[l.obsIndices]    
                    if l.getStatus() == RegionState.ACTIVE.value and l.agentId == m:
                        if len(l.obsIndices ) == 0:
                            print('len(l.obsIndices ): ', l, l.input_space )
                            exit(1)

                        xtr = l.samples.x_train[l.smpIndices]
                        smpEIs = (-1 * ei_roll.cost(xtr, ytr, l.model, "multiple"))
                        maxEI = np.array([xtr[np.argmin(smpEIs), :]])

                        # minidx = globalGP.dataset._getMinIdx(l.obsIndices)
                        # fmin = globalGP.dataset.y_train[minidx]

                        # x_opt = ei_roll._opt_acquisition(fmin, l.model, l.input_space, rng)
                        yofEI, _ = compute_robustness(maxEI, test_function, behavior, agent_sample=True)

                        print('agent actual sample decision : ', l.agentId, maxEI, yofEI)

                        globalGP.dataset = globalGP.dataset.appendSamples(maxEI, yofEI)
                        
                        localGP = Prior(globalGP.dataset, l.input_space)
                        # X_root.gp = globalGP
                        model , indices = localGP.buildModel()
                        l.updateModel(indices, model)
            # if m == 2:
            # print_tree(minXroot, 'MAIN')
            # print('-'*20)
            # print_tree(roots[0])
                # print_tree(minXroot)
                # exit(1)
            
            # Fix the configuration for agents 1 to (i-1) using rollout policy and get configs for agents i to m
            xroots, agentModels, globalGP = genSamplesForConfigsinParallel(m, globalGP, configs['configs']['smp'], num_agents, roots, "lhs_sampling", tf_dim, test_function, behavior, rng)
            xroots  = np.hstack((xroots))
            agentModels  = np.hstack((agentModels))

            print('xroots : ', xroots)
            for i in xroots:
                # print(f'tree after {m} minz')
                # print_tree(i)
                # print(globalGP)
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


        fincfgIdx = np.random.randint(len(roots))
        # print('m b4 reass part minXroot : ', m)
        # print_tree(minXroot, 'MAIN')
        # print()
        # print_tree(roots[fincfgIdx])
        # exit(1)
        # reassignAndPartition(m, minXroot, globalGP, num_agents, dim, tf_dim, test_function, behavior, rng)
        
        return roots[fincfgIdx] , globalGP#, smpGP

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
        print('res of rollout', Xs_roots)
        
        rollout  = RolloutRoutine()
        Xs_roots, F_nc = rollout.run(m, Xs_roots, globalGP, num_agents, test_function, tf_dim, behavior, rng)
        F_nc = np.hstack((F_nc))
        print('Fnc : ',F_nc)
        minXroot = Xs_roots[np.argmin(F_nc)]
        
        return minXroot #, smpGP
    