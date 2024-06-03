from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from .bointerface import BO_Interface
from .rolloutEI import RolloutEI
from ..utils import compute_robustness
from ..behavior import Behavior

from ..agent.treeOperations import * 
from ..agent.partition import Node
from ..agent.constants import *
from ..agent.localgp import Prior
from ..agent.observation import Observations
from ..utils.savestuff import *
import multiprocessing as mp

from .mainRoutine import MainRoutine

with open('config.yml', 'r') as file:
    configs = yaml.safe_load(file)

class RolloutBO(BO_Interface):
    def __init__(self):
        pass
    
    def sample(
        self,
        test_function: Callable,
        num_samples: int,
        x_train: NDArray,
        y_train: NDArray,
        region_support: NDArray,
        gpr_model: Callable,
        rng,
        num_agents: int,
        init_sampling_type = "lhs_sampling",
        behavior:Behavior = Behavior.MINIMIZATION
    ) -> Tuple[NDArray]:

        """Rollout BO Model

        Args:
            test_function: Function of System Under Test.
            num_samples: Number of samples to generate from BO.
            x_train: Samples from Training set.
            y_train: Evaluated values of samples from Trainig set.
            region_support: Min and Max of all dimensions
            rng: RNG object from numpy

        Returns:
            X_root: New set of regions and their assigned agents
            x_m: Agent x_train
            y_m: Agent y_train
        """
        self.horizon = configs['horizon']
        falsified = False
        self.tf = test_function
        self.region_support = region_support
        self.behavior = behavior
        self.x_train = x_train
        self.y_train = y_train
        self.numthreads = int(mp.cpu_count()/2)
        self.mc_iters = configs['sampling']['mc_iters']

        tf_dim = region_support.shape[0]
        X_root = Node(self.region_support, RegionState.ACTIVE)

        globalObs = Observations(x_train, y_train)
        # Build the model over the entire region initially
        globalGP = Prior(globalObs, region_support)
        print(globalGP)
        model , indices = globalGP.buildModel()
        X_root.updateModel(indices, model)

        # Get the initial set of configs by partitioning among m agents
        m = -1
        roots = getRootConfigs(m, X_root, globalGP, 0, num_agents, tf_dim, test_function, behavior, rng)
        
        # Sample points using the multi agent routine 
        for sampleIdx in tqdm(range(num_samples)):
            print('globalXtrain, globalYtrain :', min(globalObs.y_train))
            print('_____________________________________', sampleIdx)
            print('_____________________________________')
            print('global dataset : ', globalObs.x_train.shape, globalObs.y_train.shape)
            print('_____________________________________')

            
            xroots, agentModels, globalGP = genSamplesForConfigsinParallel(m, globalGP, configs['configs']['smp'], num_agents, roots, init_sampling_type, tf_dim, self.tf, self.behavior, rng)
            xroots  = np.hstack((xroots))
            agentModels  = np.hstack((agentModels))

            # print('xroots : ', xroots)
            for i in xroots:
                print_tree(i)
                print(globalGP)
                for id, l in enumerate(i.find_leaves()):
                    localGP = Prior(globalGP.dataset, l.input_space)
                    # print(l.__dict__)
                    try:
                        assert localGP.checkPoints(globalGP.dataset.x_train[l.obsIndices]) == True
                    except AssertionError:
                        print(l.__dict__)
                        exit(1)
            
            # Calling the main routine to get the configuration as a result of multi agent rollout
            main = MainRoutine()
            X_root, globalGP = main.sample(sampleIdx, xroots, globalGP, num_agents, tf_dim, test_function, behavior, rng)

            ei_roll = RolloutEI()
            agents = []
            for id, l in enumerate(X_root.find_leaves()):    
                if l.getStatus() == RegionState.ACTIVE.value:
                    agents.append(l)

            agents = sorted(agents, key=lambda x: x.agentId)
                
            # Sample new set of agent locations from the winning configuration
            for l in agents: 
                xtr = l.samples.x_train[l.smpIndices]
                ytr = globalGP.dataset.y_train[l.obsIndices]
                smpEIs = (-1 * ei_roll.cost(xtr, ytr, l.model, "multiple"))
                maxEI = np.array([xtr[np.argmin(smpEIs), :]])
                # maxEI = ei_roll._opt_acquisition(ytr, l.model, l.input_space, rng)
                # maxEI = np.array([maxEI])
                fmin, _ = compute_robustness(maxEI, test_function, behavior, agent_sample=False)
                print('pred x  : ', maxEI, fmin)

                # Append the new locations to the entire observations set
                globalGP.dataset = globalGP.dataset.appendSamples(maxEI, fmin)
                
                # Update the respective local GPs
                localGP = Prior(globalGP.dataset, l.input_space)
                model , indices = localGP.buildModel()
                l.updateModel(indices, model)

                l.resetavgRewardDist(0)

            roots = [X_root]
            
        return falsified



    
    