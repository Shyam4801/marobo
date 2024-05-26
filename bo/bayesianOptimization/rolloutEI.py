from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm
import copy
from copy import deepcopy
import time 

from .internalBO import InternalBO
# from .rolloutAllatOnce import RolloutBO
from ..agent.partition import Node
from ..agent.treeOperations import * #reassign, find_close_factor_pairs, print_tree, accumulate_rewards_and_update, find_min_leaf

from ..gprInterface import GPR
from bo.gprInterface import InternalGPR
from ..sampling import uniform_sampling, sample_from_discontinuous_region, lhs_sampling
from ..utils import compute_robustness
from ..behavior import Behavior
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

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
import os, datetime, pandas as pd
import pickle
import csv, random
from ..utils.savestuff import *

from dask.distributed import Client, LocalCluster
import dask
# from ..agent.treeOperations import saveRegionState, restoreRegionState


def unwrap_self(arg, **kwarg):
    # print('inside unwrap_self :',arg)
    return RolloutEI.get_pt_reward(*arg, **kwarg)

with open('config.yml', 'r') as file:
    configs = yaml.safe_load(file)

def intersect(x,y, x_values):
    x_tuples = [tuple(row) for row in x]
    y_tuples = [tuple(row) for row in y]

    # Find the intersection between the arrays while preserving order
    intersection_tuples = [row for row in x_tuples if row in y_tuples]

    # Convert back to numpy arrays
    intersection = np.array(intersection_tuples)

    # Find the corresponding 1D array values
    # x_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # y_values = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

    x_values_intersection = []
    y_values_intersection = []
    for row in intersection_tuples:
        index_x = x_tuples.index(row)
        index_y = y_tuples.index(row)
        x_values_intersection.append(x_values[index_x])
        # y_values_intersection.append(y_values[index_y])

    return intersection, np.asarray(x_values_intersection)
  
class RolloutEI(InternalBO):
    def __init__(self) -> None:
        pass

    # @logtime(LOGPATH)
    # def sample(
    #     self,
    #     root,
    #     agents,
    #     num_agents,
    #     horizon: int,
    #     region_support: NDArray,
    #     rng
    # ) -> Tuple[NDArray]:

    #     """Rollout with EI

    #     Args:
    #         test_function: Function of System Under Test.
    #         horizon: Number of steps to look ahead
    #         y_train: Evaluated values of samples from Training set.
    #         region_support: Min and Max of all dimensions
    #         gpr_model: Gaussian Process Regressor Model developed using Factory
    #         rng: RNG object from numpy

    #     Raises:
    #         TypeError: If y_train is not (n,) numpy array

    #     Returns:
    #         next x values that have minimum h step reward 
    #     """
    #     self.mc_iters = configs['sampling']['mc_iters']
    #     print('below yml read mc_iters',self.mc_iters)
    #     self.root = root
    #     self.numthreads = int(mp.cpu_count()/2)
    #     self.horizon = horizon
    #     self.region_support = region_support
    #     self.rng = rng
    #     self.tf_dim = region_support.shape[0]
    #     self.num_agents = num_agents

    #     lf = self.root.find_leaves()

    #     agents = []
        
    #     # assert len(a.region_support.agentList) == 1
    #     for i,lv in enumerate(lf):
    #         lv.resetavgRewardDist(num_agents)
    #         # lv.resetSmps()
    #         lv.avgsmpYtr = np.zeros((1,len(lv.smpYtr)))
    #         lv.smpYtr = np.zeros((len(lv.smpYtr)))
    #         lv.mcsmpYtr = deepcopy(lv.smpYtr)
    #         lv.mcsmpXtr = deepcopy(lv.smpXtr)

    #         lv.resetStatus()
    #         if lv.getStatus(MAIN) == 1:
    #             print('lv.smpxtr b4 rollout :',lv.smpXtr, lv.smpYtr,lv.input_space)
    #             # print('lv.xtr.all() == lv.agent.x_train.all(): ',lv.xtr.all() == lv.agent.x_train.all())
    #             # print('xtr ytr : ',lv.input_space,lv.agent.x_train,lv.agent.y_train)
    #             assert lv.xtr.all() == lv.agent.x_train.all()
    #             assert lv.agent.x_train.all() == lv.agent.simXtrain.all()
    #             # ag = Agent(gpr_model, xtr, ytr, l)
    #             # ag(MAIN)
    #             agents.append(lv.agent)
    #             # print('- START '*100)
    #             savetotxt(self.savetxtpath+f'rl_start_agent_{i}', lv.agent.__dict__)
    #             # print(lv)
    #             # print(lv.__dict__)
    #             # print('.'*100)
    #             # print(lv.agent)
    #             # print(lv.agent.__dict__)
    #             # print('-START'*100)
    #             savetotxt(self.savetxtpath+f'rl_start_reg_{i}', lv.__dict__)
    #         # else:
    #         #     lv.smpXtr = []
    #         #     lv.smpYtr = []
    #     agents = sorted(agents, key=lambda x: x.id)

    #     self.num_agents = num_agents
    #     serial_mc_iters = [int(int(self.mc_iters)/self.numthreads)] * self.numthreads
    #     print("serial_mc_iters: ",serial_mc_iters)
    #     # print(' agents samples : ',[(i.x_train, i.y_train) for i in agents])
    #     print('min obs so far from samples: ', [(i.x_train[np.argmin(i.y_train),:], np.min(i.y_train)) for i in agents])

    #     root = self.get_exp_values(agents)

    #     return None, root , agents 
           
    # Rollout from the current state of the agents
    def rollout(self, m, root, globalGP, horizon, num_agents, tf, tf_dim, behavior, rng):

        """Simulated agent outcomes (Step 1: Configuration generation followed by Step 2: Configuration rollout)

        Args: 
            m: ith agent to rollout fixing (1 to i-1) agent configurations
            root: Configuration to rollout 
            horizon: Steps to simulate the agent outcomes
            num_agents: Number of agents
            test_function_dimension: The dimensionality of the region. (Dimensionality of the test function)

        Return:
            f_ro: Approximate Q-factor of the configuration after simulating for 'h' horizons
                    
        """
        self.num_agents = num_agents
        self.tf = tf
        self.tf_dim = tf_dim
        self.behavior = behavior
        reward = 0
        actm = m 
        h = horizon
        self.tf_dim = len(root.input_space)
        rl_root = root 
        
        
        # Simulate the state of the agents by keeping a copy of the locations sampled
        tmpGP = deepcopy(globalGP)
        ytr = np.min(tmpGP.dataset.y_train)
        # Rollout the agent movements for 'h' horizons
        while(True):
            totalVolume = 0
            xt = rl_root.find_leaves() 
            for reg in xt:
                totalVolume += reg.getVolume()

            agents = []
            for r in xt:
                if r.getStatus() == RegionState.ACTIVE.value:
                    # print('indside lookahead', 'status:', r.__dict__)
                    agents.append(r)
            
            agents = sorted(agents, key=lambda x: x.agentId)

            cfgroots = [rl_root]
            # Perform rollout one agent at a time
            for m in range(actm, self.num_agents):
                # Get the config with the min approx Q factor
                f_ro, rl_root = self.forward(m, tmpGP, xt, cfgroots, h, totalVolume, rng)
                print('m seq of minz :', m, f_ro, 'actm : ',actm)
                # print('% tree ')
                # print_tree(rl_root)
                # print('-'*20)
                if m == self.num_agents - 1:
                    break

                # Get the different possible splits for the chosen config rl_root
                roots = getRootConfigs(m, rl_root, tmpGP, 1, self.num_agents, self.tf_dim, tf, behavior, rng)
                # Create different possbile agent assignments and next set of samples to evaluate
                xroots, agentModels, tmpGP = genSamplesForConfigsinParallel(m, tmpGP, configs['configs']['smp'], self.num_agents, roots, "lhs_sampling", self.tf_dim, self.tf, self.behavior, rng)
                cfgroots  = np.hstack((xroots))
                agentModels  = np.hstack((agentModels))

                # Check if the observations and samples for each configuration are not empty
                # Build the local GPs for the respective configs based on locations sampled during simulation
                for crt in cfgroots:
                    for id, l in enumerate(crt.find_leaves()): 
                        localGP = Prior(tmpGP.dataset, l.input_space)
                        try:
                            assert localGP.checkPoints(tmpGP.dataset.x_train[l.obsIndices]) == True
                        except AssertionError:
                            print(l.__dict__)
                            exit(1)


                    for a in crt.find_leaves():    
                        if a.getStatus() == RegionState.ACTIVE.value:
                            if len(a.obsIndices) == 0:
                                parent = find_parent(crt, a)
                                a.model = deepcopy(parent.model)
                                
                                actregSamples = lhs_sampling(self.tf_dim*2 , a.input_space, self.tf_dim, rng)  #self.tf_dim*10
                                mu, std = self._surrogate(a.model, actregSamples)  #agent.simModel
                                actY = []
                                for i in range(len(actregSamples)):
                                    f_xt = np.random.normal(mu[i],std[i],1)
                                    actY.append(f_xt)
                                actY = np.hstack((actY))
                                
                                tmpGP.dataset = tmpGP.dataset.appendSamples(actregSamples, actY)
                            # else:
                            localGP = Prior(tmpGP.dataset, a.input_space)
                            a.model , a.obsIndices = localGP.buildModel() #updateObs()
                            

                            # xtsize = ((self.tf_dim*10)/4) - len(a.smpIndices)
                            # if len(a.smpIndices) ==0 : #xtsize > 0: 

                            #     x_train = lhs_sampling( 2, a.input_space, self.tf_dim, rng)
                            #     mu, std = self._surrogate(a.model, x_train)  #agent.simModel
                            #     actY = []
                            #     for i in range(len(x_train)):
                            #         f_xt = np.random.normal(mu[i],std[i],1)
                            #         rw = (-1 * self.reward(f_xt,ytr))  
                            #         actY.append(rw)
                            #     actY = np.hstack((actY))

                            #     a.samples.appendSamples(x_train, actY)
                            #     a.updatesmpObs()
            
            print(f'* end of {m} minz'*10)
            print('tmp gp end of horizon shape ',tmpGP.dataset.x_train.shape)

            print()
            print(f'- end of {h}'*20)
            print_tree(rl_root)
            print()
            h -= 1
            if h <= 0 :
                break
                # exit(1)
        return rl_root, f_ro
                
    # Function to get the config with min approx Q factor across the active regions
    def forward(self, m, tmpGP, xt, roots, h, totalVolume, rng):
        """Configuration evaluation 

        Args: 
            m: ith agent to rollout fixing (1 to i-1) agent configurations
            tmpGP: Copy of the observations encountered so far 
            h: Steps to simulate the agent outcomes
            xt: Partitioned regions (Active and inactive leaf nodes)

        Return:
            f_ro: Max EI among the agents
            f_g: Configurtion correcponding to the max EI
                    
        """
        f_ro = np.float64('inf')
        f_g = None
        for rl_root in roots:
            
            f_cfg, mincfg = self.get_cfg_EI(m, tmpGP, xt, rl_root, h, totalVolume, rng)
            if f_ro > f_cfg:
                f_ro = f_cfg
                f_g = mincfg

        return f_ro, f_g
    
    # Function to evaluate the samples for each config  
    def get_cfg_EI(self, m, tmpGP, xt, rl_root, h, totalVolume, rng):
        xt = sorted(xt, key=lambda x: (x.getStatus() == RegionState.INACTIVE.value, x.agentId if x.getStatus() == RegionState.ACTIVE.value else float('inf')))
        agents = []
        for r in xt:
            if r.getStatus() == RegionState.ACTIVE.value:
                agents.append(r)
        
        agents = sorted(agents, key=lambda x: x.agentId)

        # Get the f* among all the active regions 
        ytr = self.get_min_across_regions(agents, tmpGP) 
        print('min across reg : ', ytr)

        # for ix, a in enumerate(agents[m:]):
        for a in xt[m:self.num_agents]:
            # a = reg
            ix = a.agentId
            # Local GP of agent 
            model = a.model
            if model == None:
                print(' No model !')
                print('h: ', h)

                exit(1)

            for reg in xt[m:]:
                # evaluate the samples in the active region
                if reg.status == RegionState.ACTIVE.value:
                    # An extension to use the common parent GP instead of local GP
                    if a != reg:
                        commonReg = find_common_parent(rl_root, reg, a)
                        model = commonReg.model

                    if model == None:
                        print(' No common model b/w !', a.simReg.input_space, reg.input_space)
                        print('h: ', h)
                        exit(1)

                    # Calculate the cost using EI
                    xtr = reg.samples.x_train[reg.smpIndices]
                    smpEIs = (-1 * self.cost(xtr, ytr, model, "multiple"))
                    maxEI = np.array([xtr[np.argmin(smpEIs), :]])
                    # Add the location with min cost to the local GPs
                    if a == reg:
                        mu, std = self._surrogate(model, maxEI)
                        f_xt = np.random.normal(mu,std,1)
                        tmpGP.dataset.appendSamples(maxEI, f_xt)
                    
                    print('reg cost : ', a.agentId, np.min(smpEIs))
                    reg.rewardDist[ix] = np.min(smpEIs)
                    # exit(1)
                
                else:
                    # Evaluate the inactive regions by uniformly sampling 
                    smp = sample_from_discontinuous_region(configs['sampling']['num_inactive'], [reg], totalVolume, self.tf_dim, rng, volume=True ) #uniform_sampling(5, internal_inactive_subregion[0].input_space, self.tf_dim, self.rng)
                    mu, std = self._surrogate(reg.model, smp)
                    reward = float('inf')
                    for i in range(len(smp)):
                        f_xt = np.random.normal(mu[i],std[i],1)
                        smp_reward = self.reward(f_xt,ytr)
                        if reward > -1*smp_reward:
                            reward = ((-1 * smp_reward))
                    reg.rewardDist[ix] = (reward)
                    print('inact reg cost : ', reward)
            
            agents = sorted(agents, key=lambda x: x.agentId)
            print('agents after sorting :', agents)

        # Get the minimum encountered so far from the active regions based on predicted values
        f_ro = self.get_min_across_regions(agents, tmpGP)
        return f_ro, rl_root

    # Function to get the minimum across the set of active regions
    def get_min_across_regions(self, agents, tmpGP):
        minytrval = float('inf')
        minytr = []
        for ix, a in enumerate(agents):
            for ia in agents:
                try:
                    minidx = tmpGP.dataset._getMinIdx(ia.obsIndices)
                except IndexError:
                    # print('-'*20)
                    print(tmpGP.dataset, tmpGP.dataset.x_train.shape,  tmpGP.dataset.y_train.shape)
                    # print('-'*20)
                    # exit(1)
                    continue
                if minytrval > tmpGP.dataset.y_train[minidx]:
                    minytrval = tmpGP.dataset.y_train[minidx]
                    minytr = tmpGP.dataset.y_train[minidx] #min(minytrval, tmpGP.dataset.y_train[minidx])
        ytr = minytr

        return ytr
        
    # Reward calulation for inactive regions
    # Reward is the difference between the observed min and the obs from the posterior
    def reward(self,f_xt,ytr):
        ymin = np.min(ytr)
        r = max(ymin - f_xt, 0)
        # print(' each regret : ', r)
        return r
    
    # Cost function for active regions which is our base policy heuristic (EI)
    def cost(self,xt,ytr, model, sample_type='single'):
        r = self._acquisition(ytr, xt, model,sample_type)
        return r

# Function to rollout N^(RO) times 
def simulate(m, root, globalGP, mc_iters, num_agents, tf, tf_dim, behavior, horizon, rng):
    """Step 2: Simulate Configuration rollout

        Args: 
            m: ith agent to rollout fixing (1 to i-1) agent configurations
            root: Configuration to rollout
            mc_iters: Number of times to repeat the simulation (N^(RO))
            num_agents: Number of agents
            test_function_dimension: The dimensionality of the region. (Dimensionality of the test function)

        Return:
            F_nc: Average approximate Q-factor of the configuration
    """
    total_time = 0
    roll = RolloutEI()
    lvs = root.find_leaves()
    
    for l in lvs:
        l.state = State.ACTUAL
        print("obs isx b4" , l.obsIndices)
    root = saveRegionState(root)  # Save the original state of the tree
    # print('lvs : ', lvs, [(i.input_space, i.rewardDist, i.avgRewardDist) for i in lvs])
    

    print('^'*50)
    print_tree(root)
    print('^'*50)
    f_nc = 0
    for r in tqdm(range(mc_iters)):
        print(f'= MC {r}'*50)
        start_time = time.time()  # Start time

        # Rollout the current configuration
        root, f_ro = roll.rollout(m, root, globalGP, horizon, num_agents, tf, tf_dim, behavior, rng)  # Execute the operation and build the tree
        
        # Sum up the min ecountered over N^(RO) times 
        f_nc += f_ro

        # Get actual leaves
        lvs = getActualState(root)
        
        for sima in lvs:
            # Accumulate the improvements across regions to make the actual jumps
            accumulate_rewardDist(sima, num_agents)
            sima.rewardDist = np.asarray(sima.rewardDist, dtype="object").reshape((1, num_agents))
            sima.avgRewardDist = np.vstack((sima.avgRewardDist, sima.rewardDist))
            # print('sima retrieve: ',sima.avgRewardDist)
            sima.avgRewardDist = np.sum(sima.avgRewardDist, axis=0)
            # print('sima retrieve: ', sima, (sima.input_space, sima.rewardDist)) #, sima.avgRewardDist) )#, root, sima.saved_state)
            

        end_time = time.time()  # End time
        total_time += end_time - start_time  
        # print('^'*50)
        # print_tree(root)
        # print('^'*50)
        # Restore region state
        # restoreRegionState(root, ['avgRewardDist','yOfsmpIndices'])  
        for l in lvs:
            restoreRegionState(l, ['avgRewardDist','yOfsmpIndices'])  
            # print('b4 state : ', l, l.input_space, root, root.input_space)
            l.state = State.ACTUAL
            saveRegionState(l)
            # print('l after obsinx :', l.obsIndices)

        # print('^ after restore'*10)
        # print_tree(root, routine="MAIN")
        # print_tree(root)
        # exit(1)
    print('total time ', total_time)
    # exit(1)

    # Compute the average improvements across regions over N^(RO) times 
    for lf in lvs:
            lf.avgRewardDist = lf.avgRewardDist / mc_iters
            lf.rewardDist = lf.avgRewardDist
    # exit(1)
    # Average the min ecountered over N^(RO) times 
    F_nc = f_nc/mc_iters
    return root, F_nc

