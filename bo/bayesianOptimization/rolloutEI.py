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
from ..sampling import uniform_sampling, sample_from_discontinuous_region
from ..utils import compute_robustness
from ..behavior import Behavior
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

from ..agent.agent import Agent
# from ..agent.constants import *
from ..utils.volume import compute_volume
from ..utils.logger import logtime, LOGPATH
import yaml
from joblib import Parallel, delayed
from ..agent.constants import ROLLOUT, MAIN
from ..utils.treeExport import export_tree_image
from ..utils.plotlyExport import exportTreeUsingPlotly

def unwrap_self(arg, **kwarg):
    return RolloutEI.get_pt_reward(*arg, **kwarg)

with open('config.yml', 'r') as file:
    configs = yaml.safe_load(file)
  
class RolloutEI(InternalBO):
    def __init__(self) -> None:
        pass

    # @logtime(LOGPATH)
    def sample(
        self,
        root,
        agents,
        num_agents,
        test_function: Callable,
        x_train,
        horizon: int,
        y_train: NDArray,
        region_support: NDArray,
        gpr_model: Callable,
        rng
    ) -> Tuple[NDArray]:

        """Rollout with EI

        Args:
            test_function: Function of System Under Test.
            horizon: Number of steps to look ahead
            y_train: Evaluated values of samples from Training set.
            region_support: Min and Max of all dimensions
            gpr_model: Gaussian Process Regressor Model developed using Factory
            rng: RNG object from numpy

        Raises:
            TypeError: If y_train is not (n,) numpy array

        Returns:
            next x values that have minimum h step reward 
        """
        self.mc_iters = configs['sampling']['mc_iters']
        print('below yml read mc_iters',self.mc_iters)
        self.root = root
        self.numthreads = int(mp.cpu_count()/2)
        self.tf = test_function
        self.x_train = x_train
        self.gpr_model = gpr_model
        self.horizon = horizon
        self.region_support = region_support
        self.rng = rng
        self.tf_dim = region_support.shape[0]
        self.y_train = y_train #copy.deepcopy(np.asarray(test_function.point_history,dtype=object)[:,-1])
        self.num_agents = num_agents

        lf = self.root.find_leaves()
        xtr = deepcopy(x_train)
        ytr = deepcopy(y_train)
        agents = agents

        # for l in lf:
        #     l.setRoutine(MAIN)
        #     if l.getStatus(MAIN) == 1:
        #         ag = Agent(gpr_model, xtr, ytr, l)
        #         ag(MAIN)
        #         agents.append(ag)
        
        # print('_______________________________ AGENTS AT WORK ___________________________________')  

        self.num_agents = num_agents
        serial_mc_iters = [int(int(self.mc_iters)/self.numthreads)] * self.numthreads
        print("serial_mc_iters: ",serial_mc_iters)
        # print(' initial agents obj: ',agents)
        for currentAgentIdx in range(len(agents)):
            self.root = self.get_exp_values(agents)

            print('b4 reassign MAIN [i.region_support for i in agents]: ',[i.region_support.input_space for i in agents])
            agents = reassign(root, MAIN, agents, currentAgentIdx, gpr_model, xtr, ytr)
            print('after reassign MAIN [i.region_support for i in agents]: ',[i.region_support.input_space for i in agents])
            export_tree_image(root, MAIN, f"results/trees/main/mainroot_after_{currentAgentIdx}_reassign.png")
            exportTreeUsingPlotly(root)
            assert len(agents) == num_agents
            
        #     print('<<<<<<<<<<<<<<<<<<<<<<<< Main routine tree <<<<<<<<<<<<<<<<<<<<<<<<')
        #     print_tree(self.root, MAIN)
        #     # print('Rollout tree in main routine ')
        #     print_tree(self.root, ROLLOUT)
        #     print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        # print('########################### End of MA #################################################')
        # # print('final subx : ',subx)
        # print('############################################################################')
        # for i, preds in enumerate(subx[:num_agents]):
        #         self.agent[i].point_history.append(preds)

        x_opt_from_all = []
        # print('final agent regions ', [i.region_support.input_space for i in agents])
        for i,a in enumerate(agents):
            # smp = uniform_sampling(5, a.region_support, tf_dim, rng)
            x_opt = self._opt_acquisition(y_train, a.model, a.region_support.input_space, rng) 
            # smp = np.vstack((smp, x_opt))
            x_opt_from_all.append(x_opt)
            # a.xtr = np.vstack((a.xtr, pred_sample_x))
            # a.ytr = np.hstack((a.ytr, (pred_sample_y)))
        subx = np.hstack((x_opt_from_all)).reshape((num_agents,self.tf_dim))

        for i, preds in enumerate(subx[:num_agents]):
                agents[i].point_history.append(preds)
        assert subx.shape[0] == num_agents
        return subx[:num_agents], root, agents #, self.assignments, self.agent

    # Get expected value for each point after rolled out for h steps 
    def get_exp_values(self, agents):
        self.agents = agents
        self.root = self.get_pt_reward(2)
        # self.root = self._evaluate_at_point_list(agents)
        # print("Tree after MC iters get_exp_values: ")
        # print_tree(self.root, MAIN)
        # print_tree(self.root, ROLLOUT)
        return self.root
    
    def _evaluate_at_point_list(self, agents):
        results = []
        self.agents = agents
        serial_mc_iters = [int(int(self.mc_iters)/self.numthreads)] * self.numthreads
        # print('serial_mc_iters using job lib',serial_mc_iters)
        results = Parallel(n_jobs= -1, backend="loky")\
            (delayed(unwrap_self)(i) for i in zip([self]*len(serial_mc_iters), serial_mc_iters))
        # print('_evaluate_at_point_list results',results)
        # for i in results:
        #     print('MAIN Tree returned after joblib')
        #     print_tree(i, MAIN)
        # # len(results[0].find_leaves())
        fin = np.zeros((1, len(results[0].find_leaves())))
        for lf in results:
            lvs = lf.find_leaves()
            # print('leaves reward :', [i.avgReward for i in lvs])
            tmp = np.array([i.avgReward for i in lvs], dtype='object')
            tmp = np.hstack(tmp)
            # print('tmp :',tmp)
            fin = np.vstack((fin, tmp))
        fin = fin[1:]
        # print('fin leaves ', fin)
        fin = np.mean(fin, axis=0)
        # print('avg fin',fin)
        self.root.setAvgRewards(fin.tolist())

        return self.root
        # lvs, avgval = find_leaves_and_compute_avg(results)
        # print([i.input_space for i in lvs], avgval)
        # # for i in range(len(lf)):
        #     lf[i].avgReward = lf[i].avgReward / iters
        # rewards = np.hstack((results))
        # return np.sum(rewards)/self.numthreads

    # Perform Monte carlo itegration
    # @logtime(LOGPATH)
    # @numba.jit(nopython=True, parallel=True)
    def get_pt_reward(self,iters):
        reward = []
        agents = self.agents
        lf = self.root.find_leaves()
        for a in agents:
            a.resetRegions()
            a(MAIN)
            assert a.simReg == a.region_support
        for i in range(iters):
            # rw = 
            print('agents in mc iter : ', [i.region_support.input_space for i in agents])
            self.get_h_step_with_part(agents)
            export_tree_image(self.root, ROLLOUT, f"results/trees/rollout/MCtree_after_{i}_reassign.png")
            assert(len(agents) == self.num_agents)
            
            for sima in lf:
                # print('sima region and status, ',sima.input_space, sima.mainStatus, sima.rolloutStatus)
                assert sima.routine == MAIN
                accumulate_rewards_and_update(sima)
                sima.avgReward += sima.reward
                sima.child = []
                sima.resetStatus()
            
            for a in agents:
                a.resetRegions()
                a(MAIN)
                assert a.simReg == a.region_support
            

            print(f'########################### End of MC iter {i} #################################################')
        # reward = np.array(reward, dtype='object')
        # print(">>>>>>>>>>>>>>>>>>>>>>>> Tree just b4 avg reward update >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print_tree(self.root, ROLLOUT)
        # (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        for i in range(len(lf)):
            lf[i].avgReward = lf[i].avgReward / iters
        
        
        # print()
        # print(">>>>>>>>>>>>>>>>>>>>>>>> Tree after avg reward update >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print_tree(self.root, MAIN)
        # (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        return self.root
           
    
    def get_h_step_with_part(self, agents):
        reward = 0
        # Temporary Gaussian prior 
        tmp_gpr = copy.deepcopy(self.gpr_model)
        ytr = copy.deepcopy(self.y_train)
        h = self.horizon
        rl_root = self.root #copy.deepcopy(self.root)
        xt = rl_root.find_leaves()  #self.subregions
        # agents = deepcopy(agents)
        # print()
        # print('b4 while agents_to_subregion',[(i.input_space,i.rolloutStatus, i.reward) for i in xt])
        # print()

        reward = []
        # print('empty reward: ',reward)
        while(True):
            # print(f">>>>>>>>>>>>>>>>>>>>>>>> horizon: {h} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            # print_tree(rl_root, ROLLOUT)
            # print_tree(rl_root, MAIN)
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            # print('inside while xt : ', [i.input_space for i in xt])
            totalVolume = 0
            for reg in xt:
                totalVolume += reg.getVolume()
            for reg in xt:
                if reg.rolloutStatus == 1:
                    next_xt = self._opt_acquisition(self.y_train,tmp_gpr,reg.input_space,self.rng)  # reg.agent.model
                    next_xt = np.asarray([next_xt])
                    mu, std = self._surrogate(tmp_gpr, next_xt)
                    f_xt = np.random.normal(mu,std,1)
                    reg.reward += (-1 * self.reward(f_xt,ytr))
                
                else:
                    smp = sample_from_discontinuous_region(configs['sampling']['num_inactive'], [reg], totalVolume, self.tf_dim, self.rng, volume=True ) #uniform_sampling(5, internal_inactive_subregion[0].input_space, self.tf_dim, self.rng)
                    mu, std = self._surrogate(tmp_gpr, smp)
                    for i in range(len(smp)):
                        f_xt = np.random.normal(mu[i],std[i],1)
                        smp_reward = self.reward(f_xt,ytr)
                        if reg.reward > -1*smp_reward:
                            reg.reward += (-1 * smp_reward)
                            reg.addSample(smp[i])
                # print("reward of all leaves : ",reg.input_space, reg.reward)
                    
            # print()
            print('Rollout reassignments in the tree directly ')
            # print()
            currentAgentIdx = self.num_agents - h
            print_tree(rl_root, ROLLOUT)
            print('b4 reassign rollout [i.simReg for i in agents]: ',[i.simReg.input_space for i in agents])
            agents = reassign(rl_root, ROLLOUT, agents, currentAgentIdx, tmp_gpr, self.x_train, ytr)
            print('after reassign rollout [i.simReg for i in agents]: ',[i.simReg.input_space for i in agents])
            export_tree_image(rl_root, ROLLOUT, f"results/trees/rollout/rlroot_after_{h}_reassign.png")
            print_tree(rl_root, ROLLOUT)
            assert len(agents) == self.num_agents

            xt = rl_root.find_leaves() 
            h -= 1
            if h <= 0 :
                break

        # exit(0)
    
    # Reward is the difference between the observed min and the obs from the posterior
    def reward(self,f_xt,ytr):
        ymin = np.min(ytr)
        r = max(ymin - f_xt, 0)
        # print(' each regret : ', r)
        return r
    