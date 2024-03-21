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
from bo.agent.treeOperations import saveRegionState, restoreRegionState


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
    def sample(
        self,
        root,
        agents,
        num_agents,
        horizon: int,
        region_support: NDArray,
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
        self.horizon = horizon
        self.region_support = region_support
        self.rng = rng
        self.tf_dim = region_support.shape[0]
        self.num_agents = num_agents

        lf = self.root.find_leaves()

        agents = []
        
        # assert len(a.region_support.agentList) == 1
        for i,lv in enumerate(lf):
            lv.resetavgRewardDist(num_agents)
            # lv.resetSmps()
            lv.avgsmpYtr = np.zeros((1,len(lv.smpYtr)))
            lv.smpYtr = np.zeros((len(lv.smpYtr)))
            lv.mcsmpYtr = deepcopy(lv.smpYtr)
            lv.mcsmpXtr = deepcopy(lv.smpXtr)

            lv.resetStatus()
            if lv.getStatus(MAIN) == 1:
                print('lv.smpxtr b4 rollout :',lv.smpXtr, lv.smpYtr,lv.input_space)
                # print('lv.xtr.all() == lv.agent.x_train.all(): ',lv.xtr.all() == lv.agent.x_train.all())
                # print('xtr ytr : ',lv.input_space,lv.agent.x_train,lv.agent.y_train)
                assert lv.xtr.all() == lv.agent.x_train.all()
                assert lv.agent.x_train.all() == lv.agent.simXtrain.all()
                # ag = Agent(gpr_model, xtr, ytr, l)
                # ag(MAIN)
                agents.append(lv.agent)
                # print('- START '*100)
                savetotxt(self.savetxtpath+f'rl_start_agent_{i}', lv.agent.__dict__)
                # print(lv)
                # print(lv.__dict__)
                # print('.'*100)
                # print(lv.agent)
                # print(lv.agent.__dict__)
                # print('-START'*100)
                savetotxt(self.savetxtpath+f'rl_start_reg_{i}', lv.__dict__)
            # else:
            #     lv.smpXtr = []
            #     lv.smpYtr = []
        agents = sorted(agents, key=lambda x: x.id)

        self.num_agents = num_agents
        serial_mc_iters = [int(int(self.mc_iters)/self.numthreads)] * self.numthreads
        print("serial_mc_iters: ",serial_mc_iters)
        # print(' agents samples : ',[(i.x_train, i.y_train) for i in agents])
        print('min obs so far from samples: ', [(i.x_train[np.argmin(i.y_train),:], np.min(i.y_train)) for i in agents])

        root = self.get_exp_values(agents)

        return None, root , agents #, self.assignments, self.agent, subx[:num_agents]

    # Get expected value for each point after rolled out for h steps 
    def get_exp_values(self, agents):
        self.agents = agents
        # print('inside ')
        if not configs['parallel']:
            self.root = self.get_pt_reward(2)
        else:
            self.root = self._evaluate_at_point_list(agents)
        # print("Tree after MC iters get_exp_values: ")
        # print_tree(self.root, MAIN)
        # print_tree(self.root, ROLLOUT)
        return self.root
    
    
    def _evaluate_at_point_list(self, agents):
        results = []
        self.agents = agents
        serial_mc_iters = [int(int(self.mc_iters)/self.numthreads)] * self.numthreads
        # print('serial_mc_iters using job lib',serial_mc_iters)
        results = Parallel(n_jobs= -1, backend="loky", prefer='threads')\
            (delayed(unwrap_self)(i) for i in zip([self]*len(serial_mc_iters), serial_mc_iters))
        # print('_evaluate_at_point_list results',results)
        # for i in results:
        #     print('MAIN Tree returned after joblib')
        #     print_tree(i, MAIN)
        # # len(results[0].find_leaves())
        fin = np.zeros((1,len(results[0].find_leaves()),self.num_agents))#len(results[0].find_leaves()))) #len(results[0].find_leaves())
        for lf in results:
            lvs = lf.find_leaves()
            # print('len leaves : ', len(lvs))
            # print('leaves reward :', [i.avgReward for i in lvs])
            tmp = np.array([i.avgRewardDist for i in lvs], dtype='object')
            # print('tmp b4 :',tmp)
            tmp = tmp.reshape((1,len(lvs),self.num_agents)) #np.hstack(tmp)
            # print('tmp after :',tmp)
            fin = np.vstack((fin, tmp))
        fin = fin[1:,:,:]
        # print('fin leaves ', fin)
        fin = np.mean(fin, axis=0)#, keepdims=True)
        # print('avg fin',fin)
        self.root.setAvgRewards(fin)

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
        
        # smplen = len(lf[0].smpYtr)
        for a in agents:
            a.resetRegions()
            a(MAIN)
            assert a.simReg == a.region_support

        # smpy = np.zeros((1,smplen))
        for i in range(iters):
            self.mc = i+1
            # print('agents in mc iter : ', [i.region_support.input_space for i in agents])
            # for ix, a in enumerate(agents):
                # print(f">>>>>>>>>>>>>>>>>>>>>>>>>>> {self.mc}  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                # print('agent xtr ytr in start of MC of rollout  :',a.simReg.smpXtr, a.simReg.smpYtr)
                # print(">>>>>>>>>>>>>>>>>>>>>>>   >>>>>>>>>>>>>>>>>>>    >>>>>>>>>>>>>>>>>>>>>>")
            # save_node(self.root, '/Users/shyamsundar/ASU/sem2/RA/partmahpc/partma/results/'+configs['testfunc']+f'/nodes/rl_root_{self.sample}_MC_{self.mc}.pkl')
            self.get_h_step_with_part(agents)
            # save_node(self.root, f'/Users/shyamsundar/ASU/sem2/RA/partmahpc/partma/results/nodes/rl_root_{self.sample}_MCb4_{self.mc}.pkl')
            # export_tree_image(self.root, ROLLOUT, f"results/trees/rollout/MCtree_after_{i}_reassign.png")
            assert(len(agents) == self.num_agents)
            # assert(len([]))
            for sima in lf:
                # print('sima region and status, ',sima.input_space, sima.mainStatus, sima.rolloutStatus)
                assert sima.routine == MAIN
                accumulate_rewardDist(sima, self.num_agents)
                # sima.avgRewardDist = sima.rewardDist
                # print('sima region to accumuulate rewardDist:', sima.input_space, sima.avgRewardDist)#, sima.avgRewardDist.shape,np.asarray(sima.rewardDist).shape )
                # print('--------------------------------------')
                sima.rewardDist = np.asarray(sima.rewardDist, dtype="object").reshape((1, self.num_agents))
                sima.avgRewardDist = np.vstack((sima.avgRewardDist, sima.rewardDist))
                # print('sima.smpXtr.shape: b4',sima.smpXtr.shape,sima.smpYtr.shape)
                Xtr, Ytr = accumulateSamples(sima)
                # print('sima.smpXtr.shape: ',sima.smpXtr.shape,sima.smpYtr.shape)
                # if Ytr.shape
                # Ytr = np.hstack((Ytr))
                # print('b4 mc smp ytr : ', sima.mcsmpXtr, sima.mcsmpYtr,' Xtr, Ytr :', Xtr, Ytr)
                # smpy = np.vstack((smpy, Ytr))
                # smpy = smpy[1:]
                commonX, commonY = intersect(sima.mcsmpXtr, Xtr, Ytr)
                # print('commonX, commonY: ',commonX, commonY)
                # print('Avg Samples b4 accumulating: ',sima.avgsmpXtr, sima.avgsmpYtr)
                sima.avgsmpYtr = np.vstack((sima.avgsmpYtr, commonY))
                # sima.avgsmpYtr = sima.avgsmpYtr[1:]
                sima.avgsmpYtr = np.sum(sima.avgsmpYtr, axis=0)
                sima.avgsmpXtr = commonX #Xtr
                print('Accumulated sum of Samples : ',sima.input_space,sima.avgsmpXtr, sima.avgsmpYtr)
                sima.smpYtr = deepcopy(sima.mcsmpYtr)
                sima.smpXtr = deepcopy(sima.mcsmpXtr)
                
                # save_node(self.root, f'/Users/shyamsundar/ASU/sem2/RA/partmahpc/partma/results/nodes/rl_root_{self.sample}_MCb4reset_{self.mc}.pkl')
                # sima.avgRewardDist = np.asarray((sima.avgRewardDist))
                # print('b4 sima avg reward Dist calc : ', sima.avgRewardDist, 'sima.rewardDist: ',sima.rewardDist)
                # print('--------------------------------------')
                sima.avgRewardDist = np.sum(sima.avgRewardDist, axis=0) #[sum(i) for i in zip(sima.avgRewardDist, sima.rewardDist)]
                # print(sima.avgRewardDist.shape, sima.avgRewardDist.reshape((1, self.num_agents)))
                # sima.avgRewardDist = np.hstack((sima.avgRewardDist))
                # sima.avgRewardDist.tolist()
                # print('sima avg reward Dist : ', sima.avgRewardDist)
                # print('--------------------------------------')
                sima.resetRewardDist(self.num_agents)
                # sima.avgReward += sima.reward
                sima.child = []
                sima.resetStatus()
                sima.resetTrace(MAIN)
                # sima.resetSmps()
            
            for a in agents:
                a.resetRegions()
                a(MAIN)
                a.resetAgentList(MAIN)
                # print(f">>>>>>>>>>>>>>>>>>>>>>>>>>> {self.mc}  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                
                # print('B4 reset agent xtr ytr in MC of rollout  :',a.simXtrain, a.simYtrain)
                # print(">>>>>>>>>>>>>>>>>>>>>>>   >>>>>>>>>>>>>>>>>>>    >>>>>>>>>>>>>>>>>>>>>>")
                a.resetModel()
                # print(f">>>>>>>>>>>>>>>>>>>>>>>>>>> {self.mc}  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                # print('after reset agent xtr ytr in MC of rollout  :',a.x_train, 'ytr : ',a.y_train)
                # print(f">>>>>>>>>>>>>>>>>>>>>>>>>>> {self.mc}  >>>>>>>>> MAIN xtr ytr >>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                # print('after reset agent xtr ytr in MC of rollout  :',a.simXtrain, 'ytr : ',a.simYtrain)
                # print(">>>>>>>>>>>>>>>>>>>>>>>   >>>>>>>>>>>>>>>>>>>    >>>>>>>>>>>>>>>>>>>>>>")
                assert a.simReg == a.region_support
            

        #     print(f'########################### End of MC iter {i} #################################################')
        # # reward = np.array(reward, dtype='object')
        #     print(">>>>>>>>>>>>>>>>>>>>>>>> Tree just b4 avg reward update >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        #     print_tree(self.root, ROLLOUT)
        #     print('-'*100+'MAIN')
        #     print_tree(self.root, MAIN)
        #     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # smpXtr, smpYtr = [], np.zeros((1,smplen))
        for i in range(len(lf)):
            lf[i].avgRewardDist = lf[i].avgRewardDist / iters
            
            # lf[i].avgsmpYtr = lf[i].avgsmpYtr[1:]
            lf[i].avgsmpYtr = lf[i].avgsmpYtr / iters
            # smpYtr = np.mean(lf[i].avgsmpYtr, axis=0)
            # print('mean Accumulated Samples : ',lf[i].avgsmpXtr, lf[i].avgsmpYtr)
            # lf[i].smpXtr = smpXtr
            # assert lf[i].smpXtr.shape[0] == len(Ytr)
            lf[i].smpYtr = lf[i].avgsmpYtr
            # print('lf[i].smpYtr: ', lf[i].input_space, lf[i].smpYtr)
            print('--------------------------------------')
            # if lf[i].mainStatus == 1:
            #     print('final mc iteration :', lf[i].smpYtr, lf[i].smpYtr)
        # print_tree(self.root, MAIN)
        # exportTreeUsingPlotly(self.root, MAIN)
        # print()
        # print(">>>>>>>>>>>>>>>>>>>>>>>> Tree after avg reward update >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print_tree(self.root, MAIN)
        # exit(1)
        # (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # save_node(self.root, f'/Users/shyamsundar/ASU/sem2/RA/partmahpc/partma/results/'+configs['testfunc']+f'/nodes/rl_root_{self.sample}_MCafter_{self.mc}.pkl')
        return self.root
           

    def lookahead(self, root, globalGP, horizon, rng):
        reward = 0
        h = horizon
        self.tf_dim = len(root.input_space)
        rl_root = root #copy.deepcopy(self.root)
         #self.subregions
        reward = []

        tmpGP = deepcopy(globalGP)

        print('tmp gp data b4 loop ', tmpGP.dataset)
        # exit(1)
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
            print('agents: ',agents)

            minytrval = float('inf')
            minytr = []
            for ix, a in enumerate(agents):
                for ia in agents:
                    try:
                        minidx = tmpGP.dataset._getMinIdx(ia.obsIndices)
                    except IndexError:
                        print('-'*20)
                        print(tmpGP.dataset, tmpGP.dataset.x_train.shape,  tmpGP.dataset.y_train.shape)
                        print('-'*20)
                        exit(1)
                    # if min(ia.gp.dataset.y_train) < minytrval:
                        # minytrval = min(ia.y_train)
                    minytr = min(minytrval, tmpGP.dataset.y_train[minidx])
            ytr = minytr
            
            for ix, a in enumerate(agents):
                model = a.model
                if model == None:
                    print(' No model !')
                    print('h: ', h)

                    exit(1)

                for reg in xt:
                    
                    if reg.status == RegionState.ACTIVE.value:
                        if a != reg:
                            commonReg = find_common_parent(rl_root, reg, a)
                            # print('common parent : ', commonReg.input_space, a.simReg, reg )
                            # cmPrior = commonReg.rolloutPrior
                            model = commonReg.model
                            #self.gprFromregionPairs(a, reg.agent)
                        if model == None:
                            print(' No common model b/w !', a.simReg.input_space, reg.input_space)
                            print('h: ', h)
                            # save_node(rl_root, f'/Users/shyamsundar/ASU/sem2/RA/partmahpc/partma/results/'+configs['testfunc']+'/node_with_no_common_model.pkl')
                            exit(1)

                        # assert reg.model == reg.agent.simModel

                        next_xt = self._opt_acquisition(ytr,model,reg.input_space,rng)  # reg.agent.model
                        next_xt = np.asarray([next_xt])
                        test_next_xt = next_xt
                        # min_xt = next_xt
                        mu, std = self._surrogate(model, next_xt)
                        f_xt = np.random.normal(mu,std,1)
                        # min_yxt = f_xt
                        reward = (-1 * self.reward(f_xt,ytr)) #float('inf')
                        
                        xtr = reg.samples.x_train[reg.smpIndices]
                        mu, std = self._surrogate(model, xtr)  #agent.simModel
                        actY = []
                        for i in range(len(xtr)):
                            f_xt = np.random.normal(mu[i],std[i],1)
                            rw = (-1 * self.reward(f_xt,ytr))        # ?? ytr
                            if a == reg:
                                reg.yOfsmpIndices[reg.smpIndices[i]] = rw
                            actY.append(rw)
                        actY = np.hstack((actY))
                    
                        # if a == reg:
                        #     # print('reg.yOfsmpIndices.values: ',reg.yOfsmpIndices.values)
                        #     reward = min(reward, list(reg.yOfsmpIndices.values()))
                            
                        # else:
                        reward = min(reward, np.min(actY))
                            # if reward > np.min(actY):
                            #     reward = np.min(actY)


                        reg.rewardDist[ix] += reward
                    
                    else:
                        smp = sample_from_discontinuous_region(configs['sampling']['num_inactive'], [reg], totalVolume, self.tf_dim, rng, volume=True ) #uniform_sampling(5, internal_inactive_subregion[0].input_space, self.tf_dim, self.rng)
                        mu, std = self._surrogate(reg.model, smp)
                        reward = float('inf')
                        for i in range(len(smp)):
                            f_xt = np.random.normal(mu[i],std[i],1)
                            smp_reward = self.reward(f_xt,ytr)
                            if reward > -1*smp_reward:
                                reward = ((-1 * smp_reward))
                                # reg.addSample(smp[i])
                        # reg.rewardDist.append(reg.reward)
                        reg.rewardDist[ix] += (reward)

            # print()
            agents = sorted(agents, key=lambda x: x.agentId)
            print('agents after sorting :', agents)
            jump = random.random()
            dim = np.random.randint(self.tf_dim)
            subregions = reassignUsingRewardDist(rl_root, RegionState, agents, jump)
            rl_root = partitionRegions(rl_root, tmpGP, subregions, RegionState, dim)


            for a in rl_root.find_leaves():    
                if a.getStatus() == RegionState.ACTIVE.value:
                    if len(a.obsIndices) == 0:
                        parent = find_parent(rl_root, a)
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
                    

                    xtsize = ((self.tf_dim*10)/4) - len(a.smpIndices)
                    if len(a.smpIndices) ==0 : #xtsize > 0: 
                        # print('reg and filtered pts len in Actual:',  a.region_support.input_space, a.id)

                        x_train = lhs_sampling( 2, a.input_space, self.tf_dim, rng)
                        mu, std = self._surrogate(a.model, x_train)  #agent.simModel
                        actY = []
                        for i in range(len(x_train)):
                            f_xt = np.random.normal(mu[i],std[i],1)
                            rw = (-1 * self.reward(f_xt,ytr))  
                            actY.append(rw)
                        actY = np.hstack((actY))

                        a.samples.appendSamples(x_train, actY)
                        a.updatesmpObs()
                
            print('tmp gp end of horizon shape ',tmpGP.dataset.x_train.shape)

            # xt = rl_root.find_leaves() 
            # exit(1)
            print()
            print(f'- end of {h}'*20)
            print_tree(rl_root)
            print()
            h -= 1
            if h <= 0 :
                break
        return rl_root
                

        # exit(0)

    # Reward is the difference between the observed min and the obs from the posterior
    def reward(self,f_xt,ytr):
        ymin = np.min(ytr)
        r = max(ymin - f_xt, 0)
        # print(' each regret : ', r)
        return r


def simulate(root, globalGP, mc_iters, num_agents, horizon, rng):
    total_time = 0
    roll = RolloutEI()
    root = saveRegionState(root)  # Save the original state of the tree
    print('^'*50)
    print_tree(root)
    print('^'*50)
    lvs = root.find_leaves()
    # for sima in lvs:
    #         sima = saveRegionState(sima)
    #         # print(sima.__dict__)
    #         print('sima.children: b4 mc iters ', sima.saved_state['children'])

    # exit(1)
    # b4root = root
    for m in range(mc_iters):
        print(f'= MC {m}'*50)
        
        start_time = time.time()  # Start time
        # print('b4 roll : ', id(root), root)
        root = roll.lookahead(root, globalGP, horizon, rng)  # Execute the operation and build the tree
        for sima in lvs:
            
            accumulate_rewardDist(sima, num_agents)
            sima.rewardDist = np.asarray(sima.rewardDist, dtype="object").reshape((1, num_agents))
            sima.avgRewardDist = np.vstack((sima.avgRewardDist, sima.rewardDist))

            accumulate_all_keys(sima)

            # sima.children = []
            # print('sima.children: after reset s',sima, sima.children, sima.input_space, sima.__dict__['saved_state'])
            

        end_time = time.time()  # End time
        total_time += end_time - start_time  # Accumulate computation time
        # print('a4 roll : ', id(root), root)
        # assert b4root == root
        root = restoreRegionState(root, ['avgRewardDist','yOfsmpIndices'])  # Restore state except 'avgRewardDist'
        # assert b4root == root
        # print('a4 restore roll : ', id(root), root)
        print('^ after restore'*50)
        print(lvs[0].__dict__)
        print_tree(root)
        # print_tree(b4root)
        print('^'*50)
    print('total time ', total_time)
    exit(1)
    return root

