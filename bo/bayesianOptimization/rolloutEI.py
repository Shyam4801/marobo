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

from ..agent.agent import Agent
# from ..agent.constants import *
from ..utils.volume import compute_volume
from ..utils.logger import logtime, LOGPATH
import yaml
from joblib import Parallel, delayed
from ..agent.constants import ROLLOUT, MAIN
from ..utils.treeExport import export_tree_image
from ..utils.visualize import plot_convergence
from ..utils.plotlyExport import exportTreeUsingPlotly
# from tests.log_test import logrolldf
import os, datetime, pandas as pd
import pickle
import csv, random
from ..utils.savestuff import *
from ..agent.prior import Prior

from dask.distributed import Client, LocalCluster
import dask

def logrolldf(xtr,ytr,aidx,h,init_samp, mc, rollout=True):
    # df = pd.DataFrame(np.array(data.history, dtype='object'))
    # df = df.iloc[:,1].apply(lambda x: x[0])
    # print(df)
    print('_____________Inside log roll__________________')
    print('xtr ytr :',[xtr,ytr])
    xcoord = pd.DataFrame(xtr)
    xcoord['y'] = ytr #pd.DataFrame({'x':xtr,'y': ytr})
    # xcoord['y'] = df.iloc[:,2]
    xcoord['ysofar'] = [min(xcoord['y'].iloc[:i]) for i in range(1,len(xcoord)+1)] #xcoord['y'].apply(lambda x : min([x - y for y in yofmins]))
    print('df : ', xcoord)
    if rollout:
        rl='rollout'
    else:
        rl = 'n'
    
    timestmp = 'results/rollResults/'+str(mc)+'/' +str(h)+'/'+ str(aidx)+'/'+ str(int(time.time())) + '/' #datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/'
    if not os.path.exists(timestmp):
        os.makedirs(timestmp)
    
    xcoord.to_csv(timestmp+str(h) + str(aidx)+'_'+rl+'.csv')
    plot_convergence(xcoord.iloc[init_samp:], timestmp+str(h) + str(aidx)+'_'+rl)
    xcoord = xcoord.to_numpy()

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
        sample,
        root,
        agents,
        num_agents,
        # test_function: Callable,
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
        # self.tf = test_function
        self.x_train = x_train
        self.gpr_model = gpr_model
        self.horizon = horizon
        self.region_support = region_support
        self.rng = rng
        self.tf_dim = region_support.shape[0]
        self.y_train = y_train #copy.deepcopy(np.asarray(test_function.point_history,dtype=object)[:,-1])
        self.num_agents = num_agents
        self.sample = sample

        # if not os.path.exists('results/'+configs['testfunc']+'/nodes'):
        #     os.makedirs('results/'+configs['testfunc']+'/nodes')
        #     os.makedirs('results/'+configs['testfunc']+'/reghist')
            
        # if not os.path.exists(f'results/dict/iter_{sample}'):
        #     os.makedirs(f'results/dict/iter_{sample}')
        
        # self.savetxtpath = f'results/dict/iter_{sample}/'

        lf = self.root.find_leaves()
        xtr = deepcopy(x_train)
        ytr = deepcopy(y_train)
        agents = []
        
        # assert len(a.region_support.agentList) == 1
        for i,lv in enumerate(lf):
            lv.resetavgRewardDist(num_agents)
            # lv.resetSmps()
            lv.avgsmpYtr = np.zeros((1,len(lv.smpYtr)))
            lv.smpYtr = np.zeros((len(lv.smpYtr)))
            lv.mcsmpYtr = deepcopy(lv.smpYtr)
            lv.mcsmpXtr = deepcopy(lv.smpXtr)
            # print('lv.smpxtr b4 rollout :',lv.smpXtr, lv.smpYtr,lv.input_space)

            # mu, std = self._surrogate(lv.model, lv.smpXtr)  #agent.simModel
            # actY = []
            # for i in range(len(lv.smpXtr)):
            #     f_xt = np.random.normal(mu[i],std[i],1)
            #     rw = (-1 * self.reward(f_xt,ytr))        # ?? ytr
            #     actY.append(rw)
            # actY = np.hstack((actY))
        
            # lv.smpYtr = actY

            lv.resetStatus()
            if lv.getStatus(MAIN) == 1:
                # print('lv.smpxtr b4 rollout :',lv.smpXtr, lv.smpYtr,lv.input_space)
                # print('lv.xtr.all() == lv.agent.x_train.all(): ',lv.xtr.all() == lv.agent.x_train.all())
                # print('xtr ytr : ',lv.input_space,lv.agent.x_train,lv.agent.y_train)
                assert lv.xtr.all() == lv.agent.x_train.all()
                assert lv.agent.x_train.all() == lv.agent.simXtrain.all()
                # ag = Agent(gpr_model, xtr, ytr, l)
                # ag(MAIN)
                agents.append(lv.agent)
                # print('- START '*100)
                # savetotxt(self.savetxtpath+f'rl_start_agent_{i}', lv.agent.__dict__)
                # print(lv)
                # print(lv.__dict__)
                # print('.'*100)
                # print(lv.agent)
                # print(lv.agent.__dict__)
                # print('-START'*100)
                # savetotxt(self.savetxtpath+f'rl_start_reg_{i}', lv.__dict__)
            # else:
            #     lv.smpXtr = []
            #     lv.smpYtr = []
        agents = sorted(agents, key=lambda x: x.id)
        # print('entering rollour status check-  '*10)
        # print_tree(self.root, MAIN)
        # print('--')
        # print_tree(self.root, ROLLOUT)
        # print('entering rollour -  '*10)
        # print('agents in rollout EI start : ', agents)
        # print('_______________________________ AGENTS AT WORK ___________________________________')  

        self.num_agents = num_agents
        serial_mc_iters = [int(int(self.mc_iters)/self.numthreads)] * self.numthreads
        # print("serial_mc_iters: ",serial_mc_iters)
        # print(' agents samples : ',[(i.x_train, i.y_train) for i in agents])
        # print('min obs so far from samples: ', [(i.x_train[np.argmin(i.y_train),:], np.min(i.y_train)) for i in agents])
        # from ..behavior import Behavior
        # for currentAgentIdx in range(len(agents)):
            # if currentAgentIdx != 0:
            #     agent.simXtrain = np.vstack((agent.simXtrain , agents[currentAgentIdx-1].simXtrain[-1]))
            #     agent.simYtrain = np.hstack((agent.simYtrain, agents[currentAgentIdx-1].simYtrain[-1]))
            #     agent.updatesimModel()
                

        root = self.get_exp_values(agents)

        # print('b4 reassign MAIN [i.region_support for i in agents]: ',[i.region_support.input_space for i in agents])
        # agents = reassign(root, MAIN, agents, currentAgentIdx, gpr_model, xtr, ytr)
        # subregions = reassignUsingRewardDist( root, MAIN, agents)
        # agents = partitionRegions(root, subregions, MAIN)

        # exportTreeUsingPlotly(root, MAIN)
        # assert len(agents) == num_agents
        #     # print(f'############################## End of Main iter ##############################################')
        #     # if currentAgentIdx == 1:
        #     #     exit(1)
        # for i, preds in enumerate(subx[:num_agents]):
        #         agents[i].point_history.append(preds)
        # assert subx.shape[0] == num_agents
        return None, root , agents #, self.assignments, self.agent, subx[:num_agents]

    # Get expected value for each point after rolled out for h steps 
    def get_exp_values(self, agents):
        self.agents = agents
        # print('inside ')
        if not configs['parallel']:
            self.root = self.get_pt_reward(configs['sampling']['mc_iters'])
        else:
            self.root = self._evaluate_at_point_list(agents)
        # print("Tree after MC iters get_exp_values: ")
        # print_tree(self.root, MAIN)
        # print_tree(self.root, ROLLOUT)
        return self.root
    
    def _evaluate_at_point_list_dask(self, agents):
        self.ei_roll = RolloutEI()
        self.agents = agents
        serial_mc_iters = [int(int(self.mc_iters)/self.numthreads)] * self.numthreads
        # Set up a local cluster
        dask.config.set({'distributed.worker.daemon': False})
        cluster = LocalCluster()
        client = Client(cluster)

        # Retrieve logs from all workers
        worker_logs = client.get_worker_logs()
    
        # Define a helper function to be executed in parallel
        def evaluate_in_parallel(Xs_root_item, serial_mc_iters):
            return unwrap_self((Xs_root_item, serial_mc_iters))
        
        # for worker, logs in worker_logs.items():
        #     print(f"Logs from worker {worker}:")
        #     for log in logs:
        #         print(log)
                
        # Execute the evaluation function in parallel for each Xs_root item
        results = []
        Xs_root = [self]*len(serial_mc_iters)
        for Xs_root_item in tqdm(Xs_root):
            result = client.submit(evaluate_in_parallel, Xs_root_item, 1)
            results.append(result)
    
        # Gather results
        results = client.gather(results)
    
        # Close the client and cluster
        client.close()
        cluster.close()
    
        print("results : " ,results)
        # exit(1)
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
        # return results
    
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
                # print('Accumulated sum of Samples : ',sima.input_space,sima.avgsmpXtr, sima.avgsmpYtr)
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
            # print('--------------------------------------')
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
           
    
    def get_h_step_with_part(self, agents):
        reward = 0
        # Temporary Gaussian prior 
        tmp_gpr = copy.deepcopy(self.gpr_model)
        globytr = copy.deepcopy(self.y_train)
        globxtr = copy.deepcopy(self.x_train)
        h = self.horizon
        rl_root = self.root #copy.deepcopy(self.root)
        xt = rl_root.find_leaves()  #self.subregions
        # agents = deepcopy(agents)
        # print()
        # print('b4 while agents subregion',[(i.simReg.input_space) for i in agents])
        # print('b4 while agents main subregion',[(i.region_support.input_space) for i in agents])
        # print()
        # for a in agents:
        #     ytr = a.y_train
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
                # reg.resetRewardDist()
                # reg.reward = []
            for l in rl_root.find_leaves():
                if l.rolloutStatus == 1:
                    assert l.xtr.all() == l.agent.simXtrain.all()
            # for ix, a in enumerate(agents):
                # print(f">>>>>>>>>>>>>>>>>>>>>>>>>>> {h}  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                # print('agent xtr ytr in rollout  :',a.simXtrain, a.simYtrain)
                # print(">>>>>>>>>>>>>>>>>>>>>>>   >>>>>>>>>>>>>>>>>>>    >>>>>>>>>>>>>>>>>>>>>>")
            minytrval = float('inf')
            minytr = []

            for ia in agents:
                if (ia.simYtrain).all() == None:
                    print('min(ia.simYtrain) < minytrval: ' , minytrval)
                # if ia.simYtrain != []:
                if min(ia.simYtrain) < minytrval:
                    minytrval = min(ia.simYtrain)
                    minytr = ia.simYtrain
                # else:
                #     continue
            ytr = minytr
            
            # for reg in xt:
            #     mu, std = self._surrogate(reg.model, reg.smpXtr)  #agent.simModel
            #     actY = []
            #     for i in range(len(reg.smpXtr)):
            #         f_xt = np.random.normal(mu[i],std[i],1)
            #         rw = (-1 * self.reward(f_xt,ytr))        # ?? ytr
            #         actY.append(rw)
            #     actY = np.hstack((actY))
            
            #     reg.smpYtr = actY
            for ix, a in enumerate(agents):
                # print(f">>>>>>>>>>>>>>>>>>>>>>>>>>> {h}  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                # print('agent xtr ytr in rollout  :',a.__dict__)
                # print(">>>>>>>>>>>>>>>>>>>>>>>   >>>>>>>>>>>>>>>>>>>    >>>>>>>>>>>>>>>>>>>>>>")
                model = a.simModel
                if model == None:
                    print(' No model !')
                    print('h: ', h)

                    exit(1)
                
                # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                # print(' agent sim y train ', ytr, min(ytr))
                # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                # exit(1)
                for reg in xt:
                    
                    # actregSamples = uniform_sampling(self.tf_dim*10, reg.input_space, self.tf_dim, self.rng)
                    # mu, std = self._surrogate(model, reg.smpXtr)
                    # actY = []
                    # for i in range(len(reg.smpXtr)):
                    #     f_xt = np.random.normal(mu[i],std[i],1)
                    #     rw = (-1 * self.reward(f_xt,ytr))        # ?? ytr
                    #     actY.append(rw)
                    # reg.smpYtr = actY
                    # reg.smpYtr = np.hstack((reg.smpYtr))
                    # actY = np.hstack((actY))
                    # # # print('act Y ',actY)
                    # ixtr = np.vstack((agent.simXtrain , actregSamples))
                    # iytr = np.hstack((agent.simYtrain, actY))
                    # model = GPR(InternalGPR())
                    # model.fit(ixtr, iytr)
                    # print(f">>>>>>>>>>>>>>>>>>>>>>>>>>> {h}  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    # print('reg xtr ytr in rollout  :',reg.__dict__)
                    # print(">>>>>>>>>>>>>>>>>>>>>>>   >>>>>>>>>>>>>>>>>>>    >>>>>>>>>>>>>>>>>>>>>>")
                    # exit(1)
                    if reg.rolloutStatus == 1:
                        if a.simReg != reg:
                            commonReg = find_common_parent(rl_root, reg, a.simReg)
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

                        next_xt = self._opt_acquisition(ytr,model,reg.input_space,self.rng)  # reg.agent.model
                        next_xt = np.asarray([next_xt])
                        test_next_xt = next_xt
                        # min_xt = next_xt
                        mu, std = self._surrogate(model, next_xt)
                        f_xt = np.random.normal(mu,std,1)
                        # min_yxt = f_xt
                        reward = (-1 * self.reward(f_xt,ytr)) #float('inf')
                        
                        mu, std = self._surrogate(model, reg.smpXtr)  #agent.simModel
                        actY = []
                        for i in range(len(reg.smpXtr)):
                            f_xt = np.random.normal(mu[i],std[i],1)
                            rw = (-1 * self.reward(f_xt,ytr))        # ?? ytr
                            actY.append(rw)
                        actY = np.hstack((actY))
                    
                        if a.simReg == reg:
                            
                        
                            reg.smpYtr = actY
                            # print(f">>>>>>>>>>>>>>>>>>>>>>>>>>> {h}  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                            # print('reg xtr ytr in rollout  :',reg.__dict__)
                            # print(">>>>>>>>>>>>>>>>>>>>>>>   >>>>>>>>>>>>>>>>>>>    >>>>>>>>>>>>>>>>>>>>>>")
                            # reg.smpXtr = np.vstack((reg.smpXtr , next_xt))
                            # reg.smpYtr = np.hstack((reg.smpYtr, reward))
                        
                        
                            # print('reg.smpYtr :',reg.input_space,reg.smpYtr)
                            if reward > np.min(reg.smpYtr):
                                reward = np.min(reg.smpYtr)
                                next_xt = reg.smpXtr[np.argmin(reg.smpYtr),:]
                        else:
                            if reward > np.min(actY):
                                reward = np.min(actY)


                        reg.rewardDist[ix] += reward
                        # assert test_next_xt.all() == next_xt.all()
                        # if a.simReg == reg:
                        #     a.simXtrain = np.vstack((a.simXtrain , next_xt))
                        #     a.simYtrain = np.hstack((a.simYtrain, f_xt))

                        # print('agent ytr :', ytr, reg.input_space)
                    
                    else:
                        smp = sample_from_discontinuous_region(configs['sampling']['num_inactive'], [reg], totalVolume, self.tf_dim, self.rng, volume=True ) #uniform_sampling(5, internal_inactive_subregion[0].input_space, self.tf_dim, self.rng)
                        mu, std = self._surrogate(reg.model, smp)
                        reward = float('inf')
                        for i in range(len(smp)):
                            f_xt = np.random.normal(mu[i],std[i],1)
                            smp_reward = self.reward(f_xt,ytr)
                            if reward > -1*smp_reward:
                                reward = ((-1 * smp_reward))
                                reg.addSample(smp[i])
                        # reg.rewardDist.append(reg.reward)
                        reg.rewardDist[ix] += (reward)
                    # reg.rewardDist = np.vstack((reg.rewardDist, reg.reward))
                    # reg.rewardDist = np.sum(reg.rewardDist, axis=0)
                    # print('agent ytr :', ytr, reg.input_space, min(ytr), f_xt)
                    # print("reward of all leaves : ",reg.input_space, reg.reward)
                    # reg.rewardDist = np.hstack((reg.rewardDist))
                    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    # print('agent reward dist : ', reg.rewardDist)
                    # print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>> {ix} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            # exit(1)       
            # print()
            # print('Rollout reassignments in the tree directly ')
            # print()
            currentAgentIdx = self.num_agents - h
            agents = sorted(agents, key=lambda x: x.id)
            # print_tree(rl_root, ROLLOUT)
            # print('b4 reassign rollout [i.simReg for i in agents]: ',[i.simReg.input_space for i in agents])
            # agents = reassign(rl_root, ROLLOUT, agents, currentAgentIdx, tmp_gpr, self.x_train, ytr)
            # print('after reassign rollout [i.simReg for i in agents]: ',[i.simReg.input_space for i in agents])
            jump = random.random()
            dim = np.random.randint(self.tf_dim)
            subregions = reassignUsingRewardDist(rl_root, ROLLOUT, agents, jump)
            agents = partitionRegions(rl_root, subregions, ROLLOUT, dim)
            # print('after reassign rollout [i.simReg for i in agents]: ',[i.simReg.input_space for i in agents])
            # export_tree_image(rl_root, ROLLOUT, f"results/trees/rollout/rlroot_after_{h}_reassign.png")
            # exportTreeUsingPlotly(rl_root)
            # for agent in agents:  
                # print(f'b4 splitting obs upadting from parent Rollout: ',agent.id, agent.x_train, agent.y_train)
                # print(f'b4 splitting sim obs upadting from parent Rollout: ',agent.id, agent.simXtrain, agent.simYtrain)
            self.tf = None
            agents = splitObs(agents, self.tf_dim, self.rng, ROLLOUT, self.tf, Behavior.MINIMIZATION)
            

            # print_tree(rl_root, ROLLOUT)
            # print('self.num_agents: ',self.num_agents, len(agents))
            assert len(agents) == self.num_agents

            xt = rl_root.find_leaves() 
            newx_opt = []
            for aidx, agent in enumerate(agents):
                # print(f'after splitting from parent Rollout: ',agent.id, agent.x_train, agent.y_train)
                # print('checking pts in rollout')
                assert check_points(agent, ROLLOUT) == True
                assert agent.simReg.getStatus(ROLLOUT) == 1
                # if reg.rolloutStatus == 1:
                if len(agent.simXtrain) != 0:
                    agent.updatesimModel()
                else:
                    # parentnode = find_parent(rl_root, agent.simReg)
                    # agent.simModel = deepcopy(parentnode.model)
                    actregSamples = lhs_sampling(self.tf_dim*10 , agent.simReg.input_space, self.tf_dim, self.rng)  #self.tf_dim*10
                    mu, std = self._surrogate(agent.simModel, actregSamples)  #agent.simModel
                    actY = []
                    for i in range(len(actregSamples)):
                        f_xt = np.random.normal(mu[i],std[i],1)
                        actY.append(f_xt)
                    actY = np.hstack((actY))
                    # # print('act Y ',actY)
                    agent.simXtrain = np.vstack((agent.simXtrain , actregSamples))
                    agent.simYtrain = np.hstack((agent.simYtrain, actY))
                    agent.updatesimModel()
                
                # print('-'*100)
                # print(lv)
                # print(lv.__dict__)
                # savetotxt(self.savetxtpath+f'rl_h{h}_agent_{aidx}', agent.__dict__)
                # print(f'- {h} ROLLOUT'*100)
                # print(agent)
                # print(agent.__dict__)
                # print('-ROLLOUT'*100)
                # simPrior = Prior(agent.simXtrain, agent.simYtrain, agent.simModel, ROLLOUT)
                agent.simReg.addFootprint(agent.simXtrain, agent.simYtrain, agent.simModel)
                agent.simReg.model = deepcopy(agent.simModel)
                # savetotxt(self.savetxtpath+f'rl_h{h}_agent_{aidx}', agent.__dict__)
                # assert agent.simReg.checkFootprint() == True
                
                # smp = sample_from_discontinuous_region(10*self.tf_dim, [reg], totalVolume, self.tf_dim, self.rng, volume=True ) #uniform_sampling(5, internal_inactive_subregion[0].input_space, self.tf_dim, self.rng)
                # print('a.simReg.rewardDist: ',self.mc, a.id, a.simReg.rewardDist[a.id])
                # writetocsv(f'results/'+configs['testfunc']+f'/reghist/SimA_{agent.id}', [[self.sample, self.mc, agent.id, agent.simReg.input_space.tolist(), min(agent.simReg.rewardDist.tolist()), np.argmin(agent.simReg.rewardDist.tolist())]])
            #     next_xt = self._opt_acquisition(globytr, agent.simModel,agent.simReg.input_space,self.rng)
            #     next_xt = np.asarray([next_xt])
            #     mu, std = self._surrogate(agent.simModel, next_xt)
            #     f_xt = np.random.normal(mu,std,1)
                # print(">>>>>>>>>>>>>>>>>>>>>>>>>>> inside end of horizon  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                # print('agent xtr ytr in rollout  :',agent.id, agent.x_train, agent.y_train)
                # print(">>>>>>>>>>>>>>>>>>>>>>>   >>>>>>>>>>>>>>>>>>>    >>>>>>>>>>>>>>>>>>>>>>")
            #     globxtr = np.vstack((globxtr , next_xt))
            #     globytr = np.hstack((globytr, f_xt))
            # tmp_gpr.fit(globxtr, globytr)
                # logrolldf(agent.simXtrain, agent.simYtrain, aidx, h,20, self.mc)
            # save_node(rl_root, f'/Users/shyamsundar/ASU/sem2/RA/partmahpc/partma/results/'+configs['testfunc']+f'/nodes/rl_root_{self.sample}_{currentAgentIdx}.pkl')
            # if h < 3:
            #     exit(1)
            lfs = rl_root.find_leaves() 
            for i,lv in enumerate(lfs):
                # if lv.rolloutStatus == 1:
                    # print(f'lv.smpXtr after {h}:',lv.smpXtr, lv.smpYtr, lv.input_space)
                if len(lv.smpXtr) == 0: # or lv.smpXtr.all() == None:
                    actregSamples = lhs_sampling(self.tf_dim*10 , lv.input_space, self.tf_dim, self.rng)  #self.tf_dim*10
                    lv.smpXtr = actregSamples
                # mu, std = self._surrogate(lv.model, lv.smpXtr)  #agent.simModel
                # actY = []
                # for i in range(len(lv.smpXtr)):
                #     f_xt = np.random.normal(mu[i],std[i],1)
                #     rw = (-1 * self.reward(f_xt,ytr))
                #     actY.append(rw)
                # actY = np.hstack((actY))
                # lv.smpYtr = actY
                # if lv.rolloutStatus == 1:
                #     savetotxt(self.savetxtpath+f'rl_h{h}_reg_{i}', lv.__dict__)
                assert lv.check_points() == True
                try:
                    assert lv.checkFootprint() == True
                except AssertionError:
                    print(lv.__dict__)
            # print(f'> h {h}'*100)
            # print_tree(rl_root, ROLLOUT)
            # print(f'> h {h}'*100)
            h -= 1
            if h <= 0 :
                break
        # exportTreeUsingPlotly(rl_root, ROLLOUT)
            
            

        # exit(0)
    
    # Reward is the difference between the observed min and the obs from the posterior
    def reward(self,f_xt,ytr):
        ymin = np.min(ytr)
        r = max(ymin - f_xt, 0)
        # print(' each regret : ', r)
        return r
    

    def gprFromregionPairs(self, agent1, agent2):
        # actregSamples = uniform_sampling(self.tf_dim*10, reg.input_space, self.tf_dim, self.rng)
        # mu, std = self._surrogate(agent.simModel, actregSamples)
        # actY = []
        # for i in range(len(actregSamples)):
        #     f_xt = np.random.normal(mu[i],std[i],1)
        #     actY.append(f_xt)
        # actY = np.hstack((actY))
        # # print('act Y ',actY)
        xtr = np.vstack((agent1.simXtrain , agent2.simXtrain))
        ytr = np.hstack((agent1.simYtrain, agent2.simYtrain))
        model = GPR(InternalGPR())
        model.fit(xtr, ytr)
        return model