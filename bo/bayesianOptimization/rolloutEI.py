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
import csv
from ..utils.savestuff import *

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
    return RolloutEI.get_pt_reward(*arg, **kwarg)

with open('config.yml', 'r') as file:
    configs = yaml.safe_load(file)

def min_diagonal(matrix):
    # Number of 4x4 matrices in the nx4x4 matrix
    n = matrix.shape[0]
    
    # Initialize the resulting 4x4 matrix with max values
    result_matrix = np.full((4, 4), np.inf)
    index_matrix = np.zeros((4,), dtype=int)
    
    # Iterate over each 4x4 matrix
    for i in range(n):
        # Extract the current 4x4 matrix
        current_matrix = matrix[i]
        
        # Iterate over each row
        for j in range(4):
            # Compare diagonal elements and retain the row with the minimum diagonal element
            if current_matrix[j, j] < result_matrix[j, j]:
                result_matrix[j] = current_matrix[j]
                index_matrix[j] = i
    
    return result_matrix, index_matrix
  
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
        self.sample = sample

        if not os.path.exists('results/'+configs['testfunc']+'/nodes'):
            os.makedirs('results/'+configs['testfunc']+'/nodes')
            os.makedirs('results/'+configs['testfunc']+'/reghist')

        lf = self.root.find_leaves()
        xtr = deepcopy(x_train)
        ytr = deepcopy(y_train)
        agents = agents
        # assert len(a.region_support.agentList) == 1
        for lv in lf:
            all(element.all() == 0 for element in lv.avgRewardDist)
            all(element.all() == 0 for element in lv.rewardDist)
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
        # print(' agents samples : ',[(i.x_train, i.y_train) for i in agents])
        print('min obs so far from samples: ', [(i.x_train[np.argmin(i.y_train),:], np.min(i.y_train)) for i in agents])
        # from ..behavior import Behavior
        # for currentAgentIdx in range(len(agents)):
            # if currentAgentIdx != 0:
            #     agent.simXtrain = np.vstack((agent.simXtrain , agents[currentAgentIdx-1].simXtrain[-1]))
            #     agent.simYtrain = np.hstack((agent.simYtrain, agents[currentAgentIdx-1].simYtrain[-1]))
            #     agent.updatesimModel()
                

        root, avgrewards = self.get_exp_values(agents)
        avgrewards = avgrewards[1:]
        print('avgrewards: ',avgrewards)
        # avgrewards = np.hstack((avgrewards))
        minrewardDist, minrewardDistIdx = min_diagonal(avgrewards)
        print('minrewardDist, minrewardDistIdx: ',minrewardDist, minrewardDistIdx, minrewardDist.shape, minrewardDistIdx.shape)
        # exit(1)
        # print('b4 reassign MAIN [i.region_support for i in agents]: ',[i.region_support.input_space for i in agents])
        # agents = reassign(root, MAIN, agents, currentAgentIdx, gpr_model, xtr, ytr)
        for ix, a in enumerate(agents):
            print('b4 a.id, a.avgRewardDist: ',a.id, a.region_support.avgRewardDist)
            a.avgRewardDist = minrewardDist[ix]
            print('a.id, a.avgRewardDist: ',a.id, a.region_support.avgRewardDist)

        subregions = reassignUsingRewardDist( root, MAIN, agents)
        agents = partitionRegions(root, subregions, MAIN)

            # exportTreeUsingPlotly(root, MAIN)
        assert len(agents) == num_agents
            # print(f'############################## End of Main iter ##############################################')
            # if currentAgentIdx == 1:
            #     exit(1)
                
        # print('<<<<<<<<<<<<<<<<<<<<<<<< Main routine tree <<<<<<<<<<<<<<<<<<<<<<<<')
        # print_tree(self.root, MAIN)
            #     # print('Rollout tree in main routine ')
            #     print_tree(self.root, ROLLOUT)
            #     print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            # print('########################### End of MA #################################################')
            # # print('final subx : ',subx)
            # print('############################################################################')
            # for i, preds in enumerate(subx[:num_agents]):
            #         self.agent[i].point_history.append(preds)

        x_opt_from_all = []
        print('final agent regions ', [i.region_support.input_space for i in agents])
        for i,a in enumerate(agents):
            assert len(agents) == num_agents
            # actregSamples = uniform_sampling(self.tf_dim*10, a.region_support.input_space, self.tf_dim, self.rng)
            # mu, std = self._surrogate(a.simModel, actregSamples)
            # actY = []
            # for i in range(len(actregSamples)):
            #     f_xt = np.random.normal(mu[i],std[i],1)
            #     actY.append(f_xt)
            # actY = np.hstack((actY))
            # # # print('act Y ',actY)
            # a.simXtrain = actregSamples #np.vstack((agent.simXtrain , actregSamples))
            # a.simYtrain = actY #np.hstack((agent.simYtrain, actY))
            # a.updatesimModel()

            # minytrval = float('inf')
            # minytr = a.simYtrain
            # for ia in agents:
            #     if min(ia.simYtrain) < minytrval:
            #         minytrval = min(ia.simYtrain)
            #         minytr = ia.simYtrain
            # x_opt = self._opt_acquisition(minytr, a.simModel, a.region_support.input_space, rng) 
            # mu, std = self._surrogate(a.simModel, np.array([x_opt]))
            # f_xt = np.random.normal(mu,std,1)
            
            # a.simXtrain = np.vstack((a.simXtrain , np.array([x_opt])))
            # a.simYtrain = np.hstack((a.simYtrain, f_xt))
            x_opt = a.pointsToeval[minrewardDistIdx[i]]
            # x_opt = self._opt_acquisition(self.y_train, self.gpr_model, a.region_support.input_space, rng)
            # if f_xt < minytrval:
            #     minytrval = min(a.simYtrain)
            #     minytr = a.simYtrain
            # print(i)
            writetocsv('results/'+configs['testfunc']+f'/reghist/MainA_{a.id}', [[sample, a.region_support.input_space.tolist(), min(a.region_support.avgRewardDist.tolist()), np.argmin(a.region_support.avgRewardDist.tolist())]])
            # smp = np.vstack((smp, x_opt))
            # x_opt_from_all.append(a.simXtrain[np.argmin(a.simYtrain),:])   # x_opt
            
            x_opt_from_all.append(x_opt)

            a.resetRegions()
            if a.region_support.getStatus(MAIN) == 0:
                    a.region_support.agentList = []
            assert len(a.region_support.agentList) == 1
            a.pointsToeval = None
            # a.updateModel()
                # # smp = uniform_sampling(5, a.region_support, tf_dim, rng)
                # x_opt = self._opt_acquisition(a.y_train, a.model, a.region_support.input_space, rng) 
                # # smp = np.vstack((smp, x_opt))
                # x_opt_from_all.append(x_opt)

                # a.xtr = np.vstack((a.xtr, pred_sample_x))
                # a.ytr = np.hstack((a.ytr, (pred_sample_y)))
            print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            print(f'min obs so far from samples using gaussian mean: {i}', (a.simXtrain[np.argmin(a.simYtrain),:], np.min(a.simYtrain)))
            print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            lf = root.find_leaves()
            for lv in lf:
                lv.resetavgRewardDist(num_agents)
            save_node(root, '/Users/shyamsundar/ASU/sem2/RA/partmahpc/partma/results/'+configs['testfunc']+f'/nodes/root_{sample}_{a.id}.pkl')
            # exit(1)
        subx = np.hstack((x_opt_from_all)).reshape((num_agents,self.tf_dim))

        for i, preds in enumerate(subx[:num_agents]):
                agents[i].point_history.append(preds)
        assert subx.shape[0] == num_agents
        return subx[:num_agents], root, agents #, self.assignments, self.agent

    # Get expected value for each point after rolled out for h steps 
    def get_exp_values(self, agents):
        self.agents = agents
        samples = 5
        for a in agents:
            if a.pointsToeval == None:
                a.getSamplesToeval(samples, self.tf_dim, self.rng)
                next_xt = self._opt_acquisition(self.y_train,self.gpr_model,a.region_support.input_space,self.rng)  # reg.agent.model
                next_xt = np.asarray([next_xt])
                a.pointsToeval = np.vstack((a.pointsToeval , next_xt))
                # print('aid - pointsToeval: ',a.id, a.pointsToeval)
        avgrewards = np.zeros((1,self.num_agents,self.num_agents))
        for smp in range(samples):
            self.smp = smp
            # self.root = self.get_pt_reward(2)
            # for i,a in enumerate(agents):
                # if smp == 10:
                #     # minrewardDist, minrewardDistIdx = min_diagonal(avgrewards)
                #     x_opt = a.pointsToeval
                #     a.simXtrain = np.array([x_opt]) #np.vstack((a.simXtrain, ))
                #     a.simYtrain = np.array(a.evalRewards) #a.np.vstack((a.simYtrain, minrewardDist[i]))
                #     a.updatesimModel()
                #     print('aid, a.simXtrain, a.simYtrain after collecting 10 hstep rewards: ', a.id, a.simXtrain, a.simYtrain)

            avgAgentrewards = np.zeros((1,self.num_agents))
            self.root = self._evaluate_at_point_list(agents)
            for a in agents:
                # avgAgentrewards.append(a.region_support.avgRewardDist)
                avgAgentrewards = np.vstack((avgAgentrewards, a.region_support.avgRewardDist.reshape((1,self.num_agents))))
                a.appendevalReward(avgAgentrewards[a.id][a.id])
                # print('a.pointsToeval[:smp]: ',a.pointsToeval[:smp+1], a.pointsToeval[:smp+1].shape )
                a.x_train = np.vstack((a.x_train, a.pointsToeval[:smp+1]))
                a.y_train = np.hstack((a.y_train, np.array(a.evalRewards[:smp+1])))
                a.updateModel()
                a.resetModel()
                # print('aid xtrain ,ytrain: ',a.id, a.x_train, a.y_train)
                print('a.pointsToeval[:smp+1]: ',a.id , a.pointsToeval[:smp+1], a.evalRewards[:smp+1])

            # print(avgAgentrewards, avgAgentrewards.shape)
            avgrewards = np.vstack((avgrewards, avgAgentrewards[1:].reshape((1,self.num_agents,self.num_agents))))
        # print("Tree after MC iters get_exp_values: ",avgrewards)
        # print_tree(self.root, MAIN)
        # print_tree(self.root, ROLLOUT)
        return self.root, avgrewards
    
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
        for a in agents:
            a.resetRegions()
            a(MAIN)
            assert a.simReg == a.region_support
            # a.x_train = np.vstack((a.x_train , a.pointsToeval[self.smp]))
        
        for i in range(iters):
            self.mc = i+1
            # print('agents in mc iter : ', [i.region_support.input_space for i in agents])
            save_node(self.root, '/Users/shyamsundar/ASU/sem2/RA/partmahpc/partma/results/'+configs['testfunc']+f'/nodes/rl_root_{self.sample}_MC_{self.mc}.pkl')
            # for a in agents:
            #     # a.simXtrain = np.zeros((1,self.tf_dim))
            #     # a.simYtrain = []
            #     # print('inside MC a.pointsToeval[self.smp]: ',a.pointsToeval[self.smp])
                
            #     a.x_train = np.vstack((a.x_train , a.pointsToeval[self.smp]))
            #     print()
            #     a.resetModel()
            # print('a.x_train, a.simXtrain: inside MC iter: ',a.x_train, a.simXtrain)
                # a.simXtrain = a.simXtrain[1:]

            self.get_h_stepPerSmp_with_part(agents)
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
            
            for a in agents:
                a.resetRegions()
                a(MAIN)
                a.resetAgentList(MAIN)
                a.resetModel()
                # print('a.x_train, a.simXtrain: inside MC iter: ',a.x_train, a.simXtrain)
                assert a.simReg == a.region_support
            

        #     print(f'########################### End of MC iter {i} #################################################')
        # # reward = np.array(reward, dtype='object')
        #     print(">>>>>>>>>>>>>>>>>>>>>>>> Tree just b4 avg reward update >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        #     print_tree(self.root, ROLLOUT)
        #     print('-'*100+'MAIN')
        #     print_tree(self.root, MAIN)
        #     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        for i in range(len(lf)):
            lf[i].avgRewardDist = lf[i].avgRewardDist / iters
            # print('lf[i].avgRewardDist: ', lf[i].avgRewardDist)
            # print('--------------------------------------')
        # print_tree(self.root, MAIN)
        # exportTreeUsingPlotly(self.root, MAIN)
        # print()
        # print(">>>>>>>>>>>>>>>>>>>>>>>> Tree after avg reward update >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print_tree(self.root, MAIN)
        # (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        save_node(self.root, f'/Users/shyamsundar/ASU/sem2/RA/partmahpc/partma/results/'+configs['testfunc']+f'/nodes/rl_root_{self.sample}_MCafter_{self.mc}.pkl')
        return self.root
           
    
    def get_h_stepPerSmp_with_part(self, agents):
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
        # print('b4 while agents_to_subregion',[(i.input_space,i.rolloutStatus, i.reward) for i in xt])
        # print()
        # for a in agents:
        #     ytr = a.y_train
        for a in agents:
            a.simXtrain = np.vstack((a.simXtrain , np.array([a.pointsToeval[self.smp]]))) 
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
            minytrval = float('inf')
            minytr = []
            for ix, a in enumerate(agents):
                # if h == 4:
                #     axtr = deepcopy(a.simXtrain)
                #     aytr = deepcopy(a.simYtrain)
                model = deepcopy(a.simModel)
                for ia in agents:
                    if min(ia.simYtrain) < minytrval:
                        minytrval = min(ia.simYtrain)
                        minytr = ia.simYtrain
                ytr = minytr
                # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                # print(' agent sim y train ', ytr, min(ytr))
                # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                # exit(1)
                for reg in xt:
                    # actregSamples = uniform_sampling(self.tf_dim*10, reg.input_space, self.tf_dim, self.rng)
                    # mu, std = self._surrogate(model, actregSamples)
                    # actY = []
                    # for i in range(len(actregSamples)):
                    #     f_xt = np.random.normal(mu[i],std[i],1)
                    #     actY.append(f_xt)
                    # actY = np.hstack((actY))
                    # # # print('act Y ',actY)
                    # ixtr = np.vstack((agent.simXtrain , actregSamples))
                    # iytr = np.hstack((agent.simYtrain, actY))
                    # model = GPR(InternalGPR())
                    # model.fit(ixtr, iytr)
                    
                    if reg.rolloutStatus == 1:
                        if a.simReg != reg:
                            model = self.gprFromregionPairs(a, reg.agent)

                            next_xt = self._opt_acquisition(ytr,model,reg.input_space,self.rng)  # reg.agent.model
                            next_xt = np.asarray([next_xt])
                            mu, std = self._surrogate(model, next_xt)
                            f_xt = np.random.normal(mu,std,1)
                            reward = (-1 * self.reward(f_xt,ytr)) #float('inf')
                            actregSamples = uniform_sampling(self.tf_dim*10, reg.input_space, self.tf_dim, self.rng)
                            mu, std = self._surrogate(model, actregSamples)
                            for i in range(len(actregSamples)):
                                f_xt = np.random.normal(mu[i],std[i],1)
                                smp_reward = self.reward(f_xt,ytr)
                                if reward > -1*smp_reward:
                                    reward = ((-1 * smp_reward))
                                    reg.addSample(actregSamples[i])
                        else:
                            pointToeval = a.simXtrain[-1] 
                            # print('a.simYtrain.shape[0]) == a.simXtrain.shape[0]: ',a.simYtrain.shape[0] , a.simXtrain.shape[0])
                            assert (a.simYtrain.shape[0]) == a.simXtrain.shape[0] - 1
                            # print('pointToeval: ',pointToeval)
                            mu, std = self._surrogate(model, np.array([pointToeval]))
                            f_xt = np.random.normal(mu,std,1)
                            reward = (-1 * self.reward(f_xt,ytr))
                            # aytr = np.hstack((aytr, f_xt))
    
                            # print('aid region b4 partition:', a.id, a.simReg.input_space)
                            # axtr = np.vstack((axtr, np.array([pointToeval])))
                            # a.simModel.fit(axtr, aytr)
                            a.simYtrain = np.hstack((a.simYtrain, f_xt))
                            # a.simXtrain = np.vstack((a.simXtrain, np.array([pointToeval])))
                            a.updatesimModel()
                            # print('axtr shape : ','horizon: ',h, a.simXtrain.shape,a.simYtrain.shape, np.array([pointToeval]).shape)

                        
                        # reg.reward.append( (-1 * self.reward(f_xt,ytr)))
                        # reg.rewardDist.append((-1 * self.reward(f_xt,ytr)))
                        # reg.rewardDist[ix] += (-1 * self.reward(f_xt,ytr))
                        reg.rewardDist[ix] += reward

                        # print('agent ytr :', ytr, reg.input_space)
                    
                    else:
                        smp = sample_from_discontinuous_region(configs['sampling']['num_inactive'], [reg], totalVolume, self.tf_dim, self.rng, volume=True ) #uniform_sampling(5, internal_inactive_subregion[0].input_space, self.tf_dim, self.rng)
                        mu, std = self._surrogate(model, smp)
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
                    
            # print()
            # print('Rollout reassignments in the tree directly ')
            # print()
            currentAgentIdx = self.num_agents - h
            # print_tree(rl_root, ROLLOUT)
            # print('b4 reassign rollout [i.simReg for i in agents]: ',[i.simReg.input_space for i in agents])
            # agents = reassign(rl_root, ROLLOUT, agents, currentAgentIdx, tmp_gpr, self.x_train, ytr)
            subregions = reassignUsingRewardDist(rl_root, ROLLOUT, agents)
            agents = partitionRegions(rl_root, subregions, ROLLOUT)
            # print('after reassign rollout [i.simReg for i in agents]: ',[i.simReg.input_space for i in agents])
            # export_tree_image(rl_root, ROLLOUT, f"results/trees/rollout/rlroot_after_{h}_reassign.png")
            # exportTreeUsingPlotly(rl_root)

            # print_tree(rl_root, ROLLOUT)
            assert len(agents) == self.num_agents

            xt = rl_root.find_leaves() 
            newx_opt = []
            for aidx, agent in enumerate(agents):
                assert agent.simReg.getStatus(ROLLOUT) == 1
                minytrval = float('inf')
                minytr = []
                for ia in agents:
                    if min(a.simYtrain) < minytrval:
                        minytrval = min(a.simYtrain)
                        minytr = a.simYtrain
                ytr = minytr
                # if reg.rolloutStatus == 1:
                # actregSamples = uniform_sampling(self.tf_dim*10, agent.simReg.input_space, self.tf_dim, self.rng)
                # mu, std = self._surrogate(agent.simModel, actregSamples)  #agent.simModel
                # actY = []
                # for i in range(len(actregSamples)):
                #     f_xt = np.random.normal(mu[i],std[i],1)
                #     actY.append(f_xt)
                # actY = np.hstack((actY))
                # print('agent.simXtrain agent.simYtrain ',agent.simXtrain, agent.simYtrain)
                # agent.simXtrain = np.vstack((agent.simXtrain , actregSamples))
                # agent.simYtrain = np.hstack((agent.simYtrain, actY))
                # agent.updatesimModel()
                # smp = sample_from_discontinuous_region(10*self.tf_dim, [reg], totalVolume, self.tf_dim, self.rng, volume=True ) #uniform_sampling(5, internal_inactive_subregion[0].input_space, self.tf_dim, self.rng)
                # print('a.simReg.rewardDist: ',self.mc, a.id, a.simReg.rewardDist[a.id])
                writetocsv(f'results/'+configs['testfunc']+f'/reghist/SimA_{agent.id}', [[self.sample, self.mc, agent.id, agent.simReg.input_space.tolist(), min(agent.simReg.rewardDist.tolist()), np.argmin(agent.simReg.rewardDist.tolist())]])
                next_xt = self._opt_acquisition(ytr, agent.simModel,agent.simReg.input_space,self.rng)
                agent.simXtrain = np.vstack((agent.simXtrain , next_xt))
                # print('aid region after partition:', agent.id, agent.simReg.input_space)
            #     next_xt = np.asarray([next_xt])
            #     mu, std = self._surrogate(agent.simModel, next_xt)
            #     f_xt = np.random.normal(mu,std,1)
            #     # # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            #     # # print('fxt :',f_xt)
            #     # # print(">>>>>>>>>>>>>>>>>>>>>>>   >>>>>>>>>>>>>>>>>>>    >>>>>>>>>>>>>>>>>>>>>>")
            #     globxtr = np.vstack((globxtr , next_xt))
            #     globytr = np.hstack((globytr, f_xt))
            # tmp_gpr.fit(globxtr, globytr)
                # logrolldf(agent.simXtrain, agent.simYtrain, aidx, h,20, self.mc)
            save_node(rl_root, f'/Users/shyamsundar/ASU/sem2/RA/partmahpc/partma/results/'+configs['testfunc']+f'/nodes/rl_root_{self.sample}_{currentAgentIdx}.pkl')
            
            h -= 1
            if h <= 0 :
                break
        # exportTreeUsingPlotly(rl_root, ROLLOUT)
    
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
        # print('b4 while agents_to_subregion',[(i.input_space,i.rolloutStatus, i.reward) for i in xt])
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
            minytrval = float('inf')
            minytr = []
            for ix, a in enumerate(agents):
                model = a.simModel
                for ia in agents:
                    if min(ia.simYtrain) < minytrval:
                        minytrval = min(ia.simYtrain)
                        minytr = ia.simYtrain
                ytr = minytr
                # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                # print(' agent sim y train ', ytr, min(ytr))
                # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                # exit(1)
                for reg in xt:
                    # actregSamples = uniform_sampling(self.tf_dim*10, reg.input_space, self.tf_dim, self.rng)
                    # mu, std = self._surrogate(model, actregSamples)
                    # actY = []
                    # for i in range(len(actregSamples)):
                    #     f_xt = np.random.normal(mu[i],std[i],1)
                    #     actY.append(f_xt)
                    # actY = np.hstack((actY))
                    # # # print('act Y ',actY)
                    # ixtr = np.vstack((agent.simXtrain , actregSamples))
                    # iytr = np.hstack((agent.simYtrain, actY))
                    # model = GPR(InternalGPR())
                    # model.fit(ixtr, iytr)
                    
                    if reg.rolloutStatus == 1:
                        if a.simReg != reg:
                            model = self.gprFromregionPairs(a, reg.agent)

                        next_xt = self._opt_acquisition(ytr,model,reg.input_space,self.rng)  # reg.agent.model
                        next_xt = np.asarray([next_xt])
                        mu, std = self._surrogate(model, next_xt)
                        f_xt = np.random.normal(mu,std,1)
                        reward = (-1 * self.reward(f_xt,ytr)) #float('inf')
                        actregSamples = uniform_sampling(self.tf_dim*10, reg.input_space, self.tf_dim, self.rng)
                        mu, std = self._surrogate(model, actregSamples)
                        for i in range(len(actregSamples)):
                            f_xt = np.random.normal(mu[i],std[i],1)
                            smp_reward = self.reward(f_xt,ytr)
                            if reward > -1*smp_reward:
                                reward = ((-1 * smp_reward))
                                reg.addSample(actregSamples[i])
                        
                        # reg.reward.append( (-1 * self.reward(f_xt,ytr)))
                        # reg.rewardDist.append((-1 * self.reward(f_xt,ytr)))
                        # reg.rewardDist[ix] += (-1 * self.reward(f_xt,ytr))
                        reg.rewardDist[ix] += reward

                        # print('agent ytr :', ytr, reg.input_space)
                    
                    else:
                        smp = sample_from_discontinuous_region(configs['sampling']['num_inactive'], [reg], totalVolume, self.tf_dim, self.rng, volume=True ) #uniform_sampling(5, internal_inactive_subregion[0].input_space, self.tf_dim, self.rng)
                        mu, std = self._surrogate(model, smp)
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
                    
            # print()
            # print('Rollout reassignments in the tree directly ')
            # print()
            currentAgentIdx = self.num_agents - h
            # print_tree(rl_root, ROLLOUT)
            # print('b4 reassign rollout [i.simReg for i in agents]: ',[i.simReg.input_space for i in agents])
            # agents = reassign(rl_root, ROLLOUT, agents, currentAgentIdx, tmp_gpr, self.x_train, ytr)
            subregions = reassignUsingRewardDist(rl_root, ROLLOUT, agents)
            agents = partitionRegions(rl_root, subregions, ROLLOUT)
            # print('after reassign rollout [i.simReg for i in agents]: ',[i.simReg.input_space for i in agents])
            # export_tree_image(rl_root, ROLLOUT, f"results/trees/rollout/rlroot_after_{h}_reassign.png")
            # exportTreeUsingPlotly(rl_root)

            # print_tree(rl_root, ROLLOUT)
            assert len(agents) == self.num_agents

            xt = rl_root.find_leaves() 
            newx_opt = []
            for aidx, agent in enumerate(agents):
                assert agent.simReg.getStatus(ROLLOUT) == 1
                # if reg.rolloutStatus == 1:
                actregSamples = uniform_sampling(self.tf_dim*10, agent.simReg.input_space, self.tf_dim, self.rng)
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
                # smp = sample_from_discontinuous_region(10*self.tf_dim, [reg], totalVolume, self.tf_dim, self.rng, volume=True ) #uniform_sampling(5, internal_inactive_subregion[0].input_space, self.tf_dim, self.rng)
                # print('a.simReg.rewardDist: ',self.mc, a.id, a.simReg.rewardDist[a.id])
                writetocsv(f'results/'+configs['testfunc']+f'/reghist/SimA_{agent.id}', [[self.sample, self.mc, agent.id, agent.simReg.input_space.tolist(), min(agent.simReg.rewardDist.tolist()), np.argmin(agent.simReg.rewardDist.tolist())]])
            #     next_xt = self._opt_acquisition(globytr, agent.simModel,agent.simReg.input_space,self.rng)
            #     next_xt = np.asarray([next_xt])
            #     mu, std = self._surrogate(agent.simModel, next_xt)
            #     f_xt = np.random.normal(mu,std,1)
            #     # # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            #     # # print('fxt :',f_xt)
            #     # # print(">>>>>>>>>>>>>>>>>>>>>>>   >>>>>>>>>>>>>>>>>>>    >>>>>>>>>>>>>>>>>>>>>>")
            #     globxtr = np.vstack((globxtr , next_xt))
            #     globytr = np.hstack((globytr, f_xt))
            # tmp_gpr.fit(globxtr, globytr)
                # logrolldf(agent.simXtrain, agent.simYtrain, aidx, h,20, self.mc)
            save_node(rl_root, f'/Users/shyamsundar/ASU/sem2/RA/partmahpc/partma/results/'+configs['testfunc']+f'/nodes/rl_root_{self.sample}_{currentAgentIdx}.pkl')
            
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
        numObs = agent1.y_train.shape[0]
        xtr = np.vstack((agent1.x_train[:numObs] , agent2.x_train[:numObs]))
        ytr = np.hstack((agent1.y_train, agent2.y_train))
        model = GPR(InternalGPR())
        model.fit(xtr, ytr)
        return model