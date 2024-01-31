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
from ..agent.agent import Agent
from ..agent.constants import *
from ..utils.logger import logtime, LOGPATH
from ..utils.savestuff import *
import random
import os

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

class RolloutBO(BO_Interface):
    def __init__(self):
        pass
    
    # @logtime(LOGPATH)
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
        # init_sampling_type,
        # init_budget,
        behavior:Behavior = Behavior.MINIMIZATION
    ) -> Tuple[NDArray]:

        """Internal BO Model

        Args:
            test_function: Function of System Under Test.
            num_samples: Number of samples to generate from BO.
            x_train: Samples from Training set.
            y_train: Evaluated values of samples from Trainig set.
            region_support: Min and Max of all dimensions
            gpr_model: Gaussian Process Regressor Model developed using Factory
            rng: RNG object from numpy

        Raises:
            TypeError: If x_train is not 2 dimensional numpy array or does not match dimensions
            TypeError: If y_train is not (n,) numpy array
            TypeError: If there is a mismatch between x_train and y_train

        Returns:
            x_complete
            y_complete
            x_new
            y_new
        """
        self.horizon = 4
        falsified = False
        self.tf = test_function
        self.agent_point_hist = []
        # Domain reduces after each BO iteration 
        self.region_support = region_support
        self.assignments = []
        self.assignmentsDict = []
        self.status = []
        self.behavior = behavior
        maintrace = []

        tf_dim = region_support.shape[0]
        X_root = Node(self.region_support, 1)
        factorized = find_prime_factors(num_agents) #sorted(find_close_factor_pairs(num_agents), reverse=True)
        agents_to_subregion = get_subregion(deepcopy(X_root), num_agents, factorized, np.random.randint(tf_dim))
        X_root.add_child(agents_to_subregion)
        # assignments = {value: 1 for value in agents_to_subregion} 
        testv=0
        for i in agents_to_subregion:
            testv += i.getVolume()

        assert X_root.getVolume() == testv
        # print('set([i.getVolume() for i in agents_to_subregion]) :', set([i.getVolume() for i in agents_to_subregion]))
        
        # agents = []
        mainAgents = []
        globalXtrain = np.empty((1,tf_dim))
        globalYtrain = np.empty((1))

        init_sampling_type = "uniform_sampling"
        init_budget = configs['sampling']['initBudget'] * tf_dim#x_train.shape[0]

        for id, l in enumerate(agents_to_subregion):
            l.setRoutine(MAIN)
            if l.getStatus(MAIN) == 1:
                xtr, ytr = self.initAgents(l.input_space, init_sampling_type, int(init_budget/num_agents), tf_dim, rng, store=False)
                # print(f'agent xtr ', xtr, ytr)
                globalXtrain = np.vstack((globalXtrain, xtr))
                globalYtrain = np.hstack((globalYtrain, ytr))

        # for id, l in enumerate(agents_to_subregion):
        #     l.setRoutine(MAIN)
        #     if l.getStatus(MAIN) == 1:
                mainag = Agent(id, None, xtr, ytr, l)
                mainag.updateModel()
                mainag(MAIN)
                mainAgents.append(mainag)
                l.addAgentList(mainag, MAIN)
                # print('new subr : ', l.input_space, 'agent : ', l.agent.id)
        
        globalXtrain = globalXtrain[1:]
        globalYtrain = globalYtrain[1:]
        if not os.path.exists(f'results/'+configs['testfunc']):
            os.makedirs(f'results/'+configs['testfunc'])
        writetocsv(f'results/'+configs['testfunc']+'/initSmp',[[globalXtrain[:,0], globalXtrain[:,1],  globalYtrain]])

        agentAftereachSample = []
        for sample in tqdm(range(num_samples)):
            print('globalXtrain, globalYtrain :', min(globalYtrain))
            # for sample in tqdm(range(num_samples)):
            print('_____________________________________', sample)
            # print(f"INPUT SPACE : {GREEN}{self.region_support}{END}")
            print('_____________________________________')
            print('global dataset : ', globalXtrain.shape, globalYtrain.shape)
            print('_____________________________________')
            model = GPR(gpr_model)
            model.fit(globalXtrain, globalYtrain)
            
            avgrewards = np.zeros((1,num_agents,num_agents))
            agentModels = []
            for regsamples in range(1):
                Xs_root = X_root #deepcopy(X_root) #Node(self.region_support, 1)
                Xs_root.model = deepcopy(model)
                # print_tree(X_root, MAIN)
                # print('-'*100)
                # print_tree(Xs_root, MAIN)
                # assert X_root != Xs_root
                # factorized = find_prime_factors(num_agents) #sorted(find_close_factor_pairs(num_agents), reverse=True)
                # internal_agents_to_subregion = get_subregion(deepcopy(Xs_root), num_agents, factorized, np.random.randint(tf_dim))
                # Xs_root.add_child(internal_agents_to_subregion)
                agents = []
                # internal_agents_to_subregion = deepcopy(agents_to_subregion)
                for id, l in enumerate(Xs_root.find_leaves()):
                    l.setRoutine(MAIN)
                    if l.getStatus(MAIN) == 1:
                        # xtr, ytr = self.initAgents(l.input_space, init_sampling_type, int(init_budget/num_agents), tf_dim, rng, store=True)
                        print(f'agent xtr ', l.agent.x_train, l.agent.y_train, l.agent.id, l.input_space)
                        # if sample == 0:
                        #     xtr = globalXtrain
                        #     ytr = globalYtrain
                        # else:
                        #     xtr = 
                        # print(f'agent xtr ytr sample {regsamples}', xtr, ytr)
                        # globalXtrain = np.vstack((globalXtrain, xtr))
                        # globalYtrain = np.hstack((globalYtrain, ytr))

                # for id, l in enumerate(agents_to_subregion):
                #     l.setRoutine(MAIN)
                #     if l.getStatus(MAIN) == 1:
                        ag = l.agent
                        # ag.x_train = xtr
                        # ag.y_train = ytr
                        # ag = Agent(id, None, xtr, ytr, l)
                        ag.updateModel()
                        ag(MAIN)
                        agents.append(ag)

                        # add footprint 
                        l.addFootprint(ag.x_train, ag.y_train, ag.model)
                        # l.addAgentList(ag, MAIN)
                        # print('new subr : ', l.input_space, 'agent : ', l.agent.id)
                
                # globalXtrain = globalXtrain[1:]
                # globalYtrain = globalYtrain[1:]
                # print('globalXtrain, globalYtrain :', min(globalYtrain))
                # for sample in tqdm(range(num_samples)):
                # print('_____________________________________', sample)
                # # print(f"INPUT SPACE : {GREEN}{self.region_support}{END}")
                # print('_____________________________________')
                # print('global dataset : ', globalXtrain.shape, globalYtrain.shape)
                print('_____________________________________')
                # model = GPR(gpr_model)
                # model.fit(globalXtrain, globalYtrain)
                self.ei_roll = RolloutEI()

                _, _, agents = self.ei_roll.sample(sample, Xs_root, agents, num_agents, self.tf, globalXtrain, self.horizon, globalYtrain, region_support, model, rng) #self._opt_acquisition(agent.y_train, agent.model, agent.region_support, rng) 
                print(f'End of agents reg sample {regsamples}')
                print('agent regions: ',[i.region_support.input_space for i in agents])
                
                # for smp in range(samples):
                    # self.smp = smp
                avgAgentrewards = np.zeros((1,num_agents))
                # self.root = self._evaluate_at_point_list(agents)
                agents = sorted(agents, key=lambda x: x.id)
                for a in agents:
                    print('aid inside reward acc: ', a.id)
                    # avgAgentrewards.append(a.region_support.avgRewardDist)
                    avgAgentrewards = np.vstack((avgAgentrewards, a.region_support.avgRewardDist.reshape((1,num_agents))))
                    # print(avgAgentrewards.shape)
                    # a.appendevalReward(avgAgentrewards[a.id][a.id])
                    # print('a.pointsToeval[:smp]: ',a.pointsToeval[:smp+1], a.pointsToeval[:smp+1].shape )
                    # a.x_train = np.vstack((a.x_train, a.pointsToeval[:smp+1]))
                    # a.y_train = np.hstack((a.y_train, np.array(a.evalRewards[:smp+1])))
                    # a.updateModel()
                    # a.resetModel()
                    # print('aid xtrain ,ytrain: ',a.id, a.x_train, a.y_train)
                    # print('a.pointsToeval[:smp+1]: ',a.id , a.pointsToeval[:smp+1], a.evalRewards[:smp+1])

                # print(avgAgentrewards, avgAgentrewards.shape)
                avgrewards = np.vstack((avgrewards, avgAgentrewards[1:].reshape((1,num_agents,num_agents))))
                agentModels.append(agents)

                # X_root = deepcopy(Xs_root)
                
            avgrewards = avgrewards[1:]
            print('avgrewards: ',avgrewards)
            # avgrewards = np.hstack((avgrewards))
            minrewardDist, minrewardDistIdx = min_diagonal(avgrewards)
            print('minrewardDist, minrewardDistIdx: ',minrewardDist, minrewardDistIdx, minrewardDist.shape, minrewardDistIdx.shape)
            minregtojump = np.argmin(minrewardDist[:len(agents)], axis=0)
            print('minregtojump: ',minregtojump)
            # exit(1)
            # print('b4 reassign MAIN [i.region_support for i in agents]: ',[i.region_support.input_space for i in agents])
            # agents = reassign(root, MAIN, agents, currentAgentIdx, gpr_model, xtr, ytr)
            mainAgents = sorted(mainAgents, key=lambda x: x.id)

            agentsWithminSmps = agents #self.get_nextXY(agentModels, minrewardDistIdx)
            print('agentsWithminSmps: ',[(i.id, i.region_support.input_space )for i in agentsWithminSmps], len(agentsWithminSmps))
            agentsWithminSmps = sorted(agentsWithminSmps, key=lambda x: x.id)

            for ix, a in enumerate(agentsWithminSmps):
                # print('aid inside agentsWithminSmps assign: ', a.id)
                # print('b4 a.id, a.avgRewardDist: ',a.id, a.region_support.avgRewardDist)
                a.region_support.avgRewardDist = minrewardDist[ix]
                # print('a.id, a.avgRewardDist: ',a.id, a.region_support.avgRewardDist)
                
            print('b4 reward dist : ', [i.region_support.input_space for i in agentsWithminSmps])
            jump = random.random()
            subregions = reassignUsingRewardDist( X_root, MAIN, agentsWithminSmps, jump_prob=jump)
            agentsWithminSmps = partitionRegions(X_root, subregions, MAIN)
            print('after reward dist : ', [i.region_support.input_space for i in agentsWithminSmps])
            print('agentsWithminSmps after partition : ',[(i.id, i.region_support.input_space )for i in agentsWithminSmps], len(agentsWithminSmps))
            x_opt_from_all = []

            agentsWithminSmps = splitObs(agentsWithminSmps, tf_dim, rng, MAIN, self.tf, self.behavior)
            for a in agentsWithminSmps:
                print('obs after splitting :', a.x_train, a.y_train , a.id)

            minytrval = float('inf')
            minytr = []
            for ix, a in enumerate(agentsWithminSmps):
                model = a.model
                for ia in agents:
                    if min(ia.y_train) < minytrval:
                        minytrval = min(ia.y_train)
                        minytr = ia.y_train
            ytr = minytr
            agentminytr = float('inf')
            agentminxtr = None
            assert len(agentsWithminSmps) == num_agents
            for a in agentsWithminSmps:
                assert check_points(a, MAIN) == True
                x_opt = self.ei_roll._opt_acquisition(a.y_train, a.model, a.region_support.input_space, rng)
                x_opt_from_all.append(x_opt)
                # check_pred_y, falsified = compute_robustness(x_opt, test_function, behavior, agent_sample=True)
                # idx = int(init_budget/num_agents)
                # if np.min(a.y_train[idx:]) < check_pred_y:
                #     x_opt_from_all.append(a.x_train[idx:][np.argmin(a.y_train[idx:]),:])
                # else:
                #     x_opt_from_all.append(x_opt)
                    # agentminytr = np.min(a.y_train)
                    # agentminxtr = a.x_train[np.argmin(a.y_train),:]

            subx = np.hstack((x_opt_from_all)).reshape((num_agents,tf_dim))
            pred_sample_x = subx[:num_agents]

            
            pred_sample_y, falsified = compute_robustness(pred_sample_x, test_function, behavior, agent_sample=False)
            # if pred_sample_y > agentminytr:


            print('pred_sample_x, pred_sample_y: ', pred_sample_x, pred_sample_y)
            globalXtrain = np.vstack((globalXtrain, pred_sample_x))
            globalYtrain = np.hstack((globalYtrain, (pred_sample_y)))
            print('min obs so far : ', pred_sample_x[np.argmin(pred_sample_y),:], np.min(pred_sample_y))
            # print('np.asarray([pred_sample_x[i]]).shape : ', np.asarray([pred_sample_x[0]]).shape)
            assignments=[]
            assignmentsDict=[]
            status = []

            # finalAgentsmodel = GPR(gpr_model)
            # finalAgentsmodel.fit(pred_sample_x, pred_sample_y)
            # local_xopt = self.ei_roll._opt_acquisition(ytr, finalAgentsmodel, a.region_support.input_space, rng)
            # local_yopt, falsified = compute_robustness(local_xopt, test_function, behavior, agent_sample=False)
            # print('local_xopt: ',local_xopt, local_yopt)
            
            for i,a in enumerate(agentsWithminSmps):
            #     # x_opt = self.ei_roll._opt_acquisition(globalYtrain, model, a.region_support.input_space, rng) 
            #     # pred_xopt, falsified = compute_robustness(np.asarray([x_opt]), test_function, behavior, agent_sample=True)
            #     # actregSamples = uniform_sampling(tf_dim*10, a.region_support.input_space, tf_dim, rng)
            #     # pred_act, falsified = compute_robustness(actregSamples, test_function, behavior, agent_sample=True)
            #     # # mu, std = self._surrogate(a.model, actregSamples)
            #     # # actY = []
            #     # # for i in range(len(actregSamples)):
            #     # #     f_xt = np.random.normal(mu[i],std[i],1)
            #     # #     actY.append(f_xt)
            #     # # actY = np.hstack((actY))
            #     # # # # print('act Y ',actY)
            #     # # a.x_train = actregSamples #np.vstack((agent.simXtrain , actregSamples))
            #     # # a.y_train = actY #np.hstack((agent.simYtrain, actY))
            #     # # a.updateModel()
            #     # # print(f'b4 appendign agent {i} xtrain :', a.x_train)
            #     # a.x_train = np.vstack((a.x_train, np.asarray(x_opt)))
            #     # a.y_train = np.hstack((a.y_train, pred_xopt))

                a.x_train = np.vstack((a.x_train, np.asarray([pred_sample_x[i]])))
                a.y_train = np.hstack((a.y_train, pred_sample_y[i]))
            #     # a.model = a.simModel
                # a.updateModel()
            #     # globalXtrain = np.vstack((globalXtrain, np.asarray(x_opt)))
            #     # globalYtrain = np.hstack((globalYtrain, pred_xopt))
            #     # x_opt = self.ei_roll._opt_acquisition(a.y_train, a.model, a.region_support.input_space, rng) 
            #     # pred_xopt, falsified = compute_robustness(np.asarray([x_opt]), test_function, behavior, agent_sample=False)
            #     # a.x_train = np.vstack((a.x_train, np.asarray(x_opt)))
            #     # a.y_train = np.hstack((a.y_train, pred_xopt))
            #     # print('a.id: ', a.id)
            #     # print('agent rewards after main :', a.region_support.input_space , a.region_support.avgRewardDist, 'print(a.getStatus(MAIN)): ',(a.region_support.getStatus(MAIN)))
                
                a.resetRegions()
            #     # a(MAIN)
                # a.resetAgentList(MAIN)
                if a.region_support.getStatus(MAIN) == 0:
                    a.region_support.agentList = []
                a.region_support.resetavgRewardDist(num_agents)
            #     # writetocsv(f'results/reghist/a{i}_{sample}',a.regHist)
            #     maintrace.append(a.simregHist)
            #     print(f'a.simregHist: {sample}',a.simregHist)
                
                a.resetModel()
            #     # print('agent rewards after reset :', a.region_support.input_space , a.region_support.avgRewardDist)
                assignments.append(a.region_support)
                assignmentsDict.append(a.region_support.input_space)
                status.append(a.region_support.mainStatus)

        #     # print('pred_sample_x, pred_sample_y: ', globalXtrain[-num_agents:], globalYtrain[-num_agents:])
        #     # print('min obs so far : ', globalXtrain[-num_agents:][np.argmin(globalYtrain[-num_agents:]),:], np.min(globalYtrain[-num_agents:]))
            self.assignments.append(assignments)
            self.assignmentsDict.append(assignmentsDict)
            agentAftereachSample.append([(i.x_train, i.y_train) for i in agents])
        #     pred_sample_x = globalXtrain[-num_agents:]
        #     # print('shape : ',len(self.assignments))
        #     # print('final agent regions in rolloutBO', [i[j].input_space for i in self.assignments for j in range(4)])
            self.status.append(status)

            inactive_subregion_samples = []   
            self.agent_point_hist.extend(pred_sample_x)
        #     # writetocsv(f'results/reghist/aSim{i}_{sample}',a.simregHist)

        # plot_dict = {'agents':self.agent_point_hist,'assignments' : self.assignments, 'status': status, 'region_support':region_support, 'test_function' : test_function, 'inactive_subregion_samples' : inactive_subregion_samples, 'sample': num_samples}
        save_plot_dict = {'agentmodels':agentAftereachSample, 'agents':self.agent_point_hist,'assignments' : self.assignmentsDict, 'status': status, 'region_support':region_support, 'test_function' : [], 'inactive_subregion_samples' : inactive_subregion_samples, 'sample': num_samples}
        
        # # print(plot_dict)
        save_node(save_plot_dict, '/Users/shyamsundar/ASU/sem2/RA/partmahpc/partma/results/'+configs['testfunc']+f'/plot_dict.pkl')
        return falsified, self.region_support , None #plot_dict


    def initAgents(self, region_support, init_sampling_type, init_budget, tf_dim, rng, store):
        if init_sampling_type == "lhs_sampling":
            x_train = lhs_sampling(init_budget, region_support, tf_dim, rng)
        elif init_sampling_type == "uniform_sampling":
            x_train = uniform_sampling(init_budget, region_support, tf_dim, rng)
        else:
            raise ValueError(f"{init_sampling_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")
        
        y_train, falsified = compute_robustness(x_train, self.tf, self.behavior, agent_sample=store)
        if not falsified:
            print("No falsification in Initial Samples. Performing BO now")

        return x_train, y_train
    

    def get_nextXY(self, agentmodels, minRewardIdx): #, rng, test_function, behavior):
        agents = []
        for i, minidx in enumerate(minRewardIdx):
            agents.append(agentmodels[minidx][i])

        return agents

        # for a in agents:
        #     x_opt = self.ei_roll._opt_acquisition(a.y_train, a.model, a.region_support.input_space, rng)
        #     pred_sample_y, falsified = compute_robustness(x_opt, test_function, behavior, agent_sample=False)
        

    
