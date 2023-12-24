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

with open('config.yml', 'r') as file:
    configs = yaml.safe_load(file)

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
        
        agents = []
        globalXtrain = np.empty((1,tf_dim))
        globalYtrain = np.empty((1))

        init_sampling_type = "uniform_sampling"
        init_budget = configs['sampling']['initBudget'] * tf_dim#x_train.shape[0]

        for id, l in enumerate(agents_to_subregion):
            l.setRoutine(MAIN)
            if l.getStatus(MAIN) == 1:
                xtr, ytr = self.initAgents(l.input_space, init_sampling_type, int(init_budget/num_agents), tf_dim, rng)
                print(f'agent xtr ', xtr, ytr)
                globalXtrain = np.vstack((globalXtrain, xtr))
                globalYtrain = np.hstack((globalYtrain, ytr))

        # for id, l in enumerate(agents_to_subregion):
        #     l.setRoutine(MAIN)
        #     if l.getStatus(MAIN) == 1:
                ag = Agent(id, None, xtr, ytr, l)
                ag.updateModel()
                ag(MAIN)
                agents.append(ag)
                l.addAgentList(ag, MAIN)
                # print('new subr : ', l.input_space, 'agent : ', l.agent.id)
        
        globalXtrain = globalXtrain[1:]
        globalYtrain = globalYtrain[1:]
        print('globalXtrain, globalYtrain :', min(globalYtrain))
        for sample in tqdm(range(num_samples)):
            print('_____________________________________', sample)
            # print(f"INPUT SPACE : {GREEN}{self.region_support}{END}")
            print('_____________________________________')
            print('global dataset : ', globalXtrain.shape, globalYtrain.shape)
            print('_____________________________________')
            model = GPR(gpr_model)
            model.fit(globalXtrain, globalYtrain)
            self.ei_roll = RolloutEI()

            pred_sample_x, X_root, agents = self.ei_roll.sample(sample, X_root, agents, num_agents, self.tf, globalXtrain, self.horizon, globalYtrain, region_support, model, rng) #self._opt_acquisition(agent.y_train, agent.model, agent.region_support, rng) 
            pred_sample_y, falsified = compute_robustness(pred_sample_x, test_function, behavior, agent_sample=True)
            print('pred_sample_x, pred_sample_y: ', pred_sample_x, pred_sample_y)
            globalXtrain = np.vstack((globalXtrain, pred_sample_x))
            globalYtrain = np.hstack((globalYtrain, (pred_sample_y)))
            print('min obs so far : ', pred_sample_x[np.argmin(pred_sample_y),:], np.min(pred_sample_y))
            # print('np.asarray([pred_sample_x[i]]).shape : ', np.asarray([pred_sample_x[0]]).shape)
            assignments=[]
            assignmentsDict=[]
            status = []
            for i,a in enumerate(agents):
                # x_opt = self.ei_roll._opt_acquisition(globalYtrain, model, a.region_support.input_space, rng) 
                # pred_xopt, falsified = compute_robustness(np.asarray([x_opt]), test_function, behavior, agent_sample=True)
                # actregSamples = uniform_sampling(tf_dim*10, a.region_support.input_space, tf_dim, rng)
                # pred_act, falsified = compute_robustness(actregSamples, test_function, behavior, agent_sample=True)
                # # mu, std = self._surrogate(a.model, actregSamples)
                # # actY = []
                # # for i in range(len(actregSamples)):
                # #     f_xt = np.random.normal(mu[i],std[i],1)
                # #     actY.append(f_xt)
                # # actY = np.hstack((actY))
                # # # # print('act Y ',actY)
                # # a.x_train = actregSamples #np.vstack((agent.simXtrain , actregSamples))
                # # a.y_train = actY #np.hstack((agent.simYtrain, actY))
                # # a.updateModel()
                # # print(f'b4 appendign agent {i} xtrain :', a.x_train)
                # a.x_train = np.vstack((a.x_train, np.asarray(x_opt)))
                # a.y_train = np.hstack((a.y_train, pred_xopt))

                a.x_train = np.vstack((a.x_train, np.asarray([pred_sample_x[i]])))
                a.y_train = np.hstack((a.y_train, pred_sample_y[i]))
                # a.model = a.simModel
                a.updateModel()
                # globalXtrain = np.vstack((globalXtrain, np.asarray(x_opt)))
                # globalYtrain = np.hstack((globalYtrain, pred_xopt))
                # x_opt = self.ei_roll._opt_acquisition(a.y_train, a.model, a.region_support.input_space, rng) 
                # pred_xopt, falsified = compute_robustness(np.asarray([x_opt]), test_function, behavior, agent_sample=False)
                # a.x_train = np.vstack((a.x_train, np.asarray(x_opt)))
                # a.y_train = np.hstack((a.y_train, pred_xopt))
                # print('a.id: ', a.id)
                # print('agent rewards after main :', a.region_support.input_space , a.region_support.avgRewardDist, 'print(a.getStatus(MAIN)): ',(a.region_support.getStatus(MAIN)))
                
                a.resetRegions()
                # a(MAIN)
                # a.resetAgentList(MAIN)
                if a.region_support.getStatus(MAIN) == 0:
                    a.region_support.agentList = []
                a.region_support.resetavgRewardDist(num_agents)
                # writetocsv(f'results/reghist/a{i}_{sample}',a.regHist)
                maintrace.append(a.simregHist)
                print(f'a.simregHist: {sample}',a.simregHist)
                
                # a.resetModel()
                # print('agent rewards after reset :', a.region_support.input_space , a.region_support.avgRewardDist)
                assignments.append(a.region_support)
                assignmentsDict.append(a.region_support.input_space)
                status.append(a.region_support.mainStatus)

            # print('pred_sample_x, pred_sample_y: ', globalXtrain[-num_agents:], globalYtrain[-num_agents:])
            # print('min obs so far : ', globalXtrain[-num_agents:][np.argmin(globalYtrain[-num_agents:]),:], np.min(globalYtrain[-num_agents:]))
            self.assignments.append(assignments)
            self.assignmentsDict.append(assignmentsDict)
            pred_sample_x = globalXtrain[-num_agents:]
            # print('shape : ',len(self.assignments))
            # print('final agent regions in rolloutBO', [i[j].input_space for i in self.assignments for j in range(4)])
            self.status.append(status)

            inactive_subregion_samples = []   
            self.agent_point_hist.extend(pred_sample_x)
            # writetocsv(f'results/reghist/aSim{i}_{sample}',a.simregHist)

        plot_dict = {'agents':self.agent_point_hist,'assignments' : self.assignments, 'status': status, 'region_support':region_support, 'test_function' : test_function, 'inactive_subregion_samples' : inactive_subregion_samples, 'sample': num_samples}
        save_plot_dict = {'agents':self.agent_point_hist,'assignments' : self.assignmentsDict, 'status': status, 'region_support':region_support, 'test_function' : [], 'inactive_subregion_samples' : inactive_subregion_samples, 'sample': num_samples}
        
        # print(plot_dict)
        save_node(save_plot_dict, '/Users/shyamsundar/ASU/sem2/RA/partmahpc/partma/results/'+configs['testfunc']+f'/plot_dict.pkl')
        return falsified, self.region_support, plot_dict


    def initAgents(self, region_support, init_sampling_type, init_budget, tf_dim, rng):
        if init_sampling_type == "lhs_sampling":
            x_train = lhs_sampling(init_budget, region_support, tf_dim, rng)
        elif init_sampling_type == "uniform_sampling":
            x_train = uniform_sampling(init_budget, region_support, tf_dim, rng)
        else:
            raise ValueError(f"{init_sampling_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")
        
        y_train, falsified = compute_robustness(x_train, self.tf, self.behavior)
        if not falsified:
            print("No falsification in Initial Samples. Performing BO now")

        return x_train, y_train