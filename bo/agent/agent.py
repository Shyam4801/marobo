
from typing import Any
from .constants import MAIN
from ..gprInterface import GPR, InternalGPR
from ..sampling import lhs_sampling, uniform_sampling
from ..utils import compute_robustness
from copy import deepcopy

class Agent():
    def __init__(self, id, model, x_train, y_train, region_support) -> None:
        self.id = id
        self.model = model
        self.simModel = model
        self.point_history = []
        self.x_train = deepcopy(x_train)
        self.simXtrain = deepcopy(x_train)
        self.y_train = deepcopy(y_train)
        self.simYtrain = deepcopy(y_train)
        self.ActualXtrain = deepcopy(x_train)
        self.ActualYtrain = deepcopy(y_train)
        self.region_support = region_support
        self.simReg = region_support
        self.simregHist = [region_support.input_space]
        self.regHist = [region_support.input_space]
        self.evalRewards = []

    def initAgent(self, init_sampling_type, init_budget, tf_dim, rng):
        if init_sampling_type == "lhs_sampling":
            x_train = lhs_sampling(init_budget, self.region_support, tf_dim, rng)
        elif init_sampling_type == "uniform_sampling":
            x_train = uniform_sampling(init_budget, self.region_support, tf_dim, rng)
        else:
            raise ValueError(f"{init_sampling_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")
        
        y_train, falsified = compute_robustness(x_train, self.tf_wrapper, self.behavior)

        if not falsified:
            print("No falsification in Initial Samples. Performing BO now")


    def __call__(self, routine):
        if routine == MAIN:
            region = self.region_support
        else:
            region = self.simReg
        if region.getStatus(routine) == 1:
            region.agent = self
            # region.addAgentList(self, routine)
            # print('active region gets assigned the agent using self')
        else:
            region.agent = None 

    def updateObs(self, parent, routine):
        if routine == MAIN:
            self.x_train = deepcopy(parent.agent.x_train)
            self.y_train = deepcopy(parent.agent.y_train)
            self.model = deepcopy(parent.agent.model)
        else:
            self.simXtrain = deepcopy(parent.agent.simXtrain)
            self.simYtrain = deepcopy(parent.agent.simYtrain)
            self.simModel = deepcopy(parent.agent.simModel)

    def testupdateObsFromRegion(self, parent, routine):
        if routine == MAIN:
            xtr, ytr , model = parent.mainPrior.getData(routine)
            self.x_train = deepcopy(xtr)
            self.y_train = deepcopy(ytr)
            self.model = deepcopy(model)
        else:
            xtr, ytr , model = parent.rolloutPrior.getData(routine)
            self.simXtrain = deepcopy(xtr)
            self.simYtrain = deepcopy(ytr)
            self.simModel = deepcopy(model)

    def updateObsFromRegion(self, parent, routine):
        if routine == MAIN:
            self.x_train = deepcopy(parent.xtr)
            self.y_train = deepcopy(parent.ytr)
            self.model = deepcopy(parent.model)
        else:
            self.simXtrain = deepcopy(parent.xtr)
            self.simYtrain = deepcopy(parent.ytr)
            self.simModel = deepcopy(parent.model) 


    def appendevalReward(self, reward):
        self.evalRewards.append(reward)

    def resetAgentList(self, routine):
        if routine == MAIN:
            region = self.region_support
        else:
            region = self.simReg
        if region.getStatus(routine) == 1:
            region.addAgentList(self, routine)
            # print('Agent list reset after MC iter')
        else:
            region.agentList = []

    def updatesimModel(self):
        self.simModel = GPR(InternalGPR())
        self.simModel.fit(self.simXtrain, self.simYtrain)

    def updateModel(self):
        self.model = GPR(InternalGPR())
        self.model.fit(self.x_train, self.y_train)
        self.simModel = deepcopy(self.model)

    def resetModel(self):
        self.simModel = deepcopy(self.model)
        self.simXtrain = deepcopy(self.x_train)
        self.simYtrain = deepcopy(self.y_train)

    def resetActual(self):
        self.x_train = deepcopy(self.ActualXtrain)
        self.y_train = deepcopy(self.ActualYtrain)

    def resetRegions(self):
        self.simReg = self.region_support

    def getRegion(self, routine):
        if routine == MAIN:
            return self.region_support
        else:
            return self.simReg

    def add_point(self, point):
        self.point_history.append(point)

    # def update_model(self, model):
    #     self.model = model

    def updateBounds(self, region_support, routine):
        if routine == MAIN:
            self.region_support = region_support
            self.regHist.append(region_support.input_space)
        else:
            self.simReg = region_support
            self.simregHist.append(region_support.input_space)
            # print('self.simregHist: ',self.simregHist)

