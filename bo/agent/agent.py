
from typing import Any
from .constants import MAIN
from ..gprInterface import GPR, InternalGPR
from ..sampling import lhs_sampling, uniform_sampling
from ..utils import compute_robustness

class Agent():
    def __init__(self, model, x_train, y_train, region_support) -> None:
        self.model = model
        self.point_history = []
        self.x_train = x_train
        self.y_train = y_train
        self.region_support = region_support
        self.simReg = region_support

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
            print('active region gets assigned the agent using self')
        else:
            region.agent = None 

    def updateModel(self):
        self.model = GPR(InternalGPR())
        self.model.fit(self.x_train, self.y_train)

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
        else:
            self.simReg = region_support

