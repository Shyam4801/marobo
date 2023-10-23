
from typing import Any
from .constants import MAIN
from ..gprInterface import GPR, InternalGPR

class Agent():
    def __init__(self, model, x_train, y_train, region_support) -> None:
        self.model = model
        self.point_history = []
        self.x_train = x_train
        self.y_train = y_train
        self.region_support = region_support
        self.simReg = region_support

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

    def update_model(self, model):
        self.model = model

    def updateBounds(self, region_support, routine):
        if routine == MAIN:
            self.region_support = region_support
        else:
            self.simReg = region_support

