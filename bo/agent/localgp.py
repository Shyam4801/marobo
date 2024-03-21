from typing import Any
import numpy as np
from .constants import *
from dataclasses import dataclass
from bo.agent.observation import Observations
from ..gprInterface import GPR, InternalGPR

@dataclass
class Prior:
    dataset: Observations
    # indices: list
    region: np.ndarray
    # model: Any
    
    def __post_init__(self):
        assert isinstance(self.dataset, Observations), "dataset must be an instance of Observations class"

    def buildModel(self):
        indices = self.getSubset(self.region)
        assert self.checkPoints(self.dataset.x_train[indices]) == True
        model = GPR(InternalGPR())
        model.fit(self.dataset.x_train[indices], self.dataset.y_train[indices])

        return model, indices

    def getSubset(self, region):
        indices = self.dataset.filter_point_indices(region)
        # if len(indices) == 0:
        return indices

    def sampleMore(self):
        pass


    def checkPoints(self, points):
        # Check if each coordinate of the point is within the corresponding bounds of the region
        res = True
        for point in points:
            res = res and all(np.logical_and(self.region[:, 0] <= point, point <= self.region[:, 1]))
        return res
    
    # def getMinIdx(self):
    #     idx = self.dataset._getMinIdx(self.indices)
    #     return idx
