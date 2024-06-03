from typing import Any
import numpy as np
from .constants import *
from dataclasses import dataclass
from bo.agent.observation import Observations
from ..gprInterface import GPR, InternalGPR

""" 
A class representing a Gaussian Process model.

Attributes:
    dataset (Observations): An instance of the Observations class containing training data.
    region (np.ndarray): An array defining the region of interest for the GPR model.

Methods:
    __post_init__(self): Ensures that the dataset attribute is an instance of the Observations class.
    buildModel(self): Builds a GPR model using the dataset within the specified region.
    getSubset(self, region): Retrieves the subset of data points from the dataset that fall within the specified region.
    sampleMore(self): Placeholder method for sampling additional data points.
    checkPoints(self, points): Checks if the given points are within the bounds of the region.

"""
@dataclass
class Prior:
    dataset: Observations
    region: np.ndarray
    
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
        return indices

    def sampleMore(self):
        pass


    def checkPoints(self, points):
        # Check if each coordinate of the point is within the corresponding bounds of the region
        res = True
        for point in points:
            res = res and all(np.logical_and(self.region[:, 0] <= point, point <= self.region[:, 1]))
        return res
    