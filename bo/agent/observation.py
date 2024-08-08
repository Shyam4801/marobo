
from dataclasses import dataclass
import numpy as np
from ..sampling import *
from ..utils.computeRobustness import compute_robustness

"""
Class representing observations with methods for filtering point indices based on a region, checking if a point is within a region, getting the index of the minimum value, and appending new samples to the observations.

Attributes:
    x_train (np.ndarray): Array containing the training data points.
    y_train (np.ndarray): Array containing the corresponding function values.

Methods:
    filter_point_indices(region): Filter point indices based on a region.
    point_in_region(point, region): Check if a point is within a region.
    _getMinIdx(indices): Get the index of the minimum value.
    appendSamples(xtr, ytr): Append new samples to the observations.
"""
@dataclass
class Observations:
    x_train: np.ndarray
    y_train: np.ndarray

    def __post_init__(self):
        assert self.x_train.shape[0] == len(self.y_train)

    def filter_point_indices(self, region):
        """
        Filter point indices based on a region.
        
        Args:
            region (callable): A function that takes a point (x) as input and returns True if the point 
                               belongs to the region, and False otherwise.
        
        Returns:
            list: A list of indices of points that belong to the specified region.
        """
        filtered_indices = []
        for i, x in enumerate(self.x_train):
            if self.point_in_region(x, region):
                filtered_indices.append(i)
        return filtered_indices
    
    def point_in_region(self, point, region):
        res = all(np.logical_and(region[:, 0] <= point, point <= region[:, 1]))
        return res
    
    def _getMinIdx(self, indices):
        return indices[np.argmin(self.y_train[indices])]
    
    def appendSamples(self, xtr, ytr):
        self.x_train = np.vstack((self.x_train, xtr))
        if len(ytr) != 0:
            self.y_train = np.hstack((self.y_train, ytr))

        # newidxs = np.where(np.isin(self.x_train, xtr))[0]
        return self
        