from typing import Any
import numpy as np
import random
from .constants import *
from copy import deepcopy
from bo.utils.volume import compute_volume

class Prior:
    def __init__(self, xtr, ytr, model, routine):
        self.xtrain = xtr
        self.y_train = ytr
        self.model = model
        self.routine = routine 

    def getData(self, routine):
        if self.checkRoutine(routine):
            return self.xtrain, self.y_train, self.model
        else:
            return f"This is part of {routine} routine"
    
    def checkRoutine(self, routine):
        if self.routine == routine:
            return True
        else: 
            return False