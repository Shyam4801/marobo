import numpy as np
from .constants import *
from copy import deepcopy
from bo.utils.volume import compute_volume

"""
Class representing a Region Node in a tree structure.

Attributes:
    status (RegionState): The status of the node. (ACTIVE/INACTIVE)
    input_space (numpy.ndarray): The input space of the node.
    agentId (numpy.float32): The agent ID of the node.
    obsIndices (list): List of observation indices.
    smpIndices (list): List of evaluation sample indices.
    yOfsmpIndices (dict): Dictionary mapping evaluation sample indices to predicted function values.
    model (Any): The GP model associated with the node.
    samples (Any): The evaluation samples associated with the node.
    children (list): List of child nodes.
    numAgents (RegionState): The number of agents associated with the node.
    rewardDist (numpy.ndarray): Array representing reward distribution.
    avgRewardDist (numpy.ndarray): Array representing average reward distribution.
    agentList (list): List of agents associated with the node. Temporarily held before partitioning
    saved_state (Dict): The saved state of the node.
    state (Any): The state of the node. Facilitates restoration during simulation. Takes value (ACTUAL/None)

Methods:
    __init__: Initializes the Node with input_space and status.
    updateModel: Updates the model with given indices and model.
    checkFootprint: Checks if a point is within the input space region.
    updatesmpObs: Updates sample indices and values.
    updateObs: Updates observation indices.
    resetTrace: Resets the trace of the node.
    resetRewardDist: Resets the reward distribution.
    resetavgRewardDist: Resets the average reward distribution.
    getVolume: Computes the volume of the input space region.
    updateStatus: Updates the status of the node.
    getStatus: Gets the status of the node.
    resetStatus: Resets the status of the node.
    removeFromAgentList: Removes an agent from the agent list.
    increaseNumAgents: Increases the number of agents.
    reduceNumAgents: Reduces the number of agents.
    getnumAgents: Gets the number of agents.
    addAgentList: Adds an agent to the agent list.
    getAgentList: Gets the agent list.
    add_child: Adds a child node to the node.
    find_leaves: Finds the leaf nodes of the tree structure.

"""

class Node:
    status: RegionState
    def __init__(self, input_space, status) -> None:

        self.input_space = np.asarray(input_space, dtype=np.float32)
        self.status = status
        self.agentId = np.float32('inf')
        self.obsIndices = []
        self.smpIndices = []
        self.yOfsmpIndices = None
        self.model = None
        self.samples = None
        self.children = []
        self.numAgents = status
        self.rewardDist = np.zeros((4))
        self.avgRewardDist = np.zeros((1,4))
        self.agentList = []
        self.saved_state = None
        self.state = None

    def updateModel(self, indices, model):
        self.obsIndices = indices
        self.model = model
    
    def checkFootprint(self):
        def point_in_region(points, region):
            # Check if each coordinate of the point is within the corresponding bounds of the region
            res = True
            for point in points:
                res = res and all(np.logical_and(region[:, 0] <= point, point <= region[:, 1]))
            return res

        res = point_in_region(self.xtr, self.input_space)
        return res
    
    def updatesmpObs(self):
        self.smpIndices = self.samples.filter_point_indices(self.input_space)
        self.yOfsmpIndices = {key: 0 for key in self.smpIndices}
    
    def updateObs(self, globalGP):
        self.obsIndices = globalGP.dataset.filter_point_indices(self.input_space)
    
    def resetTrace(self):
         self.numAgents = self.getStatus()
         self.agentList = []

    def resetRewardDist(self, numAgents):
        self.rewardDist = np.zeros((numAgents))
    
    def resetavgRewardDist(self, m):
        self.rewardDist[m:] = 0
        self.avgRewardDist[m:] = 0

    def getVolume(self):
        vol = compute_volume(self.input_space)
        return vol
    
    def updateStatus(self, status):
        self.status = status

    def getStatus(self):
        return self.status
        
    def resetStatus(self):
        self.rolloutStatus = self.mainStatus


    def updateStatus(self, status):
        self.status = status

    def getStatus(self):
        return self.status
        
    def resetStatus(self):
        self.rolloutStatus = self.mainStatus

    def removeFromAgentList(self, agent):
        self.agentList.remove(agent) 
    
    def increaseNumAgents(self):
        self.numAgents += 1
    
    def reduceNumAgents(self):
        self.numAgents -= 1
    
    def getnumAgents(self):
        return self.numAgents
    
    def addAgentList(self, agent):
        self.agentList.append(agent) 
        self.agentList = list(set(self.agentList))
    
    def getAgentList(self):
        return self.agentList
    
    def add_child(self,c):
        for i in c:
            self.children.append(i)

    def find_leaves(self):
        if not self:
            return []

        leaves = []
        stack = [self]

        while stack:
            node = stack.pop()
            if not node.children:
                if node != self:
                    leaves.append(node)
            stack.extend(node.children)

        return leaves
