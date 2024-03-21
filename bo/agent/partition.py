from typing import Any, Type
import numpy as np
import random
from .constants import *
from copy import deepcopy
from bo.utils.volume import compute_volume
from dataclasses import dataclass, field
from .localgp import Prior
from .observation import Observations


# @dataclass
class Node:
    status: RegionState
    def __init__(self, input_space, status) -> None:
        # input_space: np.ndarray
        # status: Enum
        # agentId : int = -1
        # gp: Prior = None
        # obsIndices: Any = None
        # smpIndices: Any = None
        # yOfsmpIndices: dict = None
        # model: Any = None
        # samples: Observations = None
        # children: list = field(default_factory=list)
        # numAgents: int = 1
        # rewardDist: np.ndarray = np.zeros((4))
        # avgRewardDist: np.ndarray = np.zeros((1,4))
        # agentList: list = field(default_factory=list)
        # saved_state: dict = None

        self.input_space = np.asarray(input_space, dtype=np.float32)
        self.status = status
        self.agentId = None
        # self.gp = None
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

    # def __repr__(self):
    #     return f"TreeNode(parent={self.parent})"

    # def __post_init__(self):
    #      if self.smpIndices != None: 
    #         self.yOfsmpIndices = {key: 0 for key in self.smpIndices}

    def updateModel(self, indices, model):
        self.obsIndices = indices
        self.model = model
    
    def checkFootprint(self):
        def point_in_region(points, region):
            # Check if each coordinate of the point is within the corresponding bounds of the region
            res = True
            for point in points:
                res = res and all(np.logical_and(region[:, 0] <= point, point <= region[:, 1]))
                # if res == False:
                    # print('pt not in region :',agent.id, point, region)
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
    
    def resetavgRewardDist(self, numAgents):
        self.rewardDist = np.zeros((numAgents))
        self.avgRewardDist = np.zeros((1,numAgents))

    def getVolume(self):
        vol = compute_volume(self.input_space)
        return vol
    
    def updateStatus(self, status):
        self.status = status

    def getStatus(self):
        # if name == ROLLOUT:
        #     return self.rolloutStatus
        # elif name == MAIN:
        return self.status
        
    def resetStatus(self):
        self.rolloutStatus = self.mainStatus


    def updateStatus(self, status):
        self.status = status

    def getStatus(self):
        # if name == ROLLOUT:
        #     return self.rolloutStatus
        # elif name == MAIN:
        return self.status
        
    def resetStatus(self):
        self.rolloutStatus = self.mainStatus

    def removeFromAgentList(self, agent):
        # if routine == MAIN:
            self.agentList.remove(agent) 
        # else:
        #     self.agentListTrace.remove(agent)
    
    def increaseNumAgents(self):
        # if routine == MAIN:
            self.numAgents += 1
        # else:
        #     self.numAgentsTrace += 1
    
    def reduceNumAgents(self):
        # if routine == MAIN:
            self.numAgents -= 1
        # else:
        #     self.numAgentsTrace -= 1
    
    def getnumAgents(self):
        # if routine == MAIN:
            return self.numAgents
        # else:
        #     return self.numAgentsTrace
    
    def addAgentList(self, agent):
        # if routine == MAIN:
            self.agentList.append(agent) 
            self.agentList = list(set(self.agentList))
        # else:
        #     self.agentListTrace.append(agent)
    
    def getAgentList(self):
        # if routine == MAIN:
            return self.agentList
        # else:
        #     return self.agentListTrace
    
    def add_child(self,c):
        for i in c:
            # print("i,i.input_space: ",i.input_space)
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

    # def __init__(self, input_space, status) -> None:
    #     self.input_space = np.asarray(input_space, dtype=np.float32)
    #     self.child = []
    #     self.mainStatus = status
    #     self.rolloutStatus = status
    #     self.agent = None
    #     self.reward = []
    #     self.rewardDist = np.zeros((4))
    #     self.avgRewardDist = np.zeros((1,4))
    #     self.numAgents = status
    #     self.agentList = []
    #     self.avgReward = 0
    #     self.sampleHist = 0
    #     self.routine = None
    #     self.volume = 0
    #     self.numAgentsTrace = status
    #     self.agentListTrace = []
    #     self.xtr = None
    #     self.ytr = None
    #     self.model = None
    #     self.rolloutPrior = None 
    #     self.mainPrior = None
    #     self.smpXtr = None #deepcopy(self.mcsmpXtr)
    #     self.smpYtr = None #deepcopy(self.mcsmpYtr)
    #     self.avgsmpYtr = None
    #     self.avgsmpXtr = None
    #     self.mcsmpXtr = None
    #     self.mcsmpYtr = None
    #     self.chkobjcopy = {}

    # def __call__(self):
    #     if self.agent != None:
    #         self.agentList = [self.agent]
    #     else:
    #         self.agentList = []

    # def check_points(self):
    #     # if self.routine == MAIN:
    #     xtr = self.smpXtr
    #     reg = self.input_space
    #     # else:
    #     #     xtr = self.smpXtr
    #     #     reg = self.input_space

    #     def point_in_region(points, region):
    #         # Check if each coordinate of the point is within the corresponding bounds of the region
    #         res = True
    #         for point in points:
    #             res = res and all(np.logical_and(region[:, 0] <= point, point <= region[:, 1]))
    #             # if res == False:
    #                 # print('pt not in region :',agent.id, point, region)
    #         return res

    #     res = point_in_region(xtr, reg)
    #     return res
    
    # def resetSmps(self):
    #     self.smpXtr = deepcopy(self.mcsmpXtr)
    #     self.smpytr = deepcopy(self.mcsmpYtr)

    # def testaddFootprint(self, prior, routine): #xtr , ytr, model):
    #     if routine == MAIN:
    #          self.mainPrior = prior
    #     else:
    #          self.rolloutPrior = prior

    # def addFootprint(self, xtr , ytr, model):
    #     # if routine == MAIN:
    #     self.xtr = deepcopy(xtr)
    #     self.ytr = deepcopy(ytr)
    #     self.model = deepcopy(model)
    #     self.chkobjcopy[model] = self.model

    # def updatesmpObs(self, parent):
    #      self.smpXtr = deepcopy(parent.smpXtr)
    #      self.smpYtr = deepcopy(parent.smpYtr)

    # # def updatesmpY(self, model):

    # def checkFootprint(self):
    #     def point_in_region(points, region):
    #         # Check if each coordinate of the point is within the corresponding bounds of the region
    #         res = True
    #         for point in points:
    #             res = res and all(np.logical_and(region[:, 0] <= point, point <= region[:, 1]))
    #             # if res == False:
    #                 # print('pt not in region :',agent.id, point, region)
    #         return res

    #     res = point_in_region(self.xtr, self.input_space)
    #     return res
         

    # def resetTrace(self, routine):
    #      self.numAgents = self.getStatus(routine)
    #      self.agentList = []
    #     # self.numAgentsTrace = self.numAgents
    #     # self.agentListTrace = self.agentList

    # # def updateRewardDist(self, routine, reward):
    # #     self.rewardDist

    # def resetRewardDist(self, numAgents):
    #     self.rewardDist = np.zeros((numAgents))
    
    # def resetavgRewardDist(self, numAgents):
    #     self.avgRewardDist = np.zeros((numAgents))

    # def getVolume(self):
    #     vol = compute_volume(self.input_space)
    #     return vol

    # def setRoutine(self, name):
    #     self.routine = name

    # def updateStatus(self, status, name = ROLLOUT):
    #     if name == ROLLOUT:
    #         self.rolloutStatus = status
    #     elif name == MAIN:
    #         self.mainStatus = status

    # def getStatus(self, name=ROLLOUT):
    #     if name == ROLLOUT:
    #         return self.rolloutStatus
    #     elif name == MAIN:
    #         return self.mainStatus
        
    # def resetStatus(self):
    #     self.rolloutStatus = self.mainStatus

    # def addSample(self, sample):
    #     self.sampleHist = sample
    
    # def resetReward(self):
    #     self.reward = 0
    #     self.avgReward = 0
    #     # if not self:
    #     #     return

    #     # # leaves = []
    #     # stack = [self]

    #     # while stack:
    #     #     node = stack.pop()
    #     #     if not node.child:
    #     #         if node != self:
    #     #             node.reward = 0
    #     #             node.avgReward = 0
    #     #     stack.extend(node.child)

    #     # return leaves

    

    
    # def setAvgRewards(self, rewards):
    #     if len(rewards) == 0:
    #         return

    #     for i in range(len(self.child)):
    #         if not self.child[i].child:
    #             # Assign a value to the leaf node
    #             rw = rewards[-1]
                
    #             rewards = np.delete(rewards, [-1], axis=0)
    #             # print('rw : ',rw)
    #             # rewards = rewards[:-1]
    #             # print('rewards after pop : ', rewards)
    #             self.child[i].avgRewardDist = rw
    #         else:
    #             self.child[i].setAvgRewards(rewards)

    #     # leaves = []
    #     # self._find_leaves_helper(self, leaves)
    #     # return leaves

    # def _find_leaves_helper(self, node, leaves):
    #     if not node.child:
    #         if node != self:  # Exclude the root node
    #             leaves.append(node)
    #     else:
    #         for child in node.child:
    #             self._find_leaves_helper(child, leaves)

    # def getAssignments(self, root, routine):
    #     lf = root.find_leaves()
    #     # print('______________below find leaves_______________________')
    #     # print_tree(self.root)
    #     # print('_____________________________________')
    #     assignments = {}
    #     agents_to_subregion = []
    #     internal_inactive_subregion=[]
    #     for l in lf:
    #         if l.getStatus(routine) == 1:
    #             assignments.update({l : l.getStatus(routine)})
    #             agents_to_subregion.append(l)
    #         elif l.getStatus(routine) == 0:
    #             internal_inactive_subregion.append(l)
        
    #     return assignments, agents_to_subregion, internal_inactive_subregion
    

    

# n = Node(1,1)
# n.reward = 1
# n.add_child([Node(2,1),Node(3,1),Node(4,1)])
# l = n.find_leaves()
# for i in l:
#     i.reward = 1
#     i.add_child([Node(i.input_space-1,1),Node(i.input_space+1,1)])

# m = Node(9,9)
# m.reward = 1
# m.add_child([Node(2,1),Node(3,1),Node(4,1)])
# k = m.find_leaves()
# for i in k:
#     i.reward = 1
#     i.add_child([Node(i.input_space-1,1),Node(i.input_space+1,1)])

# l = n.find_leaves()
# k = m.find_leaves()
# for i in l:
#     print(i.input_space)

# # from treeOperations import print_tree
# # print_tree(n)
# print('next')
# for i in k:
#     print(i.input_space)

# print_tree(m)