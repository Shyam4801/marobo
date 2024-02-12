from typing import Any
import numpy as np
import random
from .constants import *
from copy import deepcopy
from bo.utils.volume import compute_volume

class Node:
    def __init__(self, input_space, status) -> None:
        self.input_space = np.asarray(input_space, dtype=np.float32)
        self.child = []
        self.mainStatus = status
        self.rolloutStatus = status
        self.agent = None
        self.reward = []
        self.rewardDist = np.zeros((4))
        self.avgRewardDist = np.zeros((1,4))
        self.numAgents = status
        self.agentList = []
        self.avgReward = 0
        self.sampleHist = 0
        self.routine = None
        self.volume = 0
        self.numAgentsTrace = status
        self.agentListTrace = []
        self.xtr = None
        self.ytr = None
        self.model = None
        self.rolloutPrior = None 
        self.mainPrior = None

    def __call__(self):
        if self.agent != None:
            self.agentList = [self.agent]
        else:
            self.agentList = []

    def testaddFootprint(self, prior, routine): #xtr , ytr, model):
        if routine == MAIN:
             self.mainPrior = prior
        else:
             self.rolloutPrior = prior

    def addFootprint(self, xtr , ytr, model):
        # if routine == MAIN:
        self.xtr = deepcopy(xtr)
        self.ytr = deepcopy(ytr)
        self.model = deepcopy(model)
         

    def resetTrace(self, routine):
         self.numAgents = self.getStatus(routine)
         self.agentList = []
        # self.numAgentsTrace = self.numAgents
        # self.agentListTrace = self.agentList

    # def updateRewardDist(self, routine, reward):
    #     self.rewardDist

    def resetRewardDist(self, numAgents):
        self.rewardDist = np.zeros((numAgents))
    
    def resetavgRewardDist(self, numAgents):
        self.avgRewardDist = np.zeros((numAgents))

    def getVolume(self):
        vol = compute_volume(self.input_space)
        return vol

    def setRoutine(self, name):
        self.routine = name

    def updateStatus(self, status, name = ROLLOUT):
        if name == ROLLOUT:
            self.rolloutStatus = status
        elif name == MAIN:
            self.mainStatus = status

    def getStatus(self, name=ROLLOUT):
        if name == ROLLOUT:
            return self.rolloutStatus
        elif name == MAIN:
            return self.mainStatus
        
    def resetStatus(self):
        self.rolloutStatus = self.mainStatus

    def addSample(self, sample):
        self.sampleHist = sample
    
    def resetReward(self):
        self.reward = 0
        self.avgReward = 0
        # if not self:
        #     return

        # # leaves = []
        # stack = [self]

        # while stack:
        #     node = stack.pop()
        #     if not node.child:
        #         if node != self:
        #             node.reward = 0
        #             node.avgReward = 0
        #     stack.extend(node.child)

        # return leaves

    def addAgentList(self, agent, routine):
        # if routine == MAIN:
            self.agentList.append(agent) 
            self.agentList = list(set(self.agentList))
        # else:
        #     self.agentListTrace.append(agent)
    
    def removeFromAgentList(self, agent, routine):
        # if routine == MAIN:
            self.agentList.remove(agent) 
        # else:
        #     self.agentListTrace.remove(agent)
    
    def increaseNumAgents(self, routine):
        # if routine == MAIN:
            self.numAgents += 1
        # else:
        #     self.numAgentsTrace += 1
    
    def reduceNumAgents(self, routine):
        # if routine == MAIN:
            self.numAgents -= 1
        # else:
        #     self.numAgentsTrace -= 1
    
    def getnumAgents(self, routine):
        # if routine == MAIN:
            return self.numAgents
        # else:
        #     return self.numAgentsTrace
    
    def getAgentList(self, routine):
        # if routine == MAIN:
            return self.agentList
        # else:
        #     return self.agentListTrace
    
    def add_child(self,c):
        for i in c:
            # print("i,i.input_space: ",i.input_space)
            self.child.append(i)
    
    def find_leaves(self):
        if not self:
            return []

        leaves = []
        stack = [self]

        while stack:
            node = stack.pop()
            if not node.child:
                if node != self:
                    leaves.append(node)
            stack.extend(node.child)

        return leaves
    
    def setAvgRewards(self, rewards):
        if len(rewards) == 0:
            return

        for i in range(len(self.child)):
            if not self.child[i].child:
                # Assign a value to the leaf node
                rw = rewards[-1]
                
                rewards = np.delete(rewards, [-1], axis=0)
                # print('rw : ',rw)
                # rewards = rewards[:-1]
                # print('rewards after pop : ', rewards)
                self.child[i].avgRewardDist = rw
            else:
                self.child[i].setAvgRewards(rewards)

        # leaves = []
        # self._find_leaves_helper(self, leaves)
        # return leaves

    def _find_leaves_helper(self, node, leaves):
        if not node.child:
            if node != self:  # Exclude the root node
                leaves.append(node)
        else:
            for child in node.child:
                self._find_leaves_helper(child, leaves)

    def getAssignments(self, root, routine):
        lf = root.find_leaves()
        # print('______________below find leaves_______________________')
        # print_tree(self.root)
        # print('_____________________________________')
        assignments = {}
        agents_to_subregion = []
        internal_inactive_subregion=[]
        for l in lf:
            if l.getStatus(routine) == 1:
                assignments.update({l : l.getStatus(routine)})
                agents_to_subregion.append(l)
            elif l.getStatus(routine) == 0:
                internal_inactive_subregion.append(l)
        
        return assignments, agents_to_subregion, internal_inactive_subregion
    

    

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