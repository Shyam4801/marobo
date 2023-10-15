import numpy as np
import random
from .constants import *
from copy import deepcopy

class Node:
    def __init__(self, input_space, status) -> None:
        self.input_space = np.asarray(input_space, dtype=np.float32)
        self.child = []
        self.mainStatus = status
        self.rolloutStatus = status
        self.agent = 0
        self.reward = 0
        self.avgReward = 0
        self.sampleHist = 0
        self.routine = None

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

    def update_agent(self, agent):
        self.agent = agent 
    
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



n = Node(1,1)
n.reward = 1
n.add_child([Node(2,1),Node(3,1),Node(4,1)])
l = n.find_leaves()
for i in l:
    i.reward = 1
    i.add_child([Node(i.input_space-1,1),Node(i.input_space+1,1)])

l = n.find_leaves()
for i in l:
    i.reward = 1 
    # print_tree(i)
