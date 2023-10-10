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
    
    def updateReward(self, reward):
        self.reward = reward

    def update_agent(self, agent):
        self.agent = agent 
    
    def add_child(self,c):
        for i in c:
            # print("i,i.input_space: ",i.input_space)
            self.child.append(i)
    
    def find_leaves(self):
        leaves = []
        self._find_leaves_helper(self, leaves)
        return leaves

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
        
        # xtr = deepcopy(self.x_train)
        # ytr = deepcopy(self.y_train)
        # agents = [Agent(self.gpr_model, xtr, ytr, agents_to_subregion[a].input_space) for a in range(self.num_agents)]
        # # print('agents_to_subregion : ',agents_to_subregion)
        # for i,sub in enumerate(agents_to_subregion):
        #     sub.update_agent(agents[i])
    
    # def accumulateReward(self):
    #     self.accumulate_rewards_and_update(self)
    #     # print(node.input_space)
    
    # def accumulate_rewards_and_update(self, node):
    #     # Base case: If the node is a leaf, return its reward
    #     if not node.child:
    #         print('term',node.input_space, node.reward)
    #         if node != self:
    #             return node.reward

    #     # Initialize the accumulated reward for this node
    #     accumulated_reward = node.reward

    #     # Recursively accumulate rewards from child nodes and update node.value
    #     for child in node.child:
    #         print(child.input_space, accumulated_reward)
    #         child_accumulated_reward = self.accumulate_rewards_and_update(child)
    #         accumulated_reward += child_accumulated_reward

    #     # Update the node.value with the accumulated reward
    #     node.reward = accumulated_reward

        # return accumulated_reward


def print_tree(node, routine, level=0, prefix=''):
    # print('node.getStatus(routine) :',node.getStatus(routine))
    if node.getStatus(routine) == 1:
        color = GREEN
    else:
        color = RED
    if node is None:
        return

    for i, child in enumerate(node.child):
        print_tree(child, routine, level + 1, '|   ' + prefix if i < len(node.child) - 1 else '    ' + prefix)
    
    print('    ' * level + prefix + f'-- {color}{node.input_space.flatten()}{node.reward}{END}')

def find_close_factor_pairs(number):
    factors = np.arange(1, int(np.sqrt(number)) + 1)
    valid_indices = np.where(number % factors == 0)[0]
    factors = factors[valid_indices]

    factor_pairs = [(factors[i], number // factors[i]) for i in range(len(factors))]
    min_gap = np.inf
    final_pair = 0
    for f1,f2 in  factor_pairs:
        if min_gap > abs(f1 - f2):
            min_gap = abs((f1 - f2))
            close_pairs = (f1,f2)

    # close_pairs = [(f1, f2) for f1, f2 in factor_pairs if abs(f1 - f2) <= 5]

    return close_pairs


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

# n.accumulateReward(n)
def accumulate_rewards_and_update(node):
    # Base case: If the node is a leaf, return its reward
    if not node.child:
        return node.reward

    # Initialize the accumulated reward for this node
    accumulated_reward = node.reward

    # Recursively accumulate rewards from child nodes and update node.value
    for child in node.child:
        child_accumulated_reward = accumulate_rewards_and_update(child)
        accumulated_reward += child_accumulated_reward

    # Update the node.value with the accumulated reward
    node.reward = accumulated_reward

    return accumulated_reward

# print(accumulate_rewards_and_update(n))
# print_tree(n)

def find_min_leaf(node, min_leaf=None):
        if min_leaf is None:
            min_leaf = [float('inf'), None]

        if not node.child:  # Check if it's a leaf node
            if node.reward < min_leaf[0]:
                min_leaf[0] = node.reward
                min_leaf[1] = node

        for child in node.child:
            find_min_leaf(child, min_leaf)

        return min_leaf

def dropChildren(node):
    if node.routine == MAIN:
        node.child = []

    for child in node.child:
        dropChildren(child)
