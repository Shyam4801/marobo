import numpy as np
import random
from .constants import *


class Node:
    def __init__(self, input_space, status) -> None:
        self.input_space = np.asarray(input_space, dtype=np.float32)
        self.child = []
        self.status = status
        self.agent = 0

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
    
def print_tree(node, level=0, prefix=''):
    if node.status:
        color = GREEN
    else:
        color = RED
    if node is None:
        return

    for i, child in enumerate(node.child):
        print_tree(child, level + 1, '|   ' + prefix if i < len(node.child) - 1 else '    ' + prefix)
    
    print('    ' * level + prefix + f'-- {color}{node.input_space.flatten()}{END}')

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
