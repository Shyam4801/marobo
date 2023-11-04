from .agent import Agent
from copy import deepcopy
from .constants import *
import numpy as np
from .partition import Node

def split_region(root,dim,num_agents):
    # print('split_region: dim',dim, num_agents)
    region = np.linspace(root.input_space[dim][0], root.input_space[dim][1], num = num_agents+1)
    final = []

    for i in range(len(region)-1):
        final.append([region[i], region[i+1]])
    regs = []
    for i in range(len(final)):
        org = root.input_space.copy()
        org[dim] = final[i]
        regs.append(org)

    regs = [Node(i, 1) for i in regs]
    return regs
    
# def get_subregion(root, num_agents,dic, dim=0):
#     q=[root]
#     while(len(q) < num_agents):
#         if len(q) % 2 == 0:
#             dim = (dim+1)% len(root.input_space)
#             print('inside: ',dim)
#         curr = q.pop(0)
#         # print('dim curr queue_size', dim, curr.input_space, len(q))
#         print('dim : ',dim, 'curr ', curr.input_space, 'q: ', [i.input_space for i in q])
#         ch = split_region(curr,dim, dic[dim])
#         # print('ch',ch)
#         curr.add_child(ch)
#         q.extend(ch)
#     # print([i.input_space for i in q])
#     return q

def get_subregion(root, num_agents, dic, dim):
    q = [root]
    while len(q) < num_agents:
        cuts = dim % len(dic) 
        if len(q) % dic[cuts] == 0:
            dim = np.random.randint(len(root.input_space)) #(dim + 1) % len(root.input_space) # #
        curr = q.pop(0)
        
        # print('dim :', dim,'q: ', [i.input_space for i in q])
        ch = split_region(curr, dim, dic[cuts])
        curr.add_child(ch)
        q.extend(ch)
        # if len(q) % dic[cuts] == 0:
        #     dim = (dim + 1) % len(root.input_space)
    return q

# def get_subregion(root, num_agents, dic, dim=0):
#     q = [root]
#     while len(q) < num_agents:
#         if len(q) % 2 == 0:
#             dim = (dim + 1) % len(root.input_space)
#         curr = q.pop(0)
        
#         cuts = (dim + 1) % len(dic)
#         print('dim : ',dim,'cuts :',cuts, 'curr ', curr.input_space, 'q: ', [i.input_space for i in q])
#         ch = split_region(curr, dim, dic[cuts])
#         curr.add_child(ch)
#         q.extend(ch)
#         print('ch : ',ch[0].input_space)
#         print('len(root.input_space): ',len(root.input_space))
        

#     return q

def print_tree(node, routine, level=0, prefix=''):
    # print('node.getStatus(routine) :',node.getStatus(routine))
    if routine == MAIN:
        reward = node.avgReward
    else:
        reward = node.reward
    if node.getStatus(routine) == 1:
        color = GREEN
    else:
        color = RED
    if node is None:
        return

    for i, child in enumerate(node.child):
        print_tree(child, routine, level + 1, '|   ' + prefix if i < len(node.child) - 1 else '    ' + prefix)
    
    print('    ' * level + prefix + f'-- {color}{node.input_space.flatten()}{reward}{END}')

# def find_close_factor_pairs(number):
#     factors = np.arange(1, int(np.sqrt(number)) + 1)
#     valid_indices = np.where(number % factors == 0)[0]
#     factors = factors[valid_indices]

#     factor_pairs = [(factors[i], number // factors[i]) for i in range(len(factors))]
#     min_gap = np.inf
#     final_pair = 0
#     for f1,f2 in  factor_pairs:
#         if min_gap > abs(f1 - f2):
#             min_gap = abs((f1 - f2))
#             close_pairs = (f1,f2)

#     # close_pairs = [(f1, f2) for f1, f2 in factor_pairs if abs(f1 - f2) <= 5]

#     return close_pairs

def find_prime_factors(number):
    prime_factors = []
    candidate = 2

    while candidate <= number:
        if number % candidate == 0:
            prime_factors.append(candidate)
            number /= candidate
        else:
            candidate += 1

    return prime_factors

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

def find_min_leaf(node, routine, min_leaf=None):
        if min_leaf is None:
            min_leaf = [float('inf'), None]

        if routine == MAIN:
            reward = node.avgReward
        else:
            reward = node.reward

        if not node.child:  # Check if it's a leaf node
            if reward < min_leaf[0]:
                min_leaf[0] = reward
                min_leaf[1] = node

        for child in node.child:
            find_min_leaf(child, routine, min_leaf)

        return min_leaf

def dropChildren(node):
    if node.routine == MAIN:
        node.child = []

    for child in node.child:
        dropChildren(child)



def reassign(root, routine, agents, currentAgentIdx, model, xtr, ytr):
    minRegval = find_min_leaf(root, routine)
    minReg = minRegval[1]
    val = minRegval[0]
    # print('agents inside reassign from:'+str(routine), [i.region_support.input_space for i in agents])
    # print('curr agents[self.num_agents-h].region_support: ',agents[currentAgentIdx].region_support.input_space, agents[currentAgentIdx].region_support)
    # print('region with min reward: find_min_leaf ',minReg.input_space, val, minReg)
    currAgentRegion = agents[currentAgentIdx].getRegion(routine)
    print('currentAgentIdx region: ',currAgentRegion.input_space, 'minreg :', minReg.input_space, 'val: ',val)

    # print(root.find_leaves())
    if currAgentRegion != minReg:  # either moves to act or inact region
        currAgentRegion.updateStatus(0, routine)
        agents[currentAgentIdx](routine)
        if minReg.getStatus(routine) == 1:
            internal_factorized = find_prime_factors(2) #sorted(find_close_factor_pairs(2), reverse=True)
            ch = get_subregion(deepcopy(minReg), 2, internal_factorized, np.random.randint(len(minReg.input_space)))
            minReg.add_child(ch)
            minReg.updateStatus(0, routine)

            agents[currentAgentIdx].updateBounds(ch[1], routine)
            print('ch[1].getStatus(routine): ',ch[1].input_space ,ch[1].getStatus(routine))
            agents[currentAgentIdx](routine)
            print('ch[1].agent.simReg.input_space: ',ch[1].agent.simReg.input_space, ch[1].agent.region_support.input_space)  # ch[1].agent.simReg.input_space = ch[1].input_space should be equal
            assert ch[1].agent.getRegion(routine) == ch[1]
            minReg.agent.updateBounds(ch[0], routine)
            minReg.agent(routine)  # ch[0].input_space == minReg.agent.simReg.input_space should be 
            assert ch[0] == minReg.agent.getRegion(routine)
            # print('minreg : ', minReg.input_space, minReg.getStatus(routine))
        else:
            minReg.updateStatus(1, routine)
            agents[currentAgentIdx].updateBounds(minReg, routine)
            agents[currentAgentIdx](routine)   # minReg.input_space == agents[currentAgentIdx].simReg.input_space)
            assert minReg == agents[currentAgentIdx].getRegion(routine)

        if routine == MAIN:
            minReg.resetStatus()
            minReg.resetReward()
        
        # print('agents inside reassign ', [i.getRegion(routine).input_space for i in agents])
        newAgents = []
        newLeaves = root.find_leaves()
        print(newLeaves)
        # print('newleaves ', [{str(i.input_space) : (i.agent.simReg, i.getStatus(ROLLOUT))} for i in newLeaves])
        # print_tree(root, ROLLOUT)
        # if routine == MAIN:
        #     print('inside reassign tree')
        #     print_tree(root, MAIN)
        #     print_tree(root, ROLLOUT)
        for reg in newLeaves:
            # reg.setRoutine(routine)
            # if reg.agent != None:
            #     reg.agent(routine)
            if reg.getStatus(routine) == 1:
                # reg.agent(routine)
                print('reg ', str(reg.input_space), reg.agent)
                print('newleaves ', {str(reg.input_space) : (reg.agent.getRegion(routine).input_space, reg.getStatus(ROLLOUT))})  # regions should be equal
                newAgents.append(reg.agent)
                # print('new agents inside for : ', newAgents)
            if routine == MAIN:
                reg.setRoutine(routine)
                reg.resetStatus()
                reg.resetReward()
        
        return newAgents
    else:
        oldLeaves = root.find_leaves()
        for reg in oldLeaves:
            # reg.region_support.setRoutine(routine)
            # print('else loop ', reg.input_space, reg.routine)
            if routine == MAIN:
                reg.resetStatus()
                reg.resetReward()
        # print('inside reassign new agents : ', newAgents)

        
    return agents


def find_leaves_and_compute_avg(trees):
    if not trees:
        return []

    leaves = []
    total_leaf_values = [0] * len(trees)
    total_leaf_counts = [0] * len(trees)

    # Iterate over each tree
    for tree_index, tree in enumerate(trees):
        stack = [tree]

        while stack:
            node = stack.pop()
            if not node.child:
                if node != tree:  # Exclude the root node of the tree
                    total_leaf_values[tree_index] += node.avgReward
                    total_leaf_counts[tree_index] += 1
                else:
                    leaves.append(node)
            stack.extend(node.child)
    
    avg_leaf_values = [total_leaf_values[i] / total_leaf_counts[i] if total_leaf_counts[i] > 0 else 0 for i in range(len(trees))]
    return leaves, avg_leaf_values



# n = Node(1,1)
# n.reward = 1
# n.add_child([Node(2,1),Node(3,1),Node(4,1)])
# l = n.find_leaves()
# for i in l:
#     i.reward = 1
#     i.add_child([Node(i.input_space-1,1),Node(i.input_space+1,1)])

# m = Node(9,9)
# m.reward = 1
# m.add_child([Node(8,1),Node(7,1),Node(6,1)])
# k = m.find_leaves()
# for i in k:
#     i.reward = 1
#     i.add_child([Node(i.input_space-1,1),Node(i.input_space+1,1)])

# l = n.find_leaves()
# k = m.find_leaves()
# for i in l:
#     print(i.input_space)

# from treeOperations import print_tree

# print('next')
# for i in k:
#     print(i.input_space)

# print_tree(n, MAIN)
# print_tree(m, MAIN)

# reg = np.array([[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]])
# n = Node(reg,1)
# a = 6
# factorized = find_prime_factors(a) #sorted(find_close_factor_pairs(a), reverse=True)
# print(factorized, 'np.random.randint(len(reg)): ',np.random.randint(len(reg)))
# ch = get_subregion(deepcopy(n), a,factorized, np.random.randint(len(reg)))
# print(len(ch))
# n.add_child(ch)

# print('[i.getVolume() for i in ch]: ',[{str(i.input_space ): i.getVolume()} for i in ch])

# v=0
# for i in ch:
#     v += i.getVolume()

# assert n.getVolume() == v
# print('set([i.getVolume() for i in ch]) :', set([i.getVolume() for i in ch]))
# print_tree(n, MAIN)