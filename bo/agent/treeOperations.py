# from .agent import Agent
from copy import deepcopy
from .constants import *
import numpy as np
from .partition import Node
from bo.sampling import uniform_sampling, lhs_sampling
from bo.utils import compute_robustness
from itertools import permutations
import yaml
from collections import defaultdict
from .localgp import Prior
from .observation import Observations

from multiprocessing import Pool
from joblib import Parallel, delayed
import random
# from bo.bayesianOptimization.rolloutEI import RolloutEI
from tqdm import tqdm

with open('config.yml', 'r') as file:
    configs = yaml.safe_load(file)


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

    regs = [Node(i, RegionState.ACTIVE.value) for i in regs]
    return regs
    
def genFactors(num_sub_regions):
    factors = []
    for i in range(1, num_sub_regions + 1):
        if num_sub_regions % i == 0:
            factors.append([i, num_sub_regions // i])
            if i != 1 and i != num_sub_regions:
                factors.append([num_sub_regions // i, i])

    return factors

def get_subregion(root, num_agents, dic, dim):
    q = [root]
    while len(q) < num_agents:
        cuts = dim % len(dic) 
        if len(q) % dic[cuts] == 0:
            dim =  np.random.randint(len(root.input_space)) #(dim + 1) % len(root.input_space) #
        curr = q.pop(0)
        
        # print('dim :', dim,'q: ', [i.input_space for i in q])
        ch = split_region(curr, dim, dic[cuts])
        curr.add_child(ch)
        q.extend(ch)
        # if len(q) % dic[cuts] == 0:
        #     dim = (dim + 1) % len(root.input_space)
    return q

def print_tree(node, level=0, prefix=''):
    # print('node.getStatus(routine) :',node.getStatus(routine))
    # if routine == MAIN:
    #     reward = node.avgRewardDist
    # else:
    reward = node.rewardDist
    if node.getStatus() == 1:
        color = GREEN
    else:
        color = RED
    if node is None:
        return
    id = -1
    if node.getStatus() == RegionState.ACTIVE.value:
        id = node.agentId

    for i, child in enumerate(node.children):
        print_tree(child, level + 1, '|   ' + prefix if i < len(node.children) - 1 else '    ' + prefix)
    
    print('    ' * level + prefix + f'-- {color}{node.input_space.flatten()}{reward}{id}{END}')

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

def accumulate_rewardDist(node, numAgents):
    # Base case: If the node is a leaf, return its reward
    if not node.children:
        return node.rewardDist

    # Initialize the accumulated reward for this node
    accumulated_reward = node.rewardDist

    # Recursively accumulate rewards from child nodes and update node.value
    for child in node.children:
        child_accumulated_reward = accumulate_rewardDist(child, numAgents)
        # print('child_accumulated_reward: ',child_accumulated_reward, child.input_space, accumulated_reward)
        if child_accumulated_reward == []:
            child_accumulated_reward = [0]* numAgents
        child_accumulated_reward = np.hstack((child_accumulated_reward))
        accumulated_reward = np.hstack((accumulated_reward))
        # accumulated_reward += child_accumulated_reward
        # print('--------------------------------------')
        
        accumulated_reward = np.vstack((accumulated_reward, child_accumulated_reward))
        accumulated_reward = np.sum(accumulated_reward, axis=0) #[sum(i) for i in zip(accumulated_reward, child_accumulated_reward)]
        # print('accumulated_reward after sum ',accumulated_reward)
        # print('--------------------------------------')

    # Update the node.value with the accumulated reward
    # print('accumulated_reward: ',accumulated_reward, node.input_space)
    # print('--------------------------------------')
    node.rewardDist = accumulated_reward#[-1]

    return accumulated_reward

# print(accumulate_rewards_and_update(n))
# print_tree(n)

def accumulateSamples(node):
    # Base case: If the node is a leaf, return its reward
    if not node.child:
        return node.smpXtr, node.smpYtr

    # Initialize the accumulated reward for this node
    accumulated_reward = node.smpXtr
    accy = node.smpYtr

    # Recursively accumulate rewards from child nodes and update node.value
    for child in node.child:
        child_accumulated_reward, chy = accumulateSamples(child)
        
        accumulated_reward = np.vstack((accumulated_reward, child_accumulated_reward))
        accy = np.hstack((accy, chy))

    dictionary = {} #{tuple(row): value for row, value in zip(array, values)}
    for row, value in zip(accumulated_reward, accy):
        if tuple(row) in dictionary:
            dictionary[tuple(row)] = value
        else:
            dictionary[tuple(row)] = value
            
    keys = np.array(list(dictionary.keys()))
    values = np.array(list(dictionary.values()))


    return keys, values #accumulated_reward, accy

def accumulate_all_keys(node):
    if not node.children:
        return node.yOfsmpIndices.copy() if node.yOfsmpIndices else {}
    
    accumulated_values = {}
    for child in node.children:
        child_values = accumulate_all_keys(child)
        for key, value in child_values.items():
            accumulated_values[key] = accumulated_values.get(key, 0) + value
    return accumulated_values

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

def find_leaf_with_min_reward_dist(node, agentIdx, routine, min_leaf=None):
        if min_leaf is None:
            min_leaf = [float('inf'), None]

        if routine == MAIN:
            reward = node.avgRewardDist[agentIdx]
        else:
            reward = node.rewardDist[agentIdx]

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

def find_level_of_leaf(root, target_leaf_value):
        if root is None:
            return None

        stack = [(root, [])]

        while stack:
            current_node, path = stack.pop()

            # Check if the current node is a leaf and has the target value
            if not current_node.child:
                if current_node == target_leaf_value:
                    # Calculate the level based on the path length
                    return len(path)

            # Add children to the stack for further exploration
            for child in current_node.child:
                stack.append((child, path + [current_node]))

        return None

def arrays_equal(arr1, arr2):
    return all(x == y for x, y in zip(arr1, arr2))

def find_common_parent(root, node1, node2):
    if root is None:
        return None

    # If either node1 or node2 has the same value as the current root, then the root is the common parent
    if (root.input_space == node1.input_space).all() or (root.input_space == node2.input_space).all():
        return root

    # Recursively search in the children
    child_results = []
    for child in root.children:
        result = find_common_parent(child, node1, node2)
        if result:
            child_results.append(result)

    # If both nodes are found in different children, the current root is the common parent
    if len(child_results) == 2:
        return root

    # If one of the nodes is found, return that node
    return child_results[0] if child_results else None


def reassignUsingRewardDist(root, routine, agents, agentID, jump_prob):
    subregs = root.find_leaves() 
    rewardStack = []
    subregs = sorted(subregs, key=lambda x: (x.getStatus() == RegionState.INACTIVE.value, x.agentId if x.getStatus() == RegionState.ACTIVE.value else float('inf')))
    for subr in subregs:
        rewardStack.append(np.hstack((subr.rewardDist)))
    minsubreg = np.asarray(rewardStack, dtype="object")
    minsubreg = minsubreg.reshape((len(subregs), 4))
    assert (len(subregs), 4) == (minsubreg.shape[0], minsubreg.shape[1])

    lower_triangular_mask = np.tril(np.ones(minsubreg.shape, dtype=bool))
    # Apply mask to the matrix
    lower_triangular_matrix = np.where(lower_triangular_mask, minsubreg, np.inf)
    
    minsubregIdxAmongAll = np.argmin(lower_triangular_matrix ,axis=0)
    minsubregIdx = minsubregIdxAmongAll

    # Reassign the agent to region with max EI
    currSubreg = agents[agentID] 
    currSubreg.reduceNumAgents()
    currSubreg.removeFromAgentList(currSubreg.agentId)
    
    subregs[minsubregIdx[agentID]].updateStatus(routine.ACTIVE)
    subregs[minsubregIdx[agentID]].increaseNumAgents()
    subregs[minsubregIdx[agentID]].addAgentList(currSubreg.agentId)
    assert len(subregs[minsubregIdx[agentID]].getAgentList()) ==  subregs[minsubregIdx[agentID]].getnumAgents()
    
    return subregs

def partitionRegions(root, globalGP, subregions, routine, dim):
    # Based on reassignments partition the respective sub regions among the agents
    for subr in subregions:
        # If the sub region has more than one agent 
        if subr.getnumAgents() > 1:
            subr.updateStatus(routine.INACTIVE)
            internal_factorized = find_prime_factors(subr.getnumAgents())
            ch = get_subregion(deepcopy(subr), subr.getnumAgents() , internal_factorized, dim)
            subr.add_child(ch)
            assert len(ch) == subr.getnumAgents()

            # assign the agents to the new childs 
            for idx, agent in enumerate(subr.agentList):
                ch[idx].agentId = agent
                ch[idx].samples = subr.samples
                ch[idx].updatesmpObs()

        # If the sub region has no agent 
        elif subr.getnumAgents() == 0:
            subr.updateStatus(routine.INACTIVE.value)
            subr.agentId = None
            assert len(subr.getAgentList()) == 0

        # If the sub region has exactly one agent 
        elif subr.getnumAgents() == 1:
            subr.updateStatus(routine.ACTIVE.value)
            assert len(subr.agentList) == 1
            aidx = subr.getAgentList()
            subr.agentId = aidx[0]
            subr.updatesmpObs()
            
    newSubreg = root.find_leaves()
    for newsub in newSubreg:
        if newsub.getStatus() == routine.ACTIVE.value:
            newsub.addAgentList(newsub.agentId)
    return root


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

def filter_points_in_region(points, values, region):
    # Create a boolean mask indicating which points fall within the region
    mask = np.all(np.logical_and(region[:, 0] <= points, points <= region[:, 1]), axis=1)

    filtered_points = points[mask]
    filtered_values = values[mask]
    
    return filtered_points, filtered_values

def splitsmpObs(region):  #, tf_dim, rng):
    filtered_points, filtered_values = filter_points_in_region(region.smpXtr, region.smpYtr, region.input_space)

    region.smpXtr = filtered_points
    region.smpYtr = filtered_values

    if len(filtered_points) == 0:
        print(' filtered pts in reg empty:',  region.input_space)


def splitObs(agents, tf_dim, rng, routine, tf, behavior):
        
        
        for a in agents:
            if routine == MAIN:
                # print()
                # print('INSIDE MAIN agent id: ',a.id)
                # print()
                # print('^'*100)
                filtered_points, filtered_values = filter_points_in_region(a.x_train, a.y_train, a.region_support.input_space)

                a.x_train = filtered_points
                a.y_train = filtered_values
            
                # if len(filtered_points) == 0:
                #     print('reg and filtered pts len :',  a.region_support.input_space, a.id)

                # x_train = uniform_sampling( 1, a.region_support.input_space, tf_dim, rng)
                # y_train, falsified = compute_robustness(x_train, tf, behavior, agent_sample=True)
            
                # a.x_train = np.vstack((filtered_points, x_train))
                # a.y_train = np.hstack((filtered_values, y_train))

                # a.updateModel()
            
            elif routine == ACTUAL:
                print('agent id: actual',a.id)
                filtered_points, filtered_values = filter_points_in_region(a.ActualXtrain, a.ActualYtrain, a.region_support.input_space)

            
                if len(filtered_points) == 0:
                    print('reg and filtered pts len in Actual:',  a.region_support.input_space, a.id)

                x_train = uniform_sampling( 1, a.region_support.input_space, tf_dim, rng)
                y_train, falsified = compute_robustness(x_train, tf, behavior, agent_sample=True)
            
                a.ActualXtrain = np.vstack((filtered_points, x_train))
                a.ActualYtrain = np.hstack((filtered_values, y_train))

                a.updateModel()
            else:
                filtered_points, filtered_values = filter_points_in_region(a.simXtrain, a.simYtrain, a.simReg.input_space)
                a.simXtrain = filtered_points
                a.simYtrain = filtered_values

            # if routine == MAIN:
                
            # else:
            #     a.updatesimModel()
        return agents
    

def check_points(agent, routine):
    if routine == MAIN:
        xtr = agent.x_train
        reg = agent.region_support.input_space
    else:
        xtr = agent.simXtrain
        reg = agent.simReg.input_space

    def point_in_region(points, region):
        # Check if each coordinate of the point is within the corresponding bounds of the region
        res = True
        for point in points:
            res = res and all(np.logical_and(region[:, 0] <= point, point <= region[:, 1]))
            # if res == False:
                # print('pt not in region :',agent.id, point, region)
        return res

    res = point_in_region(xtr, reg)
    return res

def find_parent(root, target):
    # Stack for DFS traversal
    stack = [(root, [])]

    while stack:
        node, path = stack.pop()
        if node == target:
            # If the target node is found, return the last node in the path
            return path[-1] if path else None
        for child in node.children:
            stack.append((child, path + [node]))

    return None

def find_min_diagonal_sum_matrix(matrix):
    n, _, _ = matrix.shape
    min_sum = float('inf')
    min_matrix = None
    
    for i in range(n):
        sums = np.trace(matrix[i])
        if min_sum > sums:
            min_sum = sums
            min_matrix = i

    return min_matrix


def genSamplesForConfigs(m, globalGP, num_agents, roots, init_sampling_type, tf_dim, tf, behavior, rng):
        avgrewards = np.zeros((1,num_agents,num_agents))
        agentModels = []
        xroots = []

        print("total comb of roots with assign and dim: ",len(roots))
        
        # Permute the different possible agent assignments 
        permutations_list = list(permutations(range(m+1,num_agents)))
        permToeval = configs['configs']['perm']
        permutations_list = permutations_list[:permToeval]
        for perm in range(len(permutations_list)):
            for Xs_root in roots:
                agents = []
                nid = 0
                allxtr = np.empty((0,tf_dim))
                allytr = np.empty((0))
                allsmp = Observations(allxtr , allytr)
                
                for id, l in enumerate(Xs_root.find_leaves()):
                    nid = nid % len(range(m+1,num_agents))
                    if l.getStatus() == RegionState.ACTIVE.value and l.agentId > m:
                            
                            l.samples = allsmp
                            l.agentId = permutations_list[perm][nid]
                            l.agentList = []
                            l.addAgentList(l.agentId)                       
                            
                            nid += 1
                            agents.append(l)
                
                
                agents = sorted(agents, key=lambda x: x.agentId)
                for a in agents:
                    xtsize = int((tf_dim*2)/num_agents) - len(a.obsIndices)
                    # If the observation to build a model is empty, then sample points to build local GP
                    if len(a.obsIndices) == 0: #xtsize > 0: 

                        x_train = lhs_sampling( xtsize, a.input_space, tf_dim, rng)
                        y_train, falsified = compute_robustness(x_train, tf, behavior, agent_sample=True)

                        globalGP.dataset = globalGP.dataset.appendSamples(x_train, y_train)
                    # Build the respective local GPs 
                    localGP = Prior(globalGP.dataset, a.input_space)
                    model, indices = localGP.buildModel()
                    a.updateModel(indices, model)
                    # Sample points to evaluate using Rollout
                    xtr, ytr = initAgents(a.model, a.input_space, init_sampling_type, tf_dim*10, tf_dim, tf, behavior, rng, store=True)
                    a.samples.appendSamples(xtr, ytr) 
                    a.updatesmpObs()
                

                agentModels.append(deepcopy(agents))
                xroots.append(deepcopy(Xs_root))
        return xroots, agentModels, globalGP


def initAgents(globmodel,region_support, init_sampling_type, init_budget, tf_dim, tf, behavior, rng, store):
        if init_sampling_type == "lhs_sampling":
            x_train = lhs_sampling(init_budget, region_support, tf_dim, rng)
        elif init_sampling_type == "uniform_sampling":
            x_train = uniform_sampling(init_budget, region_support, tf_dim, rng)
        else:
            raise ValueError(f"{init_sampling_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")
        
        mu, std = globmodel.predict( x_train)  
        actY = []
        for i in range(len(x_train)):
            f_xt = np.random.normal(mu[i],std[i],1)
            actY.append(f_xt)
        actY = np.hstack((actY))
        y_train = actY

        return x_train , y_train #actY

def delete_tree(node):
    if node is None:
        return
    for child in node.children:
        delete_tree(child)
    del node


# Function to save the state of the tree nodes
def saveRegionState(node):
    # Save the node status as a dict in each node
    node.saved_state = deepcopy(node.__dict__)
    del node.saved_state['saved_state']
    if node.children:
        for child in node.children:
            saveRegionState(child)

    return node

def copyRegionAttributes(root, node):
    parent = find_parent(root, node)
    if node.saved_state != None:
        state = node.__dict__.copy()
        for key, value in state.items():
            setattr(node, key, value)
    
    return node

# Function to restore the state of the tree nodes 
def restoreRegionState(node, keeplist):
    if node.saved_state != None:
        state = node.saved_state
        for key, value in state.items():
            if key not in keeplist:
                setattr(node, key, value)
    if node.children:
        for child in node.children:
            restoreRegionState(child, keeplist)
    return node

# Function to consider different possible splits of a region
def getRootConfigs(m, X_root, globalGP, sample, num_agents, tf_dim):

        roots = []
        for dim in range(tf_dim):
            # Split the region among 'm' agents and build local GPs
            if sample == 0: 
                print(dim)
                root = deepcopy(X_root)
                factorized = find_prime_factors(num_agents) 
                agents_to_subregion = get_subregion(deepcopy(root), num_agents, factorized, dim)
                root.add_child(agents_to_subregion)
                for id, l in enumerate(root.find_leaves()):
                    if l.getStatus() == RegionState.ACTIVE.value:
                        # l.agentID = id
                        localGP = Prior(globalGP.dataset, l.input_space)
                        # l.samples = root.samples
                        
                        l.model , l.obsIndices = localGP.buildModel()

                        
                roots.append(root)
                
            else:
                # Partition the sub region among 'm' agents and build local GPs
                jump = random.random()
                root = deepcopy(X_root)
                agents =[]
                for id, l in enumerate(root.find_leaves()):
                    if l.getStatus() == RegionState.ACTIVE.value:
                        # l.rewardDist = l.avgRewardDist.reshape((num_agents)) #?
                        agents.append(l)
                
                # Reassign and partition
                subregions = reassignUsingRewardDist(root, RegionState, agents, m, jump_prob=jump)
                root = partitionRegions(root, globalGP, subregions, RegionState, dim)
                print('after moving and partitioning ')
                # print_tree(root)
                
                # Check inactive regions  
                for a in agents:
                    # a.resetRegions()
                    if a.getStatus() == RegionState.INACTIVE.value:
                        a.agentList = []
                    a.resetavgRewardDist(num_agents) #?
                        
                roots.append(root)
            
        print('roots after dim: ', roots)
        for i in roots:
            print_tree(i)

        return roots
    
# @profile
def genSamplesForConfigsinParallel(m, globalGP, configSamples, num_agents, roots, init_sampling_type, tf_dim, tf, behavior, rng):
    # Define a helper function to be executed in parallel
    def genSamples_in_parallel(num_agents, roots, init_sampling_type, tf_dim, tf, behavior, rng):
        return genSamplesForConfigs(m, globalGP, num_agents, roots, init_sampling_type, tf_dim, tf, behavior, rng)

    # Execute the evaluation function in parallel for each Xs_root item
    results = Parallel(n_jobs=-1)(delayed(genSamples_in_parallel)(num_agents, roots, init_sampling_type, tf_dim, tf, behavior, np.random.default_rng(csmp+1)) for csmp in tqdm(range(configSamples)))
    
    roots = [results[i][0] for i in range(configSamples)]
    agents = [results[i][1] for i in range(configSamples)]
    globalGP = results[0][2]

    return roots , agents, globalGP

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

# reg = np.array([[0,1],[0,1],[0,1]]) #,[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]])
# n = Node(reg,1)
# a = 4
# dim = 0 #np.random.randint(len(reg))
# factorized = find_prime_factors(a) #sorted(find_close_factor_pairs(a), reverse=True)
# print(factorized, 'np.random.randint(len(reg)): ',dim)
# ch = get_subregion(deepcopy(n), a,factorized, dim)
# print(len(ch))
# n.add_child(ch)

# print_tree(n, MAIN)

# print('[i.getVolume() for i in ch]: ',[{str(i.input_space ): i.getVolume()} for i in ch])

# v=0
# for i in ch:
#     v += i.getVolume()

# assert n.getVolume() == v
# print('set([i.getVolume() for i in ch]) :', set([i.getVolume() for i in ch]))
# print_tree(n, MAIN)

