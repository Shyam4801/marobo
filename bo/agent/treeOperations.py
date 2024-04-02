from .agent import Agent
from copy import deepcopy
from .constants import *
import numpy as np
from .partition import Node
from bo.sampling import uniform_sampling, lhs_sampling
from bo.utils import compute_robustness
from itertools import permutations
import yaml, pickle
from collections import defaultdict
# from bo.bayesianOptimization.rolloutEI import RolloutEI
from joblib import wrap_non_picklable_objects
import random

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

    regs = [Node(i, 1) for i in regs]
    return regs
    
def get_subregion(root, num_agents, dic, dim):
    dic = sorted(dic, reverse=True)
    q = [root]
    while len(q) < num_agents:
        for c in range(len(dic)):
                if num_agents - len(q) >= dic[c]-1:
                    cuts = c
                    break
        dim =  np.random.randint(len(root.input_space)) #(dim + 1) % len(root.input_space) #
        curr = q.pop(0)
        ch = split_region(curr, dim, dic[cuts])
        curr.add_child(ch)
        q.extend(ch)
    return q

# def get_subregion(root, num_agents, dic, dim):
#     q = [root]
#     while len(q) < num_agents:
#         cuts = dim % len(dic) 
#         if len(q) % dic[cuts] == 0:
#             dim =  np.random.randint(len(root.input_space)) #(dim + 1) % len(root.input_space) #
#         curr = q.pop(0)
        
#         # print('dim :', dim,'q: ', [i.input_space for i in q])
#         ch = split_region(curr, dim, dic[cuts])
#         curr.add_child(ch)
#         q.extend(ch)
#         # if len(q) % dic[cuts] == 0:
#         #     dim = (dim + 1) % len(root.input_space)
#     return q

def print_tree(node, routine, level=0, prefix=''):
    # print('node.getStatus(routine) :',node.getStatus(routine))
    if routine == MAIN:
        reward = node.avgRewardDist
    else:
        reward = node.rewardDist
    if node.getStatus(routine) == 1:
        color = GREEN
    else:
        color = RED
    if node is None:
        return
    id = -1
    if node.agent != None:
        id = node.agent.id

    for i, child in enumerate(node.child):
        print_tree(child, routine, level + 1, '|   ' + prefix if i < len(node.child) - 1 else '    ' + prefix)
    
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
def remove_duplicates(lst):
    seen = set()
    unique = []

    for sublst in lst:
        subtuple = tuple(sublst)
        if subtuple not in seen:
            unique.append(sublst)
            seen.add(subtuple)

    return unique

def find_factors(num_sub_regions):
    factors = []
    for i in range(1, num_sub_regions + 1):
        if num_sub_regions % i == 0:
            factors.append([i, num_sub_regions // i])
            if i != 1 and i != num_sub_regions:
                factors.append([num_sub_regions // i, i])
    
    factors = remove_duplicates(factors)
    return factors[1]

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
    if not node.child:
        return node.rewardDist

    # Initialize the accumulated reward for this node
    accumulated_reward = node.rewardDist

    # Recursively accumulate rewards from child nodes and update node.value
    for child in node.child:
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
    node.rewardDist = accumulated_reward

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
            dictionary[tuple(row)] += value
        else:
            dictionary[tuple(row)] = value
            
    keys = np.array(list(dictionary.keys()))
    values = np.array(list(dictionary.values()))


    return keys, values #accumulated_reward, accy


# def accumulateSamples(node):
#     # Base case: if it's a leaf node, return its X and Y
#     if not node.child:
#         return node.smpXtr, node.smpYtr.reshape(-1, 1)  # Reshape Y to make it 2D
    
#     # Initialize dictionaries to store accumulated Y and counts for each X value
#     aggregated_Y = defaultdict(list)
#     counts = defaultdict(int)
    
#     # Accumulate arrays from children
#     for child in node.child:
#         child_X, child_Y = accumulateSamples(child)
#         for x, y in zip(child_X, child_Y):
#             x_key = tuple(x.tolist()) if isinstance(x, np.ndarray) else x
#             aggregated_Y[x_key].append(y)
#             counts[x_key] += 1
    
#     # Aggregate duplicate values and calculate mean of corresponding Y values
#     aggregated_X = []
#     mean_Y = []
#     for x_key, y_values in aggregated_Y.items():
#         aggregated_X.append(x_key)
#         mean_Y.append(np.mean(y_values, axis=0))
#     print('aggregated_X : ',aggregated_X, mean_Y)
    
#     # Include the current node's X and Y values
#     aggregated_X.extend(node.smpXtr)
#     mean_Y.extend(node.smpYtr)
#     print('aggregated_X fter: ',aggregated_X, mean_Y)
#     mean_Y = np.asarray(mean_Y, dtype='object')
#     mean_Y = np.hstack((mean_Y))
#     if mean_Y.shape[0] > 1:
#         mean_Y = mean_Y.reshape((-1))
#     # mean_Y = np.hstack((mean_Y))
#     return np.asarray(aggregated_X), mean_Y

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
    for child in root.child:
        result = find_common_parent(child, node1, node2)
        if result:
            child_results.append(result)

    # If both nodes are found in different children, the current root is the common parent
    if len(child_results) == 2:
        return root

    # If one of the nodes is found, return that node
    return child_results[0] if child_results else None

def reassignPerAgentUsingRewardDist(currentAgentIdx, root, routine, agents):
    subregs = root.find_leaves() 
    
    rewardStack = []
    subregs = sorted(subregs, key=lambda x: (x.getStatus(routine) == 0, x.agent.id if x.getStatus(routine) == 1 else None))
    for subr in subregs:
        if routine == MAIN:
            # if subr.agent != None:
            #     print('subr.agent.id: ',subr.agent.id, 'status :',subr.getStatus(routine))
            # print('subr.avgRewardDist: ', subr.avgRewardDist, subr.input_space)
            rewardStack.append(np.hstack((subr.avgRewardDist)))
        else:
            rewardStack.append(np.hstack((subr.rewardDist)))
        # if routine == MAIN:
        #     print(agent)
    # print('rewardStack: ',rewardStack)
    minsubreg = np.asarray(rewardStack, dtype="object")
    minsubreg = minsubreg.reshape((len(subregs), 4))
    # if routine == MAIN:
        # print('minsubreg: nx4 arr: ',minsubreg, minsubreg.shape)
    assert (len(subregs), 4) == (minsubreg.shape[0], minsubreg.shape[1])
    minsubregIdxAmongAll = np.argmin(minsubreg ,axis=0)
    minsubregIdxAmongAgents = np.argmin(minsubreg[:len(agents)], axis=0)
    
        
    for idx, a in enumerate(agents[currentAgentIdx:]):
        idx += currentAgentIdx
        a = agents[idx]
        # idx = a.id
        if idx == 0:
            minsubregIdx = minsubregIdxAmongAll
        else:
            minsubregIdx = minsubregIdxAmongAgents
        # if routine == MAIN:
        #     print('--------------------------------------')
        #     print('minsubregIdx: ',minsubregIdx, f'of agent {a.id}')
        #     print('--------------------------------------')
        # deactivate curr subreg
        currSubreg = a.getRegion(routine)
        # print('currSubreg: ',currSubreg.input_space)#, 'len(currSubreg.getAgentList(MAIN, Rollout)): ',len(currSubreg.getAgentList(MAIN)), len(currSubreg.getAgentList(ROLLOUT)))
        # print([{i: i.getRegion(MAIN).input_space} for i in currSubreg.getAgentList(MAIN)])
        # print('--------------------------------------')
        currSubreg.reduceNumAgents(routine)
        # print('currSubreg agentList region: b4 removal ',len(currSubreg.getAgentList(routine)) ) #currSubreg.getAgentList(routine)[0].getRegion(routine).input_space)
        currSubreg.removeFromAgentList(a, routine)
        # print('--------------------------------------')
        # if currSubreg.numAgents > 
        # currSubreg.agent
        # a(routine)
        # activate new sub reg
        # subregs = subregs[::-1]
        subregs[minsubregIdx[idx]].updateStatus(1, routine)
        subregs[minsubregIdx[idx]].increaseNumAgents(routine)
        subregs[minsubregIdx[idx]].addAgentList(a, routine)
        # print('subregs[minsubregIdx[idx]]: ',subregs[minsubregIdx[idx]].input_space)
        # if routine == MAIN:
        # print('--------------------------------------')
        # print(subregs[minsubregIdx[idx]].input_space)
        # print('len subregs[minsubregIdx[idx]] agentList: after appending ',len(subregs[minsubregIdx[idx]].agentList), subregs[minsubregIdx[idx]].getnumAgents(routine))
        assert len(subregs[minsubregIdx[idx]].getAgentList(routine)) ==  subregs[minsubregIdx[idx]].getnumAgents(routine)
    # print('num agents : ',[i.getnumAgents(routine) for i in subregs])
    # print('len agent list : ',[len(i.getAgentList(routine)) for i in subregs])
    # print('--------------------------------------')
    # partitionRegions(subregs)
    return subregs

def reassignUsingRewardDist(root, routine, agents, jump_prob):
    subregs = root.find_leaves() 
    numagents = configs['agents']
    
    rewardStack = []
    subregs = sorted(subregs, key=lambda x: (x.getStatus(routine) == 0, x.agent.id if x.getStatus(routine) == 1 else None))
    for subr in subregs:
        if routine == MAIN:
            # if subr.agent != None:
            #     print('subr.agent.id: ',subr.agent.id, 'status :',subr.getStatus(routine))
            # print('subr.avgRewardDist: ', subr.avgRewardDist, subr.input_space)
            rewardStack.append(np.hstack((subr.avgRewardDist)))
        else:
            rewardStack.append(np.hstack((subr.rewardDist)))
        # if routine == MAIN:
        #     print(agent)
    # print('rewardStack: ',rewardStack)
    minsubreg = np.asarray(rewardStack, dtype="object")
    minsubreg = minsubreg.reshape((len(subregs), numagents))
    # if routine == MAIN:
        # print('minsubreg: nx4 arr: ',minsubreg, minsubreg.shape)
    
    # print('level: ',level)
    # sorted_indices = sorted(enumerate(minsubreg), key=lambda x: x[1])
    # # Extract the sorted indices
    # sorted_indices_only = [index for index, value in sorted_indices]
    assert (len(subregs), numagents) == (minsubreg.shape[0], minsubreg.shape[1])

    agent0 = minsubreg[:,0]
    
    minsubregIdxAmongAll = np.argmin(minsubreg ,axis=0)
    minsubregIdxAmongAgents = np.argmin(minsubreg[:len(agents)], axis=0)

    
    # if minsubregIdxAmongAll[0] >= 4:
    #     level = [find_level_of_leaf(root, n) for n in subregs]
    #     # print('^'*100,'agent 0 ', agent0)
    #     # print('minsubregIdxAmongAll b4: ',minsubregIdxAmongAll)
    #     curragentlevel = find_level_of_leaf(root, subregs[0])
    #     leveltolook = 1
    #     filtered_indices = np.where(np.array(level) >= curragentlevel- leveltolook)[0]
    #     # print('filtered_indices: ',filtered_indices)
    #     # Sort the filtered elements of array a
    #     sorted_indices = np.argsort(agent0[filtered_indices])
    #     # print('sorted_indices: ',sorted_indices)
    #     minsubregIdxAmongAll[0] = filtered_indices[sorted_indices[0]]
    #     # print('minsubregIdxAmongAll after getting leaf in nearest level: ',minsubregIdxAmongAll)
    #     # print('^'*100)

    for idx, a in enumerate(agents):
        # idx = a.id
        if idx == 0:# and jump_prob > 0.8:
            minsubregIdx = minsubregIdxAmongAll
        else:
            minsubregIdx = minsubregIdxAmongAgents
        # if routine == MAIN:
        #     print('--------------------------------------')
        #     print('minsubregIdx: ',minsubregIdx, f'of agent {a.id}')
        #     print('--------------------------------------')
        # deactivate curr subreg
        currSubreg = a.getRegion(routine)
        # print('currSubreg: ',currSubreg.input_space)#, 'len(currSubreg.getAgentList(MAIN, Rollout)): ',len(currSubreg.getAgentList(MAIN)), len(currSubreg.getAgentList(ROLLOUT)))
        # print([{i: i.getRegion(MAIN).input_space} for i in currSubreg.getAgentList(MAIN)])
        # if routine == MAIN:
        #     print('-------------------------------------- inside reassign agent id', currSubreg.agent.id)
        currSubreg.reduceNumAgents(routine)
        # print('currSubreg agentList region: b4 removal ',len(currSubreg.getAgentList(routine)) ) #currSubreg.getAgentList(routine)[0].getRegion(routine).input_space)
        currSubreg.removeFromAgentList(a, routine)
        # print('--------------------------------------')
        # if currSubreg.numAgents > 
        # currSubreg.agent
        # a(routine)
        # activate new sub reg
        # subregs = subregs[::-1]
        subregs[minsubregIdx[idx]].updateStatus(1, routine)
        subregs[minsubregIdx[idx]].increaseNumAgents(routine)
        subregs[minsubregIdx[idx]].addAgentList(a, routine)
        # print('subregs[minsubregIdx[idx]]: ',subregs[minsubregIdx[idx]].input_space)
        # if routine == MAIN:
        # print('--------------------------------------')
        # print(subregs[minsubregIdx[idx]].input_space)
        # print_tree(root, ROLLOUT)
        # print('len subregs[minsubregIdx[idx]] agentList: after appending ',len(subregs[minsubregIdx[idx]].agentList),[(a, a.id) for a in subregs[minsubregIdx[idx]].agentList],  subregs[minsubregIdx[idx]].getnumAgents(routine))
        assert len(subregs[minsubregIdx[idx]].getAgentList(routine)) ==  subregs[minsubregIdx[idx]].getnumAgents(routine)
    # print('num agents : ',[i.getnumAgents(routine) for i in subregs])
    # print('len agent list : ',[len(i.getAgentList(routine)) for i in subregs])
    # print('--------------------------------------')
    # partitionRegions(subregs)
    return subregs
        # minsubreg[idx]()
        # update agents new sub reg 
        # a.updateBounds(subregs[minsubregIdx[idx]], routine)
        # a(routine)

def partitionRegions(root, subregions, routine, dim):
    # print('===================================================')
    # print('================= Paritioning =================')
    for subr in subregions:
        if subr.getnumAgents(routine) > 1:
            subr.updateStatus(0, routine)
            internal_factorized = find_prime_factors(subr.getnumAgents(routine))
            ch = get_subregion(deepcopy(subr), subr.getnumAgents(routine) , internal_factorized, dim)
            subr.add_child(ch)
            # print('len(ch),subr.getnumAgents(routine) : ',internal_factorized,len(ch),subr.getnumAgents(routine), subr.input_space)
            assert len(ch) == subr.getnumAgents(routine)

            # assign the agents to the new childs 
            for idx, agent in enumerate(subr.agentList):
                agent.updateBounds(ch[idx], routine)
                #assign self
                agent(routine)
                agent.getRegion(routine).addAgentList(agent, routine)
                if routine == MAIN:
                    rt = 'MAIN'
                else:
                    rt = 'ROLLOUT'
                # print('ch :',ch[idx].input_space)
                # print(f'b4 upadting from parent {rt}',agent.id, [agent.x_train if routine == MAIN else agent.simXtrain], [agent.y_train if routine == MAIN else agent.simYtrain])
                agent.updateObs(subr, routine)
                ch[idx].updatesmpObs(subr)
                splitsmpObs(ch[idx])
                # print(f'after upadting from parent {rt}',agent.id, [agent.x_train if routine == MAIN else agent.simXtrain], [agent.y_train if routine == MAIN else agent.simYtrain])
                # print('agent.getRegion(routine): ',agent.getRegion(routine))
                # print('-------------------more than 1 agent-------------------')

        elif subr.getnumAgents(routine) == 0:
            subr.updateStatus(0, routine)
            # subr.agent(routine)
            assert len(subr.getAgentList(routine)) == 0

        elif subr.getnumAgents(routine) == 1:
            assert len(subr.agentList) == 1
            alist = subr.getAgentList(routine)
            alist[0].updateBounds(subr, routine)
            alist[0](routine)
            # print('subr.agentList[0].getRegion(routine).input_space, subr.input_space : ',subr.agentList[0].getRegion(routine).input_space, subr.input_space)
            assert alist[0].getRegion(routine) == subr
            # if routine == MAIN:
            # print(f'b4 upadting from parent {routine}',[alist[0].x_train if routine == MAIN else alist[0].simXtrain], [alist[0].x_train if routine == MAIN else alist[0].simXtrain])
            # alist[0].updateObs(subr, routine)
            alist[0].updateObsFromRegion(subr, routine)
            # print(f'after upadting from parent {routine}',[alist[0].x_train if routine == MAIN else alist[0].simXtrain], [alist[0].x_train if routine == MAIN else alist[0].simXtrain])
            # print('----------------exactly 1 agent ----------------------')
    
    newSubreg = root.find_leaves()
    newAgents = []
    for newsub in newSubreg:
        if routine == MAIN:
            newsub.setRoutine(routine)
        if newsub.getStatus(routine) == 1:
            # print('agent in new active region : ', newsub.input_space, newsub.agentList)
            # print('--------------------------------------')
            assert len(newsub.getAgentList(routine)) == 1
            newAgents.append(newsub.getAgentList(routine)[0])
            # if routine == MAIN:
            #     newsub.addFootprint(newsub.agent.x_train, newsub.agent.y_train, newsub.agent.model)
            # else:
            #     newsub.addFootprint(newsub.agent.simXtrain, newsub.agent.simYtrain, newsub.agent.simModel)
        # if routine == MAIN:
            # print('new subr : ', newsub.input_space, 'agent : ', newsub.agent.id)

    newAgents = sorted(newAgents, key=lambda x: x.id)
    return newAgents






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

def filter_points_in_region(points, values, region):
            # Create a boolean mask indicating which points fall within the region
            mask = np.all(np.logical_and(region[:, 0] <= points, points <= region[:, 1]), axis=1)

            # Filter the points and corresponding values based on the mask
            # if routine == MAIN:
            #     print('idx err caught :', points, region, values)
            filtered_points = points[mask]
            # try:
            filtered_values = values[mask]
            
            # except IndexError:
            #     print('idx err caught :', points, region, filtered_points, filtered_values)

            return filtered_points, filtered_values

def splitsmpObs(region):  #, tf_dim, rng):
    filtered_points, filtered_values = filter_points_in_region(region.smpXtr, region.smpYtr, region.input_space)

    region.smpXtr = filtered_points
    region.smpYtr = filtered_values

    # if len(filtered_points) == 0:
        # print(' filtered pts in reg empty:',  region.input_space)

        # actregSamples = lhs_sampling(tf_dim*10 , region.input_space, tf_dim, rng)  #self.tf_dim*10
        # mu, std = self._surrogate(agent.simModel, actregSamples)  #agent.simModel
        # actY = []
        # for i in range(len(actregSamples)):
        #     f_xt = np.random.normal(mu[i],std[i],1)
        #     actY.append(f_xt)
        # actY = np.hstack((actY))
        # # # print('act Y ',actY)
        # agent.simXtrain = np.vstack((agent.simXtrain , actregSamples))
        # agent.simYtrain = np.hstack((agent.simYtrain, actY))

        # x_train = uniform_sampling( 1, a.region_support.input_space, tf_dim, rng)
        # y_train, falsified = compute_robustness(x_train, tf, behavior, agent_sample=True)

        # a.ActualXtrain = np.vstack((filtered_points, x_train))
        # a.ActualYtrain = np.hstack((filtered_values, y_train))


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
        for child in node.child:
            stack.append((child, path + [node]))

    return None

def genSamplesForConfigs(ei, perm, permutations_list, num_agents, roots, init_sampling_type, tf_dim, tf, behavior, rng):
        avgrewards = np.zeros((1,num_agents,num_agents))
        agentModels = []
        xroots = []

                
        print("total comb of roots with assign and dim: ",len(roots))
        # print()
        # print([obj.__dict__ for obj in roots])
        # print()
        # permutations_list = genPerm(num_agents*2) #list(permutations(range(num_agents)))
        
        for Xs_root in roots:
            # Xs_root = deepcopy(X_root) #Node(self.region_support, 1)
            # rtPrior = Prior(x_train, y_train, model, MAIN)
            # Xs_root.addFootprint(rtPrior, MAIN)
            # _,_, model = Xs_root.mainPrior.getData(MAIN)
            # Xs_root.model = deepcopy(model)
            # state = np.random.get_state()

            # # Print the seed value
            # print("Seed value:", state[1][0])
            agents = []
            # xtr, ytr = self.initAgents(model, region_support, init_sampling_type, tf_dim*5, tf_dim, rng, store=True)
            nid = 0
            totalVolume = 0
            for id, l in enumerate(Xs_root.find_leaves()):
                l.setRoutine(MAIN)
                nid = nid % num_agents
                if l.getStatus(MAIN) == 1:
                    totalVolume += l.getVolume()
                    # if l.agent.model == None:
                    #     parent = find_parent(Xs_root, l)
                    #     model = parent.model
                    # else:
                    #     model = l.agent.model
                    
                    # xtr, ytr = initAgents(l.agent.model, l.input_space, init_sampling_type, tf_dim*5, tf_dim, tf, behavior, rng, store=True)
                    # print(idx, id,i, len(permutations_list))
                    # mainag = Agent(permutations_list[perm][nid], None, xtr, ytr, l)
                    # mainag(MAIN)
                    mainag = l.agent
                    l.addAgentList(mainag, MAIN)
                    # mainag.x_train = xtr #np.vstack((mainag.x_train, xtr))
                    # mainag.y_train = ytr #np.hstack((mainag.y_train, ytr))
                    mainag.id = permutations_list[perm][nid]
                    
                    nid += 1
                    agents.append(mainag)
            # moreRoots.append(deepcopy(rt))
            
                # l.setRoutine(MAIN)
                # if l.getStatus(MAIN) == 1:
                #     # if sample != 0:
                #     xtr, ytr = initAgents(l.agent.model, l.input_space, init_sampling_type, tf_dim*5, tf_dim, tf, behavior, rng, store=True)
                    # print(f'agent xtr ', l.agent.x_train, l.agent.y_train, l.agent.id, l.input_space)
                    # print(f'agent actual xtr ', l.agent.ActualXtrain, l.agent.ActualYtrain, l.agent.id, l.input_space)
                # else:
                #     print(f'inactive xtr ', l.agent.x_train, l.agent.y_train, l.agent.id, l.input_space)
                #     print(f'agent xtr ', l.agent.ActualXtrain, l.agent.ActualYtrain, l.agent.id, l.input_space)

                #     ag = l.agent
                #     ag.x_train = xtr #np.vstack((ag.x_train, xtr))
                #     ag.y_train = ytr #np.hstack((ag.y_train, ytr))
                #     # ag = Agent(id, None, xtr, ytr, l)
                #     # ag.updateModel()
                #     ag(MAIN)
                #     agents.append(ag)
            minytrval = float('inf')
            minytr = []
            for ix, a in enumerate(agents):
                for ia in agents:
                    if min(ia.y_train) < minytrval:
                        minytrval = min(ia.y_train)
                        minytr = ia.y_train

            agents = sorted(agents, key=lambda x: x.id)
            agents = splitObs(agents, tf_dim, rng, MAIN, tf, behavior)
            for a in agents:
                # if sample == 0:
                #     a.ActualXtrain == 
                peragent = int((tf_dim*10)/(num_agents * 0.5))
                xtsize = peragent - len(a.x_train)
                # if a.region_support.getVolume() 
                if len(a.x_train) == 0 :#xtsize > 0: 
                    # print('reg and filtered pts len in Actual:',  a.region_support.input_space, a.id)

                    x_train = lhs_sampling( 5, a.region_support.input_space, tf_dim, rng)
                    y_train, falsified = compute_robustness(x_train, tf, behavior, agent_sample=True)
                
                    a.x_train = np.vstack((a.x_train, x_train))
                    a.y_train = np.hstack((a.y_train, y_train))

                    # a.updateModel()
                assert check_points(a, MAIN) == True
                a.updateModel()
                a.resetModel()
                a.region_support.addFootprint(a.x_train, a.y_train, a.model)
                
                assert check_points(a, ROLLOUT) == True
                # print(f'agent xtr config sample', Xs_root, a.x_train, a.y_train, a.id, a.region_support.input_space)

                xtr, ytr = initAgents(a.model, a.region_support.input_space, init_sampling_type, tf_dim*20, tf_dim, tf, behavior, rng, store=True)

                x_opt = ei._opt_acquisition(minytr, a.model, a.region_support.input_space, rng)
                x_opt = np.asarray([x_opt])
                mu, std = ei._surrogate(a.model, x_opt)
                f_xt = np.random.normal(mu,std,1)
                xtr = np.vstack((xtr , x_opt))
                ytr = np.hstack((ytr, f_xt))
                print('ei pt to eval : ',x_opt, f_xt)

                # if np.size(a.region_support.smpXtr) == 0:
                #     a.region_support.smpXtr = np.vstack((a.region_support.smpXtr, xtr))
                #     a.region_support.smpYtr = np.hstack((a.region_support.smpYtr, ytr))
                #     a.simReg.smpXtr = np.vstack((a.simReg.smpXtr, xtr))
                #     a.simReg.smpYtr = np.hstack((a.simReg.smpYtr, ytr))
                # else:
                a.region_support.smpXtr = xtr
                a.region_support.smpYtr = ytr
                a.simReg.smpXtr = xtr
                a.simReg.smpYtr = ytr
                assert a.region_support.check_points() == True
                assert a.simReg.check_points() == True
            

            # print(f'_________________  ____________________')
            # model = GPR(gpr_model)
            # model.fit(globalXtrain, globalYtrain)
            agentModels.append(deepcopy(agents))
            # srt = pickle.dump(deepcopy(Xs_root))
            xroots.append(deepcopy(Xs_root))
                # exit(1)
        return xroots, agentModels


def initAgents(globmodel,region_support, init_sampling_type, init_budget, tf_dim, tf, behavior, rng, store):
        # state = np.random.get_state()

        # # Print the seed value
        # print("Seed value:", state[1][0])
        if init_sampling_type == "lhs_sampling":
            x_train = lhs_sampling(init_budget, region_support, tf_dim, rng)
        elif init_sampling_type == "uniform_sampling":
            x_train = uniform_sampling(init_budget, region_support, tf_dim, rng)
        else:
            raise ValueError(f"{init_sampling_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")
        
        # y_train, falsified = compute_robustness(x_train, tf, behavior, agent_sample=store)
        # if not falsified:
        #     print("No falsification in Initial Samples. Performing BO now")
        # ei = RolloutEI()
        mu, std = globmodel.predict( x_train)  #agent.simModel
        actY = []
        for i in range(len(x_train)):
            f_xt = np.random.normal(mu[i],std[i],1)
            actY.append(f_xt)
        actY = np.hstack((actY))
        y_train = actY

        return x_train , y_train #actY

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


# @wrap_non_picklable_objects
def evaluate_in_parallel(ei_roll, Xs_root_item, sample, num_agents, globalXtrain, globalYtrain, region_support, model, horizon, rng):
            # print('Xs_root_item in eval config : ',Xs_root_item)
            print(f'Task {Xs_root_item}', flush=True)
            agents = []
            return ei_roll.sample(sample, Xs_root_item, agents, num_agents, globalXtrain, horizon, globalYtrain, region_support, model, rng)


def evalWrapper(args):
        return evaluate_in_parallel(*args)


# def evaluate_in_parallel(Xs_root_item, sample, num_agents, globalXtrain, globalYtrain, region_support, model, rng):
#             ei_roll = RolloutEI()
#             # print('Xs_root_item in eval config : ',Xs_root_item)
#             agents = []
#             return ei_roll.sample(sample, Xs_root_item, agents, num_agents, globalXtrain, self.horizon, globalYtrain, region_support, model, rng)

# def evaluate_in_parallel( arg):
#             print(arg)
#             ei_roll, Xs_root_item, sample, num_agents, globalXtrain, horizon, globalYtrain, region_support, model, rng = arg
#             # ei_roll = arg[0]
#             # Xs_root_item = pickle.load(Xs_root_item)
#             # print('Xs_root_item in eval config : ',Xs_root_item)
#             agents = []
#             return ei_roll.sample(sample, Xs_root_item, agents, num_agents, globalXtrain, horizon, globalYtrain, region_support, model, rng)
