import numpy as np
import random

RED = '\033[91m'
GREEN = '\033[92m'
END = '\033[0m'

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
            self.child.append(c)
    
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
    # RED = '\033[91m'
    # GREEN = '\033[92m'
    # END = '\033[0m'
    if node.status:
        color = GREEN
    else:
        color = RED
    if node is None:
        return

    for i, child in enumerate(node.child):
        print_tree(child, level + 1, '|   ' + prefix if i < len(node.child) - 1 else '    ' + prefix)
    
    print('    ' * level + prefix + f'-- {color}{node.input_space.flatten()}{END}')



    # def split_space(self,input_space,num_agents, tf_dim):
    #     reg = np.zeros((num_agents, tf_dim), dtype=np.int64)
    #     for dim in input_space:
    #         region = np.linspace(dim[0], dim[1], num = num_agents+1)
    #         final = []
    #         for i in range(len(region)-1):
    #             final.append([region[i], region[i+1]])

    #         region = np.asarray(final)
    #         reg = np.hstack((reg,region))
    #     child_nodes = []
    #     for child_reg in reg[:,tf_dim:]:
    #         child_nodes.append(Node(child_reg.reshape((tf_dim,2)),1))
    #     return child_nodes
    
    # def split_region(self,root,dim):
    #     region = np.linspace(root.input_space[dim][0], root.input_space[dim][1], num = 3)
    #     final = []

    #     for i in range(len(region)-1):
    #         final.append([region[i], region[i+1]])
    #     regs = []
    #     for i in range(len(final)):
    #         org = root.input_space.copy()
    #         org[dim] = final[i]
    #         regs.append(org)

    #     regs = [Node(i, 1) for i in regs]
    #     return regs

    # def get_subregion(self, root, num_agents, dim = 0):
    #     q=[root]
    #     while(len(q) < num_agents):
    #         if len(q) % 2 == 0:
    #             dim = (dim+1)% len(root.input_space)
    #         curr = q.pop(0)
    #         # print('dim curr queue_size', dim, curr.input_space, len(q))
    #         ch = self.split_region(curr,dim)
    #         curr.add_child(ch)
    #         q.extend(ch)
    #     # print([i.input_space for i in q])
    #     return q
    
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

# Example usage
number = 10
pairs = find_close_factor_pairs(number)
print(pairs)

def split_region(root,dim):
        region = np.linspace(root[dim][0], root[dim][1], num = 3)
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



# import math

# def factorize_number(number, n):
#     def get_prime_factors(num):
#         factors = []
#         while num % 2 == 0:
#             factors.append(2)
#             num //= 2
#         for i in range(3, int(math.sqrt(num)) + 1, 2):
#             while num % i == 0:
#                 factors.append(i)
#                 num //= i
#         if num > 2:
#             factors.append(num)
#         return factors

#     prime_factors = get_prime_factors(number)
#     print(prime_factors)
#     num_factors = len(prime_factors)

#     if n > num_factors:
#         print(f"Cannot factorize {number} into {n} multipliers.")
#         return []

#     # Use combinatorial techniques to select n-1 positions to split the prime factors
#     factor_indices = set()
#     while len(factor_indices) < n - 1:
#         factor_indices.add(np.random.randint(1, num_factors - 1))

#     factor_indices = sorted(factor_indices)

#     # Generate the multipliers based on the selected positions
#     multipliers = []
#     start_index = 0
#     for index in factor_indices:
#         selected_factors = prime_factors[start_index:index]
#         multiplier = math.prod(selected_factors)
#         multipliers.append(multiplier)
#         start_index = index
#     last_factors = prime_factors[start_index:]
#     last_multiplier = math.prod(last_factors)
#     multipliers.append(last_multiplier)

#     return multipliers

# # Example usage
# number = 5
# n = 2
# multipliers = factorize_number(number, n)
# # print(multipliers)

# import numpy as np

# def partition_input_space(region_support, num_agents):
#     num_dimensions = len(region_support)
#     lower_bounds = region_support[:,0]
#     upper_bounds = region_support[:,1]
#     print('lb,ub')
#     print(lower_bounds,upper_bounds)
#     input_space_volume = np.prod(np.array(upper_bounds) - np.array(lower_bounds))
#     target_volume = input_space_volume / num_agents
#     delta = np.power(target_volume, 1/num_dimensions)
#     print('delta: ',delta)
#     num_cells = []
#     for i in range(num_dimensions):
#         num_cells.append(int((upper_bounds[i] - lower_bounds[i]) / delta))
#     print('num cells ',num_cells)
#     total_cells = np.prod(num_cells)
#     print('tot cells :',total_cells)
#     while total_cells > num_agents:
#         max_ratio_index = np.argmax((np.array(upper_bounds) - np.array(lower_bounds)) / delta)
#         num_cells[max_ratio_index] -= 1
#         delta = np.power(target_volume, 1/num_dimensions)
#         total_cells = np.prod(num_cells)
#     print('below while ')
#     print('delta ',delta)
#     print('total_cells :',total_cells)
#     regions = []
#     for i in range(num_agents):
#         region = []
#         for j in range(num_dimensions):
#             cell_start = lower_bounds[j] + (num_cells[j] * (i / num_agents)) * delta
#             cell_end = lower_bounds[j] + ((num_cells[j] * (i + 1) / num_agents)) * delta
#             region.append([cell_start, cell_end])
#         regions.append(region)
    
#     return regions

# # Example usage
# lower_bounds = [-5,-5]
# upper_bounds = [5,5]
# num_agents = 5
# r = np.array([[-5,5],[-5,5]])
# regions = partition_input_space(r, num_agents)
# print('regions: ',regions)

# for i, region in enumerate(regions):
#     print(f"Region {i+1}:")
#     for j, (start, end) in enumerate(region):
#         print(f"Dimension {j+1}: {start} - {end}")
#     print()

# import numpy as np

# def partition_input_space(lower_bounds, upper_bounds, num_regions):
#     num_dimensions = len(lower_bounds)
#     input_space_volume = np.prod(np.array(upper_bounds) - np.array(lower_bounds))
#     target_volume = input_space_volume / num_regions
#     delta = np.power(target_volume, 1/num_dimensions)

#     num_cells = []
#     for i in range(num_dimensions):
#         num_cells.append(int((upper_bounds[i] - lower_bounds[i]) / delta))
    
#     total_cells = np.prod(num_cells)
#     while total_cells > num_regions:
#         max_ratio_index = np.argmax((np.array(upper_bounds) - np.array(lower_bounds)) / delta)
#         num_cells[max_ratio_index] -= 1
#         delta = np.power(target_volume, 1/num_dimensions)
#         total_cells = np.prod(num_cells)
    
#     regions = []
#     for i in range(num_regions):
#         region = []
#         for j in range(num_dimensions):
#             cell_start = lower_bounds[j] + (num_cells[j] * (i / num_regions)) * delta
#             cell_end = lower_bounds[j] + ((num_cells[j] * (i + 1) / num_regions)) * delta
#             region.append((cell_start, cell_end))
#         regions.append(region)
    
#     return regions

# # Example usage
# lower_bounds = [-5, -5]
# upper_bounds = [5, 5]
# num_regions = 5

# regions = partition_input_space(lower_bounds, upper_bounds, num_regions)

# for i, region in enumerate(regions):
#     print(f"Region {i+1}:")
#     for j, (start, end) in enumerate(region):
#         print(f"Dimension {j+1}: {start} - {end}")
#     print()


def split_space(input_space,num_agents, tf_dim):
        pairs = find_close_factor_pairs(num_agents)
        numbers = list(range(tf_dim))
        didx =random.shuffle(numbers)
        print('pairs ',pairs)
        # for d in didx:
        #     if len(pairs) > d:

        reg = np.zeros((num_agents, tf_dim), dtype=np.int64)
        prm = []
        for i,dim in enumerate(input_space):
            print(i,dim)
            if i < len(pairs):
                num_agents = pairs[i]
                # reg = np.zeros((num_agents, tf_dim), dtype=np.int64)
            else:
                num_agents = 1
            # reg = np.zeros((num_agents, tf_dim), dtype=np.int64)
            region = np.linspace(dim[0], dim[1], num = num_agents+1)
            print(region)
            final = []
            for i in range(len(region)-1):
                final.append([region[i], region[i+1]])
            print('fin: ',final)
            region = np.asarray(final)
            tot = []
            for y in prm:
                for x in final:
                    tot.append([y + [x]]) 
            prm = tot #[y + [x] for y in prm for x in final]
            print('prm', prm)
            print('r r',region)
        reg = np.hstack((reg,region))
        print('reg: ',reg)
        child_nodes = []
        for child_reg in reg[:,tf_dim:]:
            child_nodes.append(Node(child_reg.reshape((tf_dim,2)),1))
        return child_nodes

# c = split_space(np.array([[-5,5],[-5,5]]), 5 , 2)
# root = Node(np.array([[-5,5],[-5,5]]), 1)

# for child_reg in regions:
#     root.add_child(Node(child_reg,1))
# # return child_nodes

# print_tree(root)
# lf = root.find_leaves()
# print([i.input_space for i in c])


import itertools

def split_region(root,dim,num_agents):
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
    
def get_subregion(root, num_agents,dic, dim=0):
    q=[root]
    while(len(q) < num_agents):
        if len(q) % 2 == 0:
            dim = (dim+1)% len(root.input_space)
        curr = q.pop(0)
        # print('dim curr queue_size', dim, curr.input_space, len(q))
        ch = split_region(curr,dim, dic[dim])
        print('ch',ch)
        curr.add_child(ch)
        q.extend(ch)
    # print([i.input_space for i in q])
    return q

dic = {0:3,1:1}
rt = Node(np.array([[-5,5],[-5,5]]), 1)

q = get_subregion(rt, 3,dic )
print('q',[i.input_space for i in q])


# class naryNode:
#     def __init__(self, value):
#         self.value = value
#         self.children = []

#     def add_child(self, child):
#         self.children.append(child)

# def build_nary_tree(data, n):
#     if not data:
#         return None

#     root = naryNode(data[0])

#     for i in range(1, min(n + 1, len(data))):
#         child = build_nary_tree(data[i:], n)
#         root.add_child(child)

#     return root

sd = find_close_factor_pairs(5)
print(sorted(sd, reverse=True))

def split_interval(start, end, num_intervals):
    interval_size = (end - start) / num_intervals
    intervals = []
    
    for i in range(num_intervals):
        interval_start = start + i * interval_size
        interval_end = (start - (i + 1) * interval_size) - 0.00001
        intervals.append([interval_start, interval_end])
    
    return intervals

intervals = split_interval(-5, 5, 4)
print('intervals: ',intervals, random.uniform(-5, -2.5))
