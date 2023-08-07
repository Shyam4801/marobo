# import numpy as np

# class binNode:
#     def __init__(self, input_space) -> None:
#         self.input_space = input_space
#         self.left = None
#         self.right = None
#         self.num_leaves = 1

#     def add_left_child(self, left_child):
#         self.left = left_child
#         self.num_leaves += left_child.num_leaves

#     def add_right_child(self, right_child):
#         self.right = right_child
#         self.num_leaves += right_child.num_leaves
    
#     def add_child(self,c):
#         for i in c:
#             print(i.input_space)
#             self.child.append(i)

#     def find_leaves(self):
#         leaves = []
#         self._find_leaves_helper(self, leaves)
#         return leaves

#     def _find_leaves_helper(self, node, leaves):
#         if not node.child:
#             if node != self:  # Exclude the root node
#                 leaves.append(node)
#         else:
#             for child in node.child:
#                 self._find_leaves_helper(child, leaves)

# class Node:
#     def __init__(self, input_space) -> None:
#         self.input_space = np.asarray(input_space, dtype=np.float32)
#         self.child = []
#         self.lf = 1
    
#     def add_child(self,c):
#         for i in c:
#             # print(i.input_space)
#             self.child.append(i)

#     def find_leaves(self):
#         leaves = []
#         self._find_leaves_helper(self, leaves)
#         return leaves

#     def _find_leaves_helper(self, node, leaves):
#         if not node.child:
#             if node != self:  # Exclude the root node
#                 leaves.append(node)
#         else:
#             for child in node.child:
#                 self._find_leaves_helper(child, leaves)
    
#     @classmethod
#     def create_obj(self, cls, data):
#         return cls(data)

# def split_space(root,dim, num_agents):
#     if root is None:
#         return
    
#     # fun = lambda x : np.linspace(root.input_space[x][0], root.input_space[x][1], num = num_agents+1)
#     # reg = np.zeros((num_agents, 2), dtype=np.int64)
#     # print(input_space[0])
#     # if len(nlf) < 5:
#     d = dim
#     # for _ in range(num_agents):
        
#     #     # print(d)
#     #     # for i in range(len(input_space)):
#     #     dim = d % len(input_space)
#     region = np.linspace(root.input_space[d][0], root.input_space[d][1], num = 2+1)
#     # np.linspace(input_space[dim % 2][0], input_space[dim % 2][1], num = num_agents+1)
#     final = []
        
#     for i in range(len(region)-1):
        
#         final.append([region[i], region[i+1]])
#     # print('final', final)
#     regs = []
#     for i in range(len(final)):
#         org = root.input_space.copy()
#         # org = np.asarray(org,dtype=np.float32)
#         # print('final[i]',final[i])
#         org[d] = final[i]
#         regs.append(org)
#     # print( final)
#     print()
#     print(regs)
#     regs = [Node(i) for i in regs]
#     return regs
#         # nc=[]
#         # for c in regs:
#         # root.add_left_child(binNode(regs[0]))
#         # root.add_right_child(binNode(regs[1]))
#         # # nlf = 0
#         # # for lf in root.find_leaves():
#         # split_space(root.left , dim+1,nlf,num_agents)
#         # split_space(root.right , dim+1, nlf,num_agents)
    
#     # return root

#     # return root.find_leaves()
#         # d += 1
#         # 

#         # region = np.asarray(final)
#         # print(region)
#         # # reg = np.hstack((reg,region))
#         # print(region)
#     # nc = []
#     # for c in reg[:,2:]:
#     #     nc.append(Node(c.reshape((2,2))))
#     #     # self.add_child(nc)
#     # return 

# s = np.array([[-5,5],[-5,5]])
# n = Node(s)
# # n.add_child(split_space(s,5))

# q=[n]
# d = 0
# c = 1
# while(len(q) < 5):
#     print(q)
#     if len(q) % 2 == 0:
#         print('d: ',d)
#         d = (d+1)% len(s)
#     curr = q.pop(0)
#     # print('curr: ',curr.input_space, c, len(q))   
#     ch = split_space(curr,d,4)
#     curr.add_child(ch)
#     q.extend(ch)
#     # print('len q: ',len(q),d)
#     # d += 1
#     # c*=2
    
    
# print([i.input_space for i in q])
# # for lf in c.find_leaves():
# #     print(lf.input_space)
#     # # break
# # n.add_child(c)
# # print(n.child[0].input_space)

# # c2 = [Node(np.array([[-1,3],[-3,2]]))]
# # c3 = [Node(np.array([[-3,3],[-4,2]]))]

# # for i,s in enumerate(c):
# #     if i == 2:
# #         s.add_child(c2)
# #         s.add_child(c3)
# # region_support = np.array([[-1,1], [-2,2]])
# # direction_of_branching = 0

# # branching_factor = 4
# # dim_length = region_support[direction_of_branching][1] - region_support[direction_of_branching][0]
# # split_array = region_support[direction_of_branching][0] + (np.arange(branching_factor+1)/branching_factor) * dim_length

# # print('split_array: ',split_array)

# # new_bounds = []
# # for i in range(branching_factor):
# #     temp = region_support.copy()
# #     temp[direction_of_branching,0] = split_array[i]
# #     temp[direction_of_branching,1] = split_array[i+1]
    
# #     new_bounds.append(temp)

# # print(new_bounds)
# print('____________________________________')

# # class QuadTreeNode:
# #     def __init__(self, region_support):
# #         # self.lb = lb  # Lower bounds (x, y)
# #         # self.ub = ub  # Upper bounds (x, y)
# #         self.region_support =  region_support
# #         self.children = []

# # def split_space(region_support, branching_factor, direction_of_branching):
# #     root = QuadTreeNode(region_support)
# #     split_quadtree(root, branching_factor, direction_of_branching)
# #     return root

# # def split_quadtree(node, branching_factor, direction_of_branching):
# #     if branching_factor > 5:
# #         return

# #     dim_length = node.region_support[direction_of_branching % 2][1] - node.region_support[direction_of_branching % 2][0]
# #     split_array = node.region_support[direction_of_branching % 2][0] + (np.arange(branching_factor+1)/branching_factor) * dim_length

# #     for i in range(branching_factor):
# #         temp = node.region_support.copy()
# #         temp[direction_of_branching % 2,0] = split_array[i]
# #         temp[direction_of_branching%2,1] = split_array[i+1]

# #     reg = QuadTreeNode(temp)
# #     node.children.append(reg)

# #     # center_x = (node.lb[0] + node.ub[0]) / 2
# #     # center_y = (node.lb[1] + node.ub[1]) / 2

# #     # # Split the node into four quadrants
# #     # top_left = QuadTreeNode((node.lb[0], center_y), (center_x, node.ub[1]))
# #     # top_right = QuadTreeNode((center_x, center_y), (node.ub[0], node.ub[1]))
# #     # bottom_left = QuadTreeNode((node.lb[0], node.lb[1]), (center_x, center_y))
# #     # bottom_right = QuadTreeNode((center_x, node.lb[1]), (node.ub[0], center_y))

# #     # node.children = [top_left, top_right, bottom_left, bottom_right]

# #     branching_factor += 1
# #     direction_of_branching += 1
# #     # Recursively split one of the quadrants
# #     split_quadtree(reg, 0, 2)

# # # Example usage
# # r = np.array([[-1,1],[-2,2]])
# # num_regions = 5

# # root = split_space(r, num_regions, 2)

# # # Access the regions
# # regions = root.children
# # for i, region in enumerate(regions):
# #     print(f"Region {i+1}: lb={region.lb}, ub={region.ub}")

# # print('---------------')

# def count_leaves(node):

#     if node == None:
#         return 0

#     elif node.left == None and node.right == None:
#         return 1

#     else:
#         return count_leaves(node.left) + count_leaves(node.right)

# class BinaryTreeNode:
#     def __init__(self, input_space, level):
#         self.input_space = input_space
#         # self.start = start
#         # self.end = end
#         self.left = None
#         self.right = None
#         self.level = level

# def find_leaves(root):
#     leaves = []
#     inorder_traversal(root, leaves)
#     return leaves

# def inorder_traversal(node, leaves):
#     if node is None:
#         return
    
#     inorder_traversal(node.left, leaves)
    
#     if node.left is None and node.right is None:
#         leaves.append(node.input_space)
    
#     inorder_traversal(node.right, leaves)


# def split_array(root,dim, array, result, count, head):
#     # if root is None:
#     #     return

#     if count_leaves(head) >= 3:
#         print('head.input_space',head.input_space)
#         return

#     print('count_leaves(head)',count_leaves(head),find_leaves(head))
#     # if count < 3:
#     d = dim%2
#     region = np.linspace(root.input_space[d][0], root.input_space[d][1], num = 2+1)
#     final = []
        
#     for i in range(len(region)-1):
#         final.append([region[i], region[i+1]])
#     # print('final', final)
#     regs = []
#     for i in range(len(final)):
#         org = root.input_space.copy()
#         org[d] = final[i]
#         regs.append(org)
#     print('regs: ',regs)

#     # subarray = array[start:end+1]
#     result.append(regs)

#     # mid = (start + end) // 2

#     root.left = BinaryTreeNode(regs[0], root.level+1)
#     root.right = BinaryTreeNode(regs[1], root.level+1)

#     # count+=1
#     split_array(root.left,dim+1, array, result, count, head)
#     split_array(root.right,dim+1, array, result, count, head)

#     # if count_leaves(head) > 3:
#     #     root.right = None
#     #     split_array(root.left, dim+1, array, result, count, head)
#         # result.pop(0)
#     # if not root.left and not root.right:
            

# # Example usage
# array = [[-1,1],[-2,2]]
# root = BinaryTreeNode(array,0)
# result = []
# head = root
# split_array(root,0, array, result, 0, head)

# # Print the subarrays
# # print('res: ',result)
# # for i, subarray in enumerate(result):
# #     print(f"Subarray {i + 1}: {subarray}")

# def find_leaves(root):
#     leaves = []
#     inorder_traversal(root, leaves)
#     return leaves

# def inorder_traversal(node, leaves):
#     if node is None:
#         return
    
#     inorder_traversal(node.left, leaves)
    
#     if node.left is None and node.right is None:
#         leaves.append(node.input_space)
    
#     inorder_traversal(node.right, leaves)

# print('leaf:',find_leaves(root))


def split_search_space(x_min, x_max, y_min, y_max, num_subspaces):
    x_range = x_max - x_min
    y_range = y_max - y_min

    x_step = x_range / num_subspaces
    y_step = y_range / num_subspaces

    subspaces = []

    for i in range(num_subspaces):
        x_subspace_min = x_min + i * x_step
        x_subspace_max = x_subspace_min + x_step

        for j in range(num_subspaces):
            y_subspace_min = y_min + j * y_step
            y_subspace_max = y_subspace_min + y_step

            subspace = {
                'x_min': x_subspace_min,
                'x_max': x_subspace_max,
                'y_min': y_subspace_min,
                'y_max': y_subspace_max
            }

            subspaces.append(subspace)

    return subspaces

sub = split_search_space(-5,5,-5,5,5)
print(sub)