import networkx as nx
import matplotlib.pyplot as plt
import pickle

def load_node(filename):
    with open(filename, 'rb') as file:
        node = pickle.load(file)
    return node


# class Node:
#     def __init__(self, value):
#         self.value = value
#         self.children = []

def add_edge(graph, parent, child):
    graph.add_edge(str(parent.input_space), str(child.input_space))
    for grandchild in child.child:
        add_edge(graph, child, grandchild)

def visualize_tree(root):
    graph = nx.DiGraph()
    graph.add_node(str(root.input_space))
    for child in root.child:
        add_edge(graph, root, child)
    pos = nx.spring_layout(graph)  # You can use other layout algorithms
    nx.draw(graph, pos, with_labels=True, arrows=True, node_size=700, node_color='skyblue', font_size=8, font_color='black')
    plt.show()

# Example usage
# root = Node("A")
# b = Node("B")
# c = Node("C")
# d = Node("D")
# e = Node("E")

# root.children = [b, c]
# b.children = [d, e]

# rl_root = load_node('/Users/shyamsundar/ASU/sem2/RA/partmahpc/partma/results/nodes/rl_root_0_3.pkl')
# visualize_tree(root)
root = load_node('/Users/shyamsundar/ASU/sem2/RA/partmahpc/partma/results/nodes/root_0_19.pkl')

mceq = load_node('/Users/shyamsundar/ASU/sem2/RA/partmahpc/partma/results/nodes/rl_root_0_MC_1.pkl')

mcroot = load_node('/Users/shyamsundar/ASU/sem2/RA/partmahpc/partma/results/nodes/rl_root_0_MCb4reset_1.pkl')

# mcroota4 = load_node('/Users/shyamsundar/ASU/sem2/RA/partmahpc/partma/results/nodes/rl_root_0_MCafter_2.pkl')

from bo.utils.plotlyExport import exportTreeUsingPlotly
from bo.agent.constants import *

# exportTreeUsingPlotly(rl_root, ROLLOUT)
exportTreeUsingPlotly(root, MAIN)
exportTreeUsingPlotly(mcroot, MAIN)
# exportTreeUsingPlotly(mcroota4, MAIN)
# exportTreeUsingPlotly(mceq, MAIN)
# exportTreeUsingPlotly(mceq, MAIN)

def get_node_values(root):
    values = []

    def traverse(node):
        values.append([node.input_space, node.avgRewardDist])
        print(node.input_space)
        print()
        print(node.avgRewardDist)
        for child in node.child:
            traverse(child)
            

    traverse(root)
    print('-'*100)
    return values

print(get_node_values(rl_root))

print(get_node_values(mcroot))

print(get_node_values(root))