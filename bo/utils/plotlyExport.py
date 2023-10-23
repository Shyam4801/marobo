import networkx as nx
import plotly.graph_objects as go
from bo.agent.constants import MAIN

# Sample TreeNode class
# class TreeNode:
#     def __init__(self, data):
#         self.data = data
#         self.child = []

# # Create TreeNode structure
# root = TreeNode("A")
# root.child = [TreeNode("B"), TreeNode("C")]
# root.child[0].child = [TreeNode("D"), TreeNode("E")]

# Convert TreeNode to a graph structure
def tree_to_graph(root_node):
    G = nx.Graph()
    stack = [(root_node, None)]

    while stack:
        node, parent = stack.pop()
        G.add_node(node)

        if parent:
            G.add_edge(parent, node)

        for child in node.child:
            stack.append((child, node))

    return G

def exportTreeUsingPlotly(root, routine = MAIN):
    graph = tree_to_graph(root)

    # Plot the graph using Plotly
    pos = nx.spring_layout(graph)  # Define the layout of nodes

    edge_trace = go.Scatter(x=[], y=[], line={'width': 0.5}, hoverinfo='none', mode='lines')

    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_colors = ['red' if node.getStatus(MAIN) == 0 else 'green' for node in graph.nodes()]

    node_trace = go.Scatter(x=[], y=[], text=[], mode='markers', hoverinfo='text',marker=dict(size=10, color=node_colors))

    for node in graph.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        print('graph.nodes[node]: ',graph.nodes[node], node)
        
        if node.agent != None:
            xtr = node.agent.x_train
        else:
            xtr = "Inactive"
        nl = "<br>"
        node_trace['text'] += tuple([f'Node: {node.input_space},{nl} X Train:{nl} {xtr}'])

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                    ))

    fig.show()
