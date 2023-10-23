import matplotlib.pyplot as plt

# class TreeNode:
#     def __init__(self, value):
#         self.value = value
#         self.children = []

def render_tree(node, routine,  x, y, ax, x_spacing=10.0, y_spacing=10.0):
    if node is not None:
        if node.agent is not None: # or node.agent.x_train is not None or node.y_train is not None:
            reg = "Actual"+str(node.input_space) + "\nSimReg"+str(node.agent.simReg.input_space)+'\n Main region'+str(node.agent.region_support.input_space)+'\n dataset:'+ str(node.agent.x_train) + str(node.agent.y_train)
        else:
            reg = 'NA'
        if node.getStatus(routine) == 1:
            color = 'green'
        else:
            color = 'red'
        ax.text(x, y, reg, ha='center', va='center', bbox=dict(facecolor=color, edgecolor='black', boxstyle='circle'))
        num_children = len(node.child)
        if num_children > 0:
            child_x = x - 0.5 * (num_children - 1) * x_spacing
            for child in node.child:
                ax.plot([x, child_x], [y - 1, y - y_spacing], 'bo-')
                render_tree(child,routine, child_x, y - y_spacing, ax, x_spacing, y_spacing)
                child_x += x_spacing

def export_tree_image(root,routine, filename):
    fig, ax = plt.subplots(figsize=(15, 15))
    render_tree(root,routine, 0, 0, ax, x_spacing=10.0, y_spacing=10.0)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.savefig(filename, format="png")
    plt.close()

# # Example usage for an N-ary tree:
# root = TreeNode(1)
# root.children = [TreeNode(2), TreeNode(3)]
# root.children[0].children = [TreeNode(4), TreeNode(5)]
# root.children[1].children = [TreeNode(6), TreeNode(7)]

# export_tree_image(root, "nary_tree_image.png")
