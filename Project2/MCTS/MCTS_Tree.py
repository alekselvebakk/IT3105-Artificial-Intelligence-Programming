import MCTS_Node



class MCTS_Tree():
    def __init__(state):
        self.root = MCTS_Node(state = state)

    def prune_tree(node):
        self.root = node
        node.prune_parent()