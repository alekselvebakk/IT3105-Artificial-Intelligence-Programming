import MCTS_Node



class MCTS_Tree():
    def __init__(state):
        self.root = MCTS_Node(state, root = True)
