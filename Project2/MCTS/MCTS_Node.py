


class MCTS_Node():
    def __init__(state, root = False):
        self.N_s = 0
        self.state = state
        self.root = root
        self.Q_s_a = dict()
        self.N_s_a = dict()
        self.leaf = True

        