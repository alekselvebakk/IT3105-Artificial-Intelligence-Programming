class MCTS():
    def __init__(GameHandler, number_search_games, root):
        self.GameHandler = GameHandler
        self.root = root
        self.GameHandler.set_state(self.root)
        actions = self.GameHandler.get_actions
        root_node = MCTS_Node(state = state, actions = actions)
        self.tree = dict()
        self.tree[self.root] = root_node
        self.number_search_games = number_search_games


    def tree_simulation(c = 1):
        simulation_order = []
        self.GameHandler.set_state(self.root)
        action = self.tree[self.root].get_action(c)
        self.GameHandler.perform_action(action)
        s_t = GameHandler.get_action
        if 

        while not self.GameHandler.state_is_final()
            simulation_order.append()
            if 
        