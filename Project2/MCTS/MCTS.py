from MCTS.MCTS_Node import MCTS_Node

class MCTS:
    def __init__(self, GameHandler, GameBoard, exploration_weight, tree_games, time_for_rollouts = 1, stochastic_root_choice = False):
        #Setting root state
        self.root = GameHandler.get_state(GameBoard)

        #Copying GameHandler into object memory, setting its state and getting its action possibilities
        self.GameHandler = GameHandler
        
        actions = self.GameHandler.get_actions(GameBoard)
        #Initializing root node for tree
        root_node = MCTS_Node(self.root, actions)
        
        #Creating tree
        self.tree = dict()
        self.tree[self.root] = root_node
        self.simulation_history = dict()

        #Setting constants
        self.c = exploration_weight
        self.tree_games = tree_games
        self.time_for_rollouts = time_for_rollouts
        self.stochastic_root_choice = stochastic_root_choice

    def insert_new_node(self, state, actions):
        new_node = MCTS_Node(state, actions)
        self.tree[state] = new_node
    
    
    def tree_simulation(self, GameBoard):
        
        while not self.GameHandler.state_is_final(GameBoard):
            
            s_t = self.GameHandler.get_state(GameBoard)
            if s_t not in self.tree:
                actions_for_node = self.GameHandler.get_actions(GameBoard)
                self.insert_new_node(s_t, actions_for_node)
                action = self.tree[s_t].get_action(self.c)
                self.simulation_history[s_t] = action
                return action, False
            else:
                action = self.tree[s_t].get_action(self.c)
                self.simulation_history[s_t] = action
            self.GameHandler.perform_action(GameBoard, action)
        
        return None, True

    def backprop_tree(self, z):
        i = 0
        for state in self.simulation_history:
            self.tree[state].update_attributes(self.simulation_history[state], z)
            i += 1

    def update_and_reset_tree(self, board):
        self.simulation_history = dict()
        if not self.GameHandler.get_state(board) == self.root:
            self.root = self.GameHandler.get_state(board)
        
        
    
    def get_root_distribution_and_value(self):
        #Getting distribution of all actions
        state = self.root
        distribution_and_value = self.tree[self.root].get_action_distribution_and_value()
        D = [state, distribution_and_value]
        return D
    
    def get_root_distribution(self):
        state = self.root
        distribution = self.tree[self.root].get_action_distribution()
        D = [state, distribution]
        return D

    def get_RBUF_data(self, net_with_critic):
        if net_with_critic:
            return self.get_root_distribution_and_value()
        else:
            return self.get_root_distribution()


    def get_best_root_action(self):
        if self.stochastic_root_choice:
            action = self.tree[self.root].get_weighted_random_action()
        else:
            action = self.tree[self.root].get_most_frequent_action()
        return action