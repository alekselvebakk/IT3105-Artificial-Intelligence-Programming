import MCTS_Node 

class MCTS:
    def __init__(self, GameHandler, root):
        #Setting root state
        self.root = root

        #Copying GameHandler into object memory, setting its state and getting its action possibilities
        self.GameHandler = GameHandler  #   Her skal det initialiseres ny GameHandler med 
                                        #   samme settings som GameHandler
        self.GameHandler.set_state(self.root)
        actions = self.GameHandler.get_actions()
        
        #Initializing root node for tree
        root_node = MCTS_Node(state = self.root, actions = actions)
        
        #Creating tree
        self.tree = dict()
        self.tree[self.root] = root_node
        self.simulation_history = dict()


    def insert_new_node(self, state):
        if not self.GameHandler.get_state == state:
            self.GameHandler.set_state(state)
        actions = self.GameHandler.get_actions()
        new_node = MCTS_Node(state = state, actions = actions)
        self.tree[state] = new_node
    
    
    def tree_simulation(self, c):

        self.GameHandler.set_state(self.root)

        while not self.GameHandler.state_is_final():
            
            s_t = self.GameHandler.get_state()
            if s_t not in self.tree:
                self.insert_new_node(s_t)
                action = self.tree[s_t].get_action(c)
                self.simulation_history[s_t] = action
                return s_t, action, False
            else:
                action = self.tree[s_t].get_action(c)
                self.simulation_history[s_t] = action       
            self.GameHandler.perform_action(action)

        s_finished = self.GameHandler.get_state()
        return s_finished, None, True

    def backprop_tree(self, z):
        for state in self.simulation_history:
            self.tree[state].update_attributes(self.simulation_history[state], z)

    def update_and_reset_tree(self, new_root_state):
        self.simulation_history = dict()
        if not new_root_state == self.root:
            self.root = new_root_state
    
    def get_root_distribution(self):
        #Getting distribution of all actions
        state = self.root
        distribution = self.tree[self.root].get_action_distribution()
        D = [state, distribution]
        return D

    def get_best_root_action(self):
        action = self.tree[self.root].get_most_frequent_action()
        return action