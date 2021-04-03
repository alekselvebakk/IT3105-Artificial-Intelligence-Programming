import MCTS_Node 

class MCTS():
    def __init__(self, GameHandler, root):
        #Setting root state
        self.root = root
        #Copying GameHandler into object memory, setting its state and getting its action possibilities
        self.GameHandler = GameHandler
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

        #set starting state and action to root
        self.GameHandler.set_state(self.root)
        action = self.tree[self.root].get_action(c)
        self.simulation_history[self.root] = action
        
        while not self.GameHandler.state_is_final():
            self.GameHandler.perform_action(action)
            s_t = self.GameHandler.get_state()
            if s_t not in self.tree:
                self.insert_new_node(s_t)
                return self.GameHandler
            action = self.tree[s_t].get_action(c)        
            
        return self.GameHandler

    def backprop_tree(self, z):
        for state in self.simulation_history:
            self.tree[state].update_attributes(self.simulation_history[state], z)

    def update_tree(self, new_root_state, z):
        self.backprop_tree(z)
        self.simulation_history = dict()
        self.root = new_root_state