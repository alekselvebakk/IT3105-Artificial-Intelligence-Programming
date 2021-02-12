

class TableCritic:
    def __init__(self, alpha, gamma, elig_decay):
        #Mappings
        self.V = dict()
        self.elig = dict()

        #Constants
        self.gamma = gamma
        self.alpha = alpha
        self.elig_decay = elig_decay
    
    def get_delta(self, state, next_state, reward):
        if state not in self.V:
            self.V[state] = 0
        if next_state not in self.V:
            self.V[next_state] = 0
        delta = reward + self.gamma*self.V[next_state] - self.V[state]
        return delta

    def set_unit_eligibility(self, state):
        self.elig[state] = 1

    def update_value_function(self, state, delta):
        self.V[state] = self.V[state] + self.alpha*delta*self.elig[state]
        self.elig[state] = self.gamma*self.elig_decay*self.elig[state]

    def episode_reset(self):
        for state in self.elig:
            self.elig[state] = 0
        