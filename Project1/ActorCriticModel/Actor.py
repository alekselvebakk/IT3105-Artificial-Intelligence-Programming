import numpy as np 
from numpy.random import default_rng
import random

class Actor:
    ## ACTOR MODEL FOR ACTOR-CRITIC MODEL IN REINFORCEMENT LEARNING
    # Here the Actor class is implemented with on policy learning
    # using an epsilon-greedy method

    #Initialization function
    def __init__(   self, 
                    alpha, #Learning rate
                    gamma, #Discount factor
                    elig_decay): #Eligibility decay

        #Mappings
        self.table = dict()
        self.elig = dict()

        #Constants
        self.alpha = alpha
        self.gamma = gamma
        self.elig_decay = elig_decay

        #Random Number Generator
        seed = 1337
        self.rng = default_rng(seed)
        random.seed(seed)

    @staticmethod
    def epsilon_greedy_policy(action_value_pairs, epsilon, rng):
        decision = rng.uniform(0, 1)
        if decision > epsilon:
            action = max(action_value_pairs, key=action_value_pairs.get)
        else:
            action = random.choice(list(action_value_pairs.keys()))
        return action
    

    def get_action(self, state, possible_actions, epsilon):
        #Check if SAP already in value-table
        if state not in self.table:
            self.table[state] = dict()
        for action in possible_actions:
            if action not in self.table[state]:
                self.table[state][action] = 0

        #Epsilon-Greedy Policy
        action = self.epsilon_greedy_policy(self.table[state], epsilon, self.rng)
        return action
    


    
    def update_policy_table(self, state, action, delta):
        self.table[state][action] = self.table[state][action] + self.alpha*delta*self.elig[state][action]
        self.elig[state][action] = self.elig[state][action]*self.gamma*self.elig_decay

    
    def set_unit_eligibility(self, state, action):
        #Check if SAP already in eligibility-table
        if state not in self.elig:
            self.elig[state] = dict()
        self.elig[state][action] = 1


    def episode_reset(self):
        for state in self.elig:
            for action in self.elig[state]:
                self.elig[state][action] = 0

    def print_table(self):
        print(self.table)