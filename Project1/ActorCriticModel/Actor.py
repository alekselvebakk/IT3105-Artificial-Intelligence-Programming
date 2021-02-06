import numpy as np 
from numpy.random import default_rng
import random

class Actor:
    def __init__(self, epsilon, alpha, gamma):

        #Mappings
        self.table = dict()
        self.elig = dict()
        self.current_episode = dict()

        #Constants
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        #Tools
        self.rng = default_rng()

    
    
    def get_action(self, state, action):

        #Check if SAP already in table
        if state not in self.table:
            self.table[state] = dict()
            self.table[action][state] = 0
        elif action not in self.table[state]:
            self.table[action][state] = 0

        #Do the epsilon-greedy-step
        decision = self.rng.uniform([0, 1, 1])
        if decision > self.epsilon:
            action = max(self.table[state], key = self.table[state])
        else:
            action = random.choice(self.table[state].key)
        
        self.current_episode[state] = action
        return action


    def episode_reset(self):

        for state in self.elig:
            for action in self.elig[state]:
                self.elig[state][action] = 0
    

    def episode