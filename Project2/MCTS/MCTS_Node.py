import math
import numpy as np
import random

class MCTS_Node():
    def __init__(self, state = state, actions = actions):
        self.state = state
        """
        logikk for å bestemme  hvilken spiller man er
        -> player = 0 eller 1 
        """
        self.player = 0
        self.N_s = 0
        self.N_s_a = dict()
        self.Q_s_a = dict()
        self.actions = actions
        self.unexplored_actions = actions
        for action in self.actions:
            self.N_s_a[action] = 0
            self.Q_s_a[action] = 0
    
    def increment_N_s(self):
        self.N_s = self.N_s + 1
    
    def increment_N_s_a(self, action):
        self.N_s_a[action] = self.N_s_a[action] + 1

    def update_Q_s_a(self, action, z):
        self.Q_s_a[action] = self.Q_s_a[action] + (z-self.Q_s_a[action])/self.N_s_a[action]
    
    def update_attributes(self, action, z):
        self.increment_N_s()
        self.increment_N_s_a(action)
        self.update_Q_s_a(action, z)
            

    def numeric_exploration_score(self, action, c):
        if c == 0:
            return c
        else:
            log_N_s = math.log(self.N_s)
            sqrt_term = math.sqrt(log_N_s/self.N_s_a[action])
            return c*sqrt_term
        
    def score_for_maximizing(self, action, c):
        to_be_maximized = self.Q_s_a[action]+self.numeric_exploration_score(action, c)
        return to_be_maximized

    def score_for_minimizing(self, action, c):
        to_be_minimized = self.Q_s_a[action]-self.numeric_exploration_score(action, c)
        return to_be_minimized
    
    def get_action_score(self, action, c):
        if self.player == 0:
                score = self.score_for_maximizing(action, c)
        else:
                score = self.score_for_minimizing(action, c)
        return score

    
    def get_action(self, c):
        if not self.unexplored_actions: #Enters this for-loop if there are no unexplored actions
            action_scores = np.zeros_like(self.actions)
            for i in len(action_scores):
                action_scores[i] = self.get_action_score(self.actions[i], c)
            if self.player == 0:
                action = self.actions[np.argmax(action_scores)]
            else:
                action = self.actions[np.argmin(action_scores)]
        else:
            action = random.choice(self.unexplored_actions)
            self.unexplored_actions.remove(action)
        return action