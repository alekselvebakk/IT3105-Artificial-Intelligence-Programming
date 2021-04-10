import math
import numpy as np
import random

class MCTS_Node:
    def __init__(self, state, actions):
        self.state = state
        self.player = int(state[0])
        self.N_s = 0
        self.V_s = 0
        self.N_s_a = dict()
        self.Q_s_a = dict()

        self.actions = actions
        #makes this as a lookup-table to store indices of actions that can be done
        self.available_actions = {}
        self.unexplored_actions = []
        

        #Adds elements that is not "False" in the unexplored_action-list
        #And gives them a Value
        for x in range(len(self.actions)):
            self.N_s_a[self.actions[x]] = 0
            self.Q_s_a[self.actions[x]] = 0
            if self.actions[x] != False:
                self.unexplored_actions.append(self.actions[x])
                self.available_actions[x] = self.actions[x]
    
    def increment_N_s(self):
        self.N_s = self.N_s + 1
    
    def increment_N_s_a(self, action):
        self.N_s_a[action] = self.N_s_a[action] + 1

    def update_V_s(self, z):
        delta = z-self.V_s
        self.V_s = self.V_s + delta

    def update_Q_s_a(self, action, z):
        self.Q_s_a[action] = self.Q_s_a[action] + (z-self.Q_s_a[action])/self.N_s_a[action]
    
    def update_attributes(self, action, z, discount):
        self.increment_N_s()
        self.increment_N_s_a(action)
        self.update_Q_s_a(action, z)
        self.update_V_s(z*discount)
            

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

    def get_minimizing_action(self, c):
        min_score = 10000
        argmin = 0
        for i in self.available_actions:
            current_score = self.score_for_minimizing(self.actions[i], c)
            if current_score < min_score:
                min_score = current_score
                argmin = i
        return self.actions[argmin]

    def get_maximixing_action(self, c):
        max_score = 0
        argmax = 0
        for i in self.available_actions:
            current_score = self.score_for_maximizing(self.actions[i], c)
            if current_score > max_score:
                max_score = current_score
                argmax = i
        return self.actions[argmax]

    def get_action(self, c):
        if not self.unexplored_actions: #Enters this for-loop if there are no unexplored actions
            if self.player == 1:
                action = self.get_minimizing_action(c)
            elif self.player == 2:
                action = self.get_maximixing_action(c)        
        else:
            action = random.choice(self.unexplored_actions)
            self.unexplored_actions.remove(action)
        return action

    def get_action_distribution_and_value(self):
        action_distribution_and_value = np.zeros_like(self.actions, dtype = "float")
        for i in range(len(self.actions)):
            action_distribution_and_value[i] = self.N_s_a[self.actions[i]]/self.N_s
        action_distribution_and_value.append(self.V_s)
        return action_distribution_and_value
    
    def get_most_frequent_action(self):
        action_visits = 0
        for i in range(len(self.actions)):
            if self.N_s_a[self.actions[i]] > action_visits:
                best_action = self.actions[i]
                action_visits = self.N_s_a[self.actions[i]]
        return best_action