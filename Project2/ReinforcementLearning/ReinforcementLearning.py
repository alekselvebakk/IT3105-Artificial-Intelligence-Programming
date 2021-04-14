import numpy as np
from numpy.random import default_rng
import random


class ReinforcementLearning:
    def __init__(   self, 
                    number_actual_games, 
                    value_discount_factor, 
                    rollout_init_prob, 
                    rollout_final_prob, 
                    epsilon, 
                    final_epsilon, 
                    percent_before_critic, 
                    winning_reward, 
                    losing_reward,
                    net_with_critic=True,
                    minibatch_size = 25,
                    increasing_prob_data = True,
                    RBUF_max_size = 750,
                    RBUF_trimming = True):
        self.game_length = 0
        self.critic_indices = {}
        self.number_actual_games = number_actual_games
        self.gamma = value_discount_factor
        self.RBUF = []
        self.use_critic = False
        self.minibatch_size = minibatch_size
        self.increasing_prob_data = increasing_prob_data
        self.RBUF_max_size = RBUF_max_size
        self.RBUF_weight = []
        self.RBUF_trimming = RBUF_trimming

        self.net_with_critic = net_with_critic
        self.percent_before_critic = percent_before_critic/100
        self.rollout_prob = rollout_init_prob
        if self.net_with_critic:
            self.rollout_prob_decay = np.exp((np.log(rollout_final_prob/rollout_init_prob))/((1-self.percent_before_critic)*float(number_actual_games)))
        else:
            self.rollout_prob_decay = 1
        self.epsilon = epsilon
        if not epsilon == 0:
            self.epsilon_decay = np.exp((np.log(final_epsilon/epsilon))/number_actual_games)
        else:
            self.epsilon_decay = 0

        self.winning_reward = winning_reward
        self.losing_reward = losing_reward

        #Random Number Generator
        seed = 1337
        self.rng = default_rng(seed)

    def get_simulation_result(self, state_manager, board, actor_critic, action, finished):
        if not finished:
            state_manager.perform_action(board, action)
        if self.use_critic:
            state = state_manager.get_state(board)
            z = actor_critic.get_critic_value(state)
        else:
            while not state_manager.state_is_final(board):
                action = actor_critic.get_action(state_manager.get_state(board), self.epsilon)
                state_manager.perform_action(board, action)
            z = self.get_result_score(state_manager.get_result(board))

        return z

    def perform_real_move(self, state_manager, board, action):
        state_manager.perform_action(board, action)
        self.game_length += 1

    def update_critic_indices(self, RBUF):
        RBUF[-1][1][-1] = 0  # Setter value til 0
        current_iteration = self.game_length
        self.critic_indices[len(RBUF) - 1] = current_iteration  # logging index of critic-based-move

    def update_RBUF_critic_values(self, result):
        if self.net_with_critic:
            for RBUF_index in self.critic_indices:
                discount = self.gamma**(self.game_length-1-self.critic_indices[RBUF_index])
                value = self.get_result_score(result)*discount
                self.RBUF[RBUF_index][1][-1] = value
                   
    def print_progress(self, training_iteration):
        progress = str(float(training_iteration)/self.number_actual_games *100)+"%"
        print("Progress: "+progress)
    

    def add_to_RBUF(self, D, current_progress):
        self.RBUF.append(D)
        self.RBUF_weight.append(current_progress)
        if self.use_critic and self.net_with_critic: self.update_critic_indices(self.RBUF)
        if self.RBUF_trimming:
            if len(self.RBUF)>self.RBUF_max_size:
                self.RBUF.pop(0)
                self.RBUF_weight.pop(0)

    
    def train_actor_critic(self, actor_critic):
        if self.minibatch_size <= len(self.RBUF):
            indices = np.array(list(range(len(self.RBUF))))
            if self.increasing_prob_data:
                #index_dist = indices+1
                weights = np.array(self.RBUF_weight)
                index_dist = weights/weights.sum()
                index_dist[-1] += 1 - index_dist.sum()
                minibatch = np.random.choice(   indices, 
                                                size = self.minibatch_size, 
                                                p = index_dist, 
                                                replace = False)
            else:
                minibatch = random.sample(list(indices), self.minibatch_size)
        else:
            minibatch = np.array(list(range(0, len(self.RBUF))))

        inputs = np.zeros((len(minibatch), len(self.RBUF[0][0])))
        targets = np.zeros((len(minibatch), len(self.RBUF[0][1])))
        print('RBUF len:', len(self.RBUF), 'minibatch: ', minibatch)
        for i in range(len(minibatch)):
            rbuf_index = minibatch[i]
            inputs[i] = actor_critic.string_to_tensor(self.RBUF[rbuf_index][0])
            targets[i] = self.RBUF[rbuf_index][1]
        actor_critic.train(inputs, targets)
    

    def decay_epsilon(self, zero=False):
        self.epsilon = self.epsilon * self.epsilon_decay if not zero else 0
        
    def decide_use_of_critic(self):
        decision = self.rng.uniform(0, 1)
        if decision > self.rollout_prob and self.net_with_critic:
            self.use_critic = True
        else:
            self.use_critic = False
        
        
    def increase_chance_of_critic(self, progress):
        if progress/self.number_actual_games > self.percent_before_critic:
            self.rollout_prob = self.rollout_prob*self.rollout_prob_decay

    def get_result_score(self, result):
        if result == 1:
            return self.losing_reward
        elif result == 2:
            return self.winning_reward