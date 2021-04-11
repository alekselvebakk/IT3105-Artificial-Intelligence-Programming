import numpy as np
from numpy.random import default_rng


class ReinforcementLearning:
    def __init__(self, number_actual_games, value_discount_factor, rollout_init_prob, rollout_final_prob, epsilon, final_epsilon):
        self.game_length = 0
        self.critic_indices = {}
        self.rollout_probability = 1
        self.number_actual_games = number_actual_games
        self.gamma = value_discount_factor
        self.RBUF = []
        self.use_critic = False

        self.rollout_prob = rollout_init_prob
        self.rollout_prob_decay = np.exp((np.log(rollout_final_prob/rollout_init_prob))/number_actual_games)
        self.epsilon = epsilon
        self.epsilon_decay = np.exp((np.log(final_epsilon/epsilon))/number_actual_games)

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
            z = state_manager.get_result(board)

        return z

    def perform_real_move(self, state_manager, board, action):
        state_manager.perform_action(board, action)
        self.game_length += 1

    def update_critic_indices(self, RBUF):
        RBUF[-1][1][-1] = 0  # Setter value til 0
        current_iteration = self.game_length
        self.critic_indices[len(RBUF) - 1] = current_iteration  # logging index of critic-based-move

    def update_RBUF_critic_values(self, result):
        if not result == 0:
            for RBUF_index in self.critic_indices:
                discount = self.gamma**(self.game_length-1-self.critic_indices[RBUF_index])
                self.RBUF[RBUF_index][1][-1] = result*discount
    
    def print_progress(self, training_iteration):
        progress = str(float(training_iteration)/self.number_actual_games *100)+"%"
        print("Progress: "+progress)
    
    
    def train_actor_critic(self, actor_critic):
        actor_critic.train_from_RBUF(self.RBUF)

    def add_to_RBUF(self, D):
        self.RBUF.append(D)
        if self.use_critic: self.update_critic_indices(self.RBUF)

    def train_actor_critic(self, RBUF, actor_critic):
        actor_critic.train_from_RBUF(RBUF)

    def decay_epsilon(self, zero=False):
        self.epsilon = self.epsilon * self.epsilon_decay if not zero else 0
        
    def decide_use_of_critic(self):
        decision = self.rng.uniform(0, 1)
        if decision > self.rollout_prob:
            self.use_critic = True
        else:
            self.use_critic = False
        self.rollout_prob = self.rollout_prob*self.rollout_prob_decay

