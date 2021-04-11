class ReinforcementLearning:
    def __init__(self, epsilon, epsilon_decay):
        self.game_length = 0
        self.critic_indices = {}
        self.rollout_probability = 1
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def get_simulation_result(self, state_manager, board, use_critic, actor_critic, action, finished):
        if not finished:
            state_manager.perform_action(board, action)

        if use_critic:
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

    def update_RBUF_critic_values(self, state_manager, board, RBUF):
        for RBUF_index in self.critic_indices:
            discount = self.game_length - 1 - self.critic_indices[RBUF_index]
            RBUF[RBUF_index][1][-1] = state_manager.get_result(board) * discount

    def print_progress(self, RBUF):
        print("Training starts, RBUF has", len(RBUF), "samples. \nProgress: " + progress)

    def train_actor_critic(self, RBUF, actor_critic):
        actor_critic.train_from_RBUF(RBUF)

    def decay_epsilon(self, zero=False):
        self.epsilon = self.epsilon * self.epsilon_decay if not zero else 0

