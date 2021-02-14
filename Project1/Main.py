from ActorCriticModel.LearningModule import LearningModule
from SimWorld.GameHandler import GameHandler
import numpy as np
from matplotlib import pyplot as plt


def main():
    learning_module = LearningModule(neural_net_critic=False,
                        epsilon = 0.9,
                        alpha_actor = 0.6,
                        alpha_critic = 0.6,
                        gamma = 0.99,
                        elig_decay = 0.99,
                        epsilon_decay = 0.65)

    game_handler = GameHandler('diamond', 4, [[2, 1]], 5, visualization=False)
    last_game_handler = GameHandler('diamond', 4, [[2, 1]], 5, visualization=True)

    episodes = 200

    performance = np.zeros([episodes,1])
    decays = 0

    for i in range(episodes):
        run_episode(learning_module, game_handler)

        performance[i] = game_handler.board.num_pegs  # TODO: make even cleaner code
        if i > 0 and performance[i] < performance[i - 1]:
            decays = decays + 1
            learning_module.decay_epsilon()

    learning_module.decay_epsilon(zero=True)
    run_episode(learning_module, last_game_handler)

    plt.plot(performance)
    plt.show()

def run_episode(learning_module, game_handler):

    game_handler.reset_board()
    action = learning_module.initialize_episode(game_handler.get_board_state(), game_handler.get_actions())
    while not game_handler.check_if_final_state()[0]:
        game_handler.perform_action(action)
        reward = game_handler.calculate_reward()
        action = learning_module.episode_step(game_handler.get_board_state(),
                                 game_handler.get_actions(),
                                 reward,
                                 next_state_is_final=game_handler.check_if_final_state()[0])

if __name__ == '__main__':
    main()




