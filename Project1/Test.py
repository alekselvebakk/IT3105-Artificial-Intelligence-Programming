from ActorCriticModel.LearningModule import LearningModule
from SimWorld.GameHandler import GameHandler
import numpy as np
from matplotlib import pyplot as plt


def main():
    lm = LearningModule(NNCriticBool=True,
                        epsilon = 0.9,
                        alpha_actor = 0.6,
                        alpha_critic = 0.6,
                        gamma = 0.99,
                        elig_decay = 0.99,
                        epsilon_decay = 0.65,
                        hidden_layers = [5, 5],
                        input_size = 16)
    episodes = 200
    performance = np.zeros([episodes,1])

    for i in range(episodes):
        gh = GameHandler('diamond', 4, [[2, 1]],5, visualization=False)
        action = lm.initialize_episode(gh.get_board_state(), gh.get_actions())
        while not gh.check_if_final_state()[0]:
            gh.perform_action(action)
            action = lm.episode_step(   gh.get_board_state(), 
                                        gh.get_actions(), 
                                        gh.calculate_reward(), 
                                        next_state_is_final=gh.check_if_final_state()[0])
        performance[i]=gh.board.num_pegs
        if i > 0 and performance[i] < performance[i-1]:
            decays = decays +1
            lm.decay_epsilon()
    plt.plot(performance)
    plt.show()

if __name__ == '__main__':
    main()