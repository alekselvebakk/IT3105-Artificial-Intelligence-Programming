from ActorCriticModel.LearningModule import LearningModule
from SimWorld.GameHandler import GameHandler
import numpy as np
from matplotlib import pyplot as plt


def main():
    lm = LearningModule(neural_net_critic=False,
                        epsilon = 0.3,
                        alpha_actor = 0.01,
                        alpha_critic = 0.01,
                        gamma = 0.8,
                        elig_decay = 0.8,
                        epsilon_decay = 0.9)
    episodes = 1

    performance = np.zeros([episodes,1])

    for i in range(episodes):
        gh = GameHandler('diamond', 4, [[2, 2]], visualization=False)
        action = lm.initialize_episode(gh.get_board_state(), gh.get_actions())
        while not gh.check_if_final_state()[0]:
            gh.perform_action(action)
            action = lm.episode_step(   gh.get_board_state(), 
                                        gh.get_actions(), 
                                        gh.calculate_reward(), 
                                        next_state_is_final=gh.check_if_final_state()[0])
        performance[i]=gh.board.num_pegs
    
    print()
    plt.plot(performance)
    plt.show()

if __name__ == '__main__':
    main()