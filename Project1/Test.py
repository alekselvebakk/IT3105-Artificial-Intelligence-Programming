from ActorCriticModel.LearningModule import LearningModule
from SimWorld.GameHandler import GameHandler
import numpy as np
from matplotlib import pyplot as plt


def main():
    lm = LearningModule(neural_net_critic=False,
                        epsilon = 0.9,
                        alpha_actor = 0.6,
                        alpha_critic = 0.6,
                        gamma = 0.99,
                        elig_decay = 0.99,
                        epsilon_decay = 0.65)
    episodes = 200

    performance = np.zeros([episodes,1])
    decays = 0

    for i in range(episodes):
        gh = GameHandler('diamond', 4, [[2, 1]],5, visualization=False)
        action = lm.initialize_episode(gh.get_board_state(), gh.get_actions())
        while not gh.check_if_final_state()[0]:
            gh.perform_action(action)
            r = gh.calculate_reward()
            action = lm.episode_step(   gh.get_board_state(), 
                                        gh.get_actions(), 
                                        r, 
                                        next_state_is_final=gh.check_if_final_state()[0])
        performance[i]=gh.board.num_pegs
        if i > 0 and performance[i] < performance[i-1]:
            decays = decays +1
            lm.decay_epsilon()
    print("number of decays: ", decays)
    lm.decay_epsilon(zero = True)

    gh = GameHandler('diamond', 4, [[2, 1]],5,  visualization=True)
    action = lm.initialize_episode(gh.get_board_state(), gh.get_actions())   
    while not gh.check_if_final_state()[0]:
        gh.perform_action(action)
        action = lm.episode_step(   gh.get_board_state(), 
                                    gh.get_actions(), 
                                    gh.calculate_reward(), 
                                    next_state_is_final=gh.check_if_final_state()[0])
    print("number of pegs after greedy run: ",gh.board.num_pegs)
    plt.plot(performance)
    plt.show()

if __name__ == '__main__':
    main()