from Project1.ActorCriticModel.LearningModule import LearningModule
from Project1.SimWorld.GameHandler import GameHandler


def main():
    gh = GameHandler('diamond', 4, [[2, 2]], visualization=True)
    lm = LearningModule(neural_net_critic=False)
    lm.initialize_episode(gh.get_board_state(), gh.get_actions())

    while not gh.check_if_final_state()[0]:
        action = lm.episode_step(gh.get_board_state(), gh.get_actions(), gh.calculate_reward(), next_state_is_final=gh.check_if_final_state()[0])
        gh.perform_action(action)



if __name__ == '__main__':
    main()