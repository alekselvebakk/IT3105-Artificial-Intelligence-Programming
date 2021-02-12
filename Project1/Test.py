from Project1.ActorCriticModel.LearningModule import LearningModule
from Project1.SimWorld.GameHandler import GameHandler


def main():
    gh = GameHandler('diamond', 4, [[2, 2]])
    lm = LearningModule(neural_net_critic=False)
    init_state = gh.get_board_state()
    init_actions = gh.get_actions()

    print(init_state, init_actions)

if __name__ == '__main__':
    main()