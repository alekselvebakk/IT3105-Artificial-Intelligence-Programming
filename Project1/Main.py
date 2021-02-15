from ActorCriticModel.LearningModule import LearningModule
from SimWorld.GameHandler import GameHandler
import numpy as np
from matplotlib import pyplot as plt
from configparser import ConfigParser
import ast
import pathlib

def main():
    config = ConfigParser()
    config_path = str(pathlib.Path(__file__).parent.absolute())+"/config.ini"
    config.read(config_path)
    game_handler = GameHandler(config['board']['type'],
                               config.getint('board', 'size'),
                               ast.literal_eval(config['board']['open_cells']),
                               config.getint('board', 'interval'),
                               visualization=config.getboolean('board', 'visualization'),
                               board_gif_name=config['board']['board_gif_name'])

    input_size = game_handler.board.num_pegs+len(game_handler.board.empty)


    
    learning_module = LearningModule(NNCriticBool=config.getboolean('learning_module', 'nn_critic'),
                                     epsilon=config.getfloat('actor', 'epsilon'),
                                     alpha_actor=config.getfloat('actor', 'learning_rate'),
                                     alpha_critic=config.getfloat('critic', 'learning_rate'),
                                     gamma_actor=config.getfloat('actor', 'discount_factor'),
                                     gamma_critic=config.getfloat('critic', 'discount_factor'),
                                     elig_decay_actor=config.getfloat('critic', 'eligibility_factor'),
                                     elig_decay_critic=config.getfloat('critic', 'eligibility_factor'),
                                     epsilon_decay=config.getfloat('actor', 'epsilon_decay'),
                                     hidden_layers=config['learning_module']['hidden_layers'],
                                     input_size=input_size)


    last_game_handler = GameHandler(config['board']['type'],
                                    config.getint('board', 'size'),
                                    ast.literal_eval(config['board']['open_cells']),
                                    config.getint('board', 'interval'),
                                    visualization=config.getboolean('board', 'last_visualization'),
                                    board_gif_name=config['board']['board_gif_name'])

    episodes = config.getint('learning_module', 'episodes')

    performance = np.zeros([episodes,1])
    decays = 0

    for i in range(episodes):
        pros = float(i)/episodes*100
        if pros%5==0:
            print(int(pros), "% ferdig")
        run_learning_episode(learning_module, game_handler)
        performance[i] = game_handler.board.num_pegs
        if i > 0 and performance[i] < performance[i - 1]:
            decays = decays + 1
            learning_module.decay_epsilon()
    print(100, "% ferdig")
    plt.plot(performance)
    plt.show()

    run_basic_episode(learning_module, last_game_handler)

def run_learning_episode(learning_module, game_handler):
    game_handler.reset_board()
    action = learning_module.initialize_episode(game_handler.get_board_state(), 
                                                game_handler.get_actions())
    while not game_handler.check_if_final_state()[0]:
        game_handler.perform_action(action)
        reward = game_handler.calculate_reward()
        action = learning_module.episode_step(game_handler.get_board_state(),
                                              game_handler.get_actions(),
                                              reward,
                                              next_state_is_final=game_handler.check_if_final_state()[0])
    game_handler.visualize_board()

def run_basic_episode(learning_module, game_handler):
    game_handler.reset_board()
    action = learning_module.initialize_episode(game_handler.get_board_state(), 
                                                game_handler.get_actions())
    game_handler.perform_action(action)
    while not game_handler.check_if_final_state()[0]:
        action = learning_module.basic_step(game_handler.get_board_state(), 
                                            game_handler.get_actions())
        game_handler.perform_action(action)
    game_handler.visualize_board()

if __name__ == '__main__':
    main()