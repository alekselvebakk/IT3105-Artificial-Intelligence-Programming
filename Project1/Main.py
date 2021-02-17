from ActorCriticModel.LearningModule import LearningModule
from SimWorld.GameHandler import GameHandler
import numpy as np
from matplotlib import pyplot as plt
from configparser import ConfigParser
import ast
import pathlib
import time

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
    episodes = config.getint('learning_module', 'episodes')
    epsilon =config.getfloat('actor', 'epsilon') 
    epsilon_decay = np.exp((np.log(0.01/epsilon))/episodes)

    
    learning_module = LearningModule(NNCriticBool=config.getboolean('learning_module', 'nn_critic'),
                                     epsilon=epsilon,
                                     alpha_actor=config.getfloat('actor', 'learning_rate'),
                                     alpha_critic=config.getfloat('critic', 'learning_rate'),
                                     gamma_actor=config.getfloat('actor', 'discount_factor'),
                                     gamma_critic=config.getfloat('critic', 'discount_factor'),
                                     elig_decay_actor=config.getfloat('critic', 'eligibility_factor'),
                                     elig_decay_critic=config.getfloat('critic', 'eligibility_factor'),
                                     epsilon_decay=epsilon_decay,
                                     hidden_layers=config['learning_module']['hidden_layers'],
                                     input_size=input_size)


    last_game_handler = GameHandler(config['board']['type'],
                                    config.getint('board', 'size'),
                                    ast.literal_eval(config['board']['open_cells']),
                                    config.getint('board', 'interval'),
                                    visualization=config.getboolean('board', 'last_visualization'),
                                    board_gif_name=config['board']['board_gif_name'],
                                    winning_reward = config.getint('board','winning_reward'),
                                    losing_reward = config.getint('board','losing_reward'))

    episodes = config.getint('learning_module', 'episodes')

    performance = np.zeros([episodes,1])
    decays = 0

    current = time.time()
    for i in range(episodes):
        pros = float(i)/episodes*100
        if pros%5==0:
            print(int(pros), "% ferdig")
        run_learning_episode(learning_module, game_handler)
        performance[i] = game_handler.board.num_pegs
        learning_module.decay_epsilon()    
    print(100, "% ferdig")
    print(time.time()-current, " sekunder brukt")
    plt.plot(performance)
    plt.show()

    run_basic_episode(learning_module, last_game_handler)

def run_learning_episode(learning_module, game_handler):
    game_handler.reset_board()
    action = learning_module.initialize_episode(game_handler.get_board_state(), 
                                                game_handler.get_actions())
    while not game_handler.check_if_final_state():
        game_handler.perform_action(action)
        reward = game_handler.calculate_reward()
        state = game_handler.get_board_state()
        action = learning_module.episode_step(state,
                                              game_handler.get_actions(),
                                              reward,
                                              next_state_is_final=game_handler.check_if_final_state())
    game_handler.visualize_board()

def run_basic_episode(learning_module, game_handler):
    game_handler.reset_board()
    action = learning_module.initialize_episode(game_handler.get_board_state(), 
                                                game_handler.get_actions())
    game_handler.perform_action(action)
    while not game_handler.check_if_final_state():
        action = learning_module.basic_step(game_handler.get_board_state(), 
                                            game_handler.get_actions())
        game_handler.perform_action(action)
    game_handler.visualize_board()

if __name__ == '__main__':
    main()