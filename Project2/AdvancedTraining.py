from MCTS.MCTS import MCTS
from NeuralNetworks.ActorCritic import ActorCritic
from SimWorld.Board import Board
from SimWorld.StateManager import StateManager
from ReinforcementLearning.ReinforcementLearning import ReinforcementLearning
from TOPP.TOPP import TOPP

from configparser import ConfigParser
from shutil import copyfile
import ast
import pathlib
import time
import os
import random


def main():
    config = ConfigParser()
    config_path = str(pathlib.Path(__file__).parent.absolute()) + "/config.ini"
    config.read(config_path)

    preloaded_actor_id = config['AdvancedTraining']['preloaded_actor_id']


    # Extract actor settings
    if not preloaded_actor_id:
        actor_critic = ActorCritic(learning_rate=config.getfloat('anet', 'learning_rate'),
                    layers=ast.literal_eval(config['anet']['hidden_layers']),
                    opt=config['anet']['optimizer'],
                    act=config['anet']['activation'],
                    last_act=config['anet']['last_activation'],
                    input_size=config.getint('board', 'size') * config.getint('board', 'size') + 1,
                    epochs=config.getint('anet', 'epochs'),
                    batch_size=config.getint('anet', 'batch_size'),
                    validation_split=config.getfloat('anet', 'validation_split'),
                    verbosity=config.getint('anet', 'verbosity'),
                    net_with_critic = config.getboolean('anet', 'net_with_critic')
                    )
    else:
    # TEST
        actor_number = config['AdvancedTraining']['preloaded_actor_number']
        actor_path = str(pathlib.Path(str(pathlib.Path(__file__).parent.absolute()))) + \
                        "/Hall_of_Fame/"+preloaded_actor_id+"/ANET"+actor_number

        actor_critic = ActorCritic(learning_rate=config.getfloat('anet', 'learning_rate'),
                    layers=ast.literal_eval(config['anet']['hidden_layers']),
                    opt=config['anet']['optimizer'],
                    act=config['anet']['activation'],
                    last_act=config['anet']['last_activation'],
                    input_size=config.getint('board', 'size') * config.getint('board', 'size')+1,
                    epochs=config.getint('anet', 'epochs'),
                    batch_size=config.getint('anet', 'batch_size'),
                    validation_split=config.getfloat('anet', 'validation_split'),
                    verbosity=config.getint('anet', 'verbosity'),
                    net_with_critic=config.getboolean('anet', 'net_with_critic'),
                    reload_model=True,
                    reload_name=actor_path
                    )

    # Create RL mondule
    rl = ReinforcementLearning( config.getint('RL', 'actual_games'), 
                                config.getfloat('RL','reward_discount_factor'),
                                config.getfloat('RL','rollout_initial_probability'),
                                config.getfloat('RL','rollout_final_probability'),
                                config.getfloat('RL', 'epsilon_initial'),
                                config.getfloat('RL', 'epsilon_final'),
                                config.getfloat('RL', 'percent_before_critic'),
                                config.getfloat('RL', 'winning_reward'),
                                config.getfloat('RL', 'losing_reward'),
                                net_with_critic = config.getboolean('anet', 'net_with_critic'),
                                minibatch_size=config.getint('RL', 'minibatch'),
                                increasing_prob_data = config.getboolean('RL','increasing_prob_data'),
                                RBUF_max_size = config.getint('RL', 'RBUF_trimmed_max_size'),
                                RBUF_trimming = config.getboolean('RL', 'RBUF_trimming')
                                )
    
    # Board setup
    see_game = ast.literal_eval(config['board']['see_game'])
    state_manager = StateManager()
    Board_A = Board(size=config.getint('board', 'size'))  # Actual board
    Board_MC = Board(size=config.getint('board', 'size'))  # MonteCarlo Board

    # Tournament of Progressive Players Setup
    topp = TOPP(config.getint('TOPP', 'number_of_nets'),
                config.getint('TOPP', 'games_between_nets'),
                state_manager,
                Board_A,
                str(int(time.time())),
                config.getboolean('TOPP', 'save_actors'),
                str(pathlib.Path(__file__).parent.absolute()) + "/Saved_Nets/"
                )
    
    Hall_of_Fame_path = str(pathlib.Path(__file__).parent.absolute())+"/Hall_of_Fame"
    opponent_folders = next(os.walk(Hall_of_Fame_path))[1]
    opponent_folders.sort()
    opponents = []
    net_number = ast.literal_eval(config['AdvancedTraining']['net_number'])
    print(opponent_folders)

    for i in range(len(opponent_folders)):
        print(i)
        if opponent_folders[i] != preloaded_actor_id:
            actor_specific_config = ConfigParser()
            current_folder = Hall_of_Fame_path+"/"+opponent_folders[i]
            actor_specific_config.read(current_folder+"/config.ini")

            actor_path = current_folder+"/ANET"+str(net_number[i])

            opponents.append(ActorCritic(learning_rate=actor_specific_config.getfloat('anet', 'learning_rate'),
                        layers=ast.literal_eval(actor_specific_config['anet']['hidden_layers']),
                        opt=actor_specific_config['anet']['optimizer'],
                        act=actor_specific_config['anet']['activation'],
                        last_act=actor_specific_config['anet']['last_activation'],
                        input_size=actor_specific_config.getint('board', 'size') * config.getint('board', 'size') + 1,
                        epochs=actor_specific_config.getint('anet', 'epochs'),
                        batch_size=actor_specific_config.getint('anet', 'batch_size'),
                        validation_split=actor_specific_config.getfloat('anet', 'validation_split'),
                        verbosity=actor_specific_config.getint('anet', 'verbosity'),
                        net_with_critic=actor_specific_config.getboolean('anet', 'net_with_critic'),
                        reload_model=True,
                        reload_name=actor_path
                        ))


    # Training loop
    for j in range(rl.number_actual_games):
        state_manager.reset_board(Board_A)
        state_manager.reset_board(Board_MC)
        current_player = 1

        if j in see_game:
            Board_A.start_visualisation('game_episode_'+str(j)+'.gif')
        
        # Alternates starting player every game
        if j % 2 == 1:
            state_manager.change_player(Board_A)
            state_manager.change_player(Board_MC)
            current_player = 2

        #Tree setup
        MCTS_tree = MCTS(   state_manager, 
                            Board_A, config.getfloat('MCTS', 'exploration_weight'), 
                            config.getint('MCTS', 'tree_games'),
                            time_for_rollouts = config.getfloat('MCTS','time_limit'))
                            
        if j == 0:
            #Save first untrained actor
            topp.save_net(actor_critic, j, rl.number_actual_games)
            # Save config-file with information about the actor settings
            topp.save_config(config_path)
        

        while not state_manager.state_is_final(Board_A):
            state_manager.set_state(Board_MC, state_manager.get_state(Board_A))
            MCTS_tree.update_and_reset_tree(Board_A)
            rl.decide_use_of_critic()
            
            
            tree_game_start_time = time.time()
            i = 0
            #Choosing only one actor to play at the time:
            actors = opponents + [actor_critic] 
            current_opponent = random.choice(actors)

            #Choosing an opponent to play agians:
            #opponent = random.choice(opponents)

            
            while i < MCTS_tree.tree_games and (time.time()-tree_game_start_time)<MCTS_tree.time_for_rollouts:
                # Tree simulation
                action, finished = MCTS_tree.tree_simulation(Board_MC)
                # Using default policy to get to end state and returns result of game
                z = rl.get_simulation_results_with_opponent(state_manager, Board_MC, actor_critic, current_opponent, current_player, action, finished)
                
                # Updating MCTS-Tree
                MCTS_tree.backprop_tree(z)
                MCTS_tree.update_and_reset_tree(Board_A)
                state_manager.set_state(Board_MC, MCTS_tree.root)
                i += 1
            print(i)
            # Saving distribution to RBUF
            D = MCTS_tree.get_RBUF_data(rl.net_with_critic)
            rl.add_to_RBUF(D, j)

            # Choosing and performing best action
            action = MCTS_tree.get_best_root_action()
            rl.perform_real_move(state_manager, Board_A, action)
        #Parse RBUF
        rl.update_RBUF_critic_values(state_manager.get_result(Board_A))

        # Train actor after finished episode
        rl.train_actor_critic(actor_critic)

        # Save net every x game to file
        if j != 0:
            topp.save_net(actor_critic, j, rl.number_actual_games)

        # Decay epsilon for greedy policy 
        rl.decay_epsilon()
        
        # increase chance for use of critic
        rl.increase_chance_of_critic(j)

        # Print progress
        rl.print_progress(j)

        # Shows game if it is in see_game
        if j in see_game:
            state_manager.show_animation(Board_A)
            Board_A.stop_visualization()

    # Save final net
    topp.save_net(actor_critic, rl.number_actual_games, rl.number_actual_games)



if __name__ == '__main__':
    main()
