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


def main():
    config = ConfigParser()
    config_path = str(pathlib.Path(__file__).parent.absolute()) + "/config.ini"
    config.read(config_path)

    # Extract actor settings
    actor_critic = ActorCritic(learning_rate=config.getfloat('anet', 'learning_rate'),
                  layers=ast.literal_eval(config['anet']['hidden_layers']),
                  opt=config['anet']['optimizer'],
                  act=config['anet']['activation'],
                  last_act=config['anet']['last_activation'],
                  input_size=config.getint('board', 'size') * config.getint('board', 'size') + 1,
                  minibatch_size=config.getint('anet', 'minibatch'),
                  epochs=config.getint('anet', 'epochs'),
                  batch_size=config.getint('anet', 'batch_size'),
                  validation_split=config.getfloat('anet', 'validation_split'),
                  verbosity=config.getint('anet', 'verbosity')
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
                                config.getfloat('RL', 'losing_reward')
                                )

    # Board setup
    state_manager = StateManager()
    Board_A = Board(size=config.getint('board', 'size'))  # Actual board
    Board_MC = Board(size=config.getint('board', 'size'))  # MonteCarlo Board

    # Tournament of Progressive Players Setup
    topp = TOPP(config.getint('TOPP', 'number_of_nets'),
                config.getint('TOPP', 'games_between_nets'),
                state_manager,
                Board_A,
                int(time.time()),
                config.getboolean('TOPP', 'save_actors'),
                str(pathlib.Path(__file__).parent.absolute()) + "/Saved_Nets/"
                )
    
    


    # Training loop
    for j in range(rl.number_actual_games):
        state_manager.reset_board(Board_A)
        state_manager.reset_board(Board_MC)
        
        #Tree setup
        MCTS_tree = MCTS(   state_manager, 
                            Board_A, config.getfloat('MCTS', 'exploration_weight'), 
                            config.getint('MCTS', 'tree_games'))
        
        # Alternates starting player every game
        if j % 2 == 1:
            state_manager.change_player(Board_A)
            state_manager.change_player(Board_MC)

        while not state_manager.state_is_final(Board_A):
            state_manager.set_state(Board_MC, state_manager.get_state(Board_A))
            MCTS_tree.update_and_reset_tree(Board_A)

            rl.decide_use_of_critic() # TODO: skal settes til en randomizing-funksjon


            tree_game_start_time = time.time()
            i = 0
            while i < MCTS_tree.tree_games and (time.time()-tree_game_start_time)<1:
                # Tree simulation
                action, finished = MCTS_tree.tree_simulation(Board_MC)

                # Using default policy to get to end state and returns result of game
                z = rl.get_simulation_result(state_manager, Board_MC, actor_critic, action, finished)
                
                # Updating MCTS-Tree
                MCTS_tree.backprop_tree(z)
                MCTS_tree.update_and_reset_tree(Board_A)
                state_manager.set_state(Board_MC, MCTS_tree.root)

            # Saving distribution to RBUF
            D = MCTS_tree.get_root_distribution_and_value()
            rl.add_to_RBUF(D)

            # Choosing and performing best action
            action = MCTS_tree.get_best_root_action()
            rl.perform_real_move(state_manager, Board_A, action)
            i += 1

        #Parse RBUF
        rl.update_RBUF_critic_values(state_manager.get_result(Board_A))

        # Train actor after finished episode
        rl.train_actor_critic(actor_critic)

        # Save net every x game to file
        topp.save_net(actor_critic, j, rl.number_actual_games)

        # Decay epsilon for greedy policy 
        rl.decay_epsilon()
        
        # increase chance for use of critic
        rl.increase_chance_of_critic(j)

        # Print progress
        rl.print_progress(j)

    

    # Save final net
    topp.save_net(actor_critic, rl.number_actual_games, rl.number_actual_games)

    # Save config-file with information about the actor settings
    topp.save_config(config_path)


if __name__ == '__main__':
    main()
