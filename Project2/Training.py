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
    actor_critic = ActorCritic(learning_rate=config.getfloat('actor', 'learning_rate'),
                               layers=ast.literal_eval(config['actor']['hidden_layers']),
                               opt=config['actor']['optimizer'],
                               act=config['actor']['activation'],
                               last_act=config['actor']['last_activation'],
                               input_size=config.getint('board', 'size') * config.getint('board', 'size') + 1,
                               minibatch_size=config.getint('actor', 'minibatch'),
                               epochs=config.getint('actor', 'epochs'),
                               batch_size=config.getint('actor', 'batch_size'),
                               validation_split=config.getfloat('actor', 'validation_split'),
                               verbosity=config.getint('actor', 'verbosity')
                               )

    # Extract critic settings
    gamma = config.getfloat('critic', 'discount_factor')

    # Create RL module
    rl = ReinforcementLearning()

    # Extract training settings
    c = config.getfloat('MCTS', 'exploration_weight')
    tree_games = config.getint('MCTS', 'tree_games')
    number_actual_games = config.getint('MCTS', 'actual_games')
    RBUF = []

    # Board setup
    state_manager = StateManager()
    Board_A = Board(size=config.getint('board', 'size'))  # Actual board
    Board_MC = Board(size=config.getint('board', 'size'))  # MonteCarlo Board

    # Extract TOPP settings
    topp = TOPP(config.getint('TOPP', 'number_of_nets'),
                config.getint('TOPP', 'games_between_nets'),
                state_manager,
                Board_A,
                int(time.time()),
                config.getboolean('actor', 'save_actors')
                )

    # Training loop
    for j in range(number_actual_games):
        state_manager.reset_board(Board_A)
        state_manager.reset_board(Board_MC)
        MCTS_tree = MCTS(state_manager, Board_A)

        # Alternates starting player every game
        if j % 2 == 1:
            state_manager.change_player(Board_A)
            state_manager.change_player(Board_MC)

        while not state_manager.state_is_final(Board_A):
            current_state = state_manager.get_state(Board_A)
            state_manager.set_state(Board_MC, current_state)
            MCTS_tree.update_and_reset_tree(current_state)

            use_critic = True

            for i in range(tree_games):
                # Tree simulation
                state, action, finished = MCTS_tree.tree_simulation(Board_MC, c)

                # Using default policy to get to end state and returns result of game
                z = rl.get_simulation_result(state_manager, Board_MC, use_critic, actor_critic, action, finished)

                # Updating MCTS-Tree
                MCTS_tree.backprop_tree(z, gamma)
                MCTS_tree.update_and_reset_tree(current_state)
                state_manager.set_state(Board_MC, MCTS_tree.root)

            # Saving distribution to RBUF
            D = MCTS_tree.get_root_distribution_and_value()
            RBUF.append(D)
            if use_critic: rl.update_critic_indices(RBUF)

            # Choosing and performing best action
            action = MCTS_tree.get_best_root_action()
            rl.perform_real_move(state_manager, Board_A, action)

        # Parse RBUF
        rl.update_RBUF_critic_values(state_manager, Board_A, RBUF)

        # Train actor after finished episode
        rl.train_actor_critic(RBUF, actor_critic)

        # Save net every x game to file
        topp.save_net(actor_critic, j, number_actual_games)

        # Print progress
        rl.print_progress()

    # Decay epsilon for greedy policy
    rl.decay_epsilon()

    # Save final net
    topp.save_net(actor_critic, number_actual_games, number_actual_games)

    # Save config-file with information about the actor settings
    topp.save_config(config_path)


if __name__ == '__main__':
    main()
