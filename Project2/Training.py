from MCTS.MCTS import MCTS
from NeuralNetworks.Actor import Actor
from SimWorld.Board import Board
from SimWorld.StateManager import StateManager

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
    actor = Actor(learning_rate=config.getfloat('actor', 'learning_rate'),
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
    train_actors = config.getboolean('actor', 'train_actors')
    save_actors = config.getboolean('actor', 'save_actors')

    # Extract training settings
    c = config.getfloat('MCTS', 'exploration_weight')
    tree_games = config.getint('MCTS', 'tree_games')
    number_actual_games = config.getint('MCTS', 'actual_games')
    RBUF = []

    # Extract TOPP settings
    number_of_nets = config.getint('TOPP', 'number_of_nets')
    save_step = int(number_actual_games / (number_of_nets - 1))

    # Board setup
    state_manager = StateManager()
    Board_A = Board(size=config.getint('board', 'size'))  # create actual board
    Board_MC = Board(size=config.getint('board', 'size'))  # MonteCarlo Board

    # Training loop
    for j in range(number_actual_games):
        state_manager.reset_board(Board_A)
        state_manager.reset_board(Board_MC)
        MCTS_tree = MCTS(state_manager, Board_A)  # TODO: burde dette være MC board?
        # og skal vi lage et nytt tre hver gang=

        # Alternates starting player every other game
        if j % 2 == 1:
            state_manager.change_player(Board_A)
            state_manager.change_player(Board_MC)

        while not state_manager.state_is_final(Board_A):
            current_state = state_manager.get_state(Board_A)
            state_manager.set_state(Board_MC, current_state)
            MCTS_tree.update_and_reset_tree(current_state)

            for i in range(tree_games):
                # Tree simulation
                state, action, finished = MCTS_tree.tree_simulation(Board_MC, c)

                # Using default policy to get to end state
                if not finished:
                    state_manager.perform_action(Board_MC, action)
                    while not state_manager.state_is_final(Board_MC):
                        action = actor.get_action(state_manager.get_state(Board_MC))
                        state_manager.perform_action(Board_MC, action)

                # Updating MCTS-Tree
                z = state_manager.get_result(Board_MC)
                MCTS_tree.backprop_tree(z)
                MCTS_tree.update_and_reset_tree(current_state)
                state_manager.set_state(Board_MC, MCTS_tree.root)

            # Saving distribution to RBUF
            D = MCTS_tree.get_root_distribution()
            RBUF.append(D)

            # Choosing and performing best action
            action = MCTS_tree.get_best_root_action()
            state_manager.perform_action(Board_A, action)

        # Train actor after finished episode
        print("Training starts, RBUF has", len(RBUF), "samples")
        actor.train_from_RBUF(RBUF)

        # Save net every x game to file
        if j % save_step == 0 and j / save_step != (number_of_nets - 1):
            if j == 0:
                tournament_id = str(int(time.time()))
                folder_name = str(pathlib.Path(__file__).parent.absolute()) + "/Saved_Nets/" + tournament_id

            # Save net
            neural_net_name = folder_name + "/ANET" + str(j)
            actor.save_net(neural_net_name)

        # Print progress
        progress = str(float(j) / number_actual_games * 100) + "%"
        print(progress)

    # Save final net
    NeuralNetName = folder_name + "/ANET" + str(number_actual_games)
    actor.save_net(NeuralNetName)

    # Save config-file with information about the actor settings
    copyfile(config_path, folder_name + "/config.ini")


if __name__ == '__main__':
    main()
