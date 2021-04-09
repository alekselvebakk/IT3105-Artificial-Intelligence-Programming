import ast
from MCTS.MCTS import MCTS
from NeuralNetworks.Actor import Actor
from SimWorld.Board import Board
from SimWorld.StateManager import StateManager
from configparser import ConfigParser
import pathlib


def main():
    """


    #Hente ut variabler fra settings-fil
    c = settings.c #Pseudocode
    tree_games = settings.tree_games
    number_actual_games = settings.number_actual_games

    GameHandler = GameHandler() #Pseudocode
    """
    #Alt inne i bolk er pseudokode ca.

    config = ConfigParser()
    config_path = str(pathlib.Path(__file__).parent.absolute()) + "/config.ini"
    config.read(config_path)


    actor = Actor(learning_rate=config.getfloat('actor','learning_rate'),
                  layers=ast.literal_eval(config['actor']['hidden_layers']),
                  opt=config['actor']['optimizer'],
                  act=config['actor']['activation'],
                  last_act=config['actor']['last_activation'],
                  input_size=config.getint('board', 'size')+1,
                  minibatch_size=config.getint('actor', 'minibatch')
                  )
    c = config.getfloat('MCTS', 'exploration_weight')
    tree_games = config.getint('MCTS', 'tree_games')
    number_actual_games = config.getint('MCTS', 'actual_games')

    state_manager = StateManager()

    RBUF = []


    for i in range(number_actual_games):
        Board_A = Board(size=config.getint('board', 'size')) #create actual board
        Board_MC = Board(size=config.getint('board', 'size')) #MonteCarlo Board
        MCTS_tree = MCTS(state_manager, Board_A)

        while not state_manager.state_is_final(Board_MC):
            current_state = state_manager.get_state(Board_A)
            state_manager.set_state(Board_MC, current_state) #Make Monte Carlo Board
            MCTS_tree.update_and_reset_tree(current_state)

            for i in range(tree_games):
                #Tree simulation
                state, action, finished = MCTS_tree.tree_simulation(Board_MC, c)

                #Updating GameHandler to tree-state
                state_manager.set_state(Board_MC, state)

                #Using default policy to get to end state
                if not finished: 
                    state_manager.perform_action(Board_MC, action)
                    while not state_manager.state_is_final(Board_MC):
                        action = actor.get_action(state_manager.get_state(Board_MC))
                        state_manager.perform_action(Board_MC, action)
                
                #Updating MCTS-Tree
                z = state_manager.get_result(Board_MC)
                MCTS.backprop_tree(z)
                MCTS.update_and_reset_tree(current_state)


            #Saving distribution to RBUF

            
            D = MCTS.get_root_distribution()
            RBUF.append(D)


            #Choosing and performing best action
            action = MCTS_tree.get_best_root_action()
            state_manager.perform_action(Board_A, action)

        #Train actor after actual game is finished    
        actor.train_from_RBUF(RBUF)



if __name__ == '__main__':
    main()