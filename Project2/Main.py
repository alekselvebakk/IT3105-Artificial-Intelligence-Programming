from MCTS.MCTS import MCTS
from NeuralNetworks.Actor import Actor
from configparser import ConfigParser
import pathlib


def main():
    """
    
    config = ConfigParser()
    config_path = str(pathlib.Path(__file__).parent.absolute())+"/config.ini"
    config.read(config_path)

    settings = settings_file(True) #Pseudocode

    #Hente ut variabler fra settings-fil
    c = settings.c #Pseudocode
    tree_games = settings.tree_games
    number_actual_games = settings.number_actual_games

    GameHandler = GameHandler() #Pseudocode
    """
    #Alt inne i bolk er pseudokode ca.
    
    Actor = Actor()


    RBUF = []


    for i in range(number_actual_games):
        Board_A = Board() #create actual board
        Board_MC = Board() #MonteCarlo Board
        MCTS_tree = MCTS(GameHandler, Board_A)

        while not GameHandler.state_is_final(Board_MC):
            current_state = GameHandler.get_state(Board_A)
            GameHandler.set_state(Board_MC, current_state) #Make Monte Carlo Board
            MCTS_tree.update_and_reset_tree(current_state)

            for i in range(tree_games):
                #Tree simulation
                state, action, finished = MCTS_tree.tree_simulation(Board_MC, c)

                #Updating GameHandler to tree-state
                GameHandler.set_state(Board_MC, state)

                #Using default policy to get to end state
                if not finished: 
                    GameHandler.perform_action(Board_MC, action)
                    while not GameHandler.state_is_final(Board_MC):
                        action = actor.get_action(GameHandler.get_state(Board_MC))
                        GameHandler.perform_action(Board_MC, action)
                
                #Updating MCTS-Tree
                z = GameHandler.get_result(Board_MC)
                MCTS.backprop_tree(z)
                MCTS.update_and_reset_tree(current_state)


            #Saving distribution to RBUF

            
            D = MCTS.get_root_distribution()
            RBUF.append(D)


            #Choosing and performing best action
            action = MCTS_tree.get_best_root_action()
            GameHandler.perform_action(Board_A, action)

        #Train actor after actual game is finished    
        actor.train_from_RBUF(RBUF)



if __name__ == '__main__':
    main()