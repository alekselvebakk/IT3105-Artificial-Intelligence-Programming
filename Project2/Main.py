from MCTS.MCTS import MCTS
from NeuralNetworks.Actor import Actor
from configparser import ConfigParser
import pathlib


def main():
    """
    Logikk for å sette opp GameHandler
    """
    config = ConfigParser()
    config_path = str(pathlib.Path(__file__).parent.absolute())+"/config.ini"
    config.read(config_path)

    settings = settings_file(True) #Pseudocode
    GameHandler = GameHandler(True) #Pseudocode

    #Hente ut variabler fra settings-fil
    s_0 = settings.starting_state #Pseudocode
    c = settings.c #Pseudocode
    tree_games = settings.tree_games

    GameHandler.set_state(s_0)
    number_actual_games = settings.number_actual_games
    actor = Actor()
    RBUF = []


    for i in range(number_actual_games):
        MCTS_tree = MCTS(GameHandler, s_0)
        while not GameHandler.state_is_final():

            #Saving state from real training game and updating tree to match it
            current_state = GameHandler.get_state()
            MCTS_tree.update_and_reset_tree(current_state)

            for i in range(tree_games):
                #Tree simulation
                state, action, finished = MCTS_tree.tree_simulation(c)

                #Updating GameHandler to tree-state
                GameHandler.set_state(state)

                #Using default policy to get to end state
                if not finished: 
                    GameHandler.perform_action(action)
                    while not GameHandler.state_is_final():
                        action = actor.get_action(GameHandler.get_state())
                        GameHandler.perform_action(action)
                
                #Updating MCTS-Tree
                z = GameHandler.get_result()
                MCTS.backprop_tree(z)
                MCTS.update_and_reset_tree(current_state)


            #Saving distribution to RBUF

            
            D = MCTS.get_root_distribution()
            RBUF.append(D)

            #Setting Gamehandler back to real state
            GameHandler.set_state(current_state)

            #Choosing and performing best action
            action = MCTS_tree.get_best_root_action()
            GameHandler.perform_action(action)

        #Train actor after actual game is finished    
        actor.train_from_RBUF(RBUF)



if __name__ == '__main__':
    main()