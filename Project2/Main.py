import ast
from MCTS.MCTS import MCTS
from NeuralNetworks.Actor import Actor
from SimWorld.Board import Board
from SimWorld.StateManager import StateManager
from TOPP.TOPP import TOPP
from configparser import ConfigParser
import pathlib
import time
from shutil import copyfile


def main():
    config = ConfigParser()
    config_path = str(pathlib.Path(__file__).parent.absolute()) + "/config.ini"
    config.read(config_path)


    #Extract actor settings
    actor = Actor(learning_rate=config.getfloat('actor','learning_rate'),
                  layers=ast.literal_eval(config['actor']['hidden_layers']),
                  opt=config['actor']['optimizer'],
                  act=config['actor']['activation'],
                  last_act=config['actor']['last_activation'],
                  input_size=config.getint('board', 'size')*config.getint('board', 'size')+1,
                  minibatch_size=config.getint('actor', 'minibatch'),
                  epochs=config.getint('actor', 'epochs'),
                  batch_size=config.getint('actor', 'batch_size'),
                  validation_split=config.getfloat('actor','validation_split'),
                  verbosity = config.getint('actor','verbosity')
                  )
    train_actors = config.getboolean('actor','train_actors')
    save_actors = config.getboolean('actor', 'save_actors')

    #Extract training settings              
    c = config.getfloat('MCTS', 'exploration_weight')
    tree_games = config.getint('MCTS', 'tree_games')
    number_actual_games = config.getint('MCTS', 'actual_games')
    RBUF = []

    #Extract TOPP settings
    games_between_nets = config.getint('TOPP', 'games_between_nets')
    number_of_nets = config.getint('TOPP', 'number_of_nets')
    run_tournament = config.getboolean('TOPP', 'run_tournament')
    training_saving_step = number_actual_games/number_of_nets
    current_step = 0



    #Board setup
    state_manager = StateManager()
    Board_A = Board(size=config.getint('board', 'size')) #create actual board
    Board_MC = Board(size=config.getint('board', 'size')) #MonteCarlo Board

    if train_actors:
        #Training Loop
        for j in range(number_actual_games):
            state_manager.reset_board(Board_A)
            state_manager.reset_board(Board_MC)
            MCTS_tree = MCTS(state_manager, Board_A)

            while not state_manager.state_is_final(Board_A):
                current_state = state_manager.get_state(Board_A)
                state_manager.set_state(Board_MC, current_state)
                MCTS_tree.update_and_reset_tree(current_state)
                for i in range(tree_games):
                    
                    #Tree simulation
                    #current = time.time()
                    state, action, finished = MCTS_tree.tree_simulation(Board_MC, c)
                    #print((time.time()-current)*1000, " ms brukt på treesim")

                    #Updating GameHandler to tree-state
                    #current = time.time()
                    state_manager.set_state(Board_MC, state)
                    #print((time.time()-current)*1000, " ms brukt på state-update")

                    #Using default policy to get to end state
                    #current = time.time()
                    if not finished: 
                        state_manager.perform_action(Board_MC, action)
                        while not state_manager.state_is_final(Board_MC):
                            action = actor.get_action(state_manager.get_state(Board_MC))
                            state_manager.perform_action(Board_MC, action)
                    #print((time.time()-current)*1000, " ms brukt på rollout")
                    
                    #Updating MCTS-Tree
                    #current = time.time()
                    z = state_manager.get_result(Board_MC)
                    MCTS_tree.backprop_tree(z)
                    MCTS_tree.update_and_reset_tree(current_state)
                    state_manager.set_state(Board_MC, MCTS_tree.root)   
                    #print((time.time()-current)*1000, " ms brukt på tre-reset")

                #Saving distribution to RBUF
                D = MCTS_tree.get_root_distribution()
                RBUF.append(D)


                #Choosing and performing best action
                action = MCTS_tree.get_best_root_action()
                state_manager.perform_action(Board_A, action)

            #Train actor after actual game is finished
            if j > 10:
                print("training starts, RBUF has",len(RBUF),"samples")  
                actor.train_from_RBUF(RBUF)

            #Save ANET every x games to file
            if save_actors:
                if j == current_step:
                    if j==0:
                        tournament_time = str(int(time.time()))
                        folder_name = str(pathlib.Path(__file__).parent.absolute()) + "/Saved_Nets/"+tournament_time
                        NeuralNetName = folder_name+"/ANET"+str(j)
                        actor.save_net(NeuralNetName)
                        
                        config_path = str(pathlib.Path(__file__).parent.absolute())+"/config.ini"
                        copyfile(config_path, folder_name+"/config.ini")
                    else:
                        NeuralNetName = folder_name+"/ANET"+str(j)
                        actor.save_net(NeuralNetName)
                    
                    current_step = current_step+training_saving_step


            #Print progress
            progress = str(float(j)/number_actual_games *100)+"%"
            print(progress)

        #Save final net  
        NeuralNetName = str(pathlib.Path(__file__).parent.absolute()) + "/Saved_Nets/ANET"+str(number_actual_games)
        actor.save_net(NeuralNetName)


    if run_tournament:
        Tournament = TOPP(number_of_nets, games_between_nets, StateManager(), Board(size=config.getint('board', 'size')))
        actor_list = []
        for i in range(number_of_nets):
            actor_number = number_actual_games/(number_of_nets)*i
            actor_name = pathlib.Path(str(pathlib.Path(__file__).parent.absolute()) + "/Saved_Nets/ANET"+str(int(actor_number)))
            actor_list.append(  Actor(learning_rate=config.getfloat('actor','learning_rate'),
                                layers=ast.literal_eval(config['actor']['hidden_layers']),
                                opt=config['actor']['optimizer'],
                                act=config['actor']['activation'],
                                last_act=config['actor']['last_activation'],
                                input_size=config.getint('board', 'size')*config.getint('board', 'size')+1,
                                minibatch_size=config.getint('actor', 'minibatch'),
                                epochs=config.getint('actor', 'epochs'),
                                batch_size=config.getint('actor', 'batch_size'),
                                validation_split=config.getfloat('actor','validation_split'),
                                verbosity = config.getint('actor','verbosity'),
                                reload_model=True,
                                reload_name=actor_name))
        results = Tournament.run_tournament(actor_list)
        print(results)


if __name__ == '__main__':
    main()