from TOPP.TOPP import TOPP
from SimWorld.StateManager import StateManager
from SimWorld.Board import Board
from NeuralNetworks.ActorCritic import ActorCritic

from configparser import ConfigParser
import pathlib
import ast


def main():
    tournament_number = input("What tournament would you like to play?\n")

    # Reading config-file with the same settings as used in training
    config = ConfigParser()
    tournament__path = str(pathlib.Path(str(pathlib.Path(__file__).parent.absolute()))) + \
                       "/Saved_Nets/" + tournament_number + "/"
    config.read(tournament__path + "config.ini")

    # Extract TOPP settings
    number_of_nets = config.getint('TOPP', 'number_of_nets')
    games_between_nets = config.getint('TOPP', 'games_between_nets')
    board_size = config.getint('board', 'size')
    training_games = config.getint('RL', 'actual_games')

    # Extracting Actor settings
    lr = config.getfloat('anet', 'learning_rate')
    layers = ast.literal_eval(config['anet']['hidden_layers'])
    opt = config['anet']['optimizer']
    act = config['anet']['activation']
    last_act = config['anet']['last_activation']
    input_size = config.getint('board', 'size') * config.getint('board', 'size') + 1
    mb_size = config.getint('anet', 'minibatch')
    epochs = config.getint('anet', 'epochs')
    b_size = config.getint('anet', 'batch_size')
    val_split = config.getfloat('anet', 'validation_split')
    verbose = config.getint('anet', 'verbosity')

    # Creating tournament
    tournament = TOPP(number_of_nets,
                      games_between_nets,
                      StateManager(),
                      Board(board_size),
                      int(tournament_number),
                      False,
                      tournament__path)

    # Creating actors with the saved nets from training
    actor_list = []
    for i in range(number_of_nets):
        actor_number = training_games if i == (number_of_nets - 1) else int(training_games / (number_of_nets - 1)) * i
        print(actor_number)
        actor_name = pathlib.Path(tournament__path + "ANET" + str(int(actor_number)))
        actor_list.append(ActorCritic(learning_rate=lr, layers=layers, opt=opt, act=act, last_act=last_act,
                                      input_size=input_size, minibatch_size=mb_size, epochs=epochs, batch_size=b_size,
                                      validation_split=val_split, verbosity=verbose,
                                      reload_model=True, reload_name=actor_name))

    results = tournament.run_tournament(actor_list)
    print(results)


if __name__ == '__main__':
    main()