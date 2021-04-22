from TOPP.TOPP import TOPP
from SimWorld.StateManager import StateManager
from SimWorld.Board import Board
from NeuralNetworks.ActorCritic import ActorCritic

from configparser import ConfigParser
import pathlib
import ast
import os


def main():
    config = ConfigParser()
    config_path = str(pathlib.Path(__file__).parent.absolute()) + "/config.ini"
    config.read(config_path)

    games_played = config.getint('HoFTournament','games_played')
    net_number = ast.literal_eval(config['HoFTournament']['net_number'])

    Hall_of_Fame_path = str(pathlib.Path(__file__).parent.absolute())+"/DuelingChamber"
    player_folders = next(os.walk(Hall_of_Fame_path))[1]
    player_folders.sort()
    players = []
    print(player_folders)

    for i in range(len(player_folders)):
        actor_specific_config = ConfigParser()
        current_folder = Hall_of_Fame_path+"/"+player_folders[i]
        actor_specific_config.read(current_folder+"/config.ini")

        actor_path = current_folder+"/ANET"+str(net_number[i])

        players.append(ActorCritic(learning_rate=actor_specific_config.getfloat('anet', 'learning_rate'),
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

    tournament = TOPP(len(players),
                      games_played,
                      StateManager(),
                      Board(6),
                      str(1), #filler setting
                      False,
                      config_path) #filler setting

    tournament.run_tournament(players,[])
    print(player_folders)
    tournament.print_standings()

if __name__ == '__main__':
    main()