from TOPP.TOPP import TOPP
from SimWorld.StateManager import StateManager 
from SimWorld.Board import Board
from NeuralNetworks.ActorCritic import ActorCritic
from configparser import ConfigParser
import pathlib
import ast

def main():

    config = ConfigParser()
    config_path = str(pathlib.Path(__file__).parent.absolute()) + "/config.ini"
    config.read(config_path)

    actor0_id = config['ActorDuel']['actor0_id']
    actor0_number = config.getint('ActorDuel','actor0_number')
    actor1_id = config['ActorDuel']['actor1_id']
    actor1_number = config.getint('ActorDuel','actor1_number')
    number_of_games = config.getint('ActorDuel','number_of_games')

    config = ConfigParser()
    actor0_folder = str(pathlib.Path(str(pathlib.Path(__file__).parent.absolute()))) + \
                       "/DuelingChamber/" + actor0_id + "/"
    config.read(actor0_folder+"config.ini")

    actor0 = ActorCritic(   learning_rate=config.getfloat('anet', 'learning_rate'),
                            layers=ast.literal_eval(config['anet']['hidden_layers']),
                            opt=config['anet']['optimizer'],
                            act=config['anet']['activation'],
                            last_act=config['anet']['last_activation'],
                            input_size=config.getint('board', 'size') * config.getint('board', 'size') + 1,
                            epochs=config.getint('anet', 'epochs'),
                            batch_size=config.getint('anet', 'batch_size'),
                            validation_split=config.getfloat('anet', 'validation_split'),
                            verbosity=config.getint('anet', 'verbosity'),
                            net_with_critic = config.getboolean('anet', 'net_with_critic'),
                            reload_model=True,
                            reload_name=pathlib.Path(actor0_folder+"ANET"+str(actor0_number))
                            )
    
    config = ConfigParser()
    actor1_folder = str(pathlib.Path(str(pathlib.Path(__file__).parent.absolute()))) + \
                       "/DuelingChamber/" + actor1_id + "/"
    config.read(actor1_folder+"config.ini")

    actor1 = ActorCritic(   learning_rate=config.getfloat('anet', 'learning_rate'),
                            layers=ast.literal_eval(config['anet']['hidden_layers']),
                            opt=config['anet']['optimizer'],
                            act=config['anet']['activation'],
                            last_act=config['anet']['last_activation'],
                            input_size=config.getint('board', 'size') * config.getint('board', 'size') + 1,
                            epochs=config.getint('anet', 'epochs'),
                            batch_size=config.getint('anet', 'batch_size'),
                            validation_split=config.getfloat('anet', 'validation_split'),
                            verbosity=config.getint('anet', 'verbosity'),
                            net_with_critic = config.getboolean('anet', 'net_with_critic'),
                            reload_model=True,
                            reload_name=pathlib.Path(actor1_folder+"ANET"+str(actor1_number))
                            )



    tournament = TOPP(2,
                      number_of_games,
                      StateManager(),
                      Board(6),
                      actor0_id,
                      False,
                      config_path)

    tournament.run_tournament([actor0, actor1], [])
    tournament.print_standings()


if __name__ == '__main__':
    main()