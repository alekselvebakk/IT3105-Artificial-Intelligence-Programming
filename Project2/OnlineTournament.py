from OHT.BasicClientActor import BasicClientActor
from NeuralNetworks.ActorCritic import ActorCritic
from configparser import ConfigParser
import pathlib
import ast

def main():
    config = ConfigParser()
    config_path = str(pathlib.Path(__file__).parent.absolute()) + "/config.ini"
    config.read(config_path)

    neural_net_id_0 = config['OHT']['neural_net_id_0']
    neural_net_number_0 = config['OHT']['neural_net_number_0']
    neural_net_id_1 = config['OHT']['neural_net_id_1']
    neural_net_number_1 = config['OHT']['neural_net_number_1']
    verbose = config.getboolean('OHT','verbose')

    path_0 = str(pathlib.Path(__file__).parent.absolute()) + \
                       "/Hall_of_Fame/" + neural_net_id_0+"/"
    config.read(path_0+"config.ini")




    actorcritic0 = ActorCritic(  learning_rate=config.getfloat('anet', 'learning_rate'),
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
                                reload_name=pathlib.Path(path_0+"ANET"+neural_net_number_0)
                                )

    path_1 = str(pathlib.Path(__file__).parent.absolute()) + \
            "/Hall_of_Fame/" + neural_net_id_1 + "/"
    config.read(path_1 + "config.ini")
    actorcritic1 = ActorCritic(learning_rate=config.getfloat('anet', 'learning_rate'),
                               layers=ast.literal_eval(config['anet']['hidden_layers']),
                               opt=config['anet']['optimizer'],
                               act=config['anet']['activation'],
                               last_act=config['anet']['last_activation'],
                               input_size=config.getint('board', 'size') * config.getint('board', 'size') + 1,
                               epochs=config.getint('anet', 'epochs'),
                               batch_size=config.getint('anet', 'batch_size'),
                               validation_split=config.getfloat('anet', 'validation_split'),
                               verbosity=config.getint('anet', 'verbosity'),
                               net_with_critic=config.getboolean('anet', 'net_with_critic'),
                               reload_model=True,
                               reload_name=pathlib.Path(path_1 + "ANET" + neural_net_number_1)
                               )

    clientactor = BasicClientActor(actorcritic0, actorcritic1, verbose=verbose)
    clientactor.connect_to_server()

if __name__ == '__main__':
    main()

