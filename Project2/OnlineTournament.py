from OHT.BasicClientActor import BasicClientActor
from NeuralNetworks.ActorCritic import ActorCritic
from configparser import ConfigParser
import pathlib
import ast


def main():
    config = ConfigParser()
    config_path = str(pathlib.Path(__file__).parent.absolute()) + "/config.ini"
    config.read(config_path)

    neural_net_id = config['OHT']['neural_net_id']
    neural_net_number = config['OHT']['neural_net_number']
    verbose = config.getboolean('OHT','verbose')

    path = str(pathlib.Path(__file__).parent.absolute()) + \
                       "/Saved_Nets/" + neural_net_id+"/"
    config.read(path+"config.ini")




    actorcritic = ActorCritic(  learning_rate=config.getfloat('anet', 'learning_rate'),
                                layers=ast.literal_eval(config['anet']['hidden_layers']),
                                opt=config['anet']['optimizer'],
                                act=config['anet']['activation'],
                                last_act=config['anet']['last_activation'],
                                input_size=config.getint('board', 'size') * config.getint('board', 'size') + 1,
                                minibatch_size=config.getint('anet', 'minibatch'),
                                epochs=config.getint('anet', 'epochs'),
                                batch_size=config.getint('anet', 'batch_size'),
                                validation_split=config.getfloat('anet', 'validation_split'),
                                verbosity=config.getint('anet', 'verbosity'),
                                net_with_critic = config.getboolean('anet', 'net_with_critic'),
                                reload_model=True,
                                reload_name=pathlib.Path(path+"ANET"+neural_net_number)
                                )

    clientactor = BasicClientActor(actorcritic, verbose=verbose)
    clientactor.connect_to_server()

if __name__ == '__main__':
    main()

