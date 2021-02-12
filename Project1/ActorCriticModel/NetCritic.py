import tensorflow as tf 
from tensorflow import keras
from splitgd import SplitGD

class NetCritic:
    def __init__(self, alpha, layers):
        
        



        ##### MODEL aka NEURAL NET CREATION #####
        
        #Initialize model
        #Sequential model is just normal deep neural net model
        self.model = keras.models.Sequential(name = 'SequentialCriticModel')

        #Add hidden dense layers
        #dense layers are just "every neuron connected to every other neuron"-layers
        for i in len(layers)-1:
            name = "layer"+i
            current_layer = keras.layers.Dense(layers[i],activation='relu', name = name)
            self.model.add(current_layer)
            self.model.summary()
        
        #Add output layer, this will be used to classify
        output_layer = keras.layers.Dense(layers[-1], activation = 'relu', name = "outputlayer")
        self.model.add(output_layer)
        
        #Constants
        