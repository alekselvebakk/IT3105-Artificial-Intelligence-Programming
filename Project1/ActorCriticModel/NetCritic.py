import tensorflow as tf 
from tensorflow import keras
import numpy as np

class NetCritic:
    def __init__(self, alpha, gamma, elig_decay,layers):
        
        #Constants
        self.alpha = tf.constant(alpha)
        self.gamma = tf.constant(gamma)
        self.elig_decay = tf.constant(elig_decay)

        ##### MODEL aka NEURAL NET CREATION #####
        #Initialize model
        #Sequential model is just normal deep neural net model
        self.model = keras.models.Sequential(name = 'SequentialCriticModel')

        #Add hidden dense layers
        #dense layers are just "every neuron connected to every other neuron"-layers
        for i in len(layers):
            layer_name = "hidden_layer"+i
            current_layer = keras.layers.Dense( layers[i],
                                                activation='relu', 
                                                name = layer_name)
            self.model.add(current_layer)
            self.model.summary()

        #Add output layer, this will be used to get the value
        output_layer = keras.layers.Dense(  1, 
                                            activation = 'sigmoid', 
                                            name = "outputlayer")
        self.model.add(output_layer)
        optim = keras.optimizers.SGD(learning_rate=self.alpha)
        self.model.compile(optim)

        #Set initial eligibilities
        self.elig = tf.Variable()

    def episode_reset(self):
        weights = self.model.trainable_weights
        self.elig = tf.zeros_like(weights)

    def get_delta(self, state, next_state, reward):
        V_s = self.model(state)
        V_s_next = self.model(next_state)
        delta = reward + self.gamma*V_s_next-V_s
        return delta

    def update_value_function(self, state, delta):
        #This is the custom fit-function
        params = self.model.trainable_weights
        with tf.GradientTape() as tape:
            V_s = self.model(state)
            dV_dw = tape.gradient(V_s, params)
            self.elig = self.elig + dV_dw
            update = delta*self.elig
            self.model.optimizer.apply_gradients(   update, 
                                                    self.model.trainable_weights)
        self.elig = self.elig*self.elig_decay
        
