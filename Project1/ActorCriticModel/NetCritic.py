import tensorflow as tf 
from tensorflow import keras
import numpy as np

class NetCritic:
    def __init__(self, alpha, gamma, elig_decay, layers, input_size):
        
        #Constants
        self.alpha = tf.constant(alpha)
        self.gamma = tf.constant(gamma)
        self.elig_decay = tf.constant(elig_decay)

        ##### MODEL aka NEURAL NET CREATION #####
        #Initialize model
        
        self.model = keras.models.Sequential(   name = 'SequentialCriticModel')
        #Add input layer
        self.model.add(keras.Input(shape=(input_size,)))
        self.model.summary()
        #Add hidden dense layers
        for i in range(len(layers)):
            layer_name = "hidden_layer"+str(i)
            self.model.add(keras.layers.Dense( layers[i],
                                                activation='relu', 
                                                name = layer_name))
            self.model.summary()
        print(self.model.weights)
        #Add output layer, this will be used to get the value
        output_layer = keras.layers.Dense(  1, 
                                            activation = 'sigmoid', 
                                            name = "outputlayer")
        self.model.add(output_layer)
        optim = keras.optimizers.SGD(learning_rate=self.alpha)
        self.model.summary()

        #self.model.compile(optim)
        self.model.summary()

        #Set initial eligibilities
        weights = self.model.trainable_weights
        self.elig = []
        for i in range(len(weights)):
            self.elig.append(tf.zeros_like(weights[i]))

    @staticmethod
    def string_to_tensor(string_variable):
        np_array = np.array(list(string_variable), dtype=int)
        print(np_array.shape)
        #np_array = np_array.reshape(len(np_array),1)
        variable = tf.Variable(tf.constant(np_array))
        return variable

    def episode_reset(self):
        for i in range(len(self.model.trainable_weights)):
            self.elig.append(tf.zeros_like(self.model.trainable_weights[i]))


    def get_delta(self, state, next_state, reward):
        state = self.string_to_tensor(state)
        print("------",state[0])
        
        V_s = self.model(state)
        next_state = self.string_to_tensor(next_state)
        V_s_next = self.model(next_state)
        delta = reward + self.gamma*V_s_next-V_s
        return delta

    def update_value_function(self, state, delta):
        #This is the custom fit-function
        state = self.string_to_tensor(state)
        params = self.model.trainable_weights
        with tf.GradientTape() as tape:
            V_s = self.model(state)
            dV_dw = tape.gradient(V_s, params)
            self.elig = self.elig + dV_dw
            update = delta*self.elig
            self.model.optimizer.apply_gradients(   update, 
                                                    self.model.trainable_weights)
        self.elig = self.elig*self.elig_decay
        
