
import tensorflow as tf 
from tensorflow import keras
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
import numpy as np

class NetCritic:
    def __init__(self, alpha, gamma, elig_decay, layers, input_size):
        
        #Constants
        self.alpha = alpha
        self.gamma = gamma
        self.elig_decay = elig_decay
        self.layers = layers


        ##### MODEL aka NEURAL NET CREATION #####
        #Initialize model
        self.model = keras.models.Sequential(name = 'SequentialCriticModel')

        #First layer with input specified
        input_layer = keras.layers.Dense(   self.layers[0], 
                                            input_shape=(input_size,),
                                            name = 'hidden_layer0')
        self.model.add(input_layer)
        #Hidden layers
        for i in range(len(self.layers)-1):
            layer_name = "hidden_layer"+str(i+1)
            current_hidden = keras.layers.Dense(self.layers[i+1],
                                                activation='relu', 
                                                name = layer_name)
            self.model.add(current_hidden)
        
        #Outputlayer
        output_layer = keras.layers.Dense(  1, 
                                            activation = 'sigmoid', 
                                            name = "outputlayer")
        self.model.add(output_layer)
        self.model.summary()

        optim = keras.optimizers.Adam(learning_rate=self.alpha)   #SGD(learning_rate=self.alpha)
        self.model.compile(optimizer = optim)


        #Set initial eligibilities
        vars = self.model.trainable_variables
        self.elig =[]
        for var in vars:
            self.elig.append(tf.zeros_like(var))

    @staticmethod
    def string_to_tensor(string_variable):
        content = list(string_variable)
        variable = np.array([content], dtype=int)
        #variable = variable.reshape(len(variable),1)
        #variable = tf.Variable(tf.constant(variable), shape = tf.TensorShape([len(variable),1]))
        return variable

    def episode_reset(self):
        for i in range(len(self.elig)):
            self.elig[i] = tf.math.scalar_mul(0, self.elig[i])

    def get_delta(self, state, next_state, reward):
        state = self.string_to_tensor(state)
        V_s = self.model(state)
        next_state = self.string_to_tensor(next_state)
        V_s_next = self.model(next_state)
        delta = reward + self.gamma*V_s_next-V_s
        return delta.numpy()[0][0]

    def decay_elibilities(self):
        for i in range(len(self.elig)):
            self.elig[i] = tf.math.scalar_mul(self.elig_decay, self.elig[i])


    def update_value_function(self, state, delta):
        #This is the custom fit-function
        state = self.string_to_tensor(state)
        params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            V_s = self.model(state)
            gradient_update = tape.gradient(V_s, params)
            for i in range(len(self.elig)):
                self.elig[i] = self.elig[i] + gradient_update[i]
            for i in range(len(self.elig)):
                gradient_update[i] = delta*self.elig[i]
            self.model.optimizer.apply_gradients(zip(   gradient_update,
                                                        self.model.trainable_variables))
        self.decay_elibilities()
        
