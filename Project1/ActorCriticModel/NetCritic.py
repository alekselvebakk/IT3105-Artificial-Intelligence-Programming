
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

        #Input layers
        print(layers[0])
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
        #self.model.add(output_layer)
        self.model.summary()

        optim = keras.optimizers.SGD(learning_rate=self.alpha)
        self.model.compile(optimizer = optim)
        config = self.model.get_config()
        print(config["layers"][0]["config"]["batch_input_shape"])
        #Set initial eligibilities
        weights = self.model.trainable_weights
        self.elig =[]
        for i in range(len(weights)):
            self.elig.append(tf.zeros_like(weights[i]))
        print(weights)
        print(self.elig)

    @staticmethod
    def string_to_tensor(string_variable):
        content = list(string_variable)
        print(content)
        variable = np.array([content], dtype=int)
        #variable = variable.reshape(len(variable),1)
        #variable = tf.Variable(tf.constant(variable), shape = tf.TensorShape([len(variable),1]))
        return variable

    def episode_reset(self):
        a = 1#tf.math.scalar_mul(0, self.elig)

    def get_delta(self, state, next_state, reward):
        state = self.string_to_tensor(state)
        print("*******delta func: ",state)
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
            print("----------", V_s)
            print("*********", params)
            dV_dw = tape.gradient(V_s, params)
            print("................", dV_dw)
            self.elig = self.elig + dV_dw
            update = tf.math.scalar_mul(delta,self.elig)
            self.model.optimizer.apply_gradients(   update, 
                                                    self.model.trainable_weights)
        self.elig = self.elig*self.elig_decay
        
