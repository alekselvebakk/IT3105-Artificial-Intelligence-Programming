import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import time


class ActorCritic:
    def __init__(self,
                 learning_rate=0.1,
                 layers=[25, 30, 25],
                 opt='SGD',
                 act='relu',
                 last_act='softmax',
                 input_size=25+1,
                 reload_model=False,
                 reload_name=None,
                 epochs=200,
                 batch_size=32,
                 validation_split=0.1,
                 verbosity=2,
                 net_with_critic = True):

        self.alpha = learning_rate
        self.layers = layers
        self.opt = eval('keras.optimizers.' + opt)
        self.act = act
        self.last_act = last_act
        self.input_size = input_size  # The state will be playerID + board
        self.net_with_critic = net_with_critic
        if net_with_critic:
            self.output_size = input_size  # Board of actions (minus player plus value)
        else:
            self.output_size = input_size-1 # Board of actions (minus player)
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbosity = verbosity

        # makes a new NN or reloads from earlier trained model
        self.model = self.get_net() if not reload_model else keras.models.load_model(reload_name, custom_objects={'cross_entropy': self.cross_entropy})

    def get_net(self):
        input = keras.layers.Input(shape=(self.input_size,), name='input_layer')
        x = input
        for i in range(len(self.layers)):
            x = keras.layers.Dense(self.layers[i], activation=self.act, name='hidden_'+str(i))(x)
        output = keras.layers.Dense(self.output_size, activation=self.last_act, name='output_layer')(x)
        model = keras.models.Model(input, output)
        model.compile(optimizer=self.opt(lr=self.alpha), loss=self.cross_entropy)  # TODO: metrics

        return model

    def save_net(self, net_name):
        self.model.save(net_name)

    # Loss function from Keith
    def cross_entropy(self, targets, outputs):
        return tf.reduce_mean(tf.reduce_sum(-1*targets*self.safelog(outputs), axis=1))

    def safelog(self, tensor, base=0.0001):
        return tf.math.log(tf.math.maximum(tensor, base))

    def train(self, inputs, targets):
        number_of_training_samples = int((1-self.validation_split)*float(len(inputs[:][0])))
        if number_of_training_samples <= self.epochs:
            self.model.fit( inputs,
                            targets,
                            epochs = number_of_training_samples,
                            batch_size = self.batch_size,
                            validation_split = self.validation_split)
        else:
            self.model.fit( inputs,
                            targets,
                            epochs = self.epochs,
                            batch_size = self.batch_size,
                            validation_split = self.validation_split)
            

    def string_to_tensor(self, string_variable):
        converted_to_list = list(string_variable)
        numpy_variable = np.array([converted_to_list], dtype=int)
        return numpy_variable

    def get_probabilities(self, state):
        state_tensor = self.string_to_tensor(state)
        prediction = self.model(state_tensor).numpy()[0]

        # Changes the probabilities for illegals moves to 0
        for i in range(1, len(state)):
            if int(state[i]) != 0:
                prediction[i-1] = 0

        # Scaling of the remaining numbers to the sum of 1
        sum = prediction.sum()
        if sum == 0: sum = 0.00000001  # Error handling to avoid dividing by 0
        prediction = prediction/sum

        return prediction

    def get_action(self, state, epsilon=0):
        decision = random.uniform(0, 1)
        if self.net_with_critic:
            probabilities = self.get_probabilities(state)[:-1]
        else:
            probabilities = self.get_probabilities(state)
        if probabilities.sum() == 0:
            index = self.get_random_safe_index(state)
        elif decision > epsilon:
            index = np.argmax(probabilities)
        else:
            non_zero_index, = np.nonzero(probabilities)
            index = np.random.choice(non_zero_index)
        return self.find_position(index)

    def get_critic_value(self, state):
        if self.net_with_critic:
            state_tensor = self.string_to_tensor(state)
            value_prediction = self.model(state_tensor).numpy()[0][-1]
        else:
            value_prediction = 0
        return value_prediction

    def get_random_safe_index(self, state):
        safe_actions = []
        for i in range(1, len(state)):
            if state[i] == '0':
                safe_actions.append(i-1)
        return np.random.choice(safe_actions)

    def find_position(self, index):
        table_root = int(np.sqrt(self.output_size-1)) if self.net_with_critic else int(np.sqrt(self.output_size))
        row = int(index // table_root)
        col = int(index % table_root)
        return [row, col]


        


