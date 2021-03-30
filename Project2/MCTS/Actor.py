import tensorflow as tf
from tensorflow import keras
import numpy as np
import random


class Actor:
    def __init__(self,
                 learning_rate=0.1,
                 layers=[25, 30, 25],
                 opt='SGD',
                 act='relu',
                 last_act='softmax',
                 board_size=25):

        # TODO: Loss?
        self.alpha = learning_rate
        self.layers = layers
        self.opt = eval('keras.optimizers.' + opt)
        self.act = act
        self.last_act = last_act
        self.input_size = 1 + board_size #The state will be playerID + board
        self.output_size = board_size

        self.model = self.get_net()

    # TODO: Initialize values
    # TODO: Train method

    def get_net(self):
        input = keras.layers.Input(shape=(self.input_size,), name='input_layer')
        x = input
        for i in range(len(self.layers)):
            x = keras.layers.Dense(self.layers[i], activation=self.act)(x)
        output = keras.layers.Dense(self.output_size, activation=self.last_act, name='output_layer')(x)
        model = keras.models.Model(input, output)
        model.compile(optimizer=self.opt(lr=self.alpha))
        return model

    @staticmethod
    def string_to_tensor(string_variable):
        converted_to_list = list(string_variable)
        numpy_variable = np.array([converted_to_list], dtype=int)
        return numpy_variable

    def get_probabilities(self, state):
        state_tensor = self.string_to_tensor(state)
        prediction = self.model.predict(state_tensor)[0]

        # Changes the probabilities for illegals moves to 0
        for i in range(1, len(state)):
            if int(state[i]) != 0:
                prediction[i-1] = 0

        # Scaling of the remaining numbers to the sum of 1
        prediction = prediction/prediction.sum()
        print('PRED_NORM:', prediction)

        return prediction

    def get_action(self, state, epsilon):  # TODO: Discuss: should actions be 'calculated' in a game handler? for moduality? Should prob just send probabiities
        decision = random.uniform(0, 1)
        if decision > epsilon:
            index = np.argmax(self.get_probabilities(state))
        else:
            non_zero_index, = np.nonzero(self.get_probabilities(state))
            index = np.random.choice(non_zero_index)
        return self.find_position(index)

    '''def get_greedy_action(self, state):  # TODO: is this needed? could set epsilon = 0 in get_action
        prob = self.get_probabilities(state)
        index = np.argmax(prob)
        return self.find_position(index)'''

    def find_position(self, index):
        table_root = np.sqrt(self.output_size)
        row = int(index // table_root)
        col = int(index % table_root)
        return [row, col]



def main():
    a = Actor()
    #a.model('12345678901234567890123456', training=False)
    print(a.get_action('21201201201201201201201201', 1))


if __name__ == '__main__':
    main()