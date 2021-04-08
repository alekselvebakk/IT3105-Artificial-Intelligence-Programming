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
                 board_size=25,
                 reload_model=False,
                 reload_name=None,
                 minibatch_size=350):

        self.alpha = learning_rate
        self.layers = layers
        self.opt = eval('keras.optimizers.' + opt)
        self.act = act
        self.last_act = last_act
        self.input_size = 1 + board_size  # The state will be playerID + board
        self.output_size = board_size
        self.minibatch_size = minibatch_size

        # makes a new NN or reloads from earlier trained model
        self.model = self.get_net() if not reload_model else keras.models.load_model(reload_name)

    # TODO: Initialize values: skjer ikke det automatisk? satt dem til 0 nå, men vet ikke om det er det beste
    def get_net(self):  # Not finished version, will TODO: checkout CNN
        input = keras.layers.Input(shape=(self.input_size,), name='input_layer')
        x = input
        for i in range(len(self.layers)):
            x = keras.layers.Dense(self.layers[i], activation=self.act, name='hidden_'+str(i))(x)
        output = keras.layers.Dense(self.output_size, activation=self.last_act, name='output_layer')(x)
        model = keras.models.Model(input, output)
        model.compile(optimizer=self.opt(lr=self.alpha), loss=self.cross_entropy)  # TODO: metrics

        return model

    def save_net(self, net_name):
        self.model.save('Saved_models/' + net_name)

    # loss function from Keith, but TODO: what does it mean? and does it work?
    def cross_entropy(self, targets, outputs):
        return tf.reduce_mean(tf.reduce_sum(-1*targets*self.safelog(outputs), axis=1))

    @staticmethod
    def safelog(self, tensor, base=0.0001):
        return tf.math.log(tf.math.maximum(tensor, base))

    # TODO: test train method
    def train(self, inputs, targets, batch_size, epochs):
        self.model.fit(inputs, targets, batch_size, epochs, verbose=1, validation_split=0.1)

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
        # print('PRED_NORM:', prediction)

        return prediction

    # TODO: Discuss: should actions be 'calculated' in a game handler? for moduality? Should prob just send probabiities
    def get_action(self, state, epsilon = 0):
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

    def train_from_RBUF(self, RBUF):
        if self.minibatch_size <= len(RBUF):
            minibatch = random.sample(RBUF, self.minibatch_size)
        else:
            minibatch = RBUF
        inputs = [0]*len(minibatch)
        targets = inputs
        for i in range(len(minibatch)):
            inputs[i] = minibatch[i][0]
            targets[i] = minibatch[i][1]
    
        epochs = 50
        batch_size = int(len(inputs)/epochs)
        
        self.train(inputs, targets, batch_size, epochs-1)
        



def main():
    a = Actor()
    a.get_probabilities('21201201201201201201201201')


if __name__ == '__main__':
    main()