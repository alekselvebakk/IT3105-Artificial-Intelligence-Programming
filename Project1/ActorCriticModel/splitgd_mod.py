from Project1.ActorCriticModel.splitgd import SplitGD
import tensorflow as tf
from tensorflow import keras

class SplitGD_Elig:

    def __init__(self, keras_model):
        self.model = keras_model

    def fit(self, gradients, state, next_state):
        params = self.model.trainable_weights
        with tf.GradientTape() as tape:
            V_s = self.model.predict(state)
            gradients = tape.gradient(V_s, params)
        return