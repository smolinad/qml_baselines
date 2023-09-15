import numpy as np
from sklearn.neural_network import MLPClassifier

class NeuralNetwork_2():

    def __init__(self, input_shape=(16,)):
        self.model = MLPClassifier(hidden_layer_sizes=(10,), activation="tanh", max_iter=200)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, inputs):
        return self.model.predict(inputs)
    def score(self, x_test, y_test):
        return self.model.score(x_test, y_test)
