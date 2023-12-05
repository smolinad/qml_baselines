from sklearn.neural_network import MLPClassifier

class NeuralNetwork():


    """
    Implements a Neural Network as in the paper by Farhi & Neven. Citing Section 3.3:

        'As a preliminary step we present the labeled samples to a classical neural\ 
        network. Here we run a (Matlab) classifier with one internal layer consisting of\ 
        10 neurons. Each neuron has 16 coefficient weights and one bias weight so there\ 
        are 170 parameters on the internal layer and 4 on the output layer. The classical\ 
        network has no trouble finding weights that give less than one percent classification\
        error on the training set. The (Matlab) program also looks at the generalization error\
        but to do so it picks a random 15 percent of the input data to use for a test set.\
        Since the input data set has repeated occurrences of the same 16 bit strings, the\
        test set is not purely unseen examples. Still the generalization error is less than\
        one percent.'

    """
    
    def __init__(self, input_shape=(16,)):
        self.model = MLPClassifier(hidden_layer_sizes=(10,), activation="tanh", max_iter=200)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, inputs):
        return self.model.predict(inputs)
        
    def score(self, x_test, y_test):
        return self.model.score(x_test, y_test)