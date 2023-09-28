
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

class VariationalClassifier():

    def __init__(self, input_size=16, num_layers=2, batch_size=5):
        super.__init__(qml.device("lightning.qubit", wires=input_size))
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.device = qml.device("lightning.qubit", wires=self.input_size)
        self.qnode = qml.QNode(self.circuit, self.device, interface="autograd")
        self.weights = np.zeros((2, self.input_size, 3))
        self.bias = np.array(0.0, requires_grad=True)
        self.optimizer = qml.NesterovMomentumOptimizer(0.05) 
    
    def layer(self):

        for w in self.weights:
            for wire in range(self.input_size):
                qml.Rot(w[wire, 0], w[wire, 1], w[wire, 2], wires=self.device.wires[wire])

            for wire in range(self.input_size - 1):
                qml.CNOT(wires=[self.device.wires[wire], self.device.wires[wire + 1]])

            qml.CNOT(wires=[self.device.wires[self.input_size - 1], self.device.wires[0]])
   

    def circuit(self, x):

        qml.BasisState(x, wires=self.device.wires)
        self.layer()

        return qml.expval(qml.PauliZ(wires=self.device.wires[0]))
    

    def variational_classifier(self, x):
        return self.circuit(x) + self.bias
    
    def square_loss(labels, predictions):
        loss = 0
        for l, p in zip(labels, predictions):
            loss = loss + (l - p) ** 2

        loss = loss / len(labels)
        return loss
    
    def accuracy(labels, predictions):
        loss = 0
        for l, p in zip(labels, predictions):
            if abs(l - p) < 1e-5:
                loss = loss + 1
        loss = loss / len(labels)

        return loss
    
    def cost(self, X, Y):
        predictions = [self.variational_classifier(x) for x in X]
        return self.square_loss(Y, predictions)
    
    def train(self, x_train, y_train):

        for it in range(25):
            self.qnode = self.qnode = qml.QNode(self.circuit, self.device, interface="autograd")
            # Update the weights by one optimizer step
            batch_index = np.random.randint(0, len(x_train), (self.batch_size,))
            X_batch = x_train[batch_index]
            Y_batch = y_train[batch_index]
            self.weights, self.bias, _, _ = self.optimizer.step(
                self.cost, 
                X_batch, 
                Y_batch)

            # Compute accuracy
            predictions = [
                np.sign(self.variational_classifier(x)) for x in x_train
                ]
            
            acc = self.accuracy(y_train, predictions)

            print(
                "Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.7f} ".format(
                    it + 1, 
                    self.cost(self.weights, self.bias, x_train, y_train), 
                    acc
                )
            )
