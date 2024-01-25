import tensorcircuit as tc
import numpy as np
from tensorcircuit.templates.dataset import amplitude_encoding
from tensorcircuit import keras
from functools import partial
import tensorflow as tf

tc.set_backend("tensorflow")
tc.set_dtype("complex128")

class VariationalClassifier():

    def __init__(self, nqubits:int, rangeList:list):
        self.circuit = tc.Circuit(nqubits)

        self.n = self.circuit._nqubits
        self.rangeList = rangeList
        self.stepList = [self.n//np.gcd(self.n, c) for c in self.rangeList]

        self.model = tf.keras.Sequential(
            [self.quantumLayer(r, s) for r, s in zip(self.rangeList, self.stepList)]
            )
        
        self.model.append(keras.QuantumLayer(partial(self.measure)))
        
        self.model.compile(
            loss=tf.keras.losses.Hinge(), 
            optimizer=tf.keras.optimizers.legacy.Adam(0.01),
            metrics=["binary_accuracy"]
        )

    def layer(self, c_range:int, weights:np.array): #weight debe ser nqubits+steps x 3

        steps = self.n//np.gcd(self.n, c_range)
        q_i = 0

        for i in range(self.n):
            self.circuit.u(i, theta=weights[i,0], phi=weights[i,1], lbd=weights[i,2])

        for j in range(steps):
            next_q_i = (q_i - c_range) % self.n 
            self.circuit.cu(q_i, next_q_i, theta=weights[self.n+j,0], phi=weights[self.n+j,1], lbd=weights[self.n+j,2])   
            q_i = next_q_i

    def quantumLayer(self, c_range, steps):
        return keras.QuantumLayer(partial(self.layer, c_range=c_range), [(self.n + steps, 3)])
    
    def fit(self, x_train, y_train, batch_size, epoch):
        self.model.fit(x_train, y_train, batch_size, epoch)   

    def measure(self):
        return tc.backend.real(self.circuit.measure(0))
         

            
        

    

