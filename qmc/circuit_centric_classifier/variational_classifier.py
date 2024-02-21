import tensorcircuit as tc
import numpy as np
from tensorcircuit.templates.dataset import amplitude_encoding
from tensorcircuit import keras
from functools import partial
import tensorflow as tf

K = tc.set_backend("tensorflow")
tc.set_dtype("complex128")

class VariationalClassifier():

    def __init__(self, nqubits:int=10, rangeList:list=[1,2,3]):
        self.circuit = None

        self.nqubits = nqubits
        self.rangeList = rangeList
        self.stepList = [self.nqubits//np.gcd(self.nqubits, c) for c in self.rangeList]+ [1]

        self.model = tf.keras.Sequential([
            tc.keras.QuantumLayer(
                self.layer, 
                weights_shape=[len(self.rangeList)+1, self.nqubits + max(self.stepList) +1, 3]
            )
        ])


    def layer(self, x, weights): 

        self.circuit = tc.Circuit(self.nqubits, inputs=x)
        
        # Iterates over the list of "jump" ranges
        for k in range(len(self.rangeList)):

            for i in range(self.nqubits):
                self.circuit.u(
                    i, 
                    theta=weights[k,i,0], 
                    phi=weights[k,i,1], 
                    lbd=weights[k,i,2]
                )

            steps = self.nqubits//np.gcd(self.nqubits, self.rangeList[k])
            q_i = 0

            for j in range(1, steps+1):
                next_q_i = (q_i - self.rangeList[k]) % self.nqubits
                self.circuit.cu(
                    q_i, 
                    next_q_i, 
                    theta=weights[k,self.nqubits+j,0], 
                    phi=weights[k,self.nqubits+j,1], 
                    lbd=weights[k,self.nqubits+j,2]
                )   
                q_i = next_q_i

        self.circuit.u(
            0, 
            theta=weights[len(self.rangeList),0,0], 
            phi=weights[len(self.rangeList),0,1], 
            lbd=weights[len(self.rangeList),0,2]
        )      
                
        outputs = K.stack(
            [K.real(self.circuit.expectation([tc.gates.z(), [i]])) for i in range(self.nqubits)] + 
            [K.real(self.circuit.expectation([tc.gates.x(), [i]])) for i in range(self.nqubits)])
        outputs = K.reshape(outputs, [-1])
        return K.sigmoid(K.sum(outputs))
    
    def fit(self, x_train, y_train, **kwargs):
        self.model.compile(
            loss=tf.keras.losses.Hinge(), 
            optimizer=tf.keras.optimizers.legacy.Adam(0.01),
            metrics=["binary_accuracy"]
        )

        self.model.fit(x_train, y_train, **kwargs)   


         

            
        

    

