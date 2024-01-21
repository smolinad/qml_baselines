import tensorcircuit as tc
import numpy as np
from tensorcircuit.templates.dataset import amplitude_encoding

tc.backend("tensorflow")
tc.set_dtype("complex128")

class VariationalClassifier():

    def __init__(self, nqubits:int):
        self.circuit = tc.Circuit(nqubits)
        

    def gate_G(self, index, alpha, beta, gamma, phi):

        # Define G gate
        unitary_ = np.e**(1.j*phi) * np.array(
            [[np.e**(1.j * beta)*np.cos(alpha),
              np.e**(1.j * gamma)*np.sin(alpha)],
              [-np.e**(-1.j * gamma)*np.sin(alpha),
              -np.e**(-1.j * beta)*np.cos(alpha)]])
        
        # Create the unitary matrix
        #G = tc.gates.Unitary(unitary_, wires=1)
        G = self.circuit.unitary(index, unitary=unitary_, name="G")

        

    # def layer(self, c_range:int):

    #     steps = np.gcd(self.circuit._nqubits, c_range)
    #     q_i = 0
    #     for i in range(steps):
    #         next_q_i = (q_i - c_range) % self.circuit._nqubits
    #         gate_G(next_q_i, alpha, beta, gamma, phi)
        

    

