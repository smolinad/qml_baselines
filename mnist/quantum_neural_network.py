from matplotlib import pyplot as plt
import tensorflow as tf
import tensorcircuit as tc

class QuantumNeuralNetwork():
    def __init__(self, num_qubits:int=10, num_blocks:int=3) -> None:
        self.backend = tc.set_backend("tensorflow")
        self.num_qubits = num_qubits
        self.num_blocks = num_blocks
        self.batched_ae = self.backend.vmap(
            tc.templates.dataset.amplitude_encoding, 
            vectorized_argnums=0)
        self.model = tf.keras.Sequential(
            [tc.keras.QuantumLayer(
                self.circuit, 
                weights_shape=[num_blocks, num_qubits, 3]
                )]
            )

    def circuit(self, x, weights):
        wires = tc.Circuit(self.num_qubits, inputs=x)
        for j in range(self.num_blocks):
            for i in range(self.num_qubits):
                wires.rx(i, theta=weights[j, i, 0])
                wires.rz(i, theta=weights[j, i, 1])
            for i in range(self.num_qubits - 1):
                wires.exp1(i, i + 1, theta=weights[j, i, 2], unitary=tc.gates._zz_matrix)
        outputs = self.backend.stack(
            [self.backend.real(wires.expectation([tc.gates.z(), [i]])) for i in range(self.num_qubits)] + 
            [self.backend.real(wires.expectation([tc.gates.x(), [i]])) for i in range(self.num_qubits)]
        )

        outputs = self.backend.reshape(outputs, [-1])

        return self.backend.sigmoid(self.backend.sum(outputs))


    def train(self, x_train, y_train):
        self.model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.legacy.Adam(0.01), #Using legacy because running on macOS.
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )

        self.model.fit(
            self.batched_ae(x_train, self.num_qubits), 
            y_train,
            epochs=3,
            batch_size=4
            )