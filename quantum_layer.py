import pennylane as qml
from pennylane import numpy as np

# Urządzenie kwantowe
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# Obwód kwantowy
@qml.qnode(dev)
def quantum_circuit(inputs):
    # Przygotowanie stanu kwantowego
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    # Operacje kwantowe
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[n_qubits - 1, 0])

    # Pomiar wartości oczekiwanej
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Wrapper dla TensorFlow
def quantum_layer(inputs):
    outputs = np.array([quantum_circuit(x) for x in inputs])
    return outputs
