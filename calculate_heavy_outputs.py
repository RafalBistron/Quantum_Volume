import numpy as np


def get_hu_standard_perfect_circuit(vector):
    """Get heavy output frequency for the quantum state 'vector' of 'N' qubits."""
    probs = np.abs(vector)**2
    median = np.median(probs)
    return np.sum(probs[probs >= median]) / np.sum(probs)


def get_hu_standard_circuit(outputs,vector):
    """Get heavy output frequency for the distribution 'outputs'
    assuming perfect quantum state 'vector' of 'N' qubits."""

    probs = np.abs(vector)**2
    median = np.median(probs)
    
    hu = 0
    total = 0
    for result, counts in outputs.items():
        index = int(result,base = 2)        
        total += counts
        if probs[index] >= median:
            hu += counts
    return hu/total


def get_hu_parity_circuit(outputs):
    """Get heavy output frequency for the distribution 'outputs' in parity preserving circuit."""
    hu = 0
    total = 0
    for result, counts in outputs.items():
        total += counts
        #rewrite the result as table of zeros and ones on consecutive qubits
        if  sum(map(int,[*result])) %2 ==0:
            hu += counts
    return hu/total

def get_hu_double_parity_circuit(outputs, odds):
    """Get heavy output frequency for the distribution 'outputs' in double parity preserving
    circuit with qubits form one subset listed in 'odds' array."""
    hu = 0
    total = 0
    even = np.abs(odds - 1)
    for result, counts  in outputs.items():
        total += counts
        #rewrite the result as table of zeros and ones on consecutive qubits
        tab = np.zeros_like(odds).astype(int)
        tab0 = np.array(list(map(int,[*result])))
        tab[:tab0.shape[0]] = tab0[::-1]
        if sum(tab*odds) %2 == 0 and sum(tab*even) %2 == 0:
            hu += counts
    return hu/total
