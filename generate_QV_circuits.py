import numpy as np
import scipy as sp

from qiskit import QuantumCircuit
from qiskit.circuit.library import Permutation, UnitaryGate

def parameters_a_b_c():
    """Draw parameters of random 2-qubit interaction gate: 
    $U = exp(1j*(x*\sigma_x \otimes \sigma_x + y*\sigma_y \otimes \sigma_y +   z*\sigma_z \otimes \sigma_z)$
    According to the Haar measure"""
    while True:
        numbers = np.random.random((3))*np.pi
        x, y, z = numbers[::-1]
        p_xyz = (4)*np.abs(np.sin(2*(x+y)) * np.sin(2*(x+z)) * np.sin(2*(y+z)) * np.sin(2*(x-y)) * np.sin(2*(x-z)) * np.sin(2*(y-z)))
        p = np.random.random()
        if p_xyz > p:
            return x,y,z

def get_parity_preserving_u():
    """Get random 2 qubit parity preserving gate as a 4x4 matrix according to the Haar measire."""
    XX = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]).astype(complex)
    YY = np.array([[0,0,0,1],[0,0,-1,0],[0,-1,0,0],[1,0,0,0]]).astype(complex)
    ZZ = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]).astype(complex)
    
    p1,p2,p3 = parameters_a_b_c()
    U = sp.linalg.expm(1j*(p1*XX - p2*YY + p3*ZZ) )
    return U

def get_diagonal_u():
    """Get random 2 qubit diagonal gate as a 4x4 matrix."""
    ZZ = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]).astype(complex)

    p = 2*np.random.random((1))*np.pi
    U = sp.linalg.expm(1j*p*ZZ)
    return U

def get_standard_QV_circuit(N,T):
    """Get standard Quantum Volume circuit with 'N' qubits and 'T' layers."""
    qc = QuantumCircuit(N)

    for i in range(T):

        perm = np.random.permutation(N)
        qc.compose(Permutation(N,perm),qubits  = list(range(N)),inplace = True)
        
        for j in range(N//2):
            u = sp.stats.unitary_group.rvs(4)
            qc.append(UnitaryGate(u), [2*j, 2*j+1])

    qc_with_measure = qc.copy()
    qc_with_measure.measure_all()
    return qc, qc_with_measure

def get_parity_QV_circuit(N,T):
    """Get parity preserving Quantum Volume circuit with 'N' qubits and 'T' layers."""
    # N - number of qubits, T - number of layers
    qc = QuantumCircuit(N)
    
    for i in range(T):
        perm = np.random.permutation(N)
        qc.compose(Permutation(N,perm),inplace = True)
        
        for j in range(N//2):
            u = get_parity_preserving_u()
            qc.append(UnitaryGate(u), [2*j, 2*j+1])
        
    qc.measure_all()
    return qc

def get_double_parity_QV_circuit(N,T):
    """Get double parity Quantum Volume circuit with 'N' qubits and 'T' layers."""
    qc = QuantumCircuit(N)

    odds = np.array((N//2)*[1] + (N//2)*[0])
    p = np.random.permutation(N)
    odds = odds[p]
    
    for i in range(T):
        perm = np.random.permutation(N)
        qc.compose(Permutation(N,perm),inplace = True)
        odds = odds[perm]
        
        for j in range(N//2):
            if odds[2*j] == odds[2*j+1]:
                u = get_parity_preserving_u()                
            else:
                u = get_diagonal_u()
            qc.append(UnitaryGate(u), [2*j, 2*j+1])

    qc.measure_all()
    return qc, odds