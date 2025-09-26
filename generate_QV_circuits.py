import numpy as np
from scipy.linalg import expm
from scipy.stats import unitary_group as ug
from qiskit import QuantumCircuit
from qiskit.circuit.library import Permutation, UnitaryGate


def __get_parity_preserving_u():
    U = np.zeros((4,4)).astype(complex)
    U[1:3,1:3] = ug.rvs(2)
    U[::3,::3] = ug.rvs(2)
    return U


def __parameters_a_b_c():
    """Draw parameters for interaction 2-qubit  gate:
    $U = exp(1j*(x*\sigma_x \otimes \sigma_x + y*\sigma_y \otimes \sigma_y +   z*\sigma_z \otimes \sigma_z)$
    According to the Haar measure"""
    while True:
        numbers = np.random.random(3) * np.pi
        x, y, z = numbers[::-1]
        p_xyz = 4 * np.abs(np.sin(2*(x+y)) * np.sin(2*(x+z)) * np.sin(2*(y+z)) * np.sin(2*(x-y)) * np.sin(2*(x-z)) * np.sin(2*(y-z)))
        p = np.random.random()
        if p_xyz > p:
            return x,y,z


def __get_interaction_gate():
    """Get random 2 qubit interaction gate as a 4x4 matrix according to the Haar measure."""
    XX = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]).astype(complex)
    YY = np.array([[0,0,0,1],[0,0,-1,0],[0,-1,0,0],[1,0,0,0]]).astype(complex)
    ZZ = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]).astype(complex)
    p1,p2,p3 = __parameters_a_b_c()
    U = expm(1j * (p1*XX - p2*YY + p3*ZZ))
    return U


def get_diagonal_u():
    """Get random 2 qubit diagonal gate as a 4x4 matrix."""
    ZZ = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]).astype(complex)
    p = 2 * np.random.random(1) * np.pi
    U = expm(1j *p*ZZ)
    return U


def get_standard_QV_circuit(N,T):
    """Get standard Quantum Volume circuit with 'N' qubits and 'T' layers."""
    qc = QuantumCircuit(N)

    for i in range(T):

        perm = np.random.permutation(N)
        qc.compose(Permutation(N,perm), qubits=list(range(N)), inplace = True)
        
        for j in range(N//2):
            u = ug.rvs(4)
            qc.append(UnitaryGate(u), [2*j, 2*j+1])

    qc_with_measure = qc.copy()
    qc_with_measure.measure_all()
    return qc, qc_with_measure


def get_parity_QV_circuit(N, T, simplified=False, interaction_gates=False):
    """Get parity preserving Quantum Volume circuit with.
    :N: Number of qubits
    :T: Number of layers
    :simplified: Flag, if set to False the 2-qubit gates are combined with single qubit ones to locally operate outside parity subspace and sample errors in entire Hilbert space
    :interaction_gates: Flag, if set to True, general parity preserving gates are replaced by interaction ones.
    """
    qc = QuantumCircuit(N)

    single_q_tab = np.repeat(np.eye(2,dtype = complex), N).reshape(2,2,N).transpose((2,0,1))

    for i in range(T):
        perm = np.random.permutation(N)
        new_single_q_tab = np.repeat(np.eye(2,dtype = complex),N).reshape(2,2,N).transpose((2,0,1))
        qc.compose(Permutation(N,perm),inplace = True)

        for j in range(N//2):

            u = __get_interaction_gate() if interaction_gates else __get_parity_preserving_u()

            if not simplified:
                if i < T-1:
                    new_single_q_tab[2*j] = ug.rvs(2)
                    new_single_q_tab[(2*j)+1] = ug.rvs(2)

                u = np.dot(np.kron(new_single_q_tab[(2*j)+1],new_single_q_tab[2*j]),u) #qiskit has an odd way to number qubits and interpret 2 qubit gates
                u =  np.dot(u,np.kron(single_q_tab[perm[(2*j)+1]].conj().T, single_q_tab[perm[2*j]].conj().T) )
            qc.append(UnitaryGate(u), [2*j, (2*j)+1])

        if N%2 == 1 and not simplified:
            if i < T-1:
                new_single_q_tab[-1] = ug.rvs(2)
            u = new_single_q_tab[-1] @ single_q_tab[perm[-1]].conj().T
            qc.append(UnitaryGate(u), [N-1])

        single_q_tab = new_single_q_tab
        
    qc.measure_all()
    return qc


def get_double_parity_QV_circuit(N, T, simplified=False, interaction_gates=False):
    """Get Quantum Volume circuit preserving parity in two sets of qubits.
    :N: Number of qubits
    :T: Number of layers
    :simplified: Flag, if set to False the 2-qubit gates are combined with single qubit ones to locally operate outside parity subspace and sample errors in entire Hilbert space
    :interaction_gates: Flag, if set to True, general parity preserving gates are replaced by interaction ones.
    """
    qc = QuantumCircuit(N)

    odds = np.array((N//2)*[1] + (N - N//2)*[0])
    p = np.random.permutation(N)
    odds = odds[p]

    single_q_tab = np.repeat(np.eye(2,dtype = complex),N).reshape(2,2,N).transpose((2,0,1))
    
    for i in range(T):
        perm = np.random.permutation(N)
        qc.compose(Permutation(N,perm), inplace = True)
        odds = odds[perm]
        new_single_q_tab = np.repeat(np.eye(2,dtype = complex),N).reshape(2,2,N).transpose((2,0,1))
        
        for j in range(N//2):
            if odds[2*j] == odds[2*j+1]:
                u = __get_interaction_gate() if interaction_gates else __get_parity_preserving_u()
            else:
                u = get_diagonal_u()
            if not simplified:
                if i < T-1:
                    new_single_q_tab[2*j] = ug.rvs(2)
                    new_single_q_tab[2*j+1] = ug.rvs(2)
                u = np.dot(np.kron(new_single_q_tab[(2*j)+1],new_single_q_tab[2*j]), u) #qiskit has an odd way to number qubits and interpret 2 qubit gates
                u = np.dot(u, np.kron(single_q_tab[perm[(2*j)+1]].conj().T, single_q_tab[perm[2*j]].conj().T))

            qc.append(UnitaryGate(u), [2*j, (2*j)+1])

        if N % 2 == 1 and not simplified:
            if i < T-1:
                new_single_q_tab[-1] = ug.rvs(2)
            u = new_single_q_tab[-1] @ single_q_tab[perm[-1]].conj().T
            qc.append(UnitaryGate(u), [N - 1])

        single_q_tab = new_single_q_tab

    qc.measure_all()
    return qc, odds
