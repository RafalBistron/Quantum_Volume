from qiskit_ibm_runtime.fake_provider import FakeSherbrooke


def get_modified_backend(backend,
                         decrease_measurement_error_factor=1,
                         decrease_1q_gate_error_factor=1,
                         decrease_2q_gate_error_factor=1,
                         decrease_decoherence_factor=1,
                         qubit_number=127):
    modified_backend = FakeSherbrooke()

    for i in range(qubit_number):
        modified_backend.target['measure'][(i,)].error = backend.target['measure'][
                                                             (i,)].error * decrease_measurement_error_factor
        modified_backend.target['measure'][(i,)].duration = backend.target['measure'][(i,)].duration

        modified_backend.target['x'][(i,)].error = backend.target['x'][(i,)].error * decrease_1q_gate_error_factor
        modified_backend.target['sx'][(i,)].error = backend.target['sx'][(i,)].error * decrease_1q_gate_error_factor
        modified_backend.target['rz'][(i,)].error = backend.target['rz'][(i,)].error * decrease_1q_gate_error_factor

        modified_backend.target.qubit_properties[i].t1 = backend.target.qubit_properties[
                                                             i].t1 / decrease_decoherence_factor
        modified_backend.target.qubit_properties[i].t2 = backend.target.qubit_properties[
                                                             i].t1 / decrease_decoherence_factor

    for pair in modified_backend.target['ecr'].keys():
        try:
            modified_backend.target['ecr'][pair].error = backend.target['ecr'][
                                                             pair].error * decrease_2q_gate_error_factor
        except:
            new_pair = (pair[1], pair[0])
            modified_backend.target['ecr'][pair].error = backend.target['ecr'][
                                                             new_pair].error * decrease_2q_gate_error_factor

    return modified_backend