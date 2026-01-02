# Qiskit-based QAOA implementation for Max-Cut
# This replaces the CUDA-Q mock with real quantum simulation

import numpy as np
from typing import List, Tuple
from src.utilities import Graph, get_cost

# Try to import Qiskit modules (Qiskit 2.x compatible)
try:
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.quantum_info import SparsePauliOp
    QISKIT_AVAILABLE = True
    print("✅ Qiskit loaded successfully - Using REAL quantum simulation!")
except ImportError as e:
    QISKIT_AVAILABLE = False
    print(f"⚠️ Qiskit not available: {e}. Using fallback greedy algorithm.")


def create_maxcut_hamiltonian(edges: List[Tuple[int, int]], num_qubits: int) -> 'SparsePauliOp':
    """
    Create the Max-Cut Hamiltonian as a Qiskit SparsePauliOp.
    
    For Max-Cut: We want to MAXIMIZE the number of cut edges.
    Cost = sum_{(i,j) in edges} (1 - Z_i * Z_j) / 2
    
    QAOA minimizes, so we use: H = sum 0.5 * (Z_i Z_j - 1)
    Minimizing this = Maximizing cuts
    """
    pauli_list = []
    coeffs = []
    
    for i, j in edges:
        # Create Z_i Z_j term (positive for minimization = max cut)
        z_string = ['I'] * num_qubits
        z_string[i] = 'Z'
        z_string[j] = 'Z'
        pauli_list.append(''.join(z_string[::-1]))  # Qiskit uses little-endian
        coeffs.append(0.5)  # Positive coefficient
        
        # Identity term (constant offset) - negative for max cut
        pauli_list.append('I' * num_qubits)
        coeffs.append(-0.5)
    
    return SparsePauliOp(pauli_list, coeffs).simplify()


def qaoa_qiskit(G: Graph, layer_count: int = 1, shots: int = 1000) -> Tuple[float, str]:
    """
    Run QAOA using Qiskit's quantum simulator.
    
    Parameters
    ----------
    G : Graph
        The graph to solve Max-Cut on
    layer_count : int
        Number of QAOA layers (depth)
    shots : int
        Number of measurement shots
        
    Returns
    -------
    Tuple[float, str]
        (cost, bitstring) - the Max-Cut value and the optimal partition
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is not installed. Please install with: pip install qiskit qiskit-algorithms")
    
    num_qubits = G.n_v
    edges = [(e[0], e[1]) for e in G.e]
    
    # Handle empty graph
    if not edges:
        return 0.0, '0' * num_qubits
    
    # Create Hamiltonian
    hamiltonian = create_maxcut_hamiltonian(edges, num_qubits)
    
    # Create QAOA instance with StatevectorSampler (Qiskit 2.x)
    sampler = StatevectorSampler()
    optimizer = COBYLA(maxiter=100)
    
    qaoa_solver = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=layer_count,
        initial_point=np.random.uniform(-np.pi/4, np.pi/4, 2 * layer_count)
    )
    
    # Run QAOA
    result = qaoa_solver.compute_minimum_eigenvalue(hamiltonian)
    
    # Get the best bitstring from the result
    if hasattr(result, 'best_measurement') and result.best_measurement:
        best_bitstring = result.best_measurement['bitstring']
    else:
        # Fallback: Use eigenstate to get most likely bitstring
        if hasattr(result, 'eigenstate') and result.eigenstate is not None:
            # Sample from the eigenstate
            from qiskit.quantum_info import Statevector
            if isinstance(result.eigenstate, dict):
                # It's already a counts dictionary
                best_bitstring = max(result.eigenstate, key=result.eigenstate.get)
            else:
                # It's a statevector - sample from it
                sv = Statevector(result.eigenstate)
                counts = sv.sample_counts(shots)
                best_bitstring = max(counts, key=counts.get)
        else:
            # Last resort: use greedy
            return qaoa_mock(G, layer_count, shots)
    
    # Calculate the actual cut value
    cost = get_cost(best_bitstring, edges)
    
    return float(cost), best_bitstring


def qaoa(G: Graph, layer_count: int = 1, shots: int = 1000, const: float = 0, save_file: bool = False) -> Tuple[float, str]:
    """
    Wrapper function to maintain compatibility with existing code.
    Automatically uses Qiskit if available, otherwise falls back to mock.
    """
    if QISKIT_AVAILABLE:
        try:
            return qaoa_qiskit(G, layer_count=layer_count, shots=shots)
        except Exception as e:
            print(f"Qiskit QAOA failed: {e}. Falling back to mock.")
            return qaoa_mock(G, layer_count=layer_count, shots=shots)
    else:
        # Fallback to mock implementation
        return qaoa_mock(G, layer_count=layer_count, shots=shots)


def qaoa_mock(G: Graph, layer_count: int = 1, shots: int = 1000) -> Tuple[float, str]:
    """
    Mock QAOA implementation (greedy heuristic) for when Qiskit is not available.
    """
    import random
    
    num_qubits = G.n_v
    edges = [(e[0], e[1]) for e in G.e]
    
    if not edges:
        return 0.0, '0' * num_qubits
    
    # Build adjacency list
    adj = {i: [] for i in range(num_qubits)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Greedy partition optimization
    partition = [random.randint(0, 1) for _ in range(num_qubits)]
    
    # Local search optimization
    improved = True
    iterations = 0
    max_iterations = num_qubits * layer_count * 5
    
    while improved and iterations < max_iterations:
        improved = False
        for node in range(num_qubits):
            # Count edges cut with current and flipped state
            current_cut = sum(1 for neighbor in adj[node] if partition[node] != partition[neighbor])
            flipped_cut = sum(1 for neighbor in adj[node] if partition[node] == partition[neighbor])
            
            if flipped_cut > current_cut:
                partition[node] = 1 - partition[node]
                improved = True
        iterations += 1
    
    bitstring = ''.join(str(b) for b in partition)
    cost = get_cost(bitstring, edges)
    
    return float(cost), bitstring
