# Qiskit-based QAOA implementation for Max-Cut
# Supports both local simulation AND real IBM Quantum hardware!

import numpy as np
from typing import List, Tuple, Optional
from src.utilities import Graph, get_cost
import os

# Try to import Qiskit modules (Qiskit 2.x compatible)
try:
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit.quantum_info import SparsePauliOp
    QISKIT_AVAILABLE = True
except ImportError as e:
    QISKIT_AVAILABLE = False

# Try to import IBM Runtime for real hardware
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as IBMSampler
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False

# Global variable to store the service
_ibm_service = None


def get_ibm_service(token: Optional[str] = None) -> Optional['QiskitRuntimeService']:
    """
    Get or create IBM Quantum service connection.
    Token can be passed directly, via Streamlit secrets, or via IBM_QUANTUM_TOKEN env var.
    """
    global _ibm_service
    
    if not IBM_AVAILABLE:
        return None
    
    if _ibm_service is not None:
        return _ibm_service
    
    # Try to get token from: 1) parameter, 2) Streamlit secrets, 3) env var
    api_token = token
    
    if not api_token:
        # Try Streamlit secrets
        try:
            import streamlit as st
            api_token = st.secrets.get("IBM_QUANTUM_TOKEN")
        except:
            pass
    
    if not api_token:
        # Try environment variable
        api_token = os.environ.get('IBM_QUANTUM_TOKEN')
    
    try:
        if api_token:
            # Save credentials for future use
            QiskitRuntimeService.save_account(channel="ibm_quantum", token=api_token, overwrite=True)
            _ibm_service = QiskitRuntimeService(channel="ibm_quantum")
        else:
            # Try to use saved credentials
            _ibm_service = QiskitRuntimeService(channel="ibm_quantum")
        return _ibm_service
    except Exception as e:
        return None


def list_available_backends(token: Optional[str] = None) -> List[str]:
    """List available IBM Quantum backends."""
    service = get_ibm_service(token)
    if service is None:
        return ["simulator (local)"]
    
    try:
        backends = service.backends()
        backend_names = ["simulator (local)"] + [b.name for b in backends if b.status().operational]
        return backend_names
    except Exception:
        return ["simulator (local)"]


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


def qaoa_qiskit(
    G: Graph, 
    layer_count: int = 1, 
    shots: int = 1000,
    backend_name: str = "simulator",
    ibm_token: Optional[str] = None
) -> Tuple[float, str]:
    """
    Run QAOA using Qiskit's quantum simulator or IBM Quantum hardware.
    
    Parameters
    ----------
    G : Graph
        The graph to solve Max-Cut on
    layer_count : int
        Number of QAOA layers (depth)
    shots : int
        Number of measurement shots
    backend_name : str
        "simulator" for local simulation, or IBM backend name (e.g., "ibm_brisbane")
    ibm_token : str, optional
        IBM Quantum API token (can also use IBM_QUANTUM_TOKEN env var)
        
    Returns
    -------
    Tuple[float, str]
        (cost, bitstring) - the Max-Cut value and the optimal partition
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is not installed.")
    
    num_qubits = G.n_v
    edges = [(e[0], e[1]) for e in G.e]
    
    # Handle empty graph
    if not edges:
        return 0.0, '0' * num_qubits
    
    # Create Hamiltonian
    hamiltonian = create_maxcut_hamiltonian(edges, num_qubits)
    
    # Choose backend
    use_real_hardware = backend_name != "simulator" and "local" not in backend_name.lower()
    
    if use_real_hardware and IBM_AVAILABLE:
        print(f"🔬 Running on REAL IBM Quantum hardware: {backend_name}")
        service = get_ibm_service(ibm_token)
        if service is None:
            print("⚠️ Could not connect to IBM Quantum. Falling back to simulator.")
            use_real_hardware = False
        else:
            try:
                backend = service.backend(backend_name)
                sampler = IBMSampler(backend)
                # Use SPSA optimizer for noisy hardware (more robust)
                optimizer = SPSA(maxiter=50)
            except Exception as e:
                print(f"⚠️ Backend {backend_name} not available: {e}. Falling back to simulator.")
                use_real_hardware = False
    
    if not use_real_hardware:
        print("💻 Running on local quantum simulator")
        sampler = StatevectorSampler()
        optimizer = COBYLA(maxiter=100)
    
    # Create and run QAOA
    qaoa_solver = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=layer_count,
        initial_point=np.random.uniform(-np.pi/4, np.pi/4, 2 * layer_count)
    )
    
    result = qaoa_solver.compute_minimum_eigenvalue(hamiltonian)
    
    # Get the best bitstring from the result
    if hasattr(result, 'best_measurement') and result.best_measurement:
        best_bitstring = result.best_measurement['bitstring']
    else:
        # Fallback: Use eigenstate to get most likely bitstring
        if hasattr(result, 'eigenstate') and result.eigenstate is not None:
            from qiskit.quantum_info import Statevector
            if isinstance(result.eigenstate, dict):
                best_bitstring = max(result.eigenstate, key=result.eigenstate.get)
            else:
                sv = Statevector(result.eigenstate)
                counts = sv.sample_counts(shots)
                best_bitstring = max(counts, key=counts.get)
        else:
            return qaoa_mock(G, layer_count, shots)
    
    # Calculate the actual cut value
    cost = get_cost(best_bitstring, edges)
    
    return float(cost), best_bitstring


def qaoa(
    G: Graph, 
    layer_count: int = 1, 
    shots: int = 1000, 
    const: float = 0, 
    save_file: bool = False,
    backend_name: str = "simulator",
    ibm_token: Optional[str] = None
) -> Tuple[float, str]:
    """
    Wrapper function to maintain compatibility with existing code.
    Automatically uses Qiskit if available, otherwise falls back to mock.
    """
    if QISKIT_AVAILABLE:
        try:
            return qaoa_qiskit(G, layer_count=layer_count, shots=shots, 
                             backend_name=backend_name, ibm_token=ibm_token)
        except Exception as e:
            print(f"Qiskit QAOA failed: {e}. Falling back to mock.")
            return qaoa_mock(G, layer_count=layer_count, shots=shots)
    else:
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
            current_cut = sum(1 for neighbor in adj[node] if partition[node] != partition[neighbor])
            flipped_cut = sum(1 for neighbor in adj[node] if partition[node] == partition[neighbor])
            
            if flipped_cut > current_cut:
                partition[node] = 1 - partition[node]
                improved = True
        iterations += 1
    
    bitstring = ''.join(str(b) for b in partition)
    cost = get_cost(bitstring, edges)
    
    return float(cost), bitstring
