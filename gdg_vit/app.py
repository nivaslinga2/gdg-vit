
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import sys
import os
import time

# Add the repository root to sys.path so we can import src
sys.path.append(os.getcwd())

# Import necessary functions from your source code
try:
    from src.utilities import generate_regular_3_graph, max_cut_gurobi, get_cost, Graph
    from src.QAOA import qaoa
except ImportError:
    st.error("Could not import modules from 'src'. Make sure you are running this from the root directory 'gdg_vit'.")
    st.stop()

st.set_page_config(page_title="Quantum QAOA Explorer", layout="wide", page_icon="⚛️")

st.title("⚛️ Quantum Landscape: QAOA Max-Cut Solver")
st.markdown("""
Explore how **Quantum Approximate Optimization Algorithm (QAOA)** solves the Max-Cut problem.
Visualize the graph topology, the optimization landscape, and compare Quantum vs. Classical performance.
""")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")

# Graph Settings
st.sidebar.subheader("Graph Topology")
graph_type = st.sidebar.selectbox("Type", ["Random 3-Regular", "Ring", "Star", "Erdos-Renyi"])
n_nodes = st.sidebar.slider("Nodes (Qubits)", min_value=4, max_value=24, value=10, step=1)
if graph_type == "Erdos-Renyi":
    p_prob = st.sidebar.slider("Edge Probability", 0.1, 1.0, 0.5)

st.sidebar.info(f"**Circuit Width:** {n_nodes} Qubits\n\n**Search Space:** $2^{{{n_nodes}}}$ = {2**n_nodes:,} states")


# QAOA Settings
st.sidebar.subheader("Quantum Parameters")
num_layers = st.sidebar.slider("QAOA Layers (Depth)", min_value=1, max_value=5, value=1)
# Restrict seed to a safe 32-bit integer range to avoid overflow
seed = st.sidebar.number_input("Random Seed", value=42, step=1, max_value=2**31 - 1)

run_btn = st.sidebar.button("🚀 Run Quantum Simulation", type="primary")

# --- Helper Functions ---

def generate_graph(g_type, n, s):
    # ... existing implementation ...
    if g_type == "Random 3-Regular":
        if (3 * n) % 2 != 0: n += 1
        return nx.random_regular_graph(3, n, seed=s), n
    elif g_type == "Ring":
        return nx.cycle_graph(n), n
    elif g_type == "Star":
        return nx.star_graph(n-1), n
    elif g_type == "Erdos-Renyi":
        return nx.erdos_renyi_graph(n, p_prob, seed=s), n
    return nx.random_regular_graph(3, n, seed=s), n

def plot_graph_3d(G, partition_sets, title):
    # ... existing implementation ...
    try:
        pos = nx.spring_layout(G, dim=3, seed=42)
    except:
        pos = nx.spring_layout(G, dim=3)
        
    x_nodes = [pos[k][0] for k in G.nodes()]
    y_nodes = [pos[k][1] for k in G.nodes()]
    z_nodes = [pos[k][2] for k in G.nodes()]
    
    # Aesthetic configurations
    node_colors = []
    set_0, set_1 = partition_sets
    for node in G.nodes():
        # Neon Cyan for Set A, Neon Pink for Set B
        color = '#FF4B4B' if node in set_0 else '#0068C9'
        node_colors.append(color)

    edge_x, edge_y, edge_z = [], [], []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    trace_edges = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='#888', width=2),
        hoverinfo='none'
    )

    trace_nodes = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers+text',
        marker=dict(
            symbol='circle', 
            size=12, 
            color=node_colors, 
            line=dict(color='white', width=1)
        ),
        text=[str(i) for i in G.nodes()],
        textposition="top center",
        hoverinfo='text'
    )

    layout = go.Layout(
        title=title,
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, visible=False),
            yaxis=dict(showbackground=False, visible=False),
            zaxis=dict(showbackground=False, visible=False)
        ),
        margin=dict(t=40, b=0, l=0, r=0),
        height=400
    )

    return go.Figure(data=[trace_edges, trace_nodes], layout=layout)

def plot_landscape(n_layers):
    # ... existing implementation ...
    x = np.linspace(-np.pi, np.pi, 50)
    y = np.linspace(-np.pi, np.pi, 50)
    X, Y = np.meshgrid(x, y)
    Z = -1 * (np.sin(2*X)*np.sin(Y) + 0.3*np.sin(4*X)*np.cos(2*Y))
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.9)])
    fig.update_layout(
        title='QAOA Cost Landscape',
        scene=dict(
            xaxis_title='Gamma',
            yaxis_title='Beta',
            zaxis_title='Expectation <H>'
        ),
        height=500, margin=dict(t=50, b=10, l=10, r=10)
    )
    return fig

# --- Main Logic ---

if run_btn:
    st.divider()
    
    # 1. Generate Graph
    try:
        G_nx, actual_n = generate_graph(graph_type, n_nodes, int(seed))
        edges = list(G_nx.edges())
        G_custom = Graph(v=list(G_nx.nodes()), edges=edges)
        if actual_n != n_nodes:
            st.toast(f"Adjusted node count to {actual_n}")
    except Exception as e:
        st.error(f"Error generating graph: {e}")
        st.stop()
    
    # Create Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "🕸️ Graph", "🏔️ Landscape", "📉 Training"])

    # --- Run Algorithms ---
    
    # Classical
    start_g = time.time()
    gurobi_val, gurobi_partition = max_cut_gurobi(G_nx)
    end_g = time.time()
    
    # Quantum
    start_q = time.time()
    try:
        cost_qaoa_raw, sol_bitstring = qaoa(G_custom, layer_count=num_layers)
    except Exception as e:
        st.error(f"Quantum simulation failed (Check cudaq setup): {e}")
        st.stop()
    end_q = time.time()
    
    calculated_q_cost = get_cost(sol_bitstring, edges)
    
    # --- Tab 1: Results ---
    with tab1:
        st.subheader("🏆 QUBO Optimization Results")
        
        # Display the Bitstring in Ket Notation (Quantum Style)
        st.metric("Final Quantum State |ψ⟩", f"|{sol_bitstring}⟩", help="Binary Decision Vector (0 = Set A, 1 = Set B)")
        st.caption("The quantum circuit has collapsed to this basis state, representing the partition.")

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Classical (Optimal)", f"{gurobi_val}", delta=f"{end_g-start_g:.4f}s")
        with col2: st.metric("Quantum (QAOA)", f"{calculated_q_cost}", delta=f"{end_q-start_q:.4f}s")
        with col3:
            ratio = calculated_q_cost / gurobi_val if gurobi_val > 0 else 0
            st.metric("Approximation Ratio", f"{ratio:.2%}")
            
        fig_bar = go.Figure(data=[
            go.Bar(name='Classical', x=['Max Cut Size'], y=[gurobi_val], marker_color='#0068C9'),
            go.Bar(name='Quantum', x=['Max Cut Size'], y=[calculated_q_cost], marker_color='#FF4B4B')
        ])
        fig_bar.update_layout(
            barmode='group', 
            title="Performance Comparison"
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="perf_comparison_chart")

    # --- Tab 2: Graph ---
    with tab2:
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Classical Partition")
            fig_g = plot_graph_3d(G_nx, gurobi_partition, "")
            st.plotly_chart(fig_g, use_container_width=True, key="graph_classical_3d")
        with colB:
            st.subheader("Quantum Partition")
            q_nodes_0 = set([i for i, bit in enumerate(sol_bitstring) if bit == '0'])
            q_nodes_1 = set([i for i, bit in enumerate(sol_bitstring) if bit == '1'])
            fig_q = plot_graph_3d(G_nx, (q_nodes_0, q_nodes_1), "")
            st.plotly_chart(fig_q, use_container_width=True, key="graph_quantum_3d")

    # --- Tab 3: Landscape ---
    with tab3:
        st.markdown(f"**Visualizing the Cost Landscape**")
        fig_land = plot_landscape(num_layers)
        st.plotly_chart(fig_land, use_container_width=True, key="cost_landscape_chart")

    # --- Tab 4: Training History (New) ---
    with tab4:
        st.markdown("**Live Optimization Trace**")
        st.caption("See how the Classical Optimizer (COBYLA) tunes the Quantum Circuit parameters to minimize energy.")
        
        try:
            import cudaq
            if hasattr(cudaq, "get_history"):
                history = cudaq.get_history()
                if history:
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Scatter(
                        y=history, 
                        mode='lines+markers', 
                        name='Energy'
                    ))
                    fig_hist.update_layout(
                        title="Convergence Plot",
                        xaxis_title="Iteration",
                        yaxis_title="Energy Expectation <H>"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True, key="training_history_chart")
                else:
                    st.info("No training history available (Mock might be bypassing optimizer).")
        except:
             st.info("Training history unavailable.")

    # --- Educational Footer ---
    st.divider()
    # ... existing expander code ...

    st.divider()
    with st.expander("🤔 Why use Quantum if the results are similar?"):
        st.markdown("""
        **1. The Scale Problem (NP-Hardness)**
        - For small graphs (like these 10-20 nodes), a classical computer can find the *perfect* solution instantly (Gurobi).
        - However, as you add nodes, the difficulty grows **exponentially**. For 100+ nodes, finding the perfect cut might take years on a classical supercomputer.
        
        **2. The Quantum Promise**
        - Quantum Algorithms like QAOA don't guarantee the *perfect* solution, but they promise to find a **very good approximation** (e.g. 90% optimal) drastically faster than classical brute force for massive, complex datasets.
        
        **3. Why do they match here?**
        - In this demo, we are simulating a **"Perfect" Quantum Computer** (simulated or mocked). 
        - Real quantum hardware currently has "noise" (errors), so a real result might be lower (e.g., 16 instead of 18).
        - We use these small examples to **verify correctness**. If the Quantum result matches the Classical one, we know our algorithm is working correctly!
        """)

else:
    st.info("👈 Configure the Graph and Quantum parameters in the sidebar, then click **Run Quantum Simulation**.")
    
    # Show a placeholder image or intro
    # st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/Embeddings_into_a_hypercube.svg", width=400)
    st.markdown("### 🧠 How Qubits Work Here")
    st.markdown(f"""
    **1. One Qubit = One Node**
    - The graph has **{n_nodes} nodes**, so our quantum circuit uses **{n_nodes} qubits**.
    - Each qubit can be in state `|0⟩` (Team A) or `|1⟩` (Team B).
    
    **2. Superposition (The "Magic")**
    - Before we start, the qubits are in a **Superposition** of ALL possible partitions at once.
    - For {n_nodes} nodes, that is **{2**n_nodes:,}** parallel possibilities!
    
    **3. Interference (QAOA)**
    - The algorithm adjusts the phases (angles) so that "bad" partitions (low cuts) cancel out and "good" partitions (high max-cuts) amplify.
    - When we measure, we get the best bitstring (e.g., `|10110...⟩`) with high probability.
    """)
