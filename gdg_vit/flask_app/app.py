import sys
import os
import json
import time
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify

# --- Path Configuration ---
# Add the parent directory (gdg_vit) to sys.path to allow importing 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# --- Application Imports ---
try:
    from src.utilities import (
        generate_regular_3_graph, 
        max_cut_gurobi, 
        get_cost, 
        Graph, 
        brute_force_max_cut, 
        greedy_max_cut, 
        GUROBI_AVAILABLE
    )
    from src.QAOA_qiskit import qaoa, QISKIT_AVAILABLE
except ImportError as e:
    print(f"Error importing src modules: {e}")
    # We will handle this gracefully later or fail hard here if critical

app = Flask(__name__)

# --- Helper Functions (Adapted from Streamlit App) ---

def generate_graph(g_type, n, s, p_prob=0.5):
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

def create_plot_graph_3d(G, partition_sets, title):
    try:
        pos = nx.spring_layout(G, dim=3, seed=42)
    except:
        pos = nx.spring_layout(G, dim=3)
        
    x_nodes = [pos[k][0] for k in G.nodes()]
    y_nodes = [pos[k][1] for k in G.nodes()]
    z_nodes = [pos[k][2] for k in G.nodes()]
    
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
            zaxis=dict(showbackground=False, visible=False),
            bgcolor='rgba(0,0,0,0)' # Transparent background
        ),
        margin=dict(t=40, b=0, l=0, r=0),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    return go.Figure(data=[trace_edges, trace_nodes], layout=layout)

def create_plot_landscape(n_layers):
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
            zaxis_title='Expectation <H>',
            bgcolor='rgba(0,0,0,0)',
             xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="gray", showbackground=True),
             yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="gray", showbackground=True),
             zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="gray", showbackground=True),
        ),
        height=500, margin=dict(t=50, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    data = request.json
    
    # Extract params
    graph_type = data.get('graph_type', 'Random 3-Regular')
    n_nodes = int(data.get('n_nodes', 10))
    p_prob = float(data.get('p_prob', 0.5))
    num_layers = int(data.get('num_layers', 1))
    seed = int(data.get('seed', 42))
    quantum_backend = data.get('quantum_backend', 'simulator (local)')
    ibm_token = data.get('ibm_token', '')
    classical_method = data.get('classical_method', 'Greedy (Approximate)')

    # 1. Generate Graph
    try:
        G_nx, actual_n = generate_graph(graph_type, n_nodes, seed, p_prob)
        edges = list(G_nx.edges())
        G_custom = Graph(v=list(G_nx.nodes()), edges=edges)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # 2. Classical Algorithm
    start_g = time.time()
    if classical_method == "Brute Force (Exponential)":
        if n_nodes > 22:
             # Fallback
             gurobi_val, gurobi_partition = greedy_max_cut(G_nx)
             c_method_used = "Greedy (Fallback)"
        else:
             gurobi_val, gurobi_partition = brute_force_max_cut(G_nx)
             c_method_used = "Brute Force"
    elif classical_method == "Greedy (Approximate)":
        gurobi_val, gurobi_partition = greedy_max_cut(G_nx)
        c_method_used = "Greedy"
    elif classical_method == "Gurobi (Optimal)" and GUROBI_AVAILABLE:
        gurobi_val, gurobi_partition = max_cut_gurobi(G_nx)
        c_method_used = "Gurobi"
    else:
        gurobi_val, gurobi_partition = greedy_max_cut(G_nx)
        c_method_used = "Greedy (Default)"
    end_g = time.time()

    # 3. Quantum Algorithm
    start_q = time.time()
    try:
        cost_qaoa_raw, sol_bitstring = qaoa(
            G_custom, 
            layer_count=num_layers,
            backend_name=quantum_backend,
            ibm_token=ibm_token if ibm_token else None
        )
    except Exception as e:
         return jsonify({'error': f"Quantum simulation failed: {str(e)}"}), 500
    end_q = time.time()
    
    calculated_q_cost = get_cost(sol_bitstring, edges)
    
    # 4. Prepare Plots
    
    # Classical Plot
    fig_classical = create_plot_graph_3d(G_nx, gurobi_partition, f"Classical Partition ({c_method_used})")
    
    # Quantum Plot
    q_nodes_0 = set([i for i, bit in enumerate(sol_bitstring) if bit == '0'])
    q_nodes_1 = set([i for i, bit in enumerate(sol_bitstring) if bit == '1'])
    fig_quantum = create_plot_graph_3d(G_nx, (q_nodes_0, q_nodes_1), "Quantum Partition (QAOA)")
    
    # Landscape Plot
    fig_landscape = create_plot_landscape(num_layers)
    
    # Performance Chart
    fig_perf = go.Figure(data=[
        go.Bar(name='Classical', x=['Max Cut Size'], y=[gurobi_val], marker_color='#0068C9'),
        go.Bar(name='Quantum', x=['Max Cut Size'], y=[calculated_q_cost], marker_color='#FF4B4B')
    ])
    fig_perf.update_layout(
         barmode='group', title="Performance Comparison",
         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
         font=dict(color='white')
    )

    # Serialize Plots
    plots = {
        'classical': json.loads(json.dumps(fig_classical, cls=plotly.utils.PlotlyJSONEncoder)),
        'quantum': json.loads(json.dumps(fig_quantum, cls=plotly.utils.PlotlyJSONEncoder)),
        'landscape': json.loads(json.dumps(fig_landscape, cls=plotly.utils.PlotlyJSONEncoder)),
        'performance': json.loads(json.dumps(fig_perf, cls=plotly.utils.PlotlyJSONEncoder))
    }

    results = {
        'classical_cost': gurobi_val,
        'classical_time': end_g - start_g,
        'quantum_cost': calculated_q_cost,
        'quantum_time': end_q - start_q,
        'bitstring': sol_bitstring,
        'actual_n': actual_n,
        'plots': plots
    }

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
