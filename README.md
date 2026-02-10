# ⚛️ Four Wizards — Quantum QAOA Max-Cut Explorer

An interactive web application that demonstrates how the **Quantum Approximate Optimization Algorithm (QAOA)** solves the [Max-Cut](https://en.wikipedia.org/wiki/Maximum_cut) problem. Visualize graph topologies in 3D, explore the QAOA cost landscape, and compare quantum versus classical solver performance — all from your browser.

## Features

- **Interactive Graph Generation** — choose from Random 3-Regular, Ring, Star, and Erdős–Rényi topologies with configurable node counts and random seeds.
- **Quantum QAOA Solver** — powered by [Qiskit](https://qiskit.org/) with support for local statevector simulation and real IBM Quantum hardware via [Qiskit IBM Runtime](https://github.com/Qiskit/qiskit-ibm-runtime).
- **Classical Benchmarks** — compare against Greedy, Brute-Force, and (optionally) Gurobi-optimal solvers.
- **3D Visualizations** — interactive Plotly graphs showing graph partitions, QAOA cost landscapes, and performance bar charts.
- **Experiment History** — optional Firebase/Firestore integration to persist and review past runs.
- **Dual Frontend** — a Flask web app for deployment and a Streamlit app for rapid local exploration.

## Project Structure

```
├── app.py                   # Flask entry point
├── Procfile                 # Gunicorn process definition (Heroku/Render)
├── render.yaml              # Render deployment configuration
├── requirements.txt         # Python dependencies
├── gdg_vit/
│   ├── app.py               # Streamlit frontend
│   ├── flask_app/
│   │   ├── app.py            # Flask application and API routes
│   │   ├── templates/        # HTML templates
│   │   └── static/           # CSS and JavaScript assets
│   ├── src/
│   │   ├── QAOA.py           # CUDA Quantum (cudaq) QAOA implementation
│   │   ├── QAOA_qiskit.py    # Qiskit-based QAOA (simulator + IBM hardware)
│   │   ├── QAOA_square.py    # Divide-and-conquer QAOA variant
│   │   ├── utilities.py      # Graph helpers, Max-Cut solvers, cost functions
│   │   ├── firebase_handler.py # Firebase/Firestore persistence
│   │   └── general.py        # Matplotlib formatting utilities
│   ├── Getting_started/      # Introductory Jupyter notebooks
│   ├── results/              # Benchmark CSV results
│   └── Other/                # Supplementary images and documents
└── .devcontainer/            # GitHub Codespaces / VS Code dev container config
```

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/nivaslinga2/gdg-vit.git
   cd gdg-vit
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Running the Flask Web App

```bash
python app.py
```

The app will be available at `http://localhost:5000`.

For production-like serving with Gunicorn:

```bash
gunicorn app:app -b 0.0.0.0:8000 --timeout 120
```

### Running the Streamlit App

```bash
streamlit run gdg_vit/app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

1. **Configure the graph** — select a topology type, set the number of nodes, and choose a random seed.
2. **Set quantum parameters** — pick the number of QAOA layers (depth) and a quantum backend (`simulator (local)` or an IBM Quantum system).
3. **Choose a classical solver** — Greedy, Brute Force, or Gurobi (if installed).
4. **Run the simulation** — view the results, 3D graph partitions, cost landscape, and performance comparison.

### Using Real IBM Quantum Hardware

1. Sign up for a free account at [quantum.ibm.com](https://quantum.ibm.com).
2. Copy your API token.
3. Select an IBM backend (e.g., `ibm_brisbane`) and paste your token when prompted.

> **Note:** Real hardware jobs may take several minutes due to queue times.

## Optional Integrations

| Integration | Purpose | Setup |
|---|---|---|
| **Firebase / Firestore** | Persist experiment results | Set the `FIREBASE_CREDENTIALS_PATH` environment variable to your service account JSON file, or place `firebase_key.json` in the project root. |
| **Gurobi** | Optimal classical Max-Cut solver | Install `gurobipy` and configure a license. The app falls back to greedy/brute-force solvers when Gurobi is unavailable. |
| **CUDA Quantum (cudaq)** | GPU-accelerated QAOA | Install the NVIDIA CUDA Quantum SDK. Used by `src/QAOA.py` for GPU-based simulation. |

## Deployment

The project includes configuration for deploying to **Render**:

```bash
# render.yaml is already configured — connect your repo on https://render.com
```

A `Procfile` is also included for platforms like **Heroku**:

```
web: gunicorn app:app -b 0.0.0.0:$PORT --timeout 120
```

## Tech Stack

- **Backend:** Flask, Gunicorn
- **Frontend:** HTML/CSS/JS (Flask templates), Streamlit
- **Quantum:** Qiskit, Qiskit IBM Runtime, CUDA Quantum
- **Visualization:** Plotly, Matplotlib
- **Graph Theory:** NetworkX
- **Database:** Firebase / Firestore (optional)

## License

This project is provided as-is for educational and research purposes.
