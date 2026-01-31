document.addEventListener('DOMContentLoaded', () => {
    // UI Interactions
    const rangeInputs = ['p_prob', 'n_nodes'];
    rangeInputs.forEach(id => {
        const input = document.getElementById(id);
        const valSpan = document.getElementById(`${id}_val`);
        if (input && valSpan) {
            input.addEventListener('input', (e) => {
                valSpan.textContent = e.target.value;
            });
        }
    });

    const graphTypeSelect = document.getElementById('graph_type');
    const probGroup = document.getElementById('prob-group');
    if (graphTypeSelect) {
        graphTypeSelect.addEventListener('change', (e) => {
            if (e.target.value === 'Erdos-Renyi') {
                probGroup.style.display = 'flex';
            } else {
                probGroup.style.display = 'none';
            }
        });
    }

    // Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const tabId = btn.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');

            // Resize plots if needed
            window.dispatchEvent(new Event('resize'));
        });
    });

    // Run Simulation
    const form = document.getElementById('configForm');
    const runBtn = document.getElementById('runBtn');
    const loading = document.getElementById('loading');
    const resultsArea = document.getElementById('results-area');
    const introPlaceholder = document.getElementById('intro-placeholder');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // UI Updates
        runBtn.disabled = true;
        loading.classList.remove('hidden');
        resultsArea.classList.add('hidden');
        introPlaceholder.classList.add('hidden');

        // Gather Data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/run_simulation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Simulation failed');
            }

            const result = await response.json();
            displayResults(result);

        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            runBtn.disabled = false;
            loading.classList.add('hidden');
        }
    });

    function displayResults(data) {
        resultsArea.classList.remove('hidden');

        // Metrics
        document.getElementById('bitstring-val').textContent = `|${data.bitstring}⟩`;
        document.getElementById('c-cost-val').textContent = data.classical_cost;
        document.getElementById('c-time-val').textContent = `${data.classical_time.toFixed(4)}s`;

        document.getElementById('q-cost-val').textContent = data.quantum_cost;
        document.getElementById('q-time-val').textContent = `${data.quantum_time.toFixed(4)}s`;

        // Plots
        const config = { responsive: true, displayModeBar: false };

        Plotly.newPlot('plot-performance', data.plots.performance.data, data.plots.performance.layout, config);
        Plotly.newPlot('plot-classical', data.plots.classical.data, data.plots.classical.layout, config);
        Plotly.newPlot('plot-quantum', data.plots.quantum.data, data.plots.quantum.layout, config);
        Plotly.newPlot('plot-landscape', data.plots.landscape.data, data.plots.landscape.layout, config);
    }
});
