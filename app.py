from flask import Flask, render_template, request, jsonify
import numpy as np
from simulator import calculate_signal_strength, generate_users, place_towers_kmeans

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.json

    # Parse input data with default values
    size = data.get('size', 100)
    towers = data.get('towers', [])
    Pt = data.get('Pt', 0)
    n = data.get('n', 3)
    threshold = data.get('threshold', -90)
    frequency_mhz = data.get('frequency', 3500)
    users = data.get('users', [])

    if not towers:
        # Return empty grids and 0% coverage if no towers
        return jsonify({
            'heatmap': np.full((size, size), -100).tolist(),
            'weak_zones': np.ones((size, size)).tolist(),
            'coverage_percentage': 0.0,
            'max_signal': -100,
            'min_signal': -100
        })

    # Calculate signal strength using the logic from simulator.py
    grid = calculate_signal_strength(size, towers, Pt, n, frequency_mhz)

    # Calculate weak zones (1 if weak, 0 if okay)
    weak_zones = (grid < threshold).astype(int)

    # Calculate coverage percentage (not weak)
    total_area = size * size
    weak_area = np.sum(weak_zones)
    coverage_percentage = ((total_area - weak_area) / total_area) * 100

    return jsonify({
        'heatmap': grid.tolist(),
        'weak_zones': weak_zones.tolist(),
        'coverage_percentage': float(coverage_percentage),
        'max_signal': float(np.max(grid)),
        'min_signal': float(np.min(grid))
    })

@app.route('/scatter_users', methods=['POST'])
def scatter_users():
    data = request.json
    size = data.get('size', 100)
    num_users = data.get('num_users', 50)
    num_clusters = data.get('num_clusters', 3)

    users = generate_users(size, num_users, num_clusters)
    return jsonify({'users': users})

@app.route('/auto_place', methods=['POST'])
def auto_place():
    data = request.json
    users = data.get('users', [])
    num_towers = data.get('num_towers', 3)

    towers = place_towers_kmeans(users, num_towers)
    return jsonify({'towers': towers})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
