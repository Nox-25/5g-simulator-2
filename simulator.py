import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

def calculate_signal_strength(size, towers, Pt, n, frequency_mhz=3500):
    """
    Calculates the 5G signal strength on a 2D grid based on tower locations,
    transmission power (Pt), path loss exponent (n), and Frequency (MHz)
    using the FSPL model: PL = 32.44 + 20*log10(f) + 20*log10(d)
    where d is distance in km.
    """
    grid = np.zeros((size, size))

    # Assume the 100x100 grid represents a 10km x 10km area (0.1km per grid cell)
    cell_size_km = 0.1

    for i in range(size):
        for j in range(size):
            signals = []
            for tx, ty in towers:
                # Distance in grid units
                dist_units = np.sqrt((i - tx)**2 + (j - ty)**2)
                # Distance in km. Add a tiny offset to avoid log10(0) for exactly matching points
                d_km = dist_units * cell_size_km + 0.001

                # FSPL formula
                PL = 32.44 + 20 * np.log10(frequency_mhz) + 20 * np.log10(d_km)

                # We can still factor in 'n' to adjust the distance weight if desired,
                # but standard FSPL uses 20*log10(d). Let's blend user MVP 'n'
                # (where n=2 is free space, n>2 is urban) by replacing 20 with 10*n.
                # So: PL = 32.44 + 20*log10(f) + 10*n*log10(d)
                PL = 32.44 + 20 * np.log10(frequency_mhz) + 10 * n * np.log10(d_km)

                Pr = Pt - PL
                signals.append(Pr)

            if signals:
                grid[i, j] = max(signals)
            else:
                grid[i, j] = -200 # No towers, very low signal

    return grid

def generate_users(size, num_users=50, num_clusters=3):
    """
    Generates random user coordinates on the grid, clustered around random centers.
    """
    centers = np.random.rand(num_clusters, 2) * size
    users = []

    for _ in range(num_users):
        center = centers[np.random.randint(num_clusters)]
        # Add some gaussian noise around the center
        x = np.clip(np.random.normal(center[0], size/10), 0, size-1)
        y = np.clip(np.random.normal(center[1], size/10), 0, size-1)
        users.append([int(x), int(y)])

    return users

def place_towers_kmeans(users, num_towers=3):
    """
    Uses K-Means clustering to place towers at the centers of user clusters.
    """
    if not users:
        return []

    # K-means requires num_samples >= n_clusters
    n_clusters = min(num_towers, len(users))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(users)

    towers = []
    for center in kmeans.cluster_centers_:
        towers.append([int(center[0]), int(center[1])])

    return towers

def plot_heatmap(grid, output_filename="heatmap.png"):
    """
    Plots and saves the signal coverage heatmap.
    """
    plt.figure(figsize=(8, 6))
    # Use jet for visual mapping from strong (red) to weak (blue)
    plt.imshow(grid, cmap='jet')
    plt.colorbar(label="Signal Strength (dBm)")
    plt.title("5G Coverage Heatmap")
    plt.savefig(output_filename)
    plt.close()

def plot_weak_zones(grid, threshold=-90, output_filename="weak_zones.png"):
    """
    Identifies weak signal zones (e.g., < -90 dBm) and plots/saves them.
    Weak zones are highlighted, strong zones are ignored.
    """
    plt.figure(figsize=(8, 6))
    # Create a boolean mask where True (1) indicates weak signal
    weak_zones = (grid < threshold).astype(int)

    # We can plot this as a binary map (e.g., Red for weak, White for okay)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['white', 'red'])

    plt.imshow(weak_zones, cmap=cmap)
    plt.title(f"Weak Signal Zones (< {threshold} dBm)")

    # Optional: add a custom legend for clarity
    import matplotlib.patches as mpatches
    weak_patch = mpatches.Patch(color='red', label='Weak Zone')
    okay_patch = mpatches.Patch(color='white', label='Okay Coverage')
    plt.legend(handles=[weak_patch, okay_patch], bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

def main():
    # MVP Configuration
    size = 100
    towers = [(20, 30), (70, 80)]
    Pt = 0   # dBm
    n = 3    # Path loss exponent

    print(f"Simulating 5G Coverage for {len(towers)} towers on a {size}x{size} grid...")
    grid = calculate_signal_strength(size, towers, Pt, n)

    print("Generating heatmap...")
    plot_heatmap(grid, "heatmap.png")

    print("Detecting weak signal zones (threshold = -90 dBm)...")
    plot_weak_zones(grid, threshold=-90, output_filename="weak_zones.png")

    print("Simulation complete! Check 'heatmap.png' and 'weak_zones.png'.")

if __name__ == "__main__":
    main()
