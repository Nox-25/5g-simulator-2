import numpy as np
import matplotlib.pyplot as plt

def calculate_signal_strength(size, towers, Pt, n):
    """
    Calculates the 5G signal strength on a 2D grid based on tower locations,
    transmission power, and path loss exponent.
    """
    grid = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            signals = []
            for tx, ty in towers:
                # Add 1 to avoid log(0) at the tower center
                d = np.sqrt((i - tx)**2 + (j - ty)**2) + 1
                Pr = Pt - 10 * n * np.log10(d)
                signals.append(Pr)
            # The signal at a point is the max signal from all nearby towers
            grid[i, j] = max(signals)

    return grid

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
