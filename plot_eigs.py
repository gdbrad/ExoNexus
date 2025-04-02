import numpy as np
import matplotlib.pyplot as plt

def plot_extreme_eigenvalues(input_file="all_configs_eigenvalues.npy"):
    # Load the 3D array: (n_configs, 64, 96)
    data = np.load(input_file)
    n_configs, n_timeslices, n_eigs = data.shape
    print(f"Loaded data with shape: {data.shape}")

    # Average across configurations: shape becomes (64, 96)
    avg_over_configs = np.mean(data, axis=0)
    print(f"Average over configs shape: {avg_over_configs.shape}")

    # Sort eigenvalues for each timeslice (largest to smallest)
    sorted_eigs_per_timeslice = np.sort(avg_over_configs, axis=1)[:, ::-1]

    # Extract 3 largest and 3 smallest for each timeslice
    largest_3 = sorted_eigs_per_timeslice[:, :3]    # First 3 columns
    smallest_3 = sorted_eigs_per_timeslice[:, -3:]   # Last 3 columns
    timeslices = np.arange(n_timeslices)             # 0 to 63

    # Plot 1: Three largest eigenvalues
    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.scatter(timeslices, largest_3[:, i], label=f'Largest {i+1}', s=20)
    plt.xlabel('Timeslice')
    plt.ylabel('Eigenvalue')
    plt.title('Three Largest Eigenvalues per Timeslice (Averaged over Configs)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('largest_eigenvalues.png')
    plt.close()

    # Plot 2: Three smallest eigenvalues
    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.scatter(timeslices, smallest_3[:, i], s=20)
    plt.xlabel('Timeslice')
    plt.ylabel('Eigenvalue')
    plt.title('Three Smallest Eigenvalues per Timeslice (Averaged over Configs)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('smallest_eigenvalues.png')
    plt.close()

    print("Plots saved as 'largest_eigenvalues.png' and 'smallest_eigenvalues.png'")

if __name__ == "__main__":
    plot_extreme_eigenvalues(input_file="all_configs_eigenvalues.npy")