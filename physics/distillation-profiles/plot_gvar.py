import numpy as np
import matplotlib.pyplot as plt
import gvar 
from gvar import dataset

def plot_extreme_eigenvalues(input_file="all_configs_eigenvalues.npy"):
    data = np.load(input_file)
    n_configs, n_timeslices, n_eigs = data.shape
    print(f"Loaded data with shape: {data.shape}")

    # reshape to (n_configs, 64*96)
    flat_data = data.reshape(n_configs, -1)
    avg_data = dataset.avg_data(flat_data, bstrap=False)
    # reshape back to (64,96) for mean/sdev
    means = np.array([x.mean for x in avg_data]).reshape(n_timeslices, n_eigs)
    errors = np.array([x.sdev for x in avg_data]).reshape(n_timeslices, n_eigs)

    # sort eigs descending for each timeslice based on means
    sorted_indices = np.argsort(means, axis=1)[:, ::-1] 
    sorted_means = np.take_along_axis(means, sorted_indices, axis=1)
    sorted_errors = np.take_along_axis(errors, sorted_indices, axis=1)

    # get the 3 minima and maxima on each timeslice
    largest_3_means = sorted_means[:, :3]    
    largest_3_errors = sorted_errors[:, :3]
    smallest_3_means = sorted_means[:, -3:]  
    smallest_3_errors = sorted_errors[:, -3:]
    timeslices = np.arange(n_timeslices)     

    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.errorbar(
            timeslices, largest_3_means[:, i], yerr=largest_3_errors[:, i],
            label=f'Largest {i+1}', fmt='o', capsize=3, markersize=5
        )
    plt.xlabel('Timeslice')
    plt.ylabel('Eigenvalue')
    plt.title('Three Largest Eigenvalues per Timeslice (gauge averaged)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('largest_eigenvalues_with_errors.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.errorbar(
            timeslices, smallest_3_means[:, i], yerr=smallest_3_errors[:, i],
            label=f'Smallest {i+1}', fmt='o', capsize=3, markersize=5
        )
    plt.xlabel('Timeslice')
    plt.ylabel('Eigenvalue')
    plt.title('Three Smallest Eigenvalues per Timeslice (gauge averaged)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('smallest_eigenvalues_with_errors.png')
    plt.close()

    print("saved plots 'largest_eigenvalues_with_errors.png' and 'smallest_eigenvalues_with_errors.png'")

if __name__ == "__main__":
    plot_extreme_eigenvalues(input_file="all_configs_eigenvalues.npy")