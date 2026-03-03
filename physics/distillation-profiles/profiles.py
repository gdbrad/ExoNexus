import numpy as np
import gvar as gv
import matplotlib.pyplot as plt

class GaussianProfile:
    def __init__(self, input_file):
        """Initialize with the path to the eigenvalue data file."""
        self.input_file = input_file
        self.eigs = None
        self.sorted_eigs = None
        self.largest_3 = None
        self.smallest_3 = None
        self.timeslices = None
        self.widths = None
        self.gaussian_profiles = None
        self.load_data()

    def load_data(self):
        """Load and process eigenvalue data from the input file."""
        data = np.load(self.input_file)
        n_configs, n_timeslices, n_eigs = data.shape
        print(f"Loaded data with shape: {data.shape}")

        # take gauge avg
        avg_over_configs = np.mean(data, axis=0)
        print(f"Average over configs shape: {avg_over_configs.shape}")

        self.sorted_eigs = np.sort(avg_over_configs, axis=1)[:, ::-1]
        self.eigs = self.sorted_eigs 
        self.timeslices = np.arange(n_timeslices)

        # 3 extrema eigenvalues per timeslice
        self.largest_3 = self.sorted_eigs[:, :3]    
        self.smallest_3 = self.sorted_eigs[:, -3:]

    def compute_widths(self):
        """compute 7 widths \lambda based on the smallest and largest eigenvalues"""
        small = self.smallest_3[0, 2]
        large = self.largest_3[2, 0]
        self.widths = np.linspace(0.1, large, num=7)
        print("Computed widths:", self.widths)
        return self.widths

    def compute_gaussian_profiles(self):
        """compute Gaussian profiles for each eigenvalue and width"""
        if self.widths is None:
            self.compute_widths()

        self.gaussian_profiles = {}
        for eig in self.eigs.flatten():
            self.gaussian_profiles[str(eig)] = {}
            for k in self.widths:
                self.gaussian_profiles[str(eig)][k] = np.exp(-eig**2 / (2 * k**2))
        return self.gaussian_profiles

    def plot_gaussian_profiles(self, output_file="gaussian_profiles.png"):
        """Generate a plot of Gaussian profiles similar to the right panel."""
        if self.gaussian_profiles is None:
            self.compute_gaussian_profiles()

        lambda_range = np.linspace(min(self.eigs.flatten()), max(self.eigs.flatten()), 100)
        plt.figure(figsize=(8, 6))

        # plot profile for each width
        for i, k in enumerate(self.widths):
            g_values = [np.exp(-lam**2 / (2 * k**2)) for lam in lambda_range]
            plt.plot(lambda_range, g_values, label=f"$\sigma_{i}$", linestyle='-' if i < 4 else '--')

        plt.xlabel(r"$\lambda$", fontsize=12)
        plt.ylabel(r"$g(\lambda)$", fontsize=12)
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.legend(title="Widths", loc="upper right")
        plt.title("Gaussian Profiles for Different Widths")

        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    analyzer = GaussianProfile('all_configs_eigenvalues.npy')
    widths = analyzer.compute_widths()
    gaussian_profiles = analyzer.compute_gaussian_profiles()
    analyzer.plot_gaussian_profiles()