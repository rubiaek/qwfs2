import numpy as np
import matplotlib.pyplot as plt

class QWFSResult:
    def __init__(self, path=None):
        self.path = path
        self.T_methods = None
        self.configs = None
        self.N_tries = None
        self.algos = None
        self.N_pixels = None
        # results.shape == N_T_methods, N_configs, N_tries, N_algos
        self.results = None
        self.Ts = None
        # best_phases.shape == N_T_methods, N_configs, N_tries, N_algos, self.N
        self.best_phases = None
        self.sig_for_gauss_iid = np.sqrt(2)/2

        if path:
            self.loadfrom(path)

    @property
    def N_algos(self):
        return len(self.algos)

    @property
    def N_configs(self):
        return len(self.configs)

    @property
    def N_T_methods(self):
        return len(self.T_methods)

    @property
    def N_modes(self):
        return self.best_phases.shape[-1]

    def saveto(self, path):
        np.savez(path, **self.__dict__)

    def loadfrom(self, path):
        data = np.load(path, allow_pickle=True)
        self.__dict__.update(data)

    def show(self):
        # all configurations
        fig, axes = plt.subplots(len(self.configs), len(self.T_methods))
        for config_no, config in enumerate(self.configs):
            for T_method_no, T_method in enumerate(self.T_methods):
                # upper_lim = 1 if T_method == 'unitary' else 2
                # imm = axes[config_no, T_method_no].imshow(results[T_method_no, config_no], clim=(0, upper_lim))
                if len(self.configs) > 1:
                    ax = axes[config_no, T_method_no]
                else:
                    ax = axes[T_method_no]
                imm = ax.imshow(self.results[T_method_no, config_no], aspect='auto')
                ax.set_title(rf'{config}, {T_method}')
                fig.colorbar(imm, ax=ax)
        fig.show()

    def show_scatterplots(self):
        # Create a figure with N_T_methods rows and N_configs columns
        fig, axes = plt.subplots(
            len(self.T_methods),
            len(self.configs),
            figsize=(2 * len(self.configs), 3 * len(self.T_methods)),
            sharey=False, constrained_layout=True
        )

        # Ensure axes is always a 2D array, even if one dimension is 1
        if len(self.T_methods) == 1:
            axes = axes.reshape(1, -1)
        elif len(self.configs) == 1:
            axes = axes.reshape(-1, 1)

        # Color palette for algorithms
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.algos)))

        # Reference lines
        reference_lines = [
            np.pi / 4,  # π/4
            (np.pi / 4) ** 2,  # (π/4)²
            1  # 1
        ]

        # Line styles for reference lines
        line_styles = ['--', ':', '-']
        line_colors = ['gray', 'gray', 'black']

        # Loop over T_methods and configurations
        for t_method_idx, t_method in enumerate(self.T_methods):
            for config_idx, config in enumerate(self.configs):
                ax = axes[t_method_idx, config_idx]

                # Add reference lines
                for line_val, line_style, line_color in zip(reference_lines, line_styles, line_colors):
                    ax.axhline(y=line_val, color=line_color, linestyle=line_style, alpha=0.5)

                # Get data for this T_method and configuration
                data = self.results[t_method_idx, config_idx, :, :]  # Shape: (N_tries, N_algos)

                # Scatter plot for each algorithm
                for algo_idx, algo in enumerate(self.algos):
                    # Add a bit of jitter to x-position to spread out points
                    x = np.random.normal(algo_idx, 0.1, len(data[:, algo_idx]))
                    ax.scatter(
                        x,
                        data[:, algo_idx],
                        color=colors[algo_idx],
                        alpha=0.3,  # More transparent
                        s=10,  # Smaller markers
                        label=algo
                    )
                ax.set_ylim(bottom=0)

                # Set title and labels
                ax.set_title(f'{config}')
                # ax.set_xlabel('Algorithms')

                # Set x-ticks to algorithm names
                ax.set_xticks(range(len(self.algos)))
                ax.set_xticklabels(self.algos, rotation=45, ha='right')

                if config_idx == 0:
                    ax.set_ylabel(t_method, fontweight='bold',fontsize=12)
                if config_idx == self.N_configs - 1:
                    ax.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

        # Adjust layout
        plt.tight_layout()

        # Add an overall title
        fig.suptitle('Performance Across T-Methods and Configurations', y=1.02)

    def print(self, only_slm3=False):
        for config_no, config in enumerate(self.configs):
            if only_slm3 and config != 'SLM3':
                continue
            print(f'---- {config} ----')
            for T_method_no, T_method in enumerate(self.T_methods):
                print(f'-- {T_method} --')
                for algo_no, algo in enumerate(self.algos):
                    avg = self.results[T_method_no, config_no].mean(axis=0)[algo_no]
                    std = self.results[T_method_no, config_no].std(axis=0)[algo_no]

                    print(f'{algo:<25} {avg:.3f}+-{std:.2f}')
            print()