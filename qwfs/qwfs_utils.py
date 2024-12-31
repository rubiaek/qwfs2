from qwfs.qwfs_result import QWFSResult
from qwfs.qwfs_simulation import QWFSSimulation
import numpy as np
import matplotlib.pyplot as plt
import datetime


def tnow():
    return datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


def get_slm3_intensities(res, T_method='gaus_iid'):
    # This is relevant only for SLM3, and there we use only BFGS currently...
    alg = 'L-BFGS-B'
    config = 'SLM3'
    # T_method = 'unitary'
    try_nos = range(res.N_tries)

    I_outs = []
    I_middles = []
    I_focuss = []
    # for try_no in [2]:
    for try_no in try_nos:
        alg_ind = np.where(res.algos == alg)[0]
        conf_ind = np.where(res.configs == config)[0]
        T_method_ind = np.where(res.T_methods == T_method)[0]

        T_ind = res.N_T_methods * try_no + T_method_ind
        T = res.Ts[T_ind].squeeze()
        slm_phases = res.best_phases[T_method_ind, conf_ind, try_no, alg_ind].squeeze()
        N = len(slm_phases)

        sim = QWFSSimulation(N=N)
        sim.T = T
        sim.slm_phases = np.exp(1j * slm_phases)
        sim.config = config

        I_middle = np.abs(sim.T.transpose() @ (sim.slm_phases * sim.v_in)) ** 2
        I_out = np.abs(sim.propagate()) ** 2
        I_outs.append(I_out)
        I_middles.append(I_middle)
        I_focuss.append(res.results[T_method_ind, conf_ind, try_no, alg_ind].squeeze())

    I_outs = np.array(I_outs)
    I_middles = np.array(I_middles)
    I_focuss = np.array(I_focuss)

    return I_outs, I_middles, I_focuss

def get_slm1_intensities(res, config='SLM1-only-T', T_method='gaus_iid', alg='L-BFGS-B'):
    try_nos = range(res.N_tries)

    I_outs = []
    I_focuss = []
    # for try_no in [2]:
    for try_no in try_nos:
        alg_ind = np.where(res.algos == alg)[0]
        conf_ind = np.where(res.configs == config)[0]
        T_method_ind = np.where(res.T_methods == T_method)[0]

        T_ind = res.N_T_methods * try_no + T_method_ind
        T = res.Ts[T_ind].squeeze()
        slm_phases = res.best_phases[T_method_ind, conf_ind, try_no, alg_ind].squeeze()
        N = len(slm_phases)

        sim = QWFSSimulation(N=N)
        sim.T = T
        sim.slm_phases = np.exp(1j * slm_phases)
        sim.config = config

        I_out = np.abs(sim.propagate()) ** 2
        I_outs.append(I_out)
        I_focuss.append(res.results[T_method_ind, conf_ind, try_no, alg_ind].squeeze())

    I_outs = np.array(I_outs)
    I_focuss = np.array(I_focuss)

    return I_outs, I_focuss


def get_res_phases(res, config='SLM1-only-T', T_method='gaus_iid', alg='L-BFGS-B'):
    try_nos = range(res.N_tries)
    phases = []
    for try_no in try_nos:
        alg_ind = np.where(res.algos == alg)[0]
        conf_ind = np.where(res.configs == config)[0]
        T_method_ind = np.where(res.T_methods == T_method)[0]

        slm_phases = res.best_phases[T_method_ind, conf_ind, try_no, alg_ind].squeeze()
        phases.append(slm_phases)
    return np.array(phases).squeeze()


def show_tot_energy_at_planes(I_middles, I_outs):
    fig, axes = plt.subplots(1, 3, figsize=(10, 6))

    for i, ax in enumerate(axes):
        if i == 0:
            Is = I_middles.sum(axis=1)
            title = '$I_{tot}$ at Crystal plane'
        elif i == 1:
            title = '$I_{tot}$ at Output plane'
            Is = I_outs.sum(axis=1)
        elif i == 2:
            title = r'$I_{out}/I_{crystal}^2$'
            Is = I_outs.sum(axis=1) / I_middles.sum(axis=1) ** 2
        # Create the histogram
        ax.hist(Is, bins=10, edgecolor='black', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Intensity Values')
        ax.set_ylabel('Frequency')

        # Add statistical lines
        ax.axvline(np.mean(Is), color='red', linestyle='dashed', linewidth=2,
                   label=f'Mean: {np.mean(Is):.2f}')
        ax.axvline(np.median(Is), color='green', linestyle='dashed', linewidth=2,
                   label=f'Median: {np.median(Is):.2f}')

        ax.legend()
        ax.grid(True, alpha=0.3)

        # Adjust layout and display
        plt.tight_layout()


def show_I_out_I_middle(I_out, I_middle):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(I_out, label='I_out')
    axes[0].plot(I_middle, label='I_middle')
    # ax.set_ylim([0, 0.05])
    axes[0].legend()
    axes[1].plot(I_out, label='output plane')
    axes[1].plot(I_middle, label='crystal plane')
    axes[1].set_ylim([0, 0.05])
    axes[1].legend()
    axes[0].set_title('see peek')
    axes[1].set_title('zoom in on fluctuations')
    fig.suptitle('Intensity distribution')


def get_output_random_phases(config='SLM3', T_method='gaus_iid', N=1000):
    sim = QWFSSimulation(N=256)
    sim.config = config
    sim.T_method = T_method
    sim.reset_T()
    Is = []
    for i in range(N):
        random_phases = np.random.uniform(0, 2*np.pi, sim.N)
        sim.slm_phases = np.exp(1j*random_phases)
        # sim.slm_phases = np.exp(1j*slm_phases)
        v_out = sim.propagate()
        I_out = np.abs(v_out)**2
        Is.append(I_out.sum())
    return Is


def show_hist_intensities(Is):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the histogram
    ax.hist(Is, bins=100, edgecolor='black', alpha=0.7)
    ax.set_title('Histogram of Intensities (Is)')
    ax.set_xlabel('Intensity Values')
    ax.set_ylabel('Frequency')

    # Add statistical lines
    ax.axvline(np.mean(Is), color='red', linestyle='dashed', linewidth=2,
               label=f'Mean: {np.mean(Is):.2f}')
    ax.axvline(np.median(Is), color='green', linestyle='dashed', linewidth=2,
               label=f'Median: {np.median(Is):.2f}')

    ax.text(0.02, 0.98, 'Sample Histogram\nMean: {:.3f}\nStd Dev: {:.3f}'.format(np.mean(Is), np.std(Is)),
            transform=ax.transAxes,  # Use axes coordinates
            verticalalignment='top',  # Align to the top
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.legend()
    ax.grid(True, alpha=0.3)

    # Adjust layout and display
    plt.tight_layout()


def show_max_norm_Ts(N_modes=256, N_Ts=1000):
    sim = QWFSSimulation(N=N_modes)
    sim.config = 'SLM3'
    sim.T_method = 'gaus_iid'
    max_Is = []
    for i in range(N_Ts):
        sim.reset_T()
        max_I = (np.abs(sim.T) ** 2).sum(axis=0).max()
        max_Is.append(max_I)
        max_I = (np.abs(sim.T) ** 2).sum(axis=1).max()
        max_Is.append(max_I)
    max_Is = np.array(max_Is)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the histogram
    ax.hist(max_Is, bins=100, edgecolor='black', alpha=0.7)
    ax.set_title('Histogram of max norm of Ts')
    ax.set_xlabel('Intensity Values')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

    # Adjust layout and display
    plt.tight_layout()


def show_effics(effics):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the histogram
    ax.hist(effics, bins=50, edgecolor='black', alpha=0.7)
    ax.set_title('Histogram of efficiency (I_out/I_out.sum()')
    ax.set_xlabel('Intensity Values')
    ax.set_ylabel('Frequency')

    # Add statistical lines
    ax.axvline(np.mean(effics), color='red', linestyle='dashed', linewidth=2,
               label=f'Mean: {np.mean(effics):.2f}')
    ax.axvline(np.median(effics), color='green', linestyle='dashed', linewidth=2,
               label=f'Median: {np.median(effics):.2f}')

    ax.text(0.02, 0.98, 'Sample Histogram\nMean: {:.3f}\nStd Dev: {:.3f}'.format(np.mean(effics), np.std(effics)),
            transform=ax.transAxes,  # Use axes coordinates
            verticalalignment='top',  # Align to the top
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.legend()
    ax.grid(True, alpha=0.3)

    # Adjust layout and display
    plt.tight_layout()


def show_max_SVD_N_dependance_pseudo_analytic(max_power=12):
    import numpy as np
    from scipy.integrate import quad
    import matplotlib.pyplot as plt

    def marchenko_pastur_pdf(x, q):
        """
        PDF of the Marchenko-Pastur distribution.
        """
        a = (1 - np.sqrt(q)) ** 2
        b = (1 + np.sqrt(q)) ** 2
        if a <= x <= b:
            return np.sqrt((b - x) * (x - a)) / (2 * np.pi * q * x)
        else:
            return 0

    def marchenko_pastur_cdf(x, q):
        """
        CDF of the Marchenko-Pastur distribution, computed by integrating the PDF.
        """
        a = (1 - np.sqrt(q)) ** 2
        if x < a:
            return 0
        b = (1 + np.sqrt(q)) ** 2
        if x > b:
            return 1
        result, _ = quad(lambda t: marchenko_pastur_pdf(t, q), a, x)
        return result

    def expected_maximum(N, q):
        """
        Compute the expected maximum for N samples from the Marchenko-Pastur distribution.
        """
        a = (1 - np.sqrt(q)) ** 2
        b = (1 + np.sqrt(q)) ** 2

        def integrand(x):
            f_x = marchenko_pastur_pdf(x, q)
            F_x = marchenko_pastur_cdf(x, q)
            return x * N * (F_x ** (N - 1)) * f_x

        result, _ = quad(integrand, a, b)
        return result

    # Parameters
    q = 1  # Ratio M/N
    N_values = 2 ** np.linspace(1, max_power, max_power)

    # Compute E[max_s] for each N
    expected_max_values = [expected_maximum(N, q) for N in N_values]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(N_values, expected_max_values, marker="o", label=f"Marchenko-Pastur (q={q})", color="blue")
    # ax.set_xscale("log")
    ax.set_xlabel("Number of Samples (N) [log scale]")
    ax.set_ylabel("Expected Maximum (E[max_s])")
    ax.set_title("Expected Maximum vs. Number of Samples for Marchenko-Pastur")
    ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

    # Print results for reference
    for N, value in zip(N_values, expected_max_values):
        print(f"N={N}, E[max_s]={value:.6f}")


def show_max_SVD_N_dependance_numeric(max_power=8, num_sample=100, show_histogram=False, show_means=True, TT=True):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from scipy.linalg import svd

    def largest_singular_value_distribution(N, num_samples=1000, TT=True):
        largest_singular_values = np.zeros(num_samples)

        for i in range(num_samples):
            T = 1 / np.sqrt(N) * np.random.normal(loc=0, scale=np.sqrt(2) / 2, size=(N, N, 2)).view(np.complex128)[:,:, 0]

            if TT:
                s = svd(T@T.T, compute_uv=False)
            else:
                s = svd(T, compute_uv=False)

            largest_singular_values[i] = s[0]

        return largest_singular_values

    N_values = 2 ** np.linspace(1, max_power, max_power, dtype=int)

    fig, ax = plt.subplots(figsize=(12, 8))

    means = []
    stds = []
    for N in N_values:
        # Simulate largest singular values
        largest_singular_values = largest_singular_value_distribution(N, num_samples=num_sample, TT=TT)
        mu = (largest_singular_values**2).mean()
        st = (largest_singular_values**2).std()
        mx = (largest_singular_values**2).max()
        means.append(mu)
        stds.append(st)
        print(f"N={N}, sqr_mean={mu:.3f}, sqr_std={st:.3f}, sqr_max={mx:.3f}")
        # Plot histogram of normalized singular values
        if show_histogram:
            ax.hist(largest_singular_values, bins=50, density=True, alpha=0.5, label=f'N={N}')

    if show_means:
        ax.errorbar(N_values, means, yerr=stds, marker="o", label=f"mean largest singular value, TT={TT}")
        ax.set_xlabel('N_modes')
        ax.set_ylabel('Largest singular value')
    else:
        ax.set_xlabel('Largest singular value')
        ax.set_ylabel('Probability density')
    ax.set_title('Largest singular value distribution')
    ax.legend()
    ax.grid(True)
