import time
import datetime
import numpy as np
from scipy.stats import unitary_group
from scipy.optimize import minimize, dual_annealing
from functools import partial
from qwfs.qwfs_result import QWFSResult
try:
    import torch
    # so we can do T.transpose() on torch and np arrays similarly
    def _tensor_transpose(self, *args):
        if len(args) == 0:
            return self.t()
        return self.transpose(*args)

    torch.Tensor.transpose = _tensor_transpose
except ImportError:
    pass


class QWFSSimulation:
    def __init__(self, N=256, T_method='gaus_iid', config='SLM3'):
        self.N = N
        self.N_pixels = self.N
        self.DEFAULT_OUT_MODE = self.N // 2
        self.DEFAULT_ONEHOT_INPUT_MODE = 0
        self.T_method = T_method
        self.config = config
        self.sig_for_gauss_iid = np.sqrt(2)/2
        self.T = self.get_diffuser()

        self.v_in = 1/np.sqrt(self.N) * np.ones(self.N, dtype=np.complex128)
        self.slm_phases = np.exp(1j*np.zeros(self.N, dtype=np.complex128))
        self.f_calls = 0


    def get_diffuser(self):
        if self.T_method == 'gaus_iid':
            return 1/np.sqrt(self.N) * np.random.normal(loc=0, scale=self.sig_for_gauss_iid, size=(self.N, self.N, 2)).view(np.complex128)[:, :, 0]
        elif self.T_method == 'unitary':
            return unitary_group.rvs(self.N)
        elif self.T_method == 'cue':
            # More sophisticated random unitary generation
            Z = np.random.normal(0, 1, (self.N, self.N)) + 1j * np.random.normal(0, 1, (self.N, self.N))
            Q, R = np.linalg.qr(Z)
            D = np.diag(np.diag(R) / np.abs(np.diag(R)))
            return Q @ D
        elif self.T_method == 'thin':
            V = np.exp(1j*np.random.uniform(0, 2*np.pi, self.N))
            return np.diag(V)
        else:
            raise NotImplementedError()

    def reset_T(self):
        self.T = self.get_diffuser()

    def propagate(self, use_torch=False):
        fft = torch.fft.fft if use_torch else np.fft.fft

        if self.config == 'SLM1':
            after_T = self.T @ self.T.transpose() @ (self.slm_phases * self.v_in)
            v_out = fft(after_T) / np.sqrt(self.N)
        elif self.config == 'SLM1-only-T':
            v_out = self.T @ (self.slm_phases * self.v_in)
        elif self.config == 'SLM2' or self.config == 'SLM2-same-mode':
            after_T2 = self.T @ (self.slm_phases * (self.T.transpose() @ self.v_in))
            v_out = fft(after_T2) / np.sqrt(self.N)
        elif self.config == 'SLM2-simple':
            v_in_one_hot = np.zeros_like(self.v_in)
            v_in_one_hot[self.DEFAULT_ONEHOT_INPUT_MODE] = 1
            v_out = self.T @ (self.slm_phases * (self.T.transpose() @ v_in_one_hot))
        elif self.config == 'SLM2-simple-OPC':
            v_in_one_hot = np.zeros_like(self.v_in)
            v_in_one_hot[self.DEFAULT_OUT_MODE] = 1  # this assumes that we try to optimize the self.DEFAULT_OUT_MODE
            v_out = self.T @ (self.slm_phases * (self.T.transpose() @ v_in_one_hot))
        elif self.config == 'SLM3' or self.config == 'SLM3-same-mode':
            after_SLM_second_time = self.slm_phases * (self.T @ self.T.transpose() @ (self.slm_phases * self.v_in))
            v_out = fft(after_SLM_second_time) / np.sqrt(self.N)
        elif self.config == 'SLM5' or self.config == 'SLM5-same-mode':
            # a general mixing between SLM2 modes, when SLM2 is not at the crystal plane
            after_slm1 = self.slm_phases * (self.T.transpose() @ self.v_in)
            after_ft1 = fft(after_slm1) / np.sqrt(self.N)
            after_T2 = self.T @ (self.slm_phases * after_ft1)
            v_out = fft(after_T2) / np.sqrt(self.N)
        else:
            raise NotImplementedError('WAT?')

        return v_out

    def get_intensity(self, slm_phases_rad=None, out_mode=None, use_torch=False):
        exp = torch.exp if use_torch else np.exp
        abs = torch.abs if use_torch else np.abs
        slm_phases_rad = slm_phases_rad if use_torch else np.array(slm_phases_rad)

        slm_phases_rad = resize_array(slm_phases_rad, self.N, use_torch=use_torch)

        if slm_phases_rad is not None:
            self.slm_phases = exp(1j*slm_phases_rad)
        if out_mode is None:
            out_mode = self.DEFAULT_OUT_MODE
        v_out = self.propagate(use_torch=use_torch)
        I_out = abs(v_out)**2
        I_focus = I_out[out_mode]
        self.f_calls += 1

        return -I_focus


    def optimize(self, algo="autograd-lbfgs", out_mode=None):
        if out_mode is None:
            out_mode = self.DEFAULT_OUT_MODE

        cost_func = partial(self.get_intensity, out_mode=out_mode)

        self.f_calls = 0
        # Define initial phases as the current slm_phases
        self.slm_phases = np.exp(1j*np.zeros(self.N))

        if algo == "slsqp" or algo == "L-BFGS-B":
            initial_phases = np.zeros(self.N_pixels)

            if algo == "L-BFGS-B":
                # Configure L-BFGS-B to "try harder"
                options = {
                    'maxiter': 30000,  # default: 15000. Increase the maximum number of iterations
                    'ftol': 1e-12,  # default: 2.22e-9. Tighter function tolerance
                    'gtol': 1e-8, # default: 1e-5 Tighter gradient norm tolerance
                    'eps': 1e-8,  # default: 1e-8. Smaller step size for gradient estimation
                    # 'disp': True  # Display convergence messages
                }
            else:
                options = {}

            result = minimize(
                cost_func, initial_phases, method=algo, bounds=[(0, 2 * np.pi)] * self.N_pixels, options=options
            )
            self.slm_phases = resize_array(result.x, self.N, use_torch=False)
            intensity = cost_func(self.slm_phases)
            return intensity, result

        elif algo == "simulated_annealing":
            bounds = [(0, 2 * np.pi) for _ in range(self.N_pixels)]
            result = dual_annealing(cost_func, bounds=bounds)
            self.slm_phases = resize_array(result.x, self.N, use_torch=False)
            intensity = cost_func(self.slm_phases)
            return intensity, result

        elif algo == 'analytic':
            if self.config == 'SLM1-only-T':
                desired_out_vec = np.zeros(self.N)
                desired_out_vec[out_mode] = 1
                self.slm_phases = -np.angle(self.T.transpose() @ desired_out_vec)
                intensity = cost_func(self.slm_phases)
                return intensity, None
            elif self.config == 'SLM2-simple-OPC' or self.config == 'SLM2-simple':
                O1 = self.DEFAULT_OUT_MODE if self.config == 'SLM2-simple-OPC' else self.DEFAULT_ONEHOT_INPUT_MODE
                at_slm = (self.T[O1, :] * self.T[self.DEFAULT_OUT_MODE, :])
                self.slm_phases = -np.angle(at_slm)
                intensity = cost_func(self.slm_phases)
                return intensity, None
            else:
                return -0.1, None
        elif algo == 'autograd-adam' or algo == 'autograd-lbfgs':
            return self._autograd(out_mode=out_mode, optimizer_name=algo)
        else:
            raise ValueError("Unsupported optimization method")

    def _autograd(self, out_mode=None, optimizer_name='autograd-adam', lr=0.01):
        phases = torch.zeros(self.N_pixels, requires_grad=True, dtype=torch.float64)

        if optimizer_name.lower() == 'autograd-adam':
            optimizer = torch.optim.Adam([phases], lr=lr)
        elif optimizer_name.lower() == 'autograd-lbfgs':
            optimizer = torch.optim.LBFGS([phases], lr=lr, max_iter=20, line_search_fn="strong_wolfe")
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        N_iters = 10000
        prev_cost = float('inf')
        patience = 10  # Number of iterations to wait for improvement
        patience_counter = 0
        eps_stop = 1e-6

        dtype = torch.complex128
        self.T = torch.tensor(self.T, dtype=dtype, requires_grad=False)
        self.v_in = torch.tensor(self.v_in, dtype=dtype, requires_grad=False)

        def closure():
            optimizer.zero_grad()
            cost = self.get_intensity(phases, out_mode=out_mode, use_torch=True)
            cost.backward()
            return cost

        for i in range(N_iters):
            if optimizer_name.lower() == 'autograd-lbfgs':
                cost = optimizer.step(closure)
            else:
                cost = closure()
                optimizer.step()

            with torch.no_grad():
                # important to update the data and not create a new tensor that will be detached from the graph
                phases.data = phases.data % (2 * torch.pi)

            current_cost = cost.item()
            if abs(prev_cost - current_cost) < eps_stop:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            else:
                patience_counter = 0
            prev_cost = current_cost

        self.slm_phases = resize_array(torch.exp(1j * phases), self.N, use_torch=True).detach().numpy()
        self.reset_to_numpy()

        return -prev_cost, None

    def reset_to_numpy(self):
        """Ensure all attributes are reset to NumPy arrays."""
        self.T = self.T.detach().numpy() if isinstance(self.T, torch.Tensor) else self.T
        self.v_in = self.v_in.detach().numpy() if isinstance(self.v_in, torch.Tensor) else self.v_in
        self.slm_phases = self.slm_phases.detach().numpy() if isinstance(self.slm_phases,
                                                                         torch.Tensor) else self.slm_phases

    def statistics(self, algos, configs, T_methods, N_tries=1, saveto_path=None, verbose=False,
                   save_Ts=False, save_phases=True):
        saveto_path = saveto_path or f"C:\\temp\\{tnow()}_qwfs.npz"
        qres = QWFSResult()
        qres.configs = configs
        qres.T_methods = T_methods
        qres.N_tries = N_tries
        qres.algos = algos
        qres.sig_for_gauss_iid = self.sig_for_gauss_iid
        qres._N_modes = self.N
        qres.N_pixels = self.N_pixels

        N_algos = len(algos)
        N_configs = len(configs)
        N_T_methods = len(T_methods)

        qres.results = np.zeros((N_T_methods, N_configs, N_tries, N_algos))
        qres.tot_power_results = np.zeros((N_T_methods, N_configs, N_tries, N_algos))
        qres.best_phases = np.zeros((N_T_methods, N_configs, N_tries, N_algos, self.N))
        Ts = []
        max_SVDs = []

        if verbose:
            print(f"{'T_method':<12} {'algo':<20} {'config':<16} {'I_good':<8} {'f_calls':<8} {'T':<5}")
            print("-" * 75)
        for try_no in range(N_tries):
            print(f'{try_no=}')
            for T_method_no, T_method in enumerate(T_methods):
                self.T_method = T_method
                self.reset_T()
                Ts.append(self.T)
                max_SVD = np.linalg.svd(self.T @ self.T.transpose(), compute_uv=False).max()
                max_SVDs.append(max_SVD)
                for config_no, config in enumerate(configs):
                    self.config = config
                    for algo_no, algo in enumerate(algos):
                        start_t = time.time()
                        self.slm_phases = np.exp(1j * np.zeros(self.N, dtype=np.complex128))
                        if config == 'SLM3-same-mode' or config == 'SLM1-same-mode' or config == 'SLM2-same-mode' or config == 'SLM5-same-mode':
                            # this is the equivalent output mode after fourier to the default input of flat phase ones
                            out_mode = 0
                        else:
                            out_mode = self.DEFAULT_OUT_MODE
                        I, res = self.optimize(algo=algo, out_mode=out_mode)
                        v_out = self.propagate()
                        I_good = np.abs(v_out[out_mode]) ** 2
                        I_tot = (np.abs(v_out) ** 2).sum()
                        qres.results[T_method_no, config_no, try_no, algo_no] = I_good
                        qres.tot_power_results[T_method_no, config_no, try_no, algo_no] = I_tot
                        qres.best_phases[T_method_no, config_no, try_no, algo_no] = np.angle(self.slm_phases)
                        T = time.time()-start_t
                        if verbose:
                            print(f"{T_method:<12} {algo:<20} {config:<16} {I_good:<8.4f} {self.f_calls:<8} {T:<5.2f}")



            qres.Ts = np.array(Ts)
            qres.max_SVDs = np.array(max_SVDs)

            qres.saveto(saveto_path, save_Ts=save_Ts, save_phases=save_phases)

        return qres



def tnow():
    return datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


def resize_array(arr, desired_size, use_torch=False):
    """
    Resizes (stretches) an array/tensor to a new size deterministically, evenly distributing
    repetitions of elements to achieve the desired size.

    Parameters:
        arr (numpy.ndarray or torch.Tensor): Input array or tensor of size N.
        desired_size (int): Desired size of the output array/tensor (must be >= len(arr)).
        use_torch (bool): Whether to use PyTorch (True) or NumPy (False) operations.

    Returns:
        numpy.ndarray or torch.Tensor: Resized array/tensor of the specified size.
    """
    N = len(arr)
    assert desired_size >= N, "Desired size must be greater than or equal to the input size."

    # If sizes are equal, return the input array as is
    if desired_size == N:
        return arr

    # Determine repetitions
    factor = desired_size / N
    floor = int(factor)  # Most elements repeat this many times
    ceil = floor + 1  # Some elements repeat this many times
    extra_count = desired_size - floor * N  # Number of elements needing extra repetition

    # Create a deterministic repetition pattern
    if use_torch:
        repeat_func = torch.repeat_interleave
        repeats = torch.full((N,), floor, dtype=torch.long, device=arr.device)
    else:
        repeat_func = np.repeat
        repeats = np.full((N,), floor, dtype=int)  # like `np.ones(N) * floor` but more efficient

    for i in range(extra_count):
        repeats[i] += 1  # Assign extra repetitions deterministically from the start

    # Perform the stretching
    stretched_array = repeat_func(arr, repeats)

    return stretched_array