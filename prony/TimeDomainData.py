import math
import numpy as np

from collections.abc import Callable

def bose_function(energy, beta: float):
    """calculate the bose function

    Args:
        energy (float or np.ndarray): the energy
        beta (float): the inverse temperature

    Returns:
        float or np.ndarray: the bose function
    """
    return 1.0 / (1.0 - np.exp(-energy * beta))

class TimeDomainData:

    def __init__(
        spectral_function: Callable[[np.ndarray], np.ndarray],
        beta: float,
        n_Hankel: int = 2000,
        n_sample: int = 1000000,
        n_freq_in_pi: int = 3000,
        n_scale: int = 200,
    ):
    # even-space sample of frequency domain data 
    self.n_sample = n_sample
    self.omega = np.linspace(0, n_freq_in_pi)

    # estimate the sample rate:
    self.n_rate = math.floor(n_scale_fft * n_scale/ (4 * n_Hankel))





