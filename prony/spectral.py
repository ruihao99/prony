import numpy as np

def bose_function(energy, beta: float, mu: float=0.0):
    """calculate the bose function

    Args:
        energy (float or np.ndarray): the energy
        beta (float): the inverse temperature
        mu (float): the chemical potential. defaults to 0.

    Returns:
        float or np.ndarray: the bose function
    """
    return 1.0 / (1.0 - np.exp(-(energy - mu) * beta))

def fermi_function(energy, beta: float, mu: float=0.0):
    """calculate the fermi function

    Args:
        energy (float or np.ndarray): the energy
        beta (float): the inverse temperature
        mu (float): the chemical potential. defaults to 0.

    Returns:
        float or np.ndarray: the fermi function
    """
    return 1.0 / (1.0 + np.exp((energy - mu) * beta))

def get_spectral_function_from_exponentials(w, expn, etal):
    res = np.zeros_like(w, dtype=complex)
    for i in range(len(etal)):
        res += etal[i] / (expn[i] - 1.j * w)
    return res

# spectral functions

def BO(w, lams=1.0, zeta=1.0, omega_B=1.0):
    """The brownian oscillator spectral function

    Args:
        w (np.ndarray): the frequence
        lams (float, optional) The interaction strength lambda. Defaults to 1.0.
        zeta (float, optional): The friction of an brownian motion oscillator. Defaults to 1.0.
        omega_B (float, optional): The frequency of the solvation mode. Defaults to 1.0.

    Returns:
        np.ndarray: the brownian motion spectral function
    """
    return 2.0 * lams * w * omega_B**2 / ((w*zeta)**2 + (w**2 - omega_B**2)**2)


