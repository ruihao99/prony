import math
import numpy as np
import matplotlib.pyplot as plt

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
    """Curiates the time domain data -- the numerical input for Prony fitting
    
    This class generates and organizes the time domain correlation function from any analytical spectral function.
    The must need input are:
    - spectral_function: function object for the spectral function 
    - beta: the inverse temperature of the bath
    """

    def __init__(
        self,
        spectral_function: Callable[[np.ndarray], np.ndarray],
        beta: float,
        tf: float = 200.0,
        n_Hankel: int = 2000,
        n_sample: int = 1000000,
        max_freq_in_pi: int = 3000,
    ):
        """Initilize an TimeDomainData object

        Args:
            spectral_function (Callable[[np.ndarray], np.ndarray]): the spectral function
            beta (float): the inverse temperature
            tf (float, optional): time cutoff for the correlation function sampling. Defaults to 200.0.
            n_Hankel (int, optional): Dimension of the Hankel matrix. Used here to estimate the sample rate. See class Hankel for details. Defaults to 2000.
            n_sample (int, optional): number of evenly spaced frequency samples for the spectral function. Defaults to 1000000.
            max_freq_in_pi (int, optional): the maxium frequency to be considered in the spectral function. Defaults to 3000.
        """
        # even-space sample of frequency domain data 
        self.n_sample = n_sample
        self.max_freq_in_pi = max_freq_in_pi
        self.beta = beta
        self.tf = tf
        self.n_Hankel = n_Hankel
        
        # discrete sample of the spectral function 
        self.spectral = self.get_freq_domain_spectral(spectral_function)
        
        # time domain correlation function data 
        self.correlation_function = self.get_time_domain_correlation_function(*self.spectral)
        
    def __str__(self):
        return f"<TimeDomainData t0=0, tf={self.correlation_function[0][-1]}, beta={self.beta}>"
    
    def __repr__(self):
        return str(self) 
    
    def get_freq_domain_spectral(self, spectral_function: Callable[[np.ndarray], np.ndarray]):
        """generate the spectral function samples

        Args:
            spectral_function (Callable[[np.ndarray], np.ndarray]): analytical spectral functions to be decomposed

        Returns:
            (np.ndarray, np.ndarray): tuple of frequencies and corresponding spectral function
        """
        w = np.linspace(0, self.max_freq_in_pi*np.pi, self.n_sample+1)[:-1]
        jw = spectral_function(w)
        
        return (w, jw)
    
    def get_time_domain_correlation_function(self, w, jw):
        """generate time domain correlation function

        Args:
            w (np.ndarray): the frequency samples of the spectral function
            jw (np.ndarrya): the spectral function values corresponds to w

        Returns:
            (np.ndarray, np.ndarray): tuple of time and correlation function
        """
        dw = w[1] - w[0]
        # correlation function in freq domain 
        cw_pos = jw * bose_function(+w, self.beta)
        cw_neg = jw * bose_function(-w, self.beta)
        # fix zero frequncy 
        cw_pos[0] = cw_pos[1]/2
        cw_neg[0] = cw_neg[1]/2
        
        # use Fourier transform to obtain time domain data
        ct = dw / np.pi * (np.fft.fft(cw_pos) - np.fft.ifft(cw_neg)*len(cw_neg))
        time = 2.0 * np.pi * np.fft.fftfreq(len(ct), dw)
        
        # estimate the sample rate for the time domain correlation function
        # and sample the data evenly
        n_rate = math.floor(self.max_freq_in_pi * self.tf/ (4 * self.n_Hankel))
        non_negative_time_mask = (time <= self.tf) & (time >= 0)
        
        time = time[non_negative_time_mask][::n_rate]
        ct = ct[non_negative_time_mask][::n_rate]
        
        return time, ct
    
    def plot(self):
        """quick plot for the correlation function.
        
        You shall use this function to check whether the the `tf` parameter you have is long enough to ensure C(t) = 0 in the long time limit. 
        """
        self.fig = self.plot_correlation_function(*self.correlation_function)
        
    def get_correlation_function(self):
        return self.correlation_function[1]

    def get_time(self):
        return self.correlation_function[0]

    def get_n_Hankel(self):
        return self.n_Hankel
    
    def get_tf(self):
        return self.tf
    
    @staticmethod
    def plot_correlation_function(t, ct):
        """static helper method to plot the correlation function with legends

        Args:
            t (np.ndarray): the time
            ct (np.ndarray): the correlation function

        Returns:
            matplotlib.figure.Figure: the figure object for the correlation function
        """
        t, ct = td_data.correlation_function
        fig = plt.figure(dpi=200)
        ax = fig.gca()

        ax.plot(t, np.imag(ct), label=r"$\Im$")
        ax.plot(t, np.real(ct), label=r"$\Re$")
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel(r"C(t)")
        ax.set_xlim(-0.1, 50)
        return fig
         