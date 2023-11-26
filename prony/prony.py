import warnings
# Filter out RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np

from .TimeDomainData import TimeDomainData
from .Hankel import Hankel
from .fitting import (
    get_gammas_and_t,
    get_gamma_matrix, 
    get_correlation_function_matrix,
    get_freq_matrix,
    get_expn,
    optimize
)

from .spectral import BO
from .spectral import get_spectral_function_from_exponentials
from .spectral import bose_function

# for logging
import structlog
from .Timer import Timer

def prony(data: TimeDomainData, nmode_real:int, nmode_imag, tol: float = 1e-8):
    log = structlog.get_logger()
    timer = Timer()
    
    log.info("Staring the prony fitting program.")  
    log.info(f"Building the Hankel matrix and running Takagi factorization...")  
    
    timer.start() 
    H = Hankel(data.get_correlation_function(), data.get_n_Hankel(), tol)
    time_H = timer.stop()
    
    log.info(f"The Hankel process is done!")  
    log.info(f"{H}")
    log.info(f"Solving for the gamma values from the eigen values...")  
    
    timer.start() 
    gamma, t = get_gammas_and_t(H, nmode_real, nmode_imag)
    time_roots = timer.stop()
    
    log.info(f"The gamma and t values are solved!")  
    log.info(f"gamma: {gamma}; t: {t}.")  
    log.info(f"Preparing the matricies for Prony optimization...")  
    C = get_gamma_matrix(H, gamma)
    d = get_correlation_function_matrix(data)
    A = -get_freq_matrix(H, data, t)
    log.info(f"Start the minimization...")  
    log.info(f"The dimension for the Q matrix is ({C.size[1]}, {C.size[1]})")  
    
    timer.start()  
    omega_new = optimize(C, d, A)
    time_opt = timer.stop()
    
    log.info(f"Minimization done, now outputing decomposed correlation function.")
    
    etal = omega_new.copy()
    etar = np.conjugate(omega_new)
    etaa = np.abs(omega_new)
    expn = get_expn(data, t) 
    log.info(
        f"""
        Time elapsed
        - Hankel matrix and Takagi factorization: {time_H : .4f} seconds.
        - calculate the gamma and t by finding the polynomials roots: {time_roots : .4f} seconds.
        - The quadratic optimization: {time_opt : .4f} seconds.
        """
    )
    
    return expn, etal
    
if __name__ == "__main__":
    beta = 1.0
    data = TimeDomainData(BO, beta, tf=50, n_Hankel=500)
    
    nmode_real = 1
    nmode_imag = 1
    expn, etal = prony(data, nmode_real, nmode_imag)
    
    len_ = 10000
    spe_wid = 200
    
    w = np.append(np.linspace(-spe_wid, 0, len_), np.linspace(0, spe_wid, len_))
    jw_exact = BO(w) * bose_function(w, beta)
    jw_prony = get_spectral_function_from_exponentials(w, expn, etal).real
    
    diff = np.abs(jw_exact - jw_prony)