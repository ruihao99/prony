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

import structlog



def prony(data: TimeDomainData, nmode_real:int, nmode_imag, tol: float = 1e-8):
    log = structlog.get_logger()
    log.info("Staring the prony fitting program.")  
    log.info(f"Building the Hankel matrix and running Takagi factorization...")  
    H = Hankel(data.get_correlation_function(), data.get_n_Hankel(), tol)
    log.info(f"The Hankel process is done!")  
    log.info(f"{H}")
    log.info(f"Solving for the gamma values from the eigen values...")  
    gamma, t = get_gammas_and_t(H, nmode_real, nmode_imag)
    log.info(f"The gamma and t values are solved!")  
    log.info(f"gamma: {gamma}; t: {t}.")  
    log.info(f"Preparing the matricies for Prony optimization...")  
    C = get_gamma_matrix(H, gamma)
    d = get_correlation_function_matrix(data)
    A = -get_freq_matrix(H, data, t)
    log.info(f"Start the minimization...")  
    omega_new = optimize(C, d, A)
    log.info(f"Minimization done, now outputing decomposed correlation function.")
    
    etal = omega_new.copy()
    etar = np.conjugate(omega_new)
    etaa = np.abs(omega_new)
    expn = get_expn(data, t) 
    
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
    
    print(jw_prony) 
    print(jw_exact) 
    diff = np.abs(jw_exact - jw_prony)
    print(diff)
     
    print(expn)
    print(etal)
     