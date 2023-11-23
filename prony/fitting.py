import numpy as np
from scipy import sparse
from cvxopt import solvers, matrix, spmatrix, mul

from .Hankel import Hankel
from .TimeDomainData import TimeDomainData

def numpy_to_cvxopt_matrix(A):
    if A is None:
        return A
    if sparse.issparse(A):
        if isinstance(A, sparse.spmatrix):
            return scipy_sparse_to_spmatrix(A)
        else:
            return A
    else:
        if isinstance(A, np.ndarray):
            if A.ndim == 1:
                return matrix(A, (A.shape[0], 1), 'd')
            else:
                return matrix(A, A.shape, 'd')
        else:
            return A

def get_gammas(Qp: np.ndarray, n_gamma: int):
    """calculate the gamma values from the eigen vectors from Takagi factorization

    Args:
        Qp (np.ndarray): eigen vectors from Takagi factorization
        n_gamma (int): number of gammas

    Returns:
        np.ndarray: n_gamma complex values
    """
    # solve the roots of polynomial f(z):
    gamma = np.roots(np.flip(Qp[:, n_gamma]))
    argsort = np.argsort(np.abs(gamma))
    gamma_out = gamma[argsort][:n_gamma]
    return gamma_out

def get_gamma_matrix(H: Hankel, gamma: np.ndarray):
    n_col = 2 * H.dim + 1
    n_row = len(gamma)
    gamma_matrix = np.zeros((2 * n_col, 2 * n_row), dtype=float)
    
    for i in range(n_row):
        for j in range(n_col):
            gamma_matrix[j, i] = np.real(gamma[i]**j)
            gamma_matrix[n_col + j, n_row + i] = np.real(gamma[i]**j)
            gamma_matrix[j, n_row + i] = -np.imag(gamma[i]**j)
            gamma_matrix[n_col + j, i] = np.imag(gamma[i]**j)
    return numpy_to_cvxopt_matrix(gamma_matrix)

def get_correlation_function_matrix(data: TimeDomainData):
    ct = data.get_correlation_function()
    h_matrix = np.append(ct.real, ct.imag)
    return numpy_to_cvxopt_matrix(h_matrix)

def get_expn(data: TimeDomainData, t: np.array):
    return -t / data.get_tf()

def get_freq_matrix(H: Hankel, data: TimeDomainData, t: np.array):
    n_col = 2 * H.dim + 1
    n_row = len(t)
    
    hi_freq_left = np.linspace(-10000, 10, n_col//2)
    lo_freq = np.linspace(-10, 10, n_col + 1)
    hi_freq_right = np.linspace(10, 10000, n_col//2)
    freq_d = np.concatenate([hi_freq_left, lo_freq, hi_freq_right])
    
    expn = get_expn(data, t)
    freq_m = np.zeros((2 * n_col, 2 * n_row), dtype=float)
    for i in range(n_row):
        for j in range(2 * n_col):
            freq_m[j, i] = np.real(expn[i]) / (np.real(expn[i])**2 + (np.imag(expn[i]) - freq_d[j])**2)
            freq_m[j, n_row + i] = (np.imag(expn[i]) - freq_d[j]) / (np.real(expn[i])**2 + (np.imag(expn[i]) - freq_d[j])**2) 
            
    return numpy_to_cvxopt_matrix(freq_m)
     

def get_gammas_and_t(H: Hankel, n_gamma_real: int, n_gamma_imag: int):
    # get the numerical significant roots
    gamma_real = get_gammas(H.Qp_of_H_real, n_gamma_real)
    gamma_imag = get_gammas(H.Qp_of_H_imag, n_gamma_imag)
    
    # calculate the t values 
    t_real = 2.0 * H.dim * np.log(gamma_real) 
    t_imag = 2.0 * H.dim * np.log(gamma_imag) 
    
    # calculate exponents
    gamma = np.append(gamma_real, gamma_imag)
    t = np.append(t_real, t_imag)
    
    return gamma, t

def optimize(C, d, A):
    b = numpy_to_cvxopt_matrix(np.zeros(C.size[0]))
    Q = C.T * C
    q = - d.T * C
    
    opts = {'show_progress': True, 'abstol': 1e-24, 'reltol': 1e-24, 'feastol': 1e-24}
    for k, v in opts.items():
        solvers.options[k] = v
    sol = solvers.qp(Q, q.T, A, b, None, None, None, None)
    
    n_gamma = C.size[1] // 2
    omega_new_temp = np.array(sol['x']).reshape(2, n_gamma) 
    omega_new = omega_new_temp[0,:] + 1.j*omega_new_temp[1,:]
    
    return omega_new 