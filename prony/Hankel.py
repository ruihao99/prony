import numpy as np
import scipy.linalg as LA

class Hankel:
    def __init__(self, correlation_function: np.ndarray, n_Hankel: int=2000, tol: float=1e-8):

        if not self._is_valid_sample(correlation_function, n_Hankel):
            raise ValueError(f"You dimension for Hankel matrix is {n_Hankel}, which is too large for a correlation function of {len(correlation_function)} sampled points.")

        self.dim = n_Hankel
        self.H_real, self.H_imag = self.get_Hankel(correlation_function, n_Hankel)
        # perform the Takagi factorization
        self.factorize(tol)
        
    def factorize(self, tol: float):
        self.abs_evals_of_H_real, self.Qp_of_H_real = self.Takagi_factorization(self.H_real)
        self.error_real = self.get_error_of_Takagi_factorization(self.H_real, self.abs_evals_of_H_real, self.Qp_of_H_real)
        if self.error_real >= tol:
            raise ValueError(f"The Takagi factorization of the real Hankel matrix has error {error_real} > the tolerance value {tol}.")
        self.abs_evals_of_H_imag, self.Qp_of_H_imag = self.Takagi_factorization(self.H_imag)
        self.error_imag = self.get_error_of_Takagi_factorization(self.H_imag, self.abs_evals_of_H_imag, self.Qp_of_H_imag)
        if self.error_imag >= tol:
            raise ValueError(f"The Takagi factorization of the real Hankel matrix has error {error_real} > the tolerance value {tol}.")
        
    def __str__(self):
        return f"<Hankel dim={self.dim}, with Takagi factorization error (real: {self.error_real}, imag: {self.error_imag})>"

    def __repr__(self):
        return str(self)
    
    @staticmethod
    def get_Hankel(correlation_function, n_Hankel):
        """genetrate Hankel function from the correlation function

        Args:
            correlation_function (np.ndarray): the correlation function samples
            n_Hankel (int): dimension of the Hankel matrix

        Returns:
            tuple(np.ndarray, np.ndarray): a tuple of the real and imaginary Hankel matricies
        """
        H_real = np.zeros((n_Hankel, n_Hankel))
        H_imag = np.zeros((n_Hankel, n_Hankel))

        for i in range(n_Hankel):
            H_real[i, :] = np.real(correlation_function[i:n_Hankel+i])
            H_imag[i, :] = np.imag(correlation_function[i:n_Hankel+i])

        return (H_real, H_imag)
    
    @staticmethod
    def Takagi_factorization(M_hankel):
        """The Takagi factorization of the Hankel matricies

        Args:
            M_hankel (np.ndarray): the Hankel matrix

        Returns:
            tuple(np.ndarray, np.ndarray): tuple containing the sorted eigen values by their absolute values and the corresponding eigen vectors
        """
        evals, evecs = LA.eigh(M_hankel)

        # construct "phase-decorated" eigen vectors 
        phase_mat = np.diag([np.exp(-1j * np.angle(sing_v_r)/2.0) for sing_v_r in evals])
        Qp = np.matmul(evecs, phase_mat)

        # get the absolute value of eigen values
        abs_evals = np.abs(evals)

        # sorting absevals and Qp according to the absolute eigenvlaues
        argsort = np.flip(np.argsort(abs_evals))

        abs_evals = abs_evals[argsort]
        Qp = Qp[:, argsort]
        return abs_evals, Qp
        
    
    @staticmethod
    def _is_valid_sample(correlation_function, nsample):
        ndata = len(correlation_function)

        if 2.0*nsample > ndata:
            return False

        return True
    
    @staticmethod
    def get_error_of_Takagi_factorization(H, abs_evals, Q):
        """calculate the error of the Takagi factorization

        Args:
            H (np.ndarray): The Hankel matrix
            abs_evals (np.ndarray): The absolute values for the Hankel matrix eigen values
            Q (np.ndarray): The eigen vectors associated with abs_evals

        Returns:
            float: the error
        """
        H_from_Takagi = np.matmul(np.matmul(Q, np.diag(abs_evals)), np.transpose(Q))
        return np.sum(np.abs(H - H_from_Takagi))
    