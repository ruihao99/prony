import numpy as np
from scipy import integrate
# from .spectral import BO
from prony.spectral import BO

import structlog
from typing import Callable, Union

def get_BO_jw(
    lambd: float,
    zeta: float,
    omega_B: float
) -> Callable:
    return lambda w: BO(w, lams=lambd, zeta=zeta, omega_B=omega_B)

def linear_golden_rule_rate(
    jw: Callable,
    Delta: float,
    beta: float,
) -> float:
    return jw(Delta) * 2.0 / np.tanh(0.5 * Delta * beta)

def quadratic_golden_rule_rate(
    jw: Callable,
    Delta: float,
    beta: float,
) -> float:

    def integrand_01(w):
        return 4.0 / np.pi * jw(w) / (1 - np.exp(-beta*w)) * jw(w-Delta) / (np.exp(beta*(w-Delta)) - 1)

    def integrand_10(w):
        return 4.0 / np.pi * jw(w) / (1 - np.exp(-beta*w)) * jw(w+Delta) / (np.exp(beta*(w+Delta)) - 1)

    def foo(func, tol0=1e-10):
        result_pos, error_pos = integrate.quad(lambda x: func(x), -np.inf, -tol0, epsabs=1e-16, epsrel=1e-16)
        result_neg, error_neg = integrate.quad(lambda x: func(x), +tol0, +np.inf, epsabs=1e-16, epsrel=1e-16)
        return result_pos + result_neg, max(error_pos, error_neg)

    k_10, error_k10 = foo(integrand_10)
    k_01, error_k01 = foo(integrand_01)

    k = k_10 + k_01
    error = max(error_k10, error_k01)
    log = structlog.get_logger()
    log.info(f"Gauss Quadrature integraiton is {error}.")

    return k

def golden_rule_rates(
    jw: Callable,
    Delta: float,
    beta: Union[float, np.ndarray, list],
    interaction: str = 'linear',
) -> Union[float, np.ndarray]:
    if interaction == 'linear':
        return linear_golden_rule_rate(jw, Delta, beta)
    elif interaction == 'quadratic':
        if isinstance(beta, float):
            return quadratic_golden_rule_rate(jw, Delta, beta)
        elif isinstance(beta, (np.ndarray, list)):
            return np.array([quadratic_golden_rule_rate(jw, Delta, b) for b in beta])
        else:
            raise ValueError(f"Input 'beta' should be either a float scalar or an array, but you have {type(beta)}.")
    else:
        raise ValueError(f"Input 'interaction' should be either 'linear' or 'quadratic', but you have {interaction}.")

    return None


def linspace_log(lb, ub, n):
    out = np.linspace(np.log(lb), np.log(ub), n)
    return np.exp(out)

# def test():
#     # params
#     Delta = 1
#     λ = 0.001
#     OmegaB = 10
#     zeta = 0.3
#
#     T = linspace_log(0.3, 30, 100)
#     jw = get_BO_jw(lambd=λ, zeta=zeta, omega_B=OmegaB)
#
#     # k = np.array(list(map(lambda beta: quadratic_golden_rule_rate(jw, Delta, beta), 1.0/T)))
#     k_q = golden_rule_rates(jw, Delta, 1.0/T, 'quadratic')
#     k_l = golden_rule_rates(jw, Delta, 1.0/T, 'linear')
#
#     np.savetxt("rate.dat", np.array([T, k_l, k_q]).T)

# test()
